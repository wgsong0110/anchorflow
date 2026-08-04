"""Implicit (Newmark) time stepping for the anchor elastodynamics, and the
within-step momentum residual used as a learning signal.

Follows i-PhysGaussian (arXiv 2602.17117) with its background grid replaced by
our anchor set. That paper defines a within-step momentum residual as a
function of the node displacement increment du over [t_n, t_{n+1}]:

    a_{n+1}(du) = (du - dt*v_n - dt^2*(1/2 - beta)*a_n) / (beta*dt^2)      (8)
    R(du)       = f_ext(du) + f_int(du) - m * a_{n+1}(du)                  (10)

and drives R -> 0 with Newton-GMRES. The point of this module is that the SAME
residual is exactly the negative gradient of an incremental potential:

    L(du) = Psi_inertia(du) + E_elastic(p_n + du) - <f_ext, du>
    Psi_inertia(du) = m/(2*beta*dt^2) * || du - dt*v_n - dt^2*(1/2-beta)*a_n ||^2

    dL/ddu = m*a_{n+1}(du) - f_int(du) - f_ext = -R(du)

because f_int = -dE/dp, which the fused CUDA kernel already returns
analytically. So L is directly differentiable w.r.t. du at the cost of one
physics step (~0.37 ms), with no Hessian and no autograd graph through the
solver -- which is what makes it usable as a training loss for a network that
predicts du in one shot instead of Newton-iterating.

Newmark parameters default to beta=1/4, gamma=1/2 (average-acceleration /
trapezoidal rule, unconditionally stable) -- the paper states a Newmark family
parameterized by (beta, gamma) but does not print the values it used.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from anchorstep import fused_energy_force
from .dynamics import mlp, MPNNLayer
from . import graph as G


class _ElasticEnergy(torch.autograd.Function):
    """E_elastic(anchor_pos) with the analytic dE/dp = -f_elastic from the
    fused kernel, so gradients reach whatever produced anchor_pos."""

    @staticmethod
    def forward(ctx, anchor_pos, sim, gaussian_pos_prev, volume, mu, lam):
        f, pos, F, psi = fused_energy_force(
            sim.gaussian_canonical, gaussian_pos_prev, anchor_pos.detach(),
            sim.anchor_canonical, sim.nn_idx, volume, sim.radius, mu, lam)
        ctx.save_for_backward(f)
        ctx.extra = (pos, F)
        return (volume * psi).sum()

    @staticmethod
    def backward(ctx, grad_out):
        (f,) = ctx.saved_tensors
        return grad_out * (-f), None, None, None, None, None


def elastic_energy(anchor_pos, sim, gaussian_pos_prev, volume, mu, lam):
    return _ElasticEnergy.apply(anchor_pos, sim, gaussian_pos_prev, volume, mu, lam)


def newmark_predictor(v_n, a_n, dt, beta=0.25):
    """The du-independent part of eq. (8): dt*v_n + dt^2*(1/2 - beta)*a_n.
    du = this  =>  a_{n+1} = 0 (pure inertial coast)."""
    return dt * v_n + (dt ** 2) * (0.5 - beta) * a_n


def newmark_accel(du, v_n, a_n, dt, beta=0.25):
    """eq. (8): end-of-step acceleration implied by a displacement increment."""
    return (du - newmark_predictor(v_n, a_n, dt, beta)) / (beta * dt ** 2)


def newmark_velocity(a_next, v_n, a_n, dt, gamma=0.5):
    """eq. (9)."""
    return v_n + dt * ((1.0 - gamma) * a_n + gamma * a_next)


def incremental_potential(du, sim, anchor_pos_n, gaussian_pos_prev, volume,
                           mu, lam, mass, v_n, a_n, dt, f_ext=None,
                           beta=0.25, fixed_mask=None):
    """L(du); its gradient w.r.t. du is exactly -R(du) (see module docstring).

    Returns (L, R_detached) so a caller can both backprop L and monitor the
    residual it corresponds to without a second physics evaluation."""
    pred = newmark_predictor(v_n, a_n, dt, beta)
    a_next = (du - pred) / (beta * dt ** 2)
    inertia = 0.5 * (mass.unsqueeze(-1) * (du - pred) * a_next).sum()
    E = elastic_energy(anchor_pos_n + du, sim, gaussian_pos_prev, volume, mu, lam)
    L = inertia + E
    if f_ext is not None:
        L = L - (f_ext * du).sum()
    with torch.no_grad():
        f_int, _, _, _ = fused_energy_force(
            sim.gaussian_canonical, gaussian_pos_prev, (anchor_pos_n + du).detach(),
            sim.anchor_canonical, sim.nn_idx, volume, sim.radius, mu, lam)
        R = f_int - mass.unsqueeze(-1) * a_next
        if f_ext is not None:
            R = R + f_ext
        if fixed_mask is not None:
            R = torch.where(fixed_mask.unsqueeze(-1), torch.zeros_like(R), R)
    return L, R


def newton_reference(sim, anchor_pos_n, gaussian_pos_prev, volume, mu, lam,
                      mass, v_n, a_n, dt, f_ext=None, beta=0.25,
                      fixed_mask=None, iters=25, lr=1.0):
    """Gradient-descent reference solve of the same L (a stand-in for the
    paper's Newton-GMRES, which needs the Hessian we do not have). Used only
    to report how far a one-shot prediction is from an iterated solution."""
    du = torch.zeros_like(anchor_pos_n, requires_grad=True)
    opt = torch.optim.LBFGS([du], max_iter=iters, line_search_fn="strong_wolfe", lr=lr)

    def closure():
        opt.zero_grad()
        L, _ = incremental_potential(du, sim, anchor_pos_n, gaussian_pos_prev,
                                      volume, mu, lam, mass, v_n, a_n, dt,
                                      f_ext, beta, fixed_mask)
        L.backward()
        if fixed_mask is not None and du.grad is not None:
            du.grad = torch.where(fixed_mask.unsqueeze(-1), torch.zeros_like(du.grad), du.grad)
        return L

    opt.step(closure)
    with torch.no_grad():
        if fixed_mask is not None:
            du.data = torch.where(fixed_mask.unsqueeze(-1), torch.zeros_like(du), du)
    return du.detach()


def anchor_elastic_accel(sim, anchor_pos, gaussian_pos_prev, volume, mu, lam,
                          mass, fixed_mask=None):
    """f_int(p)/m per anchor -- the physical acceleration the current
    configuration is producing, one fused-kernel call (~0.1 ms).

    Needed as a network INPUT because the `a` the two settings carry are not the
    same quantity. In training, states come from the explicit simulator and
    a = (v_{n+1}-v_n)/dt IS f/m, so the elastic force is handed to the network
    for free -- and a network can then reach a low residual by the shortcut
    a_{n+1} ~ a_n, since R = f - m*a_n is then nearly zero without solving
    anything. In an autoregressive rollout a is instead Newmark's readback of
    the network's own du, which carries no force information at all and starts
    at exactly zero. Measured on a trained network: the correction it emits is
    the same size in both settings (|corr| 4.0e-4 vs 4.4e-4) yet the residual
    ratio is 0.09-0.58 on simulator states and 0.96-1.04 on its own -- and at
    frame 0, where a = 0, it under-corrects by 3x. Feeding the force explicitly
    removes both the shortcut and the mismatch.
    """
    f, _, _, _ = fused_energy_force(
        sim.gaussian_canonical, gaussian_pos_prev, anchor_pos.detach(),
        sim.anchor_canonical, sim.nn_idx, volume, sim.radius, mu, lam)
    a = f / mass.unsqueeze(-1)
    if fixed_mask is not None:
        a = torch.where(fixed_mask.unsqueeze(-1), torch.zeros_like(a), a)
    return a


class DuCorrector(nn.Module):
    """(v, a, dt) per anchor + anchor graph -> a DIMENSIONLESS per-anchor
    correction to the Newmark inertial predictor. See predict_du for how it
    becomes a displacement.

    The decoder is zero-init, so at the start of training du equals the
    predictor exactly -- i.e. the network begins as a no-op on top of an
    explicit step, and the first gradient it receives is the elastic force
    (see incremental_potential above). Lives in the library rather than in
    the training script so the rollout script can import it without running
    the trainer's argparse.
    """

    def __init__(self, hidden=128, mp_steps=4, edge_in=4, use_force=True):
        super().__init__()
        self.use_force = use_force
        # v, a, [f_int/m], [dt, log10 dt]; a and the force are divided by the
        # scene's acceleration scale by the caller so all three are O(1)
        self.node_enc = mlp([3 + 3 + (3 if use_force else 0) + 2, hidden, hidden])
        self.edge_enc = mlp([edge_in, hidden, hidden])
        self.proc = nn.ModuleList(MPNNLayer(hidden) for _ in range(mp_steps))
        self.dec = mlp([hidden, hidden, 3], layernorm=False)
        last = [m for m in self.dec.modules() if isinstance(m, nn.Linear)][-1]
        nn.init.zeros_(last.weight); nn.init.zeros_(last.bias)

    def forward(self, p, v, a, dt, edge_index, f_accel=None):
        t = torch.tensor([[dt, math.log10(dt)]], device=p.device).expand(p.shape[0], -1)
        feats = [v, a, t] if not self.use_force else [v, a, f_accel, t]
        h = self.node_enc(torch.cat(feats, -1))
        e = self.edge_enc(G.edge_features(p, edge_index))
        for layer in self.proc:
            h, e = layer(h, edge_index, e)
        return self.dec(h)


def predict_du(net, p, v, a, dt, edge_index, accel_scale, beta=0.25, fixed_mask=None,
                f_accel=None, direct=False):
    """du = predictor + beta*dt^2 * accel_scale * net(...), i.e. the network
    corrects the displacement the anchors' own velocity and acceleration
    predict, with a gain that matches how a real correction scales.

    The dt^2 is not decorative: whatever du finally is, Newmark's eq. (8) reads
    it back as an end-of-step acceleration a_{n+1} = (du - predictor)/(beta
    dt^2), so a correction of physical size a enters the displacement as
    beta*dt^2*a. Any other power of dt makes the gain wrong by dt^k at every
    step size but one. Two alternatives were measured and both are worse:

      dt * vel_scale * net -- one power of dt short. At dt = 1e-4 the gain is
        6e-5 where the true correction is ~4e-7, so the network's output is
        multiplied by ~150x too much and its noise swamps the signal: swept
        over dt, the trained network was 4.9x WORSE than not correcting at all
        at 1x, 3.0x worse at 2x, and only became useful past 20x.

      |predictor| * net -- dimensionless, but the gain then grows with the
        state itself. Measured: the autoregressive rollout went exponential and
        hit NaN by frame 40, where a constant gain merely drifted linearly. A
        correction that scales with the error it is correcting is a positive
        feedback loop.

    accel_scale is a CONSTANT characteristic anchor acceleration measured once
    from an explicit rollout. It leaves the dt^2 scaling intact (so one network
    transfers across step sizes and the [dt, log dt] node feature has something
    real to condition on) while putting the output in an O(1) range -- with
    accel_scale = 1, as an earlier version effectively had, the network has to
    learn to emit ~1e2 first.
    """
    pred = newmark_predictor(v, a, dt, beta)
    fa = None if f_accel is None else f_accel / accel_scale
    out = net(p, v, a / accel_scale, dt, edge_index, fa)
    # direct=True drops the predictor from the output path and asks the network
    # for the whole displacement. The two are equally expressive -- emitting
    # -pred/(beta dt^2 A) cancels the predictor -- so this only changes the
    # conditioning and the starting point, and it gives up the zero-init
    # guarantee that training begins at an explicit step. Measured on real
    # states the true du sits ~13% away from the predictor, so the residual
    # form is asking for the small part; kept as an ablation.
    du = (beta * dt * dt * accel_scale) * out if direct else \
        pred + (beta * dt * dt * accel_scale) * out
    if fixed_mask is not None:
        du = torch.where(fixed_mask.unsqueeze(-1), torch.zeros_like(du), du)
        pred = torch.where(fixed_mask.unsqueeze(-1), torch.zeros_like(pred), pred)
    return du, pred

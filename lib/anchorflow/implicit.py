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


class DtFiLM(nn.Module):
    """Fourier-encoded step size -> per-layer (gamma, beta) modulation.

    dt is one scalar shared by every anchor, so carrying it as a node feature
    wastes three of the encoder's input channels on a constant and lets it be
    swamped by the per-anchor signal. FiLM is the right shape for a global
    conditioning variable: it rescales and shifts every channel of the residual
    stream at every layer, so the step size can change what the processor
    computes rather than just being another number in the state.

    log10(dt) is what gets encoded, not dt: the step sizes of interest span 1e-4
    to 1.6e-2, more than two decades, so a linear input would give the small
    steps no resolution. The sin/cos octaves on top of the raw value give the
    network sharp features at several scales instead of one smooth ramp.

    The output layer is zero-init so gamma = 1, beta = 0 at the start -- the
    modulation begins as the identity and cannot disturb early training.
    """

    def __init__(self, hidden, n_sites, n_freq=6):
        super().__init__()
        self.n_freq = n_freq
        self.hidden = hidden
        self.n_sites = n_sites
        self.mlp = mlp([1 + 2 * n_freq, hidden, 2 * hidden * n_sites], layernorm=False)
        last = [m for m in self.mlp.modules() if isinstance(m, nn.Linear)][-1]
        nn.init.zeros_(last.weight); nn.init.zeros_(last.bias)

    def forward(self, dt, device):
        x = torch.tensor([math.log10(dt)], device=device)
        f = 2.0 ** torch.arange(self.n_freq, device=device) * math.pi
        enc = torch.cat([x, torch.sin(f * x), torch.cos(f * x)])
        out = self.mlp(enc).view(self.n_sites, 2, self.hidden)
        return 1.0 + out[:, 0], out[:, 1]        # gamma, beta -- each [n_sites, hidden]


class GeoAttentionBias(nn.Module):
    """Relative-geometry bias added to every attention logit.

    Plain attention is permutation-invariant and knows nothing about where the
    anchors are; the message-passing processor got its geometry from the edge
    features of the k-NN graph. Here the same relative geometry (offset +
    distance, so translation-invariant) is turned into a per-head scalar bias
    for EVERY pair, which is what lets attention be geometric without giving it
    absolute coordinates. Distances are divided by the batch's own mean pair
    distance so the bias is scale-free as well.

    At M = 512 this is a [512, 512, 4] tensor -- 1M floats, ~4 MB.
    """

    def __init__(self, heads, hidden=32):
        super().__init__()
        self.mlp = mlp([4, hidden, heads], layernorm=False)

    def forward(self, p):
        rel = p.unsqueeze(1) - p.unsqueeze(0)                  # [M,M,3]
        d = rel.norm(dim=-1, keepdim=True)                     # [M,M,1]
        s = d.mean().detach().clamp(min=1e-12)
        b = self.mlp(torch.cat([rel / s, torch.log1p(d / s)], -1))
        return b.permute(2, 0, 1)                              # [H,M,M]


class GeoAttentionBlock(nn.Module):
    """Pre-norm transformer block over the anchor set."""

    def __init__(self, hidden, heads=4):
        super().__init__()
        assert hidden % heads == 0
        self.h = heads
        self.d = hidden // heads
        self.n1 = nn.LayerNorm(hidden)
        self.n2 = nn.LayerNorm(hidden)
        self.qkv = nn.Linear(hidden, 3 * hidden)
        self.proj = nn.Linear(hidden, hidden)
        self.ffn = mlp([hidden, 2 * hidden, hidden], layernorm=False)

    def forward(self, x, bias):
        M = x.shape[0]
        q, k, v = self.qkv(self.n1(x)).chunk(3, dim=-1)
        q, k, v = (t.view(M, self.h, self.d).transpose(0, 1) for t in (q, k, v))
        att = (q @ k.transpose(-1, -2)) / math.sqrt(self.d) + bias
        out = (att.softmax(-1) @ v).transpose(0, 1).reshape(M, -1)
        x = x + self.proj(out)
        return x + self.ffn(self.n2(x))


class DuCorrector(nn.Module):
    """(v, a, dt) per anchor + anchor graph -> a DIMENSIONLESS per-anchor
    correction to the Newmark inertial predictor. See predict_du for how it
    becomes a displacement.

    processor="attention" replaces the message-passing stack with full
    self-attention over the anchors. One Newton step of the implicit solve is
    (M/(beta dt^2) + K) delta = -R, whose solution operator is a DENSE global
    inverse -- pinning the pot changes the branch tips in the same step. k-hop
    message passing is a local operator: with k=8 neighbours and 4 rounds the
    receptive field cannot reach across the plant, so the function being asked
    for is not representable, which fits the measured pattern (it fits states
    along the training trajectory and contributes nothing anywhere else).
    Attention makes every anchor see every other in a single layer, and at
    M = 512 the full 512x512 map costs nothing.

    The decoder is zero-init, so at the start of training du equals the
    predictor exactly -- i.e. the network begins as a no-op on top of an
    explicit step, and the first gradient it receives is the elastic force
    (see incremental_potential above). Lives in the library rather than in
    the training script so the rollout script can import it without running
    the trainer's argparse.
    """

    def __init__(self, hidden=128, mp_steps=4, edge_in=4, use_force=True,
                  processor="attention", heads=4, raw_io=False):
        super().__init__()
        self.use_force = use_force and not raw_io
        self.processor = processor
        self.raw_io = raw_io
        # dt is no longer a node feature anywhere: it is a global scalar and
        # enters through FiLM instead (see DtFiLM).
        if raw_io:
            # the stripped-down model: (position, velocity, acceleration) in,
            # next position out. No force feature, no predictor in the output
            # path -- so the decoder canNOT be zero-init (a zero output is the
            # origin, i.e. every anchor collapsing to a point), which gives up
            # the "training starts at an explicit step" property the other
            # parameterisations have.
            self.node_enc = mlp([3 + 3 + 3, hidden, hidden])
        else:
            # v, a, [f_int/m]; a and the force are divided by the scene's
            # acceleration scale by the caller so both are O(1)
            self.node_enc = mlp([3 + 3 + (3 if self.use_force else 0), hidden, hidden])
        # one modulation site before the encoder output and one per layer
        self.film = DtFiLM(hidden, mp_steps + 1)
        if processor == "attention":
            self.bias = GeoAttentionBias(heads)
            self.proc = nn.ModuleList(GeoAttentionBlock(hidden, heads)
                                       for _ in range(mp_steps))
        else:
            self.edge_enc = mlp([edge_in, hidden, hidden])
            self.proc = nn.ModuleList(MPNNLayer(hidden) for _ in range(mp_steps))
        self.dec = mlp([hidden, hidden, 3], layernorm=False)
        if not raw_io:
            last = [m for m in self.dec.modules() if isinstance(m, nn.Linear)][-1]
            nn.init.zeros_(last.weight); nn.init.zeros_(last.bias)

    def forward(self, p, v, a, dt, edge_index, f_accel=None):
        if self.raw_io:
            h = self.node_enc(torch.cat([p, v, a], -1))
        else:
            feats = [v, a] if not self.use_force else [v, a, f_accel]
            h = self.node_enc(torch.cat(feats, -1))
        gamma, beta = self.film(dt, p.device)
        h = gamma[0] * h + beta[0]
        if self.processor == "attention":
            # edge_index is unused: attention is over the complete anchor set,
            # which is the point -- no hop budget to run out of.
            bias = self.bias(p)
            for i, layer in enumerate(self.proc):
                h = layer(gamma[i + 1] * h + beta[i + 1], bias)
        else:
            e = self.edge_enc(G.edge_features(p, edge_index))
            for i, layer in enumerate(self.proc):
                h, e = layer(gamma[i + 1] * h + beta[i + 1], edge_index, e)
        return self.dec(h)


def predict_du(net, p, v, a, dt, edge_index, accel_scale, beta=0.25, fixed_mask=None,
                f_accel=None, direct=False, velocity=False):
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
    if getattr(net, "raw_io", False):
        # the network emitted the next POSITION; everything downstream (eq. 8/9,
        # the potential, the residual) is written in terms of du, so convert.
        du = out - p
    elif velocity:
        # the network emits du/dt -- a VELOCITY, the step's average. From eq. (8),
        # du/dt = v_n + dt[(1/2-beta) a_n + beta a_{n+1}], so it tends to v_n as
        # dt -> 0 and stays O(1) at every step size: the arbitrary accel_scale
        # constant disappears and the output range is dt-independent by
        # construction rather than by a hand-picked gain. Not to be confused with
        # the dt*vel_scale gain that failed earlier -- that multiplied a
        # CORRECTION whose true size is O(dt^2), one power of dt too many; here
        # the output stands for the whole du, whose departure from v_n is O(dt).
        # v_n enters as a skip so a zero-init decoder still starts at the
        # inertial coast instead of freezing the object.
        du = dt * (v + out)
    elif direct:
        du = (beta * dt * dt * accel_scale) * out
    else:
        du = pred + (beta * dt * dt * accel_scale) * out
    if fixed_mask is not None:
        du = torch.where(fixed_mask.unsqueeze(-1), torch.zeros_like(du), du)
        pred = torch.where(fixed_mask.unsqueeze(-1), torch.zeros_like(pred), pred)
    return du, pred

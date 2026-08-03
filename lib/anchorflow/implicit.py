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

import torch

from anchorstep import fused_energy_force


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

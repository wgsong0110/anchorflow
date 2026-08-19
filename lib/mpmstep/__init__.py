"""A differentiable MPM substep, for fitting a coarse particle set to a fine one.

The chain's middle stage has been an anchor simulator carrying positions and
velocities, with every Gaussian's deformation gradient rebuilt from the anchor
arrangement at each step. F is the one thing MPM carries forward and that state
cannot: branch MPM with identical positions and velocities but the
anchor-reconstructed F, and the trajectories separate by 2.6-5.0% within thirty
frames (exe/probe_history_gap.py). No amount of fitting removes that.

A coarse MPM has the same state as the reference and accumulates F the same way,
so what it gives up is resolution rather than a kind of information. Fitting one
means differentiating its step in the particles' rest positions, volumes and
moduli -- which the warp solver cannot do: its arrays are marked requires_grad
but it overwrites particle state in place every substep, so a tape would replay
adjoints against values that no longer exist.

DELIBERATE OMISSION. The adjoint does not carry the dependence of a step on where
the B-spline weights land -- that is, dx/dweights. The fit moves rest positions,
and that path is the stiff, high-frequency part of the derivative: a particle
crossing a cell boundary changes which 27 cells it touches, and the resulting
gradient is dominated by that discontinuity rather than by the physics. What is
kept is the dependence through the transferred quantities, which is what a
volume or a modulus moves. A fit that only adjusts positions would need this
back; exe/verify_mpmstep.py compares against finite differences so the size of
what is dropped is measured rather than assumed.
"""

import torch

try:
    from ._C import substep as _substep, substep_backward as _substep_bwd
    HAVE_CUDA = True
except Exception:
    HAVE_CUDA = False


class _Substep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, v, C, F, vol, mass, mu, lam, grav, fixed,
                G, dx, dt, polar_iters, ridge, damp):
        x2, v2, C2, F2, stress, grid_v, grid_m = _substep(
            x.contiguous(), v.contiguous(), C.contiguous(), F.contiguous(),
            vol.contiguous(), mass.contiguous(), mu.contiguous(), lam.contiguous(),
            grav.contiguous(), fixed, int(G), float(dx), float(dt),
            int(polar_iters), float(ridge), float(damp))
        ctx.save_for_backward(x, v, C, F, stress, grid_v, grid_m, vol, mass,
                               mu, lam, fixed)
        ctx.cfg = (int(G), float(dx), float(dt), int(polar_iters), float(ridge),
                    float(damp))
        return x2, v2, C2, F2

    @staticmethod
    def backward(ctx, gx2, gv2, gC2, gF2):
        x, v, C, F, stress, grid_v, grid_m, vol, mass, mu, lam, fixed = ctx.saved_tensors
        G, dx, dt, pi, ridge, damp = ctx.cfg
        gx, gv, gC, gF, gvol, gmu, glam = _substep_bwd(
            gx2.contiguous(), gv2.contiguous(), gC2.contiguous(), gF2.contiguous(),
            x, v, C, F, stress, grid_v, grid_m, vol, mass, mu, lam, fixed,
            G, dx, dt, pi, ridge, damp)
        return (gx, gv, gC, gF, gvol, None, gmu, glam,
                None, None, None, None, None, None, None, None)


def substep(x, v, C, F, vol, mass, mu, lam, grav, fixed, G, dx, dt,
            polar_iters=6, ridge=1e-6, damp=1.0):
    """one substep: (x, v, C, F) -> (x, v, C, F), differentiable in x, vol, mu, lam"""
    return _Substep.apply(x, v, C, F, vol, mass, mu, lam, grav, fixed,
                           G, dx, dt, polar_iters, ridge, damp)


def rollout(x, v, C, F, vol, mass, mu, lam, grav, fixed, G, dx, dt, n,
            polar_iters=6, ridge=1e-6, damp=1.0, checkpoint=True):
    """n substeps.

    Checkpointed by default: the grid alone is G^3 by three floats -- 12 MB at
    G=100 -- and keeping one per substep for the backward is what a coarse
    simulation is supposed to avoid. Recomputing the forward costs one extra
    evaluation and takes the memory from linear in the rollout to constant.
    """
    from torch.utils.checkpoint import checkpoint as _ck

    def one(x_, v_, C_, F_):
        return substep(x_, v_, C_, F_, vol, mass, mu, lam, grav, fixed,
                        G, dx, dt, polar_iters, ridge, damp)

    for _ in range(n):
        if checkpoint and torch.is_grad_enabled():
            x, v, C, F = _ck(one, x, v, C, F, use_reentrant=False)
        else:
            x, v, C, F = one(x, v, C, F)
    return x, v, C, F

"""Constrained as-rigid-as-possible (ARAP) solve for initial-condition completion.

Given a rest anchor pose and a few user-specified handle anchors (hard position
constraints), fill in ALL anchor positions as the *closest feasible* (as-rigid-as-
possible) deformation of the rest pose satisfying the handles. Used to turn a
partial initial-state spec into a coherent full initial pose before the rollout
(e.g. raise one leg -> the connected structure follows rigidly, body stays put).

Alternating local/global minimisation of
    E = Σ_i Σ_{j∈N(i)} w_ij || (p_i-p_j) - R_i (rest_i-rest_j) ||²
    s.t. p_i = target_i for handle anchors i.
Local: per-node rotation R_i by SVD (weighted Procrustes). Global: sparse linear
Laplacian solve with the handle rows fixed. M anchors are few (~512) -> dense torch.

Run under no_grad — this produces the initial condition; gradients flow through the
subsequent rollout, not through this solve.
"""

from __future__ import annotations

import torch

from . import geom


@torch.no_grad()
def arap_solve(rest, idx, w, handle_mask, handle_target, iters=8):
    """rest [M,3]; idx [M,K] neighbour ids; w [M,K] weights; handle_mask [M] bool;
    handle_target [M,3] (used where handle_mask). Returns p [M,3]."""
    M = rest.shape[0]
    dev = rest.device
    src = rest[idx] - rest[:, None]                          # [M,K,3] rest edges

    # graph Laplacian L [M,M] from the (kNN) weights
    L = torch.diag(w.sum(1))
    L.scatter_add_(1, idx, -w)                               # L[i, idx[i,k]] -= w[i,k]

    hnd = handle_mask
    free = ~hnd
    p = rest.clone()
    p[hnd] = handle_target[hnd]
    Lff = L[free][:, free]
    Lfh = L[free][:, hnd]
    th = handle_target[hnd]

    for _ in range(iters):
        # local: per-node rotation aligning rest edges to current edges
        tgt = p[idx] - p[:, None]                            # [M,K,3]
        R = geom.procrustes_rotation(src, tgt, w)            # [M,3,3]
        # global: b_i = Σ_k w_ik · 0.5(R_i+R_j) (rest_i-rest_j)
        Ravg = 0.5 * (R[:, None] + R[idx])                   # [M,K,3,3]
        b = (w[..., None] * torch.einsum("mkab,mkb->mka", Ravg, src)).sum(1)  # [M,3]
        rhs = b[free] - Lfh @ th
        p_free = torch.linalg.solve(Lff, rhs)
        p = handle_target.clone()
        p[free] = p_free
    return p


@torch.no_grad()
def complete_ic(rest, idx, w, handle_mask, handle_pos, handle_vel=None, dt=1.0, iters=8):
    """Complete a partial initial-condition spec -> full (init_pos, init_vel) [M,3].

    handle_pos [M,3] target positions for handles; handle_vel [M,3] optional target
    velocities. Position handles -> ARAP pose p0. Velocity handles -> a second ARAP
    pose advanced by dt, finite-differenced (rigid-consistent velocity propagation).
    Returns (init_pos = p0 - rest, init_vel)."""
    p0 = arap_solve(rest, idx, w, handle_mask, handle_pos, iters=iters)
    if handle_vel is None:
        vel = torch.zeros_like(rest)
    else:
        adv = handle_pos + dt * handle_vel
        p1 = arap_solve(rest, idx, w, handle_mask, adv, iters=iters)
        vel = (p1 - p0) / dt
    return p0 - rest, vel

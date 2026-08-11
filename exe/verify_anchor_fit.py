"""Does the differentiable simulator reproduce the one it replaces, and does it
have a gradient?

Two claims to check before fitting anything with it. At its initial parameters
-- anchors where they were sampled, isotropic, unrotated -- it has to be the
same simulator, or a fit that improves the loss is only measuring the rewrite.
And the gradient has to exist: the analytic force and the Newton polar factor
are here precisely because autograd through the old path returns NaN, and that
needs demonstrating rather than assuming.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

_lib = os.path.join(os.path.dirname(__file__), "..", "lib")
sys.path.insert(0, _lib)

import torch

from anchorflow import scene_setup
from anchorflow.anchor_fit import AnchorFit, polar_R

ap = argparse.ArgumentParser()
ap.add_argument("--ply", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--frames", type=int, default=2)
ap.add_argument("--polar_iters", type=int, default=8)
ap.add_argument("--dt_mult", type=int, default=40)
ap.add_argument("--n_anchors", type=int, default=512)
ap.add_argument("--K", type=int, default=8)
ap.add_argument("--eig_floor", type=float, default=0.02)
args = ap.parse_args()

dev = "cuda"
torch.manual_seed(0)

sc = scene_setup.build(args.ply, args.config, args.n_anchors, args.K, device=dev,
                        frozen_weights=True, rot_fallback=True, eig_floor=args.eig_floor)
AC = sc.anchor_canonical
base = None
for bc in sc.cfg.get("boundary_conditions", []):
    if bc["type"] == "particle_impulse":
        base = torch.tensor(bc["force"], device=dev)
fit = AnchorFit(sc, eig_floor=args.eig_floor, polar_iters=args.polar_iters).to(dev)
print(f"[setup] {sc.M} anchors, {sc.N} Gaussians, "
      f"{sum(p.numel() for p in fit.parameters())} parameters")

# ---- 0. the polar factor against the closed form ---------------------------
with torch.no_grad():
    from anchorflow.anchor_mpm import _polar_decompose
    G = torch.eye(3, device=dev).expand(4096, 3, 3) + 0.05 * torch.randn(4096, 3, 3, device=dev)
    R0, _ = _polar_decompose(G)
    R1 = polar_R(G)
    print(f"\n[0. polar] Newton vs the closed form: max |dR| "
          f"{(R0 - R1).abs().max():.2e}, orthogonality "
          f"{(R1.transpose(-1,-2) @ R1 - torch.eye(3, device=dev)).abs().max():.2e}")

# ---- 1. parity -------------------------------------------------------------
with torch.no_grad():
    w0, rest0 = fit.prepare()
    blocked_frac = (rest0[3].diagonal(dim1=-2, dim2=-1).sum(-1) > 0.5).float().mean()
    p_ref, v_ref, gp = AC.clone(), sc.initial_velocity(base), sc.pos.clone()
    p_fit, v_fit = AC.clone(), sc.initial_velocity(base)
    span = None
    print(f"\n[1. parity] against sc.explicit_step, {blocked_frac*100:.2f}% of Gaussians "
          f"have a blocked direction")
    print(f"  {'frame':>6} {'moved':>10} {'difference':>12} {'relative':>10}")
    for f in range(args.frames):
        p_ref, v_ref, gp = sc.explicit_step(p_ref, v_ref, gp, args.dt_mult)
        p_fit, v_fit = fit.rollout(p_fit, v_fit, args.dt_mult, cache=(w0, rest0))
        moved = (p_ref - AC).norm(dim=-1).max()
        d = (p_ref - p_fit).norm(dim=-1).max()
        print(f"  {f:6d} {moved:10.6f} {d:12.3e} {100*d/moved.clamp(min=1e-12):9.3f}%")

# ---- 2. the gradient -------------------------------------------------------
target = p_ref.detach().clone()
p, v = AC.clone(), sc.initial_velocity(base)
t0 = time.time()
w, rest = fit.prepare()
for _ in range(args.frames):
    p, v = fit.rollout(p, v, args.dt_mult, cache=(w, rest))
loss = ((p - target) ** 2).mean()
torch.cuda.synchronize(); fwd = time.time() - t0
t0 = time.time()
loss.backward()
torch.cuda.synchronize(); bwd = time.time() - t0
print(f"\n[2. gradient] loss {loss.item():.3e}, forward {fwd:.1f}s, backward {bwd:.1f}s "
      f"over {args.frames * args.dt_mult} substeps")
for n, q in fit.named_parameters():
    g = q.grad
    ok = "finite" if g is not None and torch.isfinite(g).all() else "NOT FINITE"
    print(f"  {n:8} {ok:11} |grad| mean {0.0 if g is None else g.norm(dim=-1).mean():.3e}, "
          f"max {0.0 if g is None else g.norm(dim=-1).max():.3e}")

# ---- 3. is the gradient the real one? --------------------------------------
# one finite difference on one anchor, which is slow but is the only check that
# the analytic force and the hand-derived scatter agree with the loss they claim
# to differentiate
with torch.no_grad():
    j = int(torch.nonzero(~fit.fixed, as_tuple=False)[0])
    eps = 1e-4
    g_ana = fit.pos.grad[j, 0].item()


def loss_at(delta):
    with torch.no_grad():
        fit.pos[j, 0] += delta
    with torch.no_grad():
        w_, rest_ = fit.prepare()
        p_, v_ = AC.clone(), sc.initial_velocity(base)
        for _ in range(args.frames):
            p_, v_ = fit.rollout(p_, v_, args.dt_mult, cache=(w_, rest_))
        out = ((p_ - target) ** 2).mean().item()
    with torch.no_grad():
        fit.pos[j, 0] -= delta
    return out


num = (loss_at(eps) - loss_at(-eps)) / (2 * eps)
print(f"\n[3. finite difference] anchor {j}, x: analytic {g_ana:.6e}, "
      f"numerical {num:.6e}, ratio {num / g_ana if g_ana else float('nan'):.4f}")
print(f"    the two agreeing means the force and its scatter are the derivative of "
      f"the\n    energy they are supposed to come from, not merely something "
      f"plausible.")

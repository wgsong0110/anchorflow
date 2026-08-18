"""Does the fused sparse kernel compute the same physics as the torch path?

The kernel exists to make the fitted simulator fast, and a fast simulator that
answers a slightly different question is worse than a slow one. So every
quantity it produces is compared against the reference it replaces, on the real
scene rather than on random tensors: the same anchor set, the same pair list,
the same weights.

Three things are checked, in the order they compose, because a discrepancy in
the first would explain the other two:

  F     shape matching. Two scatters and a 3x3 solve.
  x     skinning. F applied to each Gaussian's canonical offset.
  f     the elastic force. F, the polar factor, Fixed Corotated PK1, and the
        gather back onto anchors.

and then a rollout, because the interesting failure is not a wrong first step
but a small bias that compounds over forty substeps.

Agreement is reported relative to the magnitude of the quantity itself. Exact
equality is not on offer: the kernel accumulates a warp reduction in a different
order from index_add_, and the polar iteration runs in float32 either way.
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

ap = argparse.ArgumentParser()
ap.add_argument("--ply", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--dreamphysics", default="/workspace/DreamPhysics")
ap.add_argument("--fit", default=None,
                 help="a fitted anchor set. Without one this runs on the sampled "
                      "anchors, which exercises the kernel but not the anisotropy "
                      "and per-anchor stiffness it was written for.")
ap.add_argument("--n_anchors", type=int, default=512)
ap.add_argument("--K", type=int, default=8)
ap.add_argument("--c", type=float, default=0.25)
ap.add_argument("--eig_floor", type=float, default=0.02)
ap.add_argument("--substeps", type=int, default=40)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--tol", type=float, default=2e-3,
                 help="relative agreement demanded of every quantity")
args = ap.parse_args()

sys.path.insert(0, args.dreamphysics)

import sparsestep
from anchorflow.anchor_sparse import AnchorSparse, load_fitted
from anchorflow.streams import draw_impulse

if not sparsestep.HAVE_CUDA:
    print("sparsestep has no CUDA extension built -- nothing to verify")
    sys.exit(1)

dev = "cuda"
torch.manual_seed(args.seed)
sc = scene_setup.build(args.ply, args.config, args.n_anchors, args.K, device=dev,
                        frozen_weights=True, rot_fallback=True,
                        eig_floor=args.eig_floor)
if args.fit:
    fs, _ = load_fitted(sc, args.fit, device=dev)
    fit = fs.fit
else:
    fit = AnchorSparse(sc, c=args.c, eig_floor=args.eig_floor).to(dev)
print(f"[setup] {fit.pos.shape[0]} anchors, {fit.N} material Gaussians, "
      f"{fit.pair_g.shape[0]} pairs "
      f"({fit.pair_g.shape[0] / fit.N:.1f} per Gaussian)")

cache = fit.prepare()
w, rc, q, Binv, blocked, mass = cache
csr = sparsestep.build_csr(fit.pair_g, fit.pair_a, fit.N, fit.M)


def rel(a, b):
    """how far apart, against how large the thing is"""
    d = (a - b).abs()
    scale = b.abs().max().clamp(min=1e-20)
    return float(d.max() / scale), float(d.mean() / scale)


def check(name, got, ref, tol=None):
    tol = args.tol if tol is None else tol
    mx, mn = rel(got, ref)
    ok = mx < tol and torch.isfinite(got).all()
    print(f"  {name:24} max {mx:9.2e}   mean {mn:9.2e}   {'ok' if ok else 'FAILED'}")
    return ok


# a state that is actually deformed: comparing at rest would pass with F = I
base_force = next(torch.tensor(bc["force"], device=dev)
                  for bc in sc.cfg["boundary_conditions"]
                  if bc["type"] == "particle_impulse")
gen = torch.Generator(device=dev).manual_seed(args.seed)
f_imp, _ = draw_impulse(sc, base_force, gen, field=True)

with torch.no_grad():
    p0 = fit.pos.detach().clone()
    v0 = fit.impulse_dv(f_imp, cache)
    p_mid, v_mid, _ = fit.rollout(p0, v0, args.substeps // 2, cache)
print(f"[state] rolled {args.substeps // 2} substeps in, anchors moved "
      f"{(p_mid - p0).norm(dim=-1).max():.5f}")

ok = True
with torch.no_grad():
    # ---- F and the deformed centroid -------------------------------------
    F_ref, cc_ref = fit.deformation(p_mid, w, rc, q, Binv, blocked)
    F_k, cc_k = sparsestep.deform(p_mid, csr, w, q, Binv, blocked)
    ok &= check("F (shape matching)", F_k, F_ref)
    ok &= check("centroid", cc_k, cc_ref)

    # ---- skinning ---------------------------------------------------------
    x_ref = cc_ref + torch.einsum("nij,nj->ni", F_ref, fit.Xc - rc)
    x_k = sparsestep.skin(p_mid, csr, w, q, Binv, blocked, fit.Xc, rc)[0]
    ok &= check("skinned positions", x_k, x_ref)

    # ---- the force --------------------------------------------------------
    f_ref = fit.force(p_mid, w, rc, q, Binv, blocked)          # torch: grad is off
    k = fit.stiffness(w)
    f_k = sparsestep.force(p_mid, csr, w, q, Binv, blocked, fit.vol,
                            (fit.mu * k).contiguous(), (fit.lam * k).contiguous(),
                            fit.M, polar_iters=fit.polar_iters,
                            polar_ridge=fit.polar_ridge)
    ok &= check("elastic force", f_k, f_ref)

    # the fused path is what force() picks on its own with grad off, so this
    # also checks the dispatch rather than only the kernel behind it
    print(f"  {'dispatch':24} force() took the "
          f"{'fused' if fit._fused_ok() else 'TORCH'} path")
    ok &= fit._fused_ok()

# ---- and over a rollout, where a small bias would show ---------------------
def rollout_torch(n):
    saved = sparsestep.HAVE_CUDA
    sparsestep.HAVE_CUDA = False
    try:
        with torch.no_grad():
            return fit.rollout(p0.clone(), v0.clone(), n, cache)[:2]
    finally:
        sparsestep.HAVE_CUDA = saved


with torch.no_grad():
    p_k, v_k, _ = fit.rollout(p0.clone(), v0.clone(), args.substeps, cache)
p_t, v_t = rollout_torch(args.substeps)
moved = (p_t - p0).norm(dim=-1).max().clamp(min=1e-20)
drift = (p_k - p_t).norm(dim=-1).max()
print(f"\n[rollout] {args.substeps} substeps: positions differ by {drift:.3e}, "
      f"which is {float(drift / moved) * 100:.4f}% of how far the anchors moved")
ok &= float(drift / moved) < args.tol

# ---- gradients, which the fit depends on and a forward check cannot see -----
#
# A wrong backward does not crash and does not show up in any forward number; it
# just fits the discretisation to something else. So both pair reductions are
# differentiated both ways from the same inputs and compared.
print("\n[gradients]")
torch.set_grad_enabled(True)


def grads(fused):
    saved = sparsestep.HAVE_CUDA
    sparsestep.HAVE_CUDA = saved and fused
    try:
        pp = p_mid.detach().clone().requires_grad_(True)
        ww = w.detach().clone().requires_grad_(True)
        qq = q.detach().clone().requires_grad_(True)
        F, cc = fit.deformation(pp, ww, rc, qq, Binv, blocked)
        # a loss that touches every output, with weights that are not all one
        g = torch.arange(1, 10, device=dev, dtype=F.dtype).reshape(1, 3, 3)
        loss = (F * g).sum() + (cc * cc).sum()
        loss.backward()
        d = (pp.grad.clone(), ww.grad.clone(), qq.grad.clone())

        pp2 = p_mid.detach().clone().requires_grad_(True)
        ww2 = w.detach().clone().requires_grad_(True)
        qq2 = q.detach().clone().requires_grad_(True)
        f2 = fit.force(pp2, ww2, rc, qq2, Binv, blocked)
        (f2 * f2).sum().backward()
        return d + (pp2.grad.clone(), ww2.grad.clone(), qq2.grad.clone())
    finally:
        sparsestep.HAVE_CUDA = saved


gk = grads(True)
gt = grads(False)
for nm, a_, b_ in zip(("deform dL/dp", "deform dL/dw", "deform dL/dq",
                        "force  dL/dp", "force  dL/dw", "force  dL/dq"), gk, gt):
    ok &= check(nm, a_, b_, tol=5e-3)
torch.set_grad_enabled(False)

print("\nVERIFIED" if ok else "\nMISMATCH -- do not use the kernel")
sys.exit(0 if ok else 1)

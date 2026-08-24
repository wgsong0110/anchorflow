"""Does the oriented kernel path compute what the torch path computes?

The kernel now carries two things it did not: the oriented anchor's own turned
second moment in the shape matching, and the per-anchor stress moment the torque
is built from. Both are pair reductions with hand-written backwards, so a
mistake in either shows up as a wrong gradient and nothing else -- the forward
would look fine and the fit would quietly optimise something adjacent.

Everything is compared against the torch expression it replaces, on the real
scene, at a deformed state, with the orientations turned away from the identity
so the new term is actually doing something.
"""
from __future__ import annotations

import argparse
import os
import sys

_lib = os.path.join(os.path.dirname(__file__), "..", "lib")
sys.path.insert(0, _lib)

import torch

from anchorflow import scene_setup

ap = argparse.ArgumentParser()
ap.add_argument("--ply", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--dreamphysics", default="/workspace/DreamPhysics")
ap.add_argument("--n_anchors", type=int, default=512)
ap.add_argument("--K", type=int, default=8)
ap.add_argument("--c", type=float, default=0.25)
ap.add_argument("--eig_floor", type=float, default=0.02)
ap.add_argument("--tol", type=float, default=5e-3)
args = ap.parse_args()

sys.path.insert(0, args.dreamphysics)

import sparsestep

from anchorflow.anchor_sparse import AnchorSparse

if not sparsestep.HAVE_CUDA:
    print("sparsestep has no CUDA extension built -- nothing to verify")
    sys.exit(1)

dev = "cuda"
torch.manual_seed(0)
sc = scene_setup.build(args.ply, args.config, args.n_anchors, args.K, device=dev,
                        frozen_weights=True, rot_fallback=True,
                        eig_floor=args.eig_floor)
fit = AnchorSparse(sc, c=args.c, eig_floor=args.eig_floor, oriented=True).to(dev)
fit.init_from_geometry()
print(f"[setup] {fit.M} anchors, {fit.N} Gaussians, {fit.pair_g.shape[0]} pairs")

cache = fit.prepare()
w, rc, q, Binv, blocked, mass = (t.detach() for t in cache)
ca = (w, rc, q, Binv, blocked, mass)

# a deformed state, and orientations well away from the identity so the term
# the kernel gained is not silently zero
with torch.no_grad():
    p_def = fit.pos.detach() + 0.003 * torch.randn_like(fit.pos)
    ax = torch.randn(fit.M, 3, device=dev) * 0.25
    o = torch.cat([torch.ones(fit.M, 1, device=dev), ax], -1)
    o = o / o.norm(dim=-1, keepdim=True)
print(f"[state] anchors moved {float((p_def - fit.pos).norm(dim=-1).max()):.5f}, "
      f"turned by up to {float(2 * torch.asin(ax.norm(dim=-1).clamp(max=1)) ):.3f} rad")


def rel(a, b):
    d = (a - b).abs()
    return float(d.max() / b.abs().max().clamp(min=1e-20))


def both(fn):
    """the same thing through the kernel and through torch"""
    out = []
    for use in (True, False):
        saved = sparsestep.HAVE_CUDA
        sparsestep.HAVE_CUDA = saved and use
        try:
            out.append(fn())
        finally:
            sparsestep.HAVE_CUDA = saved
    return out


ok = True


def check(name, a, b, tol=None):
    global ok
    tol = args.tol if tol is None else tol
    r = rel(a, b)
    good = r < tol and torch.isfinite(a).all()
    print(f"  {name:26} rel {r:9.2e}   {'ok' if good else 'FAILED'}")
    ok &= bool(good)


print("\n[forward]")
with torch.no_grad():
    fit._o = o
    Fk, cck = both(lambda: fit.deformation(p_def, *ca[:5]))
    check("F (oriented)", Fk[0], cck[0])
    check("centroid", Fk[1], cck[1])
    fit._o = o
    fk, tk = both(lambda: fit.force_torque(p_def, *ca[:5]))
    check("force", fk[0], tk[0])
    check("torque", fk[1], tk[1])
    fit._o = o
    xk, xt = both(lambda: fit.gaussian_pos(p_def, cache))
    check("skinned positions", xk, xt)

print("\n[gradients]  (p and the orientation both reach the loss)")
torch.set_grad_enabled(True)


def grads(use):
    saved = sparsestep.HAVE_CUDA
    sparsestep.HAVE_CUDA = saved and use
    try:
        pp = p_def.detach().clone().requires_grad_(True)
        oo = o.detach().clone().requires_grad_(True)
        fit._o = oo
        f, tau = fit.force_torque(pp, *ca[:5])
        g = torch.arange(1, 4, device=dev, dtype=f.dtype).reshape(1, 3)
        ((f * g).sum() + (tau * tau).sum()).backward()
        return pp.grad.clone(), oo.grad.clone()
    finally:
        sparsestep.HAVE_CUDA = saved
        fit._o = o


gk = grads(True)
gt = grads(False)
check("dL/dp", gk[0], gt[0])
check("dL/d(orientation)", gk[1], gt[1])
torch.set_grad_enabled(False)

print("\nVERIFIED" if ok else "\nMISMATCH -- do not use the oriented kernel")
sys.exit(0 if ok else 1)

"""Where does a fit iteration actually spend its time?

The fit runs at 112 s per iteration and the obvious answer is "write a CUDA
backward", but lib/sparsestep is forward-only and a backward is a large,
error-prone piece of work -- so it is worth knowing what it would buy before
writing it. An unroll-12 sample is 480 differentiable substeps over 3.3M pairs,
and the parts are separable: the pair-level scatters are memory traffic on P,
while the polar factor and the stress are 3x3 arithmetic on N. Those want
different fixes, and only a measurement says which dominates.
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
ap.add_argument("--c", type=float, default=0.25)
ap.add_argument("--eig_floor", type=float, default=0.02)
ap.add_argument("--unroll", type=int, default=12)
ap.add_argument("--dt_mult", type=int, default=40)
ap.add_argument("--repeat", type=int, default=3)
args = ap.parse_args()

sys.path.insert(0, args.dreamphysics)
import warp as wp

from anchorflow.anchor_sparse import AnchorSparse

dev = "cuda"
wp.init()
sc = scene_setup.build(args.ply, args.config, 512, 8, device=dev, frozen_weights=True,
                        rot_fallback=True, eig_floor=args.eig_floor)
fit = AnchorSparse(sc, c=args.c, eig_floor=args.eig_floor).to(dev)
print(f"[setup] {fit.M} anchors, {fit.N} Gaussians, {fit.pair_g.shape[0]} pairs "
      f"({fit.pair_g.shape[0]/fit.N:.1f} per Gaussian)")


def t(fn, n=None, grad=True):
    n = n or args.repeat
    torch.set_grad_enabled(grad)
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e3


cache_ng = None
with torch.no_grad():
    cache_ng = fit.prepare()
p0 = fit.pos.detach().clone()
v0 = torch.zeros_like(p0)

print(f"\n{'':44} {'ms':>10}")
print(f"{'prepare()  weights + B + inv, no grad':44} {t(lambda: fit.prepare(), grad=False):10.2f}")
print(f"{'prepare()  with grad':44} {t(lambda: fit.prepare(), grad=True):10.2f}")

w, rc, q, Binv, blocked, mass = cache_ng
m_col, keep_col = mass.unsqueeze(-1), (~fit.fixed).unsqueeze(-1).float()


def fwd_deform():
    fit.deformation(p0, w, rc, q, Binv, blocked)


def fwd_force():
    fit.force(p0, w, rc, q, Binv, blocked)


print(f"{'deformation()  the two pair scatters':44} {t(fwd_deform, grad=False):10.2f}")
print(f"{'force()  = deformation + polar + stress':44} {t(fwd_force, grad=False):10.2f}")

# with grad, one substep, forward and forward+backward
def one_substep_fwd():
    pp = p0.clone().requires_grad_(True)
    fit.substep(pp, v0, w, rc, q, Binv, blocked, m_col, keep_col)


def one_substep_bwd():
    pp = p0.clone().requires_grad_(True)
    a, b, c_ = fit.substep(pp, v0, w, rc, q, Binv, blocked, m_col, keep_col)
    a.sum().backward()


print(f"{'one substep, forward with grad on':44} {t(one_substep_fwd):10.2f}")
print(f"{'one substep, forward + backward':44} {t(one_substep_bwd, n=2):10.2f}")

# and the shape the fit actually runs: an unrolled rollout under checkpointing
def unrolled(n_frames):
    def run():
        cache = fit.prepare()
        p, v = fit.pos, torch.zeros_like(fit.pos)
        for _ in range(n_frames):
            p, v, _ = fit.rollout(p, v, args.dt_mult, cache)
        g = fit.gaussian_pos(p, cache)
        g.pow(2).mean().backward()
        fit.zero_grad(set_to_none=True)
    return run


# where the gap between 40 x one-substep and the unroll-1 loss goes. prepare()
# is called once per sample, but its backward carries the gradient every substep
# accumulated into w, q, Binv and blocked -- and it still materialises a [P,3,3]
# for the rest scatter, which the kernel does not cover
def prepare_only():
    cache = fit.prepare()
    s_ = sum(t.pow(2).sum() for t in cache)
    s_.backward()
    fit.zero_grad(set_to_none=True)


def rollout_detached(n_frames):
    """the substeps alone: the cache is detached, so nothing flows into prepare"""
    det = [t.detach().requires_grad_(True) for t in cache_ng]

    def run():
        p, v = fit.pos, torch.zeros_like(fit.pos)
        for _ in range(n_frames):
            p, v, _ = fit.rollout(p, v, args.dt_mult, tuple(det))
        p.pow(2).sum().backward()
        fit.zero_grad(set_to_none=True)
        for t in det:
            t.grad = None
    return run


print(f"{'prepare()  fwd+bwd alone':44} {t(prepare_only, n=2):10.2f}")
print(f"{'40 substeps, cache detached, fwd+bwd':44} {t(rollout_detached(1), n=1):10.2f}")

for nf in (1, args.unroll):
    ms = t(unrolled(nf), n=1)
    print(f"{'unroll %d, full loss fwd+bwd (%d substeps)' % (nf, nf * args.dt_mult):44} "
          f"{ms:10.2f}")
print("\n  A fit iteration is batch 2 of the last row, so double it.")
print("  Pair-level work scales with P=%d; the 3x3 work with N=%d." %
      (fit.pair_g.shape[0], fit.N))

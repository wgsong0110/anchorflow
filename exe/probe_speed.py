"""Where does a substep actually spend its time?

The fit runs at 20 s/iteration with the fused kernel and 90-190 without it, and
"the kernel is missing" has been the standing explanation. That is a claim about
where the time goes, and it has never been measured on this machine. It matters
now because the oriented path cannot use the kernel at all, so if the cost is
somewhere else -- the polar iteration, the 3x3 inverses, the scatter, the
checkpoint recompute -- then building a kernel for the container's glibc is the
wrong thing to spend a day on.

Timed with CUDA events around each stage of one substep, at the sizes the fit
runs at, for the linear and the oriented paths. Forward and backward separately,
because checkpointing means the backward re-runs the forward and a stage that is
cheap forward is paid for twice.
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
ap.add_argument("--n_anchors", type=int, default=512)
ap.add_argument("--K", type=int, default=8)
ap.add_argument("--c", type=float, default=0.25)
ap.add_argument("--eig_floor", type=float, default=0.02)
ap.add_argument("--reps", type=int, default=20)
ap.add_argument("--dt_mult", type=int, default=40)
args = ap.parse_args()

sys.path.insert(0, args.dreamphysics)

from anchorflow.anchor_fit import closest_rotation, det3, inv3
from anchorflow.anchor_sparse import AnchorSparse

dev = "cuda"
torch.manual_seed(0)

sc = scene_setup.build(args.ply, args.config, args.n_anchors, args.K, device=dev,
                        frozen_weights=True, rot_fallback=True,
                        eig_floor=args.eig_floor)


def clock(fn, n=None, warm=3):
    n = n or args.reps
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return 1000 * (time.time() - t) / n


for oriented in (False, True):
    fit = AnchorSparse(sc, c=args.c, eig_floor=args.eig_floor,
                        oriented=oriented).to(dev)
    fit.init_from_geometry()
    P, N, M = int(fit.pair_g.shape[0]), fit.N, fit.M
    print(f"\n=== oriented={oriented}  {M} anchors, {N} Gaussians, {P} pairs ===")
    print(f"    fused kernel available: {fit._fused_ok()}")

    with torch.no_grad():
        cache = fit.prepare()
    w, rc, q, Binv, blocked, mass = cache
    p0 = fit.pos.detach().clone()

    # a state that is actually deformed, so the polar factor and the inverses
    # are doing real work rather than returning the identity
    with torch.no_grad():
        p = p0 + 0.002 * torch.randn_like(p0)

    print(f"  {'prepare()':32} {clock(lambda: fit.prepare(), 5):8.2f} ms  (refresh 당 1회)")

    with torch.no_grad():
        print(f"  {'deformation (A scatter + solve)':32} "
              f"{clock(lambda: fit.deformation(p, w, rc, q, Binv, blocked)):8.2f} ms")
        F, _ = fit.deformation(p, w, rc, q, Binv, blocked)
        print(f"  {'closest_rotation (polar)':32} "
              f"{clock(lambda: closest_rotation(F, fit.polar_iters, fit.polar_ridge)):8.2f} ms")
        print(f"  {'det3 + inv3':32} "
              f"{clock(lambda: (det3(F), inv3(F + 1e-6 * torch.eye(3, device=dev)))):8.2f} ms")
        print(f"  {'stiffness (per-Gaussian blend)':32} "
              f"{clock(lambda: fit.stiffness(w)):8.2f} ms")
        if oriented:
            print(f"  {'force_torque (whole)':32} "
                  f"{clock(lambda: fit.force_torque(p, w, rc, q, Binv, blocked)):8.2f} ms")
        print(f"  {'force (whole)':32} "
              f"{clock(lambda: fit.force(p, w, rc, q, Binv, blocked)):8.2f} ms")
        print(f"  {'gaussian_pos (skin)':32} "
              f"{clock(lambda: fit.gaussian_pos(p, cache)):8.2f} ms")

        m_ = mass.unsqueeze(-1)
        keep = (~fit.fixed).unsqueeze(-1).to(p.dtype)
        v0 = torch.zeros_like(p)
        if oriented:
            fit._o, fit._wv = fit._ident(), torch.zeros(M, 3, device=dev)
            sub = lambda: fit.substep_o(p, v0, fit._o, fit._wv, w, rc, q, Binv,
                                         blocked, m_, keep)
        else:
            sub = lambda: fit.substep(p, v0, w, rc, q, Binv, blocked, m_, keep)
        t_sub = clock(sub)
        print(f"  {'substep (forward, no grad)':32} {t_sub:8.2f} ms"
              f"   -> {args.dt_mult} substeps = {t_sub * args.dt_mult / 1000:.2f} s")

    # ---- with the graph, which is what the fit actually pays ----------------
    torch.set_grad_enabled(True)

    def one_frame():
        pp = p.detach().clone().requires_grad_(True)
        vv = v0.detach().clone()
        ca = fit.prepare()
        if oriented:
            fit._o, fit._wv = fit._ident(), torch.zeros(M, 3, device=dev)
        a, b, _ = fit.rollout(pp, vv, args.dt_mult, ca)
        loss = fit.gaussian_pos(a, ca).square().mean()
        loss.backward()
        return loss

    print(f"  {'one coarse frame fwd+bwd':32} {clock(one_frame, 3, 1):8.2f} ms"
          f"   -> 12 frames = {clock(one_frame, 3, 1) * 12 / 1000:.1f} s")

    # ---- which stage's backward is the expensive one ------------------------
    def bw(fn, name, it=None):
        def go():
            pp = p.detach().clone().requires_grad_(True)
            out = fn(pp)
            out.square().mean().backward()
        print(f"  {name:32} {clock(go, 5, 2):8.2f} ms")

    print("  -- 단계별 순+역전파 --")
    bw(lambda pp: fit.deformation(pp, w, rc, q, Binv, blocked)[0], "deformation fwd+bwd")

    def polar_only(pp):
        F_, _ = fit.deformation(pp, w, rc, q, Binv, blocked)
        return closest_rotation(F_, fit.polar_iters, fit.polar_ridge)
    bw(polar_only, f"+ polar ({fit.polar_iters} iters) fwd+bwd")

    def polar3(pp):
        F_, _ = fit.deformation(pp, w, rc, q, Binv, blocked)
        return closest_rotation(F_, 3, fit.polar_ridge)
    bw(polar3, "+ polar (3 iters) fwd+bwd")

    bw(lambda pp: fit.force(pp, w, rc, q, Binv, blocked), "force fwd+bwd")
    if oriented:
        bw(lambda pp: fit.force_torque(pp, w, rc, q, Binv, blocked)[0],
           "force_torque fwd+bwd")
    torch.set_grad_enabled(False)

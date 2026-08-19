"""What does compressing MPM's per-particle F onto the anchors cost?

The anchor state carries positions and velocities and rebuilds each Gaussian's
deformation gradient from the arrangement. F is the one thing MPM accumulates
and that state cannot, and replacing MPM's own F with the anchor-derived one
separates the trajectories by 2.6-5.0% in thirty frames
(exe/probe_history_gap.py). The obvious repair is to give each anchor an F of
its own and accumulate it.

Before building that, this asks what it could possibly buy. 588 anchors times
nine is 5,292 numbers against 171,553 times nine, a compression of three hundred
to one -- so even an anchor that accumulates perfectly can only carry what
survives that. What is measured is the compression alone, with no accumulation
scheme involved:

  take MPM's own F, average it within each anchor's support, push it back out to
  the particles, and roll from there with positions and velocities untouched.

Against the 2.6-5.0% that the anchor-DERIVED F costs, this is what an anchor
that carried F perfectly would still cost. If the two are close, anchors cannot
hold F at all and the repair is not worth building; if this is much smaller, the
gap is the derivation and accumulating would recover it.
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
ap.add_argument("--fit", required=True)
ap.add_argument("--cache", required=True)
ap.add_argument("--kind", default="field")
ap.add_argument("--n_case", type=int, default=4)
ap.add_argument("--at_frames", type=int, nargs="+", default=[10, 30])
ap.add_argument("--frames", type=int, default=30)
ap.add_argument("--n_anchors", type=int, default=512)
ap.add_argument("--K", type=int, default=8)
ap.add_argument("--eig_floor", type=float, default=0.02)
ap.add_argument("--dt_mult", type=int, default=40)
args = ap.parse_args()

sys.path.insert(0, args.dreamphysics)
import warp as wp

from anchorflow.anchor_fit import det3
from anchorflow.anchor_sparse import load_fitted
from anchorflow.mpm_teacher import MPMTeacher

dev = "cuda"
torch.set_grad_enabled(False)
wp.init()
sc = scene_setup.build(args.ply, args.config, args.n_anchors, args.K, device=dev,
                        frozen_weights=True, rot_fallback=True,
                        eig_floor=args.eig_floor)
fs, _ = load_fitted(sc, args.fit, dev)
fit = fs.fit
cache = fit.prepare()
T = MPMTeacher(sc, sparse=fit)
w, pg, pa = cache[0], fit.pair_g, fit.pair_a
M, N = fit.M, fit.N
print(f"[setup] {M} anchors over {N} material particles, "
      f"{M * 9} numbers against {N * 9} -- {N / M:.0f} to 1")

blob = torch.load(args.cache, map_location=dev, weights_only=False)
refs = blob["ref"][args.kind][: args.n_case]
forces = blob["force"][args.kind][: args.n_case]

den_a = torch.zeros(M, device=dev).index_add_(0, pa, w).clamp(min=1e-12)
den_g = torch.zeros(N, device=dev).index_add_(0, pg, w).clamp(min=1e-12)


def compress(Fp):
    """particle F -> one F per anchor -> back to the particles.

    Both directions are the blend the simulator already uses, so this is the
    anchor representation of F and not some other projection.
    """
    Fa = torch.zeros(M, 9, device=dev).index_add_(
        0, pa, w.unsqueeze(-1) * Fp[pg]) / den_a.unsqueeze(-1)
    out = torch.zeros(N, 9, device=dev).index_add_(
        0, pg, w.unsqueeze(-1) * Fa[pa]) / den_g.unsqueeze(-1)
    return out, Fa


def mpm_to(force, n):
    dv = fit.impulse_dv(force, cache)
    v0 = torch.zeros(T.n, 3, device=dev).index_add_(
        0, pg, w.unsqueeze(-1) * dv[pa]).contiguous()
    T._set(T.pos_m.clone(), v0, T.eye.clone(), torch.zeros_like(T.eye))
    for _ in range(n):
        for k in range(args.dt_mult):
            T.solver.p2g2p(None, sc.sub_dt, device=T.wp_dev)
            if (k + 1) % 8 == 0 and not T._in_domain():
                return None
    return tuple(t.clone() for t in (
        T.solver.export_particle_x_to_torch(), T.solver.export_particle_v_to_torch(),
        T.solver.export_particle_F_to_torch().reshape(-1, 9),
        T.solver.export_particle_C_to_torch().reshape(-1, 9)))


def roll(state, n):
    T._set(*[t.contiguous() for t in state])
    out = []
    for _ in range(n):
        for k in range(args.dt_mult):
            T.solver.p2g2p(None, sc.sub_dt, device=T.wp_dev)
            if (k + 1) % 8 == 0 and not T._in_domain():
                return None
        out.append(T.solver.export_particle_x_to_torch().clone())
    return torch.stack(out)


rows = []
for at in args.at_frames:
    for i, f in enumerate(forces):
        st = mpm_to(f, at)
        if st is None:
            continue
        x, v, F, C = st
        Fc, _ = compress(F)
        # how far the compressed F is from MPM's own, before any dynamics
        dF = ((F - Fc).norm(dim=-1) / F.norm(dim=-1).clamp(min=1e-12)).median()
        # and what the anchor-DERIVED F costs, for the same state and the same
        # trajectory -- the number this is being compared against
        p_ = fit.project(x, cache)
        vp_ = fit.project_v(v, cache)
        _x2, _v2, Fd, Cd = fit.lift(p_, vp_, cache)
        Fd = Fd.reshape(-1, 9)
        a = roll((x, v, F, C), args.frames)
        b = roll((x, v, Fc, C), args.frames)
        d3 = det3(Fd.reshape(-1, 3, 3))
        c = None
        if torch.isfinite(Fd).all() and d3.min() > 0.02 and d3.max() < 50:
            c = roll((x, v, Fd, Cd.reshape(-1, 9)), args.frames)
        if a is None or b is None:
            continue
        span = (a - a[0]).norm(dim=-1).max().clamp(min=1e-12)
        e_c = float(((a - b).norm(dim=-1).mean(-1) / span).mean())
        e_d = float(((a - c).norm(dim=-1).mean(-1) / span).mean()) if c is not None else float("nan")
        rows.append((at, i, float(dF), e_c, e_d))

print(f"\n{'branch at':>10} {'case':>5} {'|F-Fc|/|F|':>11} {'compressed':>12} "
      f"{'anchor-derived':>15}")
for at, i, dF, ec, ed in rows:
    print(f"{at:10d} {i:5d} {100 * dF:10.2f}% {100 * ec:11.2f}% "
          f"{'--' if ed != ed else f'{100 * ed:14.2f}%'}")
for at in args.at_frames:
    g = [r for r in rows if r[0] == at]
    if g:
        ed = [r[4] for r in g if r[4] == r[4]]
        print(f"\n  branching at frame {at}: compressed F costs "
              f"{100 * sum(r[3] for r in g) / len(g):.2f}%, the anchor-derived one "
              f"{100 * sum(ed) / len(ed):.2f}%" if ed else "")
print("\n  Positions and velocities are MPM's own in every run; only F differs.\n"
       "  'compressed' is the ceiling on what giving anchors an accumulated F\n"
       "  could reach; 'anchor-derived' is what the simulator does today.")

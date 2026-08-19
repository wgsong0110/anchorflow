"""How much of MPM's future is in information the anchor state cannot hold?

The chain assumes the anchor simulator can follow MPM. It follows it to about
7%, and the remaining gap has been treated as fitting that is not finished. But
MPM's state is per-particle (x, v, F, C) with F accumulated over the whole
history -- F <- (I + dt grad v) F -- while the anchor state is positions and
velocities, from which F is re-estimated at every step. Two MPM states with the
same particle positions and different F have different futures, and nothing in
the anchor state distinguishes them.

That is a ceiling no amount of fitting removes, and it has never been measured.
It is measurable directly and cheaply, with no learning and no differentiation:

  take MPM at a frame, keep its own x and v, and replace only F and C with what
  the anchor state reconstructs. Roll both forward. Whatever they diverge by is
  exactly the information the anchor state cannot carry.

Positions and velocities are held identical on purpose -- otherwise this would
measure the position reconstruction error too, which is the 1.1% floor already
reported and a different thing.
"""
from __future__ import annotations

import argparse
import os
import sys

_lib = os.path.join(os.path.dirname(__file__), "..", "lib")
sys.path.insert(0, _lib)

import torch
from tqdm import tqdm

from anchorflow import scene_setup

ap = argparse.ArgumentParser()
ap.add_argument("--ply", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--dreamphysics", default="/workspace/DreamPhysics")
ap.add_argument("--fit", default=None, help="the fitted anchor set; without one the "
                                             "sampled set is used")
ap.add_argument("--cache", required=True, help="eval_vs_mpm's reference cache")
ap.add_argument("--kind", default="field")
ap.add_argument("--n_case", type=int, default=4)
ap.add_argument("--at_frames", type=int, nargs="+", default=[10, 30],
                 help="frames to branch at. Later means more history for F to have "
                      "accumulated, and more for the anchor state to be missing.")
ap.add_argument("--frames", type=int, default=30, help="frames to roll after branching")
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
FIT = None
if args.fit:
    fs, _ = load_fitted(sc, args.fit, dev)
    FIT = fs.fit
T = MPMTeacher(sc, sparse=FIT)
cache = FIT.prepare() if FIT is not None else None
mat = T.mat
print(f"[setup] {'fitted ' + str(FIT.M) if FIT is not None else 'sampled 512'} anchors, "
      f"{T.n} material particles")

blob = torch.load(args.cache, map_location=dev, weights_only=False)
refs = blob["ref"][args.kind][: args.n_case]
forces = blob["force"][args.kind][: args.n_case]


def mpm_to(force, n_frames):
    """MPM from rest to a frame, returning its full state"""
    dv = sc.impulse_dv(force)
    if FIT is not None:
        w_ = cache[0]
        v0 = torch.zeros(T.n, 3, device=dev).index_add_(
            0, FIT.pair_g, w_.unsqueeze(-1) * FIT.impulse_dv(force, cache)[FIT.pair_a]
        ).contiguous()
    else:
        v0 = (T.w.unsqueeze(-1) * dv[T.idx]).sum(1).contiguous()
    T._set(T.pos_m.clone(), v0, T.eye.clone(), torch.zeros_like(T.eye))
    for _ in range(n_frames):
        for k in range(args.dt_mult):
            T.solver.p2g2p(None, sc.sub_dt, device=T.wp_dev)
            if (k + 1) % 8 == 0 and not T._in_domain():
                return None
    return (T.solver.export_particle_x_to_torch().clone(),
            T.solver.export_particle_v_to_torch().clone(),
            T.solver.export_particle_F_to_torch().reshape(-1, 9).clone(),
            T.solver.export_particle_C_to_torch().reshape(-1, 9).clone())


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
    for i, (truth, f) in enumerate(zip(refs, forces)):
        st = mpm_to(f, at)
        if st is None:
            continue
        x, v, F, C = st
        # what the anchor state reconstructs for F and C, from THIS configuration
        p = FIT.project(x, cache) if FIT is not None else T.project(x)
        vp = FIT.project_v(v, cache) if FIT is not None else T.project_v(v)
        _x2, _v2, F2, C2 = (FIT.lift(p, vp, cache) if FIT is not None
                            else T.lift(p, vp))
        F2 = F2.reshape(-1, 9); C2 = C2.reshape(-1, 9)
        d = det3(F2.reshape(-1, 3, 3))
        if not torch.isfinite(F2).all() or d.min() < 0.02 or d.max() > 50:
            rows.append((at, i, None, None, None))
            continue

        a = roll((x, v, F, C), args.frames)                 # MPM's own history
        b = roll((x, v, F2, C2), args.frames)               # ours, positions identical
        if a is None or b is None:
            rows.append((at, i, None, None, None))
            continue
        span = (a - a[0]).norm(dim=-1).max().clamp(min=1e-12)
        div = ((a - b).norm(dim=-1).mean(-1) / span)
        dF = (F - F2).reshape(-1, 9).norm(dim=-1) / F.reshape(-1, 9).norm(dim=-1).clamp(min=1e-12)
        rows.append((at, i, float(div.mean()), float(div[-1]), float(dF.mean())))

print(f"\n{'branch at':>10} {'case':>5} {'divergence':>12} {'at last frame':>15} "
       f"{'|F-F2|/|F|':>12}")
for at, i, m, l, df in rows:
    if m is None:
        print(f"{at:10d} {i:5d} {'MPM left the grid':>28}")
    else:
        print(f"{at:10d} {i:5d} {100*m:11.2f}% {100*l:14.2f}% {100*df:11.2f}%")

ok = [r for r in rows if r[2] is not None]
if ok:
    for at in args.at_frames:
        g = [r for r in ok if r[0] == at]
        if g:
            print(f"\n  branching at frame {at}: divergence {100*sum(r[2] for r in g)/len(g):.2f}% "
                  f"mean, {100*sum(r[3] for r in g)/len(g):.2f}% at the last frame, "
                  f"F differs by {100*sum(r[4] for r in g)/len(g):.1f}%")
print("\n  Positions and velocities were identical at the branch, so this is only\n"
       "  what the anchor state cannot carry. Against it, the fitted simulator's\n"
       "  own error is 6.61% (field) and the position floor is 1.13%.")

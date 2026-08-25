"""What is the error at the end of the animation, not at 5% of it?

Every number this project has quoted is a sixty coarse-frame rollout. A coarse
frame is forty MPM substeps, so sixty of them is 0.24 seconds -- and the scene
config asks for 125 frames at 40 ms, five seconds. The whole evaluation has been
measuring the first 4.8% of the motion.

That matters because error accumulates: a discretisation that tracks for a
quarter second may or may not still track after five. This runs both simulators
to the end and reports the error at several points along the way.

Neither trajectory is stored -- five seconds is 1250 frames of 171k particles,
which is 1.3 GB a trajectory in half precision. MPM and the fitted anchors are
stepped side by side and the error is accumulated as they go. The normalisation
is the evaluation's own: the mean per-particle distance over the largest
displacement MPM reaches anywhere in the whole trajectory, so it needs the run
to finish before it can be applied, which is why the per-frame means are kept
and divided at the end.
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
ap.add_argument("--fit", action="append", required=True, help="NAME=path.pt")
ap.add_argument("--n_anchors", type=int, default=512)
ap.add_argument("--K", type=int, default=8)
ap.add_argument("--eig_floor", type=float, default=0.02)
ap.add_argument("--frames", type=int, default=1250,
                 help="coarse frames; 1250 x 40 substeps x 1e-4 = 5 s, the whole "
                      "animation the scene config asks for")
ap.add_argument("--dt_mult", type=int, default=40)
ap.add_argument("--every", type=int, default=10, help="score every N frames")
ap.add_argument("--n_traj", type=int, default=4)
ap.add_argument("--impulse_range", type=float, default=10.0)
ap.add_argument("--seed", type=int, default=31337, help="the held-out stream")
ap.add_argument("--base_force", type=float, nargs=3, default=None,
                 help="scenes whose config carries no particle_impulse need the "
                      "kick named here instead")
ap.add_argument("--out", default=None)
args = ap.parse_args()

sys.path.insert(0, args.dreamphysics)
import warp as wp

from anchorflow.anchor_sparse import load_fitted
from anchorflow.mpm_teacher import MPMTeacher
from anchorflow.streams import draw_impulse

dev = "cuda"
torch.set_grad_enabled(False)
torch.manual_seed(0)
wp.init()

sc = scene_setup.build(args.ply, args.config, args.n_anchors, args.K, device=dev,
                        frozen_weights=True, rot_fallback=True,
                        eig_floor=args.eig_floor)
T = MPMTeacher(sc)
mat = T.mat
if args.base_force is not None:
    base = torch.tensor(args.base_force, device=dev)
else:
    base = next(torch.tensor(bc["force"], device=dev)
                for bc in sc.cfg["boundary_conditions"]
                if bc["type"] == "particle_impulse")

FITS = []
for spec in args.fit:
    name, path = spec.split("=", 1)
    fs, fb = load_fitted(sc, path, dev)
    print(f"[fit] {name}: iteration {fb.get('iter')}, {fs.M} anchors")
    FITS.append((name, fs))

dt_frame = args.dt_mult * sc.sub_dt
print(f"[horizon] {args.frames} coarse frames x {dt_frame * 1000:.0f} ms = "
      f"{args.frames * dt_frame:.2f} s\n")

g = torch.Generator(device=dev); g.manual_seed(args.seed)
forces = [draw_impulse(sc, base, g, args.impulse_range, field=True)[0]
          for _ in range(args.n_traj)]

# frame index -> list over trajectories of the mean distance, and of the span
marks = list(range(args.every, args.frames + 1, args.every))
acc = {n: {k: [] for k in marks} for n, _ in FITS}
spans = []
died = {n: 0 for n, _ in FITS}

for ti, force in enumerate(forces):
    # MPM starts the way the fit's own trajectory generator starts it: the
    # impulse becomes an anchor velocity, which is then spread onto the
    # particles by the same P2G weights. sc.initial_velocity is per ANCHOR.
    ref = FITS[0][1].fit
    ca0 = ref.prepare()
    dv = ref.impulse_dv(force, ca0)
    x0 = T.pos_m.clone()
    v0 = torch.zeros(ref.N, 3, device=dev).index_add_(
        0, ref.pair_g, ca0[0].unsqueeze(-1) * dv[ref.pair_a])
    T._set(x0.clone(), v0.contiguous(), T.eye.clone(), torch.zeros_like(T.eye))

    states = {}
    for name, fs in FITS:
        p, v, gp = fs.anchor_canonical.clone(), fs.initial_velocity(force), fs.pos.clone()
        states[name] = [p, v, gp, True]

    span = torch.zeros((), device=dev)
    per = {n: {} for n, _ in FITS}
    ok = True
    for k in tqdm(range(1, args.frames + 1), desc=f"  traj {ti}", ncols=90):
        for _ in range(args.dt_mult):
            T.solver.p2g2p(None, sc.sub_dt, device=T.wp_dev)
        if not T._in_domain():
            print(f"  MPM left its grid at frame {k}; trajectory dropped")
            ok = False
            break
        truth = T.solver.export_particle_x_to_torch()
        span = torch.maximum(span, (truth - x0).norm(dim=-1).max())

        for name, fs in FITS:
            st = states[name]
            if not st[3]:
                continue
            st[0], st[1], st[2] = fs.explicit_step(st[0], st[1], st[2], args.dt_mult)
            st[2] = fs.skin(st[0], st[2])
            if not torch.isfinite(st[0]).all():
                st[3] = False
                died[name] += 1
                print(f"  {name} went non-finite at frame {k}")
                continue
            if k % args.every == 0:
                per[name][k] = float((st[2][mat] - truth).norm(dim=-1).mean())
    if not ok:
        continue
    spans.append(float(span))
    for name, _ in FITS:
        for k, d in per[name].items():
            acc[name][k].append(d / max(float(span), 1e-12))

print(f"\n[span] MPM's largest displacement, per trajectory: "
      + ", ".join(f"{s:.3f}" for s in spans))
print(f"\n{'time (s)':>9} {'frame':>7} " + " ".join(f"{n:>12}" for n, _ in FITS))
rows = []
for k in marks:
    vals = []
    for n, _ in FITS:
        v = acc[n][k]
        vals.append(100 * sum(v) / len(v) if v else float("nan"))
    rows.append((k, vals))
    if k % (args.every * 5) == 0 or k == marks[-1] or k <= args.every * 6:
        print(f"{k * dt_frame:9.2f} {k:7d} " + " ".join(f"{v:12.2f}" for v in vals))

for n, _ in FITS:
    if died[n]:
        print(f"\n{n}: went non-finite on {died[n]} of {len(forces)} trajectories")

if args.out:
    torch.save({"marks": marks, "rows": rows, "names": [n for n, _ in FITS],
                "dt_frame": dt_frame, "spans": spans}, args.out)
    print(f"\n-> {args.out}")

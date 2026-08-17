"""One row of the evaluation table, as a video.

The table says the fitted discretisation beats the sampled one on the mean of
every impulse family, and on twenty of the twenty-four trajectories underneath
those means. The four it loses are the interesting ones, and a mean cannot say
whether losing is a slightly different bend or a different motion entirely.

The trajectories here are built exactly as exe/eval_vs_mpm.py builds its rows --
from rest, under the impulse it drew, with that script's own initial velocity
and stepping -- and MPM comes from the same cached reference the numbers were
computed against. So what is on screen is the number, not something adjacent to
it.

Three panels: MPM, the sampled anchor simulator, the fitted one. Each is
captioned with its error so far, against MPM, in the same normalisation the
table uses.
"""
from __future__ import annotations

import argparse
import os
import sys

_lib = os.path.join(os.path.dirname(__file__), "..", "lib")
sys.path.insert(0, _lib)

import imageio
import numpy as np
import torch
from tqdm import tqdm

from anchorflow import scene_setup

ap = argparse.ArgumentParser()
ap.add_argument("--ply", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--dreamphysics", default="/workspace/DreamPhysics")
ap.add_argument("--cache", required=True,
                 help="the reference cache eval_vs_mpm.py wrote, so the impulse and "
                      "the MPM trajectory are byte-for-byte the ones scored")
ap.add_argument("--fit", required=True)
ap.add_argument("--cases", nargs="+", default=["field:5"],
                 help="KIND:INDEX, several at a time. Building the scene and the MPM "
                      "wrapper costs more than a case does, so a batch of them shares "
                      "one setup rather than paying it per video.")
ap.add_argument("--n_anchors", type=int, default=512)
ap.add_argument("--K", type=int, default=8)
ap.add_argument("--eig_floor", type=float, default=0.02)
ap.add_argument("--frames", type=int, default=60)
ap.add_argument("--dt_mult", type=int, default=40)
ap.add_argument("--width", type=int, default=640)
ap.add_argument("--height", type=int, default=640)
ap.add_argument("--fov_x", type=float, default=0.6911)
ap.add_argument("--radius_scale", type=float, default=1.6)
ap.add_argument("--fps", type=int, default=12)
ap.add_argument("--out_dir", required=True)
args = ap.parse_args()

sys.path.insert(0, args.dreamphysics)
import warp as wp

from anchorflow.anchor_sparse import load_fitted
from anchorflow.mpm_teacher import MPMTeacher
from anchorflow.view import build_camera, label, make_renderer

dev = "cuda"
torch.set_grad_enabled(False)
torch.manual_seed(0)
wp.init()

sc = scene_setup.build(args.ply, args.config, args.n_anchors, args.K, device=dev,
                        frozen_weights=True, rot_fallback=True,
                        eig_floor=args.eig_floor)
T = MPMTeacher(sc)
mat = T.mat

blob = torch.load(args.cache, map_location=dev, weights_only=False)

fs, fb = load_fitted(sc, args.fit, dev)
print(f"[fit] {args.fit} at iteration {fb.get('iter')}: {fs.M} anchors")

# ---- the zero-volume Gaussians, carried by the material around them --------
#
# A sixth of the cloud carries zero volume and is simulated by nothing, but it
# is interleaved through the foliage and it does get drawn. Left where it
# started it stands still while the tree moves around it, which reads as a ghost
# and belongs to this script rather than to MPM. Built once: the neighbours are
# a property of the canonical scene, not of the case.
K_CARRY = 8
from scipy.spatial import cKDTree

rest = torch.nonzero(~sc.keep, as_tuple=False).squeeze(-1)
_tree = cKDTree(T.pos_m.cpu().numpy())
nd_np, ni_np = _tree.query(sc.pos[rest].cpu().numpy(), k=K_CARRY)
nd = torch.from_numpy(nd_np).float().to(dev)
ni = torch.from_numpy(ni_np).long().to(dev)
cw = 1.0 / nd.clamp(min=1e-6)
cw = cw / cw.sum(-1, keepdim=True)

cam = build_camera(sc, args.width, args.height, args.fov_x, args.radius_scale)
frame = make_renderer(sc, args.ply, cam)
names = ["MPM", "anchor simulator", "fitted to MPM"]
os.makedirs(args.out_dir, exist_ok=True)
summary = []

for case in args.cases:
    kind, idx = case.split(":")
    idx = int(idx)
    truth = blob["ref"][kind][idx][: args.frames + 1]
    force = blob["force"][kind][idx]
    print(f"\n[case] {kind} #{idx} of {len(blob['ref'][kind])}, "
          f"MPM moved {(truth - truth[0]).norm(dim=-1).max():.4f}")

    runs = {}
    p, v, gp = sc.anchor_canonical.clone(), sc.initial_velocity(force), sc.pos.clone()
    out = [gp.clone()]
    for _ in tqdm(range(args.frames), desc="  anchor 8-NN", ncols=90):
        p, v, gp = sc.explicit_step(p, v, gp, args.dt_mult)
        out.append(gp.clone())
    runs["anchor simulator"] = torch.stack(out)

    # FittedScene.explicit_step advances the anchors and hands the cloud back
    # untouched -- skinning is a separate call, and forgetting it renders sixty
    # identical frames. eval_vs_mpm.py does the same two steps in the same order.
    p, v, gp = fs.anchor_canonical.clone(), fs.initial_velocity(force), fs.pos.clone()
    out = [gp.clone()]
    for _ in tqdm(range(args.frames), desc="  fitted", ncols=90):
        p, v, gp = fs.explicit_step(p, v, gp, args.dt_mult)
        gp = fs.skin(p, gp)
        out.append(gp.clone())
    runs["fitted to MPM"] = torch.stack(out)

    mpm_full = []
    for k in range(args.frames + 1):
        q = sc.pos.clone()
        q[mat] = truth[k]
        disp = truth[k] - T.pos_m
        q[rest] = sc.pos[rest] + (cw.unsqueeze(-1) * disp[ni]).sum(1)
        mpm_full.append(q)
    runs["MPM"] = torch.stack(mpm_full)

    span = (truth - truth[0]).norm(dim=-1).max().clamp(min=1e-12)
    err = {n: (runs[n][:, mat] - truth).norm(dim=-1).mean(-1) / span for n in names[1:]}

    vid = []
    for k in tqdm(range(args.frames + 1), desc="  render", ncols=90):
        row = []
        for n in names:
            tag = n if n == "MPM" else f"{n}   {100 * err[n][k]:.1f}%"
            row.append(label(frame(runs[n][k]), tag))
        vid.append(np.concatenate(row, axis=1))
    out_path = os.path.join(args.out_dir, f"{kind}{idx}.mp4")
    imageio.mimsave(out_path, vid, fps=args.fps, quality=8)
    line = (f"{kind} #{idx}: anchor sim {100 * err['anchor simulator'].mean():.2f}% "
            f"(final {100 * err['anchor simulator'][-1]:.2f}%), "
            f"fitted {100 * err['fitted to MPM'].mean():.2f}% "
            f"(final {100 * err['fitted to MPM'][-1]:.2f}%)")
    summary.append(line)
    print(f"  [saved] {out_path}\n  {line}")

print("\n" + "\n".join(summary))

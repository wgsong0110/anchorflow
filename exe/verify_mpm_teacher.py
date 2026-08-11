"""Is MPM usable as a teacher for a student that only sees 512 anchors?

Three things have to hold, and they are separable.

  1. The projection has to be able to say what MPM did. Reducing MPM's own
     particle trajectory to anchors and skinning it back is a floor on every
     student, learned or not.

  2. The lift has to be able to restart MPM from an anchor state. The student
     will be rolled out and the teacher asked from wherever it got to, which
     means handing the solver a particle state reconstructed from 512 anchors.
     Whatever MPM's per-particle deformation gradient carries that the anchors
     cannot say is lost at that moment, and this measures it: take a real MPM
     state, go down to anchors and back up, continue, and compare against the
     continuation MPM would have had.

  3. The teacher has to differ from the anchor simulator by enough to be worth
     the 5x. If MPM-from-anchors and the anchor simulator agree, the whole
     exercise changes nothing.

The lift's affine part is measured rather than assumed: MPM's C is a velocity
gradient, and whether reconstructing it from the anchors beats setting it to
zero is not obvious in advance.
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
ap.add_argument("--frames", type=int, default=60)
ap.add_argument("--dt_mult", type=int, default=40)
ap.add_argument("--n_anchors", type=int, default=512)
ap.add_argument("--K", type=int, default=8)
ap.add_argument("--horizon", type=int, default=4,
                 help="coarse steps a DAgger label covers")
ap.add_argument("--at", type=int, nargs="+", default=[2, 10, 25, 45],
                 help="frames to restart from")
ap.add_argument("--eig_floor", type=float, default=0.02)
ap.add_argument("--anchors", default=None, help="a refitted placement to use instead")
args = ap.parse_args()

sys.path.insert(0, args.dreamphysics)
import warp as wp

from anchorflow.mpm_teacher import MPMTeacher

dev = "cuda"
torch.set_grad_enabled(False)
torch.manual_seed(0)
wp.init()

A = torch.load(args.anchors, map_location=dev)["anchor_canonical"].to(dev) \
    if args.anchors else None
sc = scene_setup.build(args.ply, args.config, args.n_anchors, args.K, device=dev,
                        frozen_weights=True, rot_fallback=True, eig_floor=args.eig_floor,
                        anchors=A)
base = None
for bc in sc.cfg.get("boundary_conditions", []):
    if bc["type"] == "particle_impulse":
        base = torch.tensor(bc["force"], device=dev)
T = MPMTeacher(sc)
mat = T.mat
print(f"[setup] {sc.M} anchors, {T.n} material particles, eig_floor {args.eig_floor}"
      + (f", anchors from {args.anchors}" if args.anchors else ""))

# ---- the reference MPM trajectory, kept as full particle state -------------
dv = sc.impulse_dv(base)
v0 = (T.w.unsqueeze(-1) * dv[T.idx]).sum(1).contiguous()
T._set(T.pos_m.clone(), v0, T.eye.clone(), torch.zeros_like(T.eye))
X, V = [T.pos_m.clone()], [v0.clone()]
for _ in tqdm(range(args.frames), desc="MPM", ncols=90):
    for _ in range(args.dt_mult):
        T.solver.p2g2p(None, sc.sub_dt, device=dev)
    X.append(T.solver.export_particle_x_to_torch().clone())
    V.append(T.solver.export_particle_v_to_torch().clone())
X, V = torch.stack(X), torch.stack(V)
span = (X - X[0]).norm(dim=-1).max().item()
print(f"[reference] peak particle displacement {span:.5f}")

# ---- 1. the projection floor ----------------------------------------------
P = torch.stack([T.project(X[t]) for t in range(X.shape[0])])
floor = torch.stack([(sc.skin(P[t], sc.pos.clone())[mat] - X[t]).norm(dim=-1).mean()
                     for t in range(X.shape[0])])
print(f"\n[1. projection floor] {100 * floor.mean() / span:.2f}% of peak "
      f"(final frame {100 * floor[-1] / span:.2f}%)")
print(f"    what 512 anchors can say about MPM's trajectory at all")

# ---- 2. the lift ----------------------------------------------------------
print(f"\n[2. restarting MPM from an anchor state] {args.horizon} coarse steps")
print(f"  {'from frame':>10} {'C fitted':>10} {'C = 0':>10} {'no restart':>12}")
for t in args.at:
    if t + args.horizon >= X.shape[0]:
        continue
    p, v = T.project(X[t]), T.project_v(V[t])
    truth = torch.stack([X[t + 1 + j] for j in range(args.horizon)])
    row = []
    for affine in (True, False):
        T.affine = affine
        got = T.query(p, v, args.horizon, args.dt_mult)          # anchors
        gx = torch.stack([sc.skin(got[j], sc.pos.clone())[mat] for j in range(args.horizon)])
        row.append(100 * (gx - truth).norm(dim=-1).mean() / span)
    T.affine = True
    # the same horizon of MPM's own trajectory, projected and skinned: the part
    # of the error that is the anchor state's size rather than the restart
    ref = torch.stack([sc.skin(T.project(X[t + 1 + j]), sc.pos.clone())[mat]
                       for j in range(args.horizon)])
    print(f"  {t:10d} {row[0]:9.2f}% {row[1]:9.2f}% "
          f"{100 * (ref - truth).norm(dim=-1).mean() / span:11.2f}%")
print(f"    'no restart' is the projection floor over the same window -- the "
      f"restart is\n    only worth what it adds on top of that")

# ---- 3. is the teacher different from the anchor simulator? ---------------
p_, v_, gp_ = sc.anchor_canonical.clone(), sc.initial_velocity(base), sc.pos.clone()
anc = [gp_[mat].clone()]
for _ in tqdm(range(args.frames), desc="anchors", ncols=90):
    p_, v_, gp_ = sc.explicit_step(p_, v_, gp_, args.dt_mult)
    anc.append(gp_[mat].clone())
ANC = torch.stack(anc)
print(f"\n[3. teacher vs the anchor simulator] "
      f"{100 * (ANC - X).norm(dim=-1).mean(-1).mean() / span:.2f}% of peak")
print(f"    if this were near the projection floor there would be nothing to gain")

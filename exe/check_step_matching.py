"""Can the anchor simulator be fitted to MPM one coarse step at a time?

Matching instantaneous accelerations does not work, and the reason is not that
the configuration is off-manifold. It is that the two quantities are not
comparable: on its own trajectory the anchor simulator's |f/m| is around 175
while MPM's projected acceleration is around 19, because the projection averages
thousands of particles and erases a high-frequency component that the anchor
simulator carries and that averages out over a coarse step anyway.

So compare what survives a coarse step. Take a state from MPM's own trajectory,
project it to anchors, hand it to the anchor simulator, run the 40 substeps that
make one frame, and ask where it lands against where MPM landed. That is the
quantity the whole project cares about, and it is what a fitting loss should be
built from.

Two references make the number readable. The projection floor over one step is
what a perfect stepper would still lose. And re-running MPM from its own state
is zero by construction, so any gap is the anchor simulator's dynamics.
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
ap.add_argument("--at", type=int, nargs="+", default=[2, 5, 10, 20, 35, 50])
ap.add_argument("--dt_mult", type=int, default=40)
ap.add_argument("--n_anchors", type=int, default=512)
ap.add_argument("--K", type=int, default=8)
ap.add_argument("--eig_floor", type=float, default=0.02)
args = ap.parse_args()

sys.path.insert(0, args.dreamphysics)
import warp as wp

from anchorflow.mpm_teacher import MPMTeacher

dev = "cuda"
torch.set_grad_enabled(False)
torch.manual_seed(0)
wp.init()

sc = scene_setup.build(args.ply, args.config, args.n_anchors, args.K, device=dev,
                        frozen_weights=True, rot_fallback=True, eig_floor=args.eig_floor)
AC, fixed = sc.anchor_canonical, sc.fixed_mask
T = MPMTeacher(sc)
base = None
for bc in sc.cfg.get("boundary_conditions", []):
    if bc["type"] == "particle_impulse":
        base = torch.tensor(bc["force"], device=dev)
mat = T.mat

dv = sc.impulse_dv(base)
v0 = (T.w.unsqueeze(-1) * dv[T.idx]).sum(1).contiguous()
T._set(T.pos_m.clone(), v0, T.eye.clone(), torch.zeros_like(T.eye))
states = {}
for f in tqdm(range(max(args.at) + 2), desc="MPM", ncols=90):
    for _ in range(args.dt_mult):
        T.solver.p2g2p(None, sc.sub_dt, device=T.wp_dev)
    if f in states or f in args.at or (f - 1) in args.at:
        states[f] = (T.project(T.solver.export_particle_x_to_torch()),
                     T.project_v(T.solver.export_particle_v_to_torch()),
                     T.solver.export_particle_x_to_torch().clone())

print(f"\n{'frame':>6} {'MPM moved':>11} {'anchor sim':>11} {'error':>9} "
      f"{'% of step':>10} {'floor':>9}")
tot = []
for f in args.at:
    if f not in states or (f + 1) not in states:
        continue
    p, v, x = states[f]
    p1, _, x1 = states[f + 1]
    # one coarse frame of the anchor simulator, from MPM's state
    q, w, g = p.clone(), v.clone(), sc.skin(p, sc.pos.clone())
    q, w, g = sc.explicit_step(q, w, g, args.dt_mult)
    step = (p1 - p)[~fixed].norm(dim=-1)
    err = (q - p1)[~fixed].norm(dim=-1)
    # what a perfect stepper still loses over the same step: MPM's own next
    # state, projected and skinned back out, against MPM's own particles
    fl = (sc.skin(p1, sc.pos.clone())[mat] - x1).norm(dim=-1).mean()
    span = (x1 - T.pos_m).norm(dim=-1).max()
    print(f"{f:6d} {step.mean():11.6f} {(q - p)[~fixed].norm(dim=-1).mean():11.6f} "
          f"{err.mean():9.6f} {100 * err.mean() / step.mean().clamp(min=1e-12):9.1f}% "
          f"{100 * fl / span:8.2f}%")
    tot.append((err.mean() / step.mean().clamp(min=1e-12)).item())

print(f"\n[verdict] the anchor simulator misses MPM's next state by "
      f"{100 * sum(tot) / len(tot):.0f}% of\n          the step itself. Under about "
      f"30% there is a signal a fitting loss can follow;\n          near or above "
      f"100% the step is uncorrelated with the target and fitting the\n          "
      f"discretisation to it will not converge.")

"""Can the anchor simulator be fitted to MPM by matching accelerations?

Fitting the anchor discretisation -- where the anchors sit, how far and in which
direction each one reaches -- needs a loss whose gradient reaches those
parameters. The trajectory cannot supply one: the force is an analytic CUDA
kernel with no backward, and the torch path differentiates through a polar
decomposition that is singular wherever the motion is near-rigid, which is most
of it.

Matching accelerations at states taken from MPM's own trajectory avoids all of
that -- one step, no unrolling. It was tried once and failed: the anchor
simulator's |f/m| at an MPM configuration came out around 59,000 against 170 on
its own trajectory, so the configuration was off-manifold and matching forces
there meant nothing.

Two things have changed since. The projection now averages displacement rather
than position, which removed a fixed offset that was in every frame including
the first (the floor went from 2.44% to 0.75%). And the deformation gradient's
unobserved directions now rotate with the body instead of freezing, which was
worth 21% to 5.5%.

So this asks the gate question again. Accelerations are compared at substep
resolution, since the anchor simulator's is instantaneous and a finite
difference over a coarse frame is not the same quantity. The anchor
simulator on its OWN trajectory is the control: whatever disagreement that
shows is the measurement's own error, and only what exceeds it belongs to MPM.
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
ap.add_argument("--at", type=int, nargs="+", default=[2, 5, 10, 20, 35, 50],
                 help="coarse frames to compare at")
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
sub = sc.sub_dt
print(f"[setup] {sc.M} anchors, substep {sub}, damping {sc.damping}, "
      f"g {sc.gravity.tolist()}")
if sc.damping != 1.0:
    print(f"[warn] the grid damping is not 1, so MPM's finite-difference "
          f"acceleration carries a drag term the elastic force does not")


def accel_of(p_prev, p_now, p_next, dt):
    return (p_next - 2 * p_now + p_prev) / (dt * dt)


def compare(a_ref, a_sim, tag):
    """magnitude ratio and direction agreement over the anchors that move"""
    m = ~fixed
    r, s = a_ref[m], a_sim[m]
    nr, ns = r.norm(dim=-1), s.norm(dim=-1)
    cos = (r * s).sum(-1) / (nr * ns).clamp(min=1e-20)
    # weighted by the reference's own magnitude: a direction is only meaningful
    # where there is an acceleration to have a direction
    w = nr / nr.sum().clamp(min=1e-20)
    print(f"  {tag:>22} |a| ref {nr.mean():9.2f}  sim {ns.mean():9.2f}  "
          f"ratio {ns.mean() / nr.mean().clamp(min=1e-20):7.2f}  "
          f"cos {(w * cos).sum():6.3f}  median cos {cos.median():6.3f}")
    return (ns.mean() / nr.mean().clamp(min=1e-20)).item(), (w * cos).sum().item()


# ---- MPM, with three consecutive substeps captured at each checkpoint -------
dv = sc.impulse_dv(base)
v0 = (T.w.unsqueeze(-1) * dv[T.idx]).sum(1).contiguous()
T._set(T.pos_m.clone(), v0, T.eye.clone(), torch.zeros_like(T.eye))
mpm_states = {}
frame = 0
for f in tqdm(range(max(args.at) + 1), desc="MPM", ncols=90):
    for _ in range(args.dt_mult):
        T.solver.p2g2p(None, sub, device=T.wp_dev)
    frame += 1
    if frame in args.at:
        trio = [T.project(T.solver.export_particle_x_to_torch())]
        for _ in range(2):
            T.solver.p2g2p(None, sub, device=T.wp_dev)
            trio.append(T.project(T.solver.export_particle_x_to_torch()))
        mpm_states[frame] = torch.stack(trio)

# ---- the anchor simulator's own trajectory, same treatment ------------------
p, v, gp = AC.clone(), sc.initial_velocity(base), sc.pos.clone()
anc_states = {}
for f in tqdm(range(max(args.at) + 1), desc="anchors", ncols=90):
    p, v, gp = sc.explicit_step(p, v, gp, args.dt_mult)
    if f + 1 in args.at:
        trio, q, w_, g2 = [p.clone()], p.clone(), v.clone(), gp.clone()
        for _ in range(2):
            q, w_, g2 = sc.explicit_step(q, w_, g2, 1)
            trio.append(q.clone())
        anc_states[f + 1] = (torch.stack(trio), gp.clone())

print(f"\n[control] the anchor simulator at its OWN states -- the finite difference "
      f"against\n          the force it just used. Whatever this misses is the "
      f"measurement's error.")
for f in args.at:
    trio, g = anc_states[f]
    compare(accel_of(trio[0], trio[1], trio[2], sub),
            sc.elastic_accel(trio[1], sc.skin(trio[1], g.clone())), f"frame {f}")

print(f"\n[test] the anchor simulator's force at MPM's configuration, against what "
      f"MPM's\n       own particles are doing there")
rat, cs = [], []
for f in args.at:
    trio = mpm_states[f]
    a_ref = accel_of(trio[0], trio[1], trio[2], sub)
    a_sim = sc.elastic_accel(trio[1], sc.skin(trio[1], sc.pos.clone()))
    r, c = compare(a_ref, a_sim, f"frame {f}")
    rat.append(r); cs.append(c)

print(f"\n[verdict] magnitude ratio {sum(rat)/len(rat):.2f} on average, "
      f"direction {sum(cs)/len(cs):.3f}")
print(f"          A ratio near 1 and a positive cosine mean the anchor simulator is "
      f"being\n          asked about a configuration it understands, and fitting its "
      f"parameters to\n          these accelerations is meaningful. A ratio in the "
      f"hundreds means the\n          configuration is off-manifold and the earlier "
      f"failure stands.")

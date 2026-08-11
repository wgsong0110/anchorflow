"""Can the anchor simulator be made to match MPM by fitting its parameters?

With the material, the impulse and the boundary conditions all matched, the
anchor simulator still moves 27% as far as MPM and differs by 21% of peak
displacement over 60 frames. The projection floor is 2.4%, so 512 anchors can
represent MPM's trajectory -- it is the dynamics that are off, and the anchor
discretisation is much stiffer than a 100^3 grid over 171k particles.

If that is a stiffness offset, one scalar fixes it, and everything downstream
(cheap data generation, DAgger against a queryable expert, the whole training
pipeline) survives unchanged. If it is not, no scalar will do and the teacher
has to become MPM itself.

The distinction is visible in what a stiffness change does: it scales the
amplitude AND the oscillation period together. If MPM is bigger and slower by
the same factor, a scalar matches both. If MPM is bigger at the same speed,
nothing does.

Fitted on one impulse and checked on others, because a scalar can always be made
to fit one trajectory.
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
from anchorflow.streams import rand_rot

ap = argparse.ArgumentParser()
ap.add_argument("--ply", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--dreamphysics", default="/workspace/DreamPhysics")
ap.add_argument("--frames", type=int, default=60)
ap.add_argument("--dt_mult", type=int, default=40)
ap.add_argument("--n_anchors", type=int, default=512)
ap.add_argument("--K", type=int, default=8)
ap.add_argument("--n_grid", type=int, default=100)
ap.add_argument("--grid_lim", type=float, default=2.0)
ap.add_argument("--n_check", type=int, default=3,
                 help="held-out impulses the fitted parameters are checked on")
ap.add_argument("--scales", type=float, nargs="+",
                 default=[1.0, 0.5, 0.25, 0.125, 0.0729, 0.05, 0.03, 0.02])
args = ap.parse_args()

sys.path.insert(0, args.dreamphysics)
import warp as wp
from mpm_solver_warp.mpm_solver_warp import MPM_Simulator_WARP

dev = "cuda"
torch.set_grad_enabled(False)
torch.manual_seed(0)
wp.init()

sc = scene_setup.build(args.ply, args.config, args.n_anchors, args.K,
                        n_grid=args.n_grid, grid_lim=args.grid_lim, device=dev,
                        frozen_weights=True)
AC, fixed, keep = sc.anchor_canonical, sc.fixed_mask, sc.keep
cfg = sc.cfg
mat = torch.nonzero(keep, as_tuple=False).squeeze(-1)
pos_m, vol_m = sc.pos[mat].contiguous(), sc.volume[mat].contiguous()
base_force = None
for bc in cfg.get("boundary_conditions", []):
    if bc["type"] == "particle_impulse":
        base_force = torch.tensor(bc["force"], device=dev)
MU0, LAM0 = sc.mu.clone(), sc.lam.clone()
W0 = sc.sim._canonical_weights()


def mpm_run(force):
    """MPM from the velocity the anchors carry, so both start from one motion"""
    solver = MPM_Simulator_WARP(10)
    solver.load_initial_data_from_torch(
        pos_m, vol_m, torch.zeros((mat.shape[0], 6), device=dev),
        n_grid=args.n_grid, grid_lim=args.grid_lim)
    mp = {k: cfg[k] for k in ("E", "nu", "density", "material") if k in cfg}
    mp.update({"n_grid": args.n_grid, "grid_lim": args.grid_lim,
               "g": cfg.get("g", [0, 0, 0]),
               "grid_v_damping_scale": cfg.get("grid_v_damping_scale", 1.0)})
    if "additional_material_params" in cfg:
        mp["additional_material_params"] = cfg["additional_material_params"]
    solver.set_parameters_dict(mp)
    solver.finalize_mu_lam()
    for bc in cfg.get("boundary_conditions", []):
        if bc["type"] == "cuboid":
            solver.set_velocity_on_cuboid(bc["point"], bc["size"], [0.0, 0.0, 0.0],
                                           start_time=0.0, end_time=999.0, reset=1)
    dv = sc.impulse_dv(force)
    v0 = (W0[mat].unsqueeze(-1) * dv[sc.sim.nn_idx[mat]]).sum(1).contiguous()
    solver.import_particle_v_from_torch(v0)
    out = [pos_m.clone()]
    for _ in range(args.frames):
        for _ in range(args.dt_mult):
            solver.p2g2p(None, sc.sub_dt, device=dev)
        out.append(solver.export_particle_x_to_torch().clone())
    return torch.stack(out)


def anchor_run(force, scale, damp=None):
    old_mu, old_lam, old_d = sc.mu, sc.lam, sc.damping
    sc.mu, sc.lam = MU0 * scale, LAM0 * scale
    if damp is not None:
        sc.damping = damp
    p, v, gp = AC.clone(), sc.initial_velocity(force), sc.pos.clone()
    out = [gp[mat].clone()]
    for _ in range(args.frames):
        p, v, gp = sc.explicit_step(p, v, gp, args.dt_mult)
        if not torch.isfinite(p).all():
            out = None
            break
        out.append(gp[mat].clone())
    sc.mu, sc.lam, sc.damping = old_mu, old_lam, old_d
    return None if out is None else torch.stack(out)


def score(A, M):
    span = (M - M[0]).norm(dim=-1).max().clamp(min=1e-12)
    return ((A - M).norm(dim=-1).mean(-1) / span).mean().item(), \
           ((A - A[0]).norm(dim=-1).max() / span).item()


MPM0 = mpm_run(base_force)
print(f"[reference] MPM peak {(MPM0 - MPM0[0]).norm(dim=-1).max():.5f}\n")
print(f"  {'stiffness x':>12} {'error':>9} {'motion vs MPM':>15}")
best, best_e = None, 1e9
for s in args.scales:
    A = anchor_run(base_force, s)
    if A is None:
        print(f"  {s:12.4f} {'diverged':>9}")
        continue
    e, m = score(A, MPM0)
    print(f"  {s:12.4f} {100*e:8.2f}% {100*m:14.0f}%")
    if e < best_e:
        best, best_e = s, e
print(f"\n[fit] stiffness x{best:.4f} gives {100*best_e:.2f}%, against "
      f"{100*score(anchor_run(base_force, 1.0), MPM0)[0]:.2f}% at the original")

# a stiffness change moves amplitude and period together; if only one of them
# needed changing, the fit will not hold at other impulses
print(f"\n[check] the same scale on {args.n_check} other impulses")
gen = torch.Generator(device=dev); gen.manual_seed(20260811)
print(f"  {'impulse':>8} {'original':>10} {'fitted':>10} {'motion vs MPM':>15}")
for i in range(args.n_check):
    s_ = 0.5 * (4.0 ** torch.rand(1, device=dev, generator=gen).item())
    f = (rand_rot(gen, dev) @ base_force) * s_
    M = mpm_run(f)
    e0, _ = score(anchor_run(f, 1.0), M)
    e1, m1 = score(anchor_run(f, best), M)
    print(f"  {i:8d} {100*e0:9.2f}% {100*e1:9.2f}% {100*m1:14.0f}%")
print(f"\n[note] a scalar can always be fitted to one trajectory. What decides "
      f"whether the\n       anchor discretisation approximates MPM at all is "
      f"whether it holds on the others.")

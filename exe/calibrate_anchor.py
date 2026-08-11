"""Can the anchor discretisation be made to match MPM, with the material fixed?

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
ap.add_argument("--eig_floors", type=float, nargs="+",
                 default=[0.2, 0.1, 0.05, 0.02, 0.01, 0.002, 0.0])
ap.add_argument("--radius_scales", type=float, nargs="+", default=[1.0, 1.5, 2.0, 3.0])
ap.add_argument("--anchors", type=int, nargs="+", default=[512])
ap.add_argument("--Ks", type=int, nargs="+", default=[8])
ap.add_argument("--rot_fallbacks", type=int, nargs="+", default=[0],
                 help="0 freezes the unspanned directions, 1 rotates them with the body")
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


_SCENES = {}


def scene_for(M, K, rs, ef, rf=0):
    """the material never changes here; only how the anchors discretise it"""
    key = (M, K, rs)
    if key not in _SCENES:
        _SCENES[key] = scene_setup.build(args.ply, args.config, M, K,
                                          n_grid=args.n_grid, grid_lim=args.grid_lim,
                                          device=dev, frozen_weights=True,
                                          radius_scale=rs)
    s_ = _SCENES[key]
    s_.sim.eig_floor = ef
    s_.sim.rot_fallback = bool(rf)
    return s_


def anchor_run(force, M, K, rs, ef, rf=0):
    s_ = scene_for(M, K, rs, ef, rf)
    p, v, gp = s_.anchor_canonical.clone(), s_.initial_velocity(force), s_.pos.clone()
    out = [gp[mat].clone()]
    for _ in range(args.frames):
        p, v, gp = s_.explicit_step(p, v, gp, args.dt_mult)
        if not torch.isfinite(p).all():
            return None
        out.append(gp[mat].clone())
    return torch.stack(out)


def score(A, M):
    span = (M - M[0]).norm(dim=-1).max().clamp(min=1e-12)
    return ((A - M).norm(dim=-1).mean(-1) / span).mean().item(), \
           ((A - A[0]).norm(dim=-1).max() / span).item()


MPM0 = mpm_run(base_force)
print(f"[reference] MPM peak {(MPM0 - MPM0[0]).norm(dim=-1).max():.5f}, "
      f"material held at the config's values throughout\n")
print(f"  {'anchors':>8} {'K':>3} {'radius x':>9} {'eig floor':>10} {'blocked':>9} "
      f"{'error':>9} {'motion vs MPM':>15}")
best, best_e = None, 1e9
for M in args.anchors:
    for K in args.Ks:
        for rs in args.radius_scales:
            for ef in args.eig_floors:
                for rf in args.rot_fallbacks:
                    A = anchor_run(base_force, M, K, rs, ef, rf)
                    tag = "rotate" if rf else "freeze"
                    if A is None:
                        print(f"  {M:8d} {K:3d} {rs:9.2f} {ef:10.4f} {tag:>9} "
                              f"{'diverged':>9}")
                        continue
                    e, m = score(A, MPM0)
                    print(f"  {M:8d} {K:3d} {rs:9.2f} {ef:10.4f} {tag:>9} "
                          f"{100*e:8.2f}% {100*m:14.0f}%")
                    if e < best_e:
                        best, best_e = (M, K, rs, ef, rf), e
print(f"\n[fit] anchors {best[0]}, K {best[1]}, radius x{best[2]}, eig floor "
      f"{best[3]}, {'rotate' if best[4] else 'freeze'} gives {100*best_e:.2f}%")

# a stiffness change moves amplitude and period together; if only one of them
# needed changing, the fit will not hold at other impulses
print(f"\n[check] the same scale on {args.n_check} other impulses")
gen = torch.Generator(device=dev); gen.manual_seed(20260811)
print(f"  {'impulse':>8} {'original':>10} {'fitted':>10} {'motion vs MPM':>15}")
for i in range(args.n_check):
    s_ = 0.5 * (4.0 ** torch.rand(1, device=dev, generator=gen).item())
    f = (rand_rot(gen, dev) @ base_force) * s_
    MM = mpm_run(f)
    e0, _ = score(anchor_run(f, 512, 8, 1.0, 0.2, 0), MM)
    e1, m1 = score(anchor_run(f, *best), MM)
    print(f"  {i:8d} {100*e0:9.2f}% {100*e1:9.2f}% {100*m1:14.0f}%")
print(f"\n[note] the material is the config's throughout. What is fitted is how "
      f"finely and\n       how forgivingly the anchors discretise it, which is "
      f"what those parameters are.")

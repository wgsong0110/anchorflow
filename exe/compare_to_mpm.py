"""How far is the anchor simulator from the MPM it replaces?

Every number in this project measures a learned stepper against the anchor
simulator. Nobody has measured the anchor simulator against PhysGaussian's MPM,
which is the thing the whole line of work is meant to reproduce. If that gap is
large then the learned model's 3.33% is an accurate reproduction of the wrong
answer.

Two quantities, and they mean different things:

  * anchor simulator vs MPM. How much of the difference from PhysGaussian is
    already spent before any network is trained.
  * the projection floor. Take MPM's own particle trajectory, reduce it to 512
    anchor positions and skin it back out. Nothing with this state can do
    better than that, learned or not, so if the floor is large the state is too
    small and no amount of training fixes it.

Both simulators are driven from the same particles in the same coordinates with
the same impulse, rather than by reproducing DreamPhysics's own preprocessing --
the point is to compare the physics, not two coordinate conventions.
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
ap.add_argument("--impulse", type=float, default=1.0)
# the config carries no grid: scene_setup.build supplies these, and the two
# simulators have to be given the same ones or they are not comparable
ap.add_argument("--match_v0", action="store_true",
                 help="start MPM from the velocity the anchors actually carry, instead of the "
                      "per-particle one PhysGaussian's impulse produces. The two are the same "
                      "momentum but not the same motion: the config's impulse is applied as "
                      "f*dt/(rho*V) per particle, and V varies enough across the cloud that "
                      "averaging it onto anchors throws away most of the kinetic energy. "
                      "Without this the comparison confounds a difference in dynamics with a "
                      "difference in how the impulse is delivered.")
ap.add_argument("--n_grid", type=int, default=100)
ap.add_argument("--grid_lim", type=float, default=2.0)
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
base_force = None
for bc in cfg.get("boundary_conditions", []):
    if bc["type"] == "particle_impulse":
        base_force = torch.tensor(bc["force"], device=dev)
force = base_force * args.impulse

mat = torch.nonzero(keep, as_tuple=False).squeeze(-1)
pos_m = sc.pos[mat].contiguous()
vol_m = sc.volume[mat].contiguous()
print(f"[setup] {sc.N} Gaussians, {mat.shape[0]} material, {sc.M} anchors; "
      f"grid {args.n_grid}^3 over {args.grid_lim}")

# the impulse as DreamPhysics applies it: per particle, v += (f/m) dt
dens = torch.full((mat.shape[0],), float(cfg["density"]), device=dev)
for reg in cfg.get("additional_material_params", []):
    c = torch.tensor(reg["point"], device=dev)
    s = torch.tensor(reg["size"], device=dev)
    dens[((pos_m - c).abs() <= s).all(-1)] = reg["density"]
v0_particle = force.view(1, 3) * sc.sub_dt / (dens * vol_m).clamp(min=1e-20).unsqueeze(-1)

# ---- MPM ------------------------------------------------------------------
solver = MPM_Simulator_WARP(10)
solver.load_initial_data_from_torch(
    pos_m, vol_m, torch.zeros((mat.shape[0], 6), device=dev),
    n_grid=args.n_grid, grid_lim=args.grid_lim)
mp = {k: cfg[k] for k in ("E", "nu", "density", "material") if k in cfg}
mp["n_grid"] = args.n_grid
mp["grid_lim"] = args.grid_lim
mp.setdefault("material", "jelly")
mp["g"] = cfg.get("g", [0, 0, 0])
mp["grid_v_damping_scale"] = cfg.get("grid_v_damping_scale", 1.0)
solver.set_parameters_dict(mp)
# region-varying material is what makes ficus behave the way it does (soft
# canopy, stiff trunk), and the solver takes it per particle
E = torch.full((mat.shape[0],), float(cfg["E"]), device=dev)
NU = torch.full((mat.shape[0],), float(cfg["nu"]), device=dev)
for reg in cfg.get("additional_material_params", []):
    c = torch.tensor(reg["point"], device=dev)
    s_ = torch.tensor(reg["size"], device=dev)
    ins = ((pos_m - c).abs() <= s_).all(-1)
    E[ins] = reg["E"]; NU[ins] = reg["nu"]
if hasattr(solver, "set_E_nu_from_torch"):
    solver.set_E_nu_from_torch(E, NU, device=dev)
solver.finalize_mu_lam()
for bc in cfg.get("boundary_conditions", []):
    if bc["type"] == "cuboid":
        solver.set_velocity_on_cuboid(bc["point"], bc["size"], [0.0, 0.0, 0.0],
                                       start_time=0.0, end_time=999.0, reset=1)
if args.match_v0:
    # the anchors' own initial velocity, put back on the particles the same way
    # skinning would: both simulators then start from identical motion
    dv_anchor = sc.impulse_dv(force)
    W0 = sc.sim._canonical_weights()
    v0_particle = (W0[mat].unsqueeze(-1) * dv_anchor[sc.sim.nn_idx[mat]]).sum(1)
    print(f"[impulse] matched: |v| max {v0_particle.norm(dim=-1).max():.4f}, "
          f"mean {v0_particle.norm(dim=-1).mean():.4f}")
else:
    print(f"[impulse] PhysGaussian's per-particle f*dt/(rho*V): "
          f"|v| max {v0_particle.norm(dim=-1).max():.4f}, "
          f"mean {v0_particle.norm(dim=-1).mean():.4f}")
solver.import_particle_v_from_torch(v0_particle.contiguous())

mpm = [pos_m.clone()]
for _ in tqdm(range(args.frames), desc="MPM", ncols=90):
    for _ in range(args.dt_mult):
        solver.p2g2p(None, sc.sub_dt, device=dev)
    mpm.append(solver.export_particle_x_to_torch().clone())
MPM = torch.stack(mpm)

# ---- the anchor simulator on the same impulse ------------------------------
p, v, gp = AC.clone(), sc.initial_velocity(force), sc.pos.clone()
anc = [gp[mat].clone()]
for _ in tqdm(range(args.frames), desc="anchors", ncols=90):
    p, v, gp = sc.explicit_step(p, v, gp, args.dt_mult)
    anc.append(gp[mat].clone())
ANC = torch.stack(anc)

# ---- the projection floor --------------------------------------------------
# MPM's own particles, reduced to anchors and skinned back out. Nothing with a
# 512-anchor state can beat this, learned or not.
W = sc.sim._canonical_weights()                        # [N,K], canonical
idx = sc.sim.nn_idx
wm = W[mat]
den = torch.zeros(sc.M, device=dev).index_add_(
    0, idx[mat].reshape(-1), wm.reshape(-1)).clamp(min=1e-12)
proj = []
for t in tqdm(range(MPM.shape[0]), desc="projection", ncols=90):
    num = torch.zeros(sc.M, 3, device=dev).index_add_(
        0, idx[mat].reshape(-1), (wm.unsqueeze(-1) * MPM[t].unsqueeze(1)).reshape(-1, 3))
    p_hat = num / den.unsqueeze(-1)
    p_hat = torch.where(fixed.unsqueeze(-1), AC, p_hat)
    proj.append(sc.skin(p_hat, sc.pos.clone())[mat])
PROJ = torch.stack(proj)

span = (MPM - MPM[0]).norm(dim=-1).max().item()
print(f"\n[reference] MPM peak particle displacement {span:.5f}")
print(f"  {'frame':>6} {'anchor vs MPM':>15} {'% of peak':>10} "
      f"{'projection floor':>18} {'% of peak':>10}")
n = MPM.shape[0]
for i in range(0, n, max(1, n // 12)):
    a = (ANC[i] - MPM[i]).norm(dim=-1).mean().item()
    q = (PROJ[i] - MPM[i]).norm(dim=-1).mean().item()
    print(f"  {i:6d} {a:15.5f} {100*a/span:9.2f}% {q:18.5f} {100*q/span:9.2f}%")
ae = (ANC - MPM).norm(dim=-1).mean(-1)
pe = (PROJ - MPM).norm(dim=-1).mean(-1)
print(f"\n[anchor simulator vs MPM] all-frame mean {100*ae.mean()/span:.2f}% of peak, "
      f"final {100*ae[-1]/span:.2f}%")
print(f"[projection floor]        all-frame mean {100*pe.mean()/span:.2f}% of peak, "
      f"final {100*pe[-1]/span:.2f}%")
print(f"[motion] MPM {span:.5f}, anchors {(ANC - ANC[0]).norm(dim=-1).max():.5f} "
      f"({100 * (ANC - ANC[0]).norm(dim=-1).max() / span:.0f}%)")
print(f"\n[note] the projection floor is what a 512-anchor state can represent of MPM's\n"
      f"       own trajectory. A learned stepper on this state cannot do better.")

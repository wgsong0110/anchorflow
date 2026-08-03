"""Check that skinning the opacity-rejected Gaussians with zero volume leaves
the physics untouched.

Path A: only the material Gaussians are in the simulation (what the filtered
        version did).
Path B: every Gaussian is in the simulation, the rejected ones with volume 0.

The anchor trajectories must agree to floating-point noise. The failure mode
worth ruling out is 0 * psi -> NaN if a passenger Gaussian ever produces a
non-finite energy density.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_lib = os.path.join(os.path.dirname(__file__), "..", "lib")
sys.path.insert(0, _lib)

import torch
from tqdm import tqdm

from anchorflow.anchors import AnchorSet
from anchorflow.anchor_mpm import AnchorElasticSim, lame_from_E_nu

ap = argparse.ArgumentParser()
ap.add_argument("--ply", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--n_anchors", type=int, default=512)
ap.add_argument("--K", type=int, default=8)
ap.add_argument("--steps", type=int, default=2000)
args = ap.parse_args()

from scene.gaussian_model import GaussianModel

dev = "cuda"
torch.set_grad_enabled(False)
cfg = json.load(open(args.config))

g = GaussianModel(3, fea_dim=0)
g.load_ply(args.ply)
xyz = g.get_xyz.detach().clone()
op = g.get_opacity.detach().clone()
keep = op[:, 0] > cfg["opacity_threshold"]
xyz_mat = xyz[keep]
pmin, pmax = xyz_mat.min(0).values, xyz_mat.max(0).values
mid = (pmin + pmax) / 2; sc = 1.0 / (pmax - pmin).max()
to_mpm = lambda p: (p - mid) * sc + torch.tensor([1., 1., 1.], device=dev)
POS_ALL = to_mpm(xyz).contiguous()
POS_MAT = POS_ALL[keep].contiguous()

ng, grid_lim = 100, 2.0
dx = grid_lim / ng
vi = (POS_MAT / dx).long().clamp(0, ng - 1)
flat = (vi[:, 0] * ng + vi[:, 1]) * ng + vi[:, 2]
cnt = torch.zeros(ng ** 3, device=dev).index_add_(0, flat, torch.ones(POS_MAT.shape[0], device=dev))
vol_mat = (dx ** 3) / cnt[flat]

# anchors + radius come from the material cloud in both paths
aset, _ = AnchorSet.from_gaussians(POS_MAT, node_num=args.n_anchors, latent_dim=0, e_dim=0, K=args.K)
AC = aset.canonical.clone().contiguous(); M = AC.shape[0]
sim_mat = AnchorElasticSim(POS_MAT, AC, K=args.K)
sim_all = AnchorElasticSim(POS_ALL, AC, K=args.K, radius=sim_mat.radius)
print(f"[setup] material={POS_MAT.shape[0]} all={POS_ALL.shape[0]} anchors={M} "
      f"radius={sim_mat.radius:.6f}")


def material_params(pos, active):
    n = pos.shape[0]
    E_p = torch.full((n,), float(cfg["E"]), device=dev)
    nu_p = torch.full((n,), float(cfg["nu"]), device=dev)
    d_p = torch.full((n,), float(cfg["density"]), device=dev)
    for reg in cfg.get("additional_material_params", []):
        c = torch.tensor(reg["point"], device=dev); s = torch.tensor(reg["size"], device=dev)
        ins = ((pos - c).abs() <= s).all(-1)
        E_p[ins] = reg["E"]; nu_p[ins] = reg["nu"]; d_p[ins] = reg["density"]
    mu, lam = lame_from_E_nu(E_p, nu_p)
    return mu, lam, d_p


MU_A, LAM_A, DENS_A = material_params(POS_MAT, None)
VOL_A = vol_mat
MU_B, LAM_B, DENS_B = material_params(POS_ALL, None)
VOL_B = torch.zeros(POS_ALL.shape[0], device=dev); VOL_B[keep] = vol_mat
VOL_B = VOL_B.contiguous()

fixed_mask = torch.zeros(M, dtype=torch.bool, device=dev)
for bc in cfg.get("boundary_conditions", []):
    if bc["type"] == "cuboid":
        c = torch.tensor(bc["point"], device=dev); s = torch.tensor(bc["size"], device=dev)
        fixed_mask |= ((AC - c).abs() <= s).all(-1)
gravity = torch.tensor(cfg["g"], dtype=torch.float32, device=dev)
sub_dt = float(cfg["substep_dt"])
damping = float(cfg.get("grid_v_damping_scale", 1.0))


def run(sim, pos0, vol, mu, lam, dens, mat_mask):
    w0 = sim._weights(pos0, AC)
    mass_p = dens * vol
    MASS = torch.zeros(M, device=dev).index_add_(
        0, sim.nn_idx.reshape(-1), (mass_p.unsqueeze(-1) * w0).reshape(-1)).clamp(min=1e-12)
    v = torch.zeros(M, 3, device=dev)
    for bc in cfg.get("boundary_conditions", []):
        if bc["type"] == "particle_impulse":
            force = torch.tensor(bc["force"], device=dev)
            wm = w0 if mat_mask is None else w0 * mat_mask.unsqueeze(-1)
            wsum = torch.zeros(M, device=dev).index_add_(0, sim.nn_idx.reshape(-1), wm.reshape(-1))
            v = v + (wsum.unsqueeze(-1) * force.unsqueeze(0) * sub_dt) / MASS.unsqueeze(-1)
    v[fixed_mask] = 0
    p, gp = AC.clone(), pos0.clone()
    traj = []
    for i in tqdm(range(args.steps), desc="sim", ncols=80, leave=False):
        p, v, gp, _ = sim.step(p, v, MASS, gp, vol, mu, lam, sub_dt,
                                gravity=(gravity if gravity.abs().sum() > 0 else None),
                                damping=damping, fixed_mask=fixed_mask)
        if (i + 1) % max(1, args.steps // 10) == 0:
            traj.append(p.clone())
    return MASS, torch.stack(traj), gp


MASS_A, TRAJ_A, GP_A = run(sim_mat, POS_MAT, VOL_A, MU_A, LAM_A, DENS_A, None)
MASS_B, TRAJ_B, GP_B = run(sim_all, POS_ALL, VOL_B, MU_B, LAM_B, DENS_B, keep)

print(f"[mass]   max |A-B| = {(MASS_A - MASS_B).abs().max():.3e} "
      f"(scale {MASS_A.abs().max():.3e})")
print(f"[finite] path B anchors finite={bool(torch.isfinite(TRAJ_B).all())} "
      f"gaussians finite={bool(torch.isfinite(GP_B).all())}")
motion = (TRAJ_A[-1] - AC).norm(dim=-1).max()
for i in range(TRAJ_A.shape[0]):
    d = (TRAJ_A[i] - TRAJ_B[i]).norm(dim=-1).max()
    print(f"  step {(i+1)*(args.steps//10):6d}: max anchor |A-B| = {d:.3e}")
print(f"[verdict] final divergence {(TRAJ_A[-1]-TRAJ_B[-1]).norm(dim=-1).max():.3e} "
      f"vs actual anchor motion {motion:.5f} "
      f"(ratio {((TRAJ_A[-1]-TRAJ_B[-1]).norm(dim=-1).max()/motion):.2e})")
# the passengers must actually be carried, not left behind
disp_pass = (GP_B[~keep] - POS_ALL[~keep]).norm(dim=-1)
disp_mat = (GP_B[keep] - POS_ALL[keep]).norm(dim=-1)
print(f"[skin] passenger Gaussian displacement: median={disp_pass.median():.5f} "
      f"max={disp_pass.max():.5f}   material: median={disp_mat.median():.5f} "
      f"max={disp_mat.max():.5f}")

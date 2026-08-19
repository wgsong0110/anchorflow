"""How well does MPM on far fewer particles follow MPM on all of them?

The chain replaces MPM with an anchor simulator: a few hundred anchors carrying
positions and velocities, with each Gaussian's deformation gradient rebuilt from
the anchor arrangement every step. Fitted, that reaches 6.61% of MPM on the field
impulses. Measuring the part of MPM's future the anchor state cannot hold put a
floor under that: branch MPM with identical positions and velocities but the
anchor-reconstructed F, and the trajectories separate by 2.6-5.0% within thirty
frames. F is the one quantity MPM accumulates and the anchor state does not.

A coarse MPM does not have that problem. Its particles carry their own F and
accumulate it exactly as the reference does, so what is lost is resolution rather
than a kind of information. This asks what that costs, before any fitting: take
the same number of particles the anchor set has, run the same solver, skin back
to the full cloud, and score on the same impulses as everything else.

Skinning uses each coarse particle's own F -- x = sum_i w_i [x_i + F_i (X - X_i)]
-- since carrying F is the whole point of the comparison; a displacement-only
blend would throw away exactly what is being tested.
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
ap.add_argument("--cache", required=True, help="eval_vs_mpm's reference cache")
ap.add_argument("--counts", type=int, nargs="+", default=[512, 2000, 8000, 32000])
ap.add_argument("--n_grid", type=int, nargs="+", default=[100],
                 help="grid resolutions to try. The particle side dominates the cost, "
                      "so this is expected to matter less than the count.")
ap.add_argument("--kinds", nargs="+", default=["uniform", "field", "poke"])
ap.add_argument("--n_case", type=int, default=8)
ap.add_argument("--frames", type=int, default=60)
ap.add_argument("--dt_mult", type=int, default=40)
ap.add_argument("--K", type=int, default=8, help="coarse particles each Gaussian follows")
ap.add_argument("--n_anchors", type=int, default=512)
ap.add_argument("--eig_floor", type=float, default=0.02)
ap.add_argument("--seed", type=int, default=7)
args = ap.parse_args()

sys.path.insert(0, args.dreamphysics)
import warp as wp
from mpm_solver_warp.mpm_solver_warp import MPM_Simulator_WARP

from anchorflow.mpm_teacher import MPMTeacher

dev = "cuda"
torch.set_grad_enabled(False)
wp.init()

sc = scene_setup.build(args.ply, args.config, args.n_anchors, args.K, device=dev,
                        frozen_weights=True, rot_fallback=True,
                        eig_floor=args.eig_floor)
T = MPMTeacher(sc)
mat = T.mat
Xfull = T.pos_m                      # canonical material positions [Nm,3]
volfull = T.vol_m
Nm = Xfull.shape[0]
print(f"[setup] {Nm} material particles, comparing against the same reference set "
      f"everything else uses")

blob = torch.load(args.cache, map_location=dev, weights_only=False)


def build(idx, n_grid):
    """a solver carrying only the chosen particles, with the volume they stand for"""
    pos = Xfull[idx].contiguous()
    # total volume is preserved: each keeps its own share scaled by how many it
    # now represents, so the material has the same mass and the same density
    vol = (volfull[idx] * (float(volfull.sum()) / float(volfull[idx].sum()))).contiguous()
    s = MPM_Simulator_WARP(10)
    s.load_initial_data_from_torch(pos, vol, torch.zeros((idx.shape[0], 6), device=dev),
                                    n_grid=n_grid, grid_lim=T.grid_lim)
    cfg = sc.cfg
    mp = {k: cfg[k] for k in ("E", "nu", "density", "material") if k in cfg}
    mp.update({"n_grid": n_grid, "grid_lim": T.grid_lim, "g": cfg.get("g", [0, 0, 0]),
               "grid_v_damping_scale": cfg.get("grid_v_damping_scale", 1.0)})
    if "additional_material_params" in cfg:
        mp["additional_material_params"] = cfg["additional_material_params"]
    s.set_parameters_dict(mp)
    s.finalize_mu_lam()
    for bc in cfg.get("boundary_conditions", []):
        if bc["type"] == "cuboid":
            s.set_velocity_on_cuboid(bc["point"], bc["size"], [0.0, 0.0, 0.0],
                                      start_time=0.0, end_time=999.0, reset=1)
    return s, pos


def skin_weights(pos):
    """each full particle follows its K nearest coarse particles, canonically"""
    from scipy.spatial import cKDTree
    tree = cKDTree(pos.cpu().numpy())
    d, i = tree.query(Xfull.cpu().numpy(), k=min(args.K, pos.shape[0]))
    d = torch.from_numpy(d).float().to(dev).reshape(Nm, -1)
    i = torch.from_numpy(i).long().to(dev).reshape(Nm, -1)
    w = 1.0 / d.clamp(min=1e-6)
    return w / w.sum(-1, keepdim=True), i


def run(s, pos, w, idx_nn, v0c):
    """coarse rollout, skinned back to the full cloud using the coarse F"""
    n = pos.shape[0]
    eye = torch.eye(3, device=dev).reshape(1, 9).repeat(n, 1).contiguous()
    s.import_particle_x_from_torch(pos.clone())
    s.import_particle_v_from_torch(v0c.contiguous())
    s.import_particle_F_from_torch(eye.clone())
    s.import_particle_C_from_torch(torch.zeros_like(eye))
    out = []
    off = Xfull.unsqueeze(1) - pos[idx_nn]                     # [Nm,K,3] canonical
    for _ in range(args.frames):
        for k in range(args.dt_mult):
            s.p2g2p(None, sc.sub_dt, device=str(dev))
        xc = s.export_particle_x_to_torch()
        Fc = s.export_particle_F_to_torch().reshape(-1, 3, 3)
        if not torch.isfinite(xc).all():
            return None
        # x = sum_i w_i [x_i + F_i (X - X_i)]
        moved = xc[idx_nn] + torch.einsum("nkij,nkj->nki", Fc[idx_nn], off)
        out.append((w.unsqueeze(-1) * moved).sum(1))
    return torch.stack(out)


def score(pred, truth):
    span = (truth - truth[0]).norm(dim=-1).max().clamp(min=1e-12)
    e = (pred - truth[1:]).norm(dim=-1).mean(-1) / span
    return float(e.mean()), float(e[-1]), float((pred - Xfull).norm(dim=-1).max() / span)


g = torch.Generator(device=dev).manual_seed(args.seed)
print(f"\n{'particles':>10} {'grid':>6} " +
      " ".join(f"{k:>16}" for k in args.kinds) + f" {'lost':>5}")
for n_grid in args.n_grid:
    for cnt in args.counts:
        cnt = min(cnt, Nm)
        idx = torch.randperm(Nm, device=dev, generator=g)[:cnt].sort().values
        s, pos = build(idx, n_grid)
        w, nn = skin_weights(pos)
        cells = []
        lost = 0
        for kind in args.kinds:
            errs = []
            for i in range(min(args.n_case, len(blob["ref"][kind]))):
                truth = blob["ref"][kind][i][: args.frames + 1]
                f = blob["force"][kind][i]
                dv = sc.impulse_dv(f)
                v0 = (T.w.unsqueeze(-1) * dv[T.idx]).sum(1)          # full particles
                pred = run(s, pos, w, nn, v0[idx])
                if pred is None:
                    lost += 1
                    continue
                errs.append(score(pred, truth)[0])
            cells.append(f"{100 * sum(errs) / max(len(errs), 1):15.2f}%" if errs
                          else f"{'diverged':>16}")
        print(f"{cnt:10d} {n_grid:6d} " + " ".join(cells) + f" {lost:5d}")

print("\n  For reference on the same impulses: the fitted anchor simulator reaches")
print("  7.38 / 6.61 / 10.14% and the unfitted one 12.19 / 10.56 / 12.59%.")
print("  Nothing here is fitted -- the particles are a random subset at their")
print("  canonical positions, with volume rescaled to preserve the total.")

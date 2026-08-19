"""How few particles can MPM be run with, and on what grid?

Fitting a coarse MPM assumes there is a coarse MPM to fit. There is not, at any
grid: 512 particles on the reference's 100^3 leaves 0.075 particles per occupied
cell, and the rollout diverges before anything is fitted. MPM needs several
particles in a cell for the grid solve to mean anything, so the grid has to
coarsen with the count.

This sweeps both, unfitted, on the same impulses and the same ruler as
everything else. It answers what the fit starts from, and whether the budget the
anchor set runs at is one MPM can be run at.
"""
from __future__ import annotations

import argparse
import os
import sys

_lib = os.path.join(os.path.dirname(__file__), "..", "lib")
sys.path.insert(0, _lib)

import torch

from anchorflow import scene_setup

ap = argparse.ArgumentParser()
ap.add_argument("--ply", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--dreamphysics", default="/workspace/DreamPhysics")
ap.add_argument("--cache", required=True)
ap.add_argument("--counts", type=int, nargs="+", default=[512, 2048, 8192, 32768])
ap.add_argument("--grids", type=int, nargs="+", default=[16, 25, 40, 64, 100])
ap.add_argument("--kind", default="field")
ap.add_argument("--n_case", type=int, default=4)
ap.add_argument("--frames", type=int, default=60)
ap.add_argument("--dt_mult", type=int, default=40)
ap.add_argument("--K", type=int, default=8)
ap.add_argument("--eig_floor", type=float, default=0.02)
ap.add_argument("--seed", type=int, default=7)
args = ap.parse_args()

sys.path.insert(0, args.dreamphysics)
import warp as wp
from mpm_solver_warp.mpm_solver_warp import MPM_Simulator_WARP
from scipy.spatial import cKDTree

from anchorflow.mpm_teacher import MPMTeacher

dev = "cuda"
torch.set_grad_enabled(False)
wp.init()
sc = scene_setup.build(args.ply, args.config, 512, 8, device=dev, frozen_weights=True,
                        rot_fallback=True, eig_floor=args.eig_floor)
T = MPMTeacher(sc)
Xfull, volfull, Nm = T.pos_m, T.vol_m, T.pos_m.shape[0]
blob = torch.load(args.cache, map_location=dev, weights_only=False)
print(f"[setup] {Nm} material particles, extent "
      f"{float((Xfull.max(0).values - Xfull.min(0).values).max()):.3f} of a "
      f"{T.grid_lim} domain")

g = torch.Generator(device=dev).manual_seed(args.seed)
print(f"\n{'particles':>9} {'grid':>5} {'per cell':>9}  " +
      "  ".join(f"{'err':>8}" for _ in range(1)) + "   note")
for cnt in args.counts:
    cnt = min(cnt, Nm)
    idx = torch.randperm(Nm, device=dev, generator=g)[:cnt].sort().values
    pos = Xfull[idx].contiguous()
    vol = (volfull[idx] * (float(volfull.sum()) / float(volfull[idx].sum()))).contiguous()
    tree = cKDTree(pos.cpu().numpy())
    d, i = tree.query(Xfull.cpu().numpy(), k=min(args.K, cnt))
    NN = torch.from_numpy(i).long().to(dev).reshape(Nm, -1)
    w = 1.0 / torch.from_numpy(d).float().to(dev).reshape(Nm, -1).clamp(min=1e-6)
    W = w / w.sum(-1, keepdim=True)
    off = Xfull.unsqueeze(1) - pos[NN]
    for G in args.grids:
        s = MPM_Simulator_WARP(10)
        s.load_initial_data_from_torch(pos, vol, torch.zeros((cnt, 6), device=dev),
                                        n_grid=G, grid_lim=T.grid_lim)
        mp = {k: sc.cfg[k] for k in ("E", "nu", "density", "material") if k in sc.cfg}
        mp.update({"n_grid": G, "grid_lim": T.grid_lim, "g": sc.cfg.get("g", [0, 0, 0]),
                   "grid_v_damping_scale": sc.cfg.get("grid_v_damping_scale", 1.0)})
        if "additional_material_params" in sc.cfg:
            mp["additional_material_params"] = sc.cfg["additional_material_params"]
        s.set_parameters_dict(mp)
        s.finalize_mu_lam()
        occupied = len(set(map(tuple, (pos / (T.grid_lim / G)).long().cpu().numpy())))
        errs, note = [], ""
        for c in range(min(args.n_case, len(blob["ref"][args.kind]))):
            truth = blob["ref"][args.kind][c][: args.frames + 1]
            f = blob["force"][args.kind][c]
            v0 = (T.w.unsqueeze(-1) * sc.impulse_dv(f)[T.idx]).sum(1)[idx].contiguous()
            eye = torch.eye(3, device=dev).reshape(1, 9).repeat(cnt, 1).contiguous()
            s.import_particle_x_from_torch(pos.clone())
            s.import_particle_v_from_torch(v0)
            s.import_particle_F_from_torch(eye.clone())
            s.import_particle_C_from_torch(torch.zeros_like(eye))
            span = (truth - truth[0]).norm(dim=-1).max().clamp(min=1e-12)
            e, dead = [], False
            for fr in range(args.frames):
                for _ in range(args.dt_mult):
                    s.p2g2p(None, sc.sub_dt, device=str(dev))
                xc = s.export_particle_x_to_torch()
                if not torch.isfinite(xc).all():
                    dead = True
                    break
                Fc = s.export_particle_F_to_torch().reshape(-1, 3, 3)
                pred = (W.unsqueeze(-1) * (xc[NN] + torch.einsum(
                    "nkij,nkj->nki", Fc[NN], off))).sum(1)
                e.append(float((pred - truth[fr + 1]).norm(dim=-1).mean() / span))
            if dead:
                note = "diverged"
            else:
                errs.append(sum(e) / len(e))
        cell = cnt / max(occupied, 1)
        val = f"{100 * sum(errs) / len(errs):7.2f}%" if errs else f"{'--':>8}"
        print(f"{cnt:9d} {G:5d} {cell:9.2f}  {val}   {note}")

print("\n  For reference on the same impulses: the fitted anchor simulator reaches")
print("  6.61% on field and the unfitted one 10.56%. Nothing here is fitted.")

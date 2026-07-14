#!/usr/bin/env python
"""Export a MoSca reconstruction into anchorflow's portable format.

Runs inside the MoSca image (PYTHONPATH=/opt/MoSca). Loads a reconstructed
DynSCFGaussian checkpoint and writes:
    node_traj.npy   [T, M, 3]   scaffold-node (anchor) trajectory
    canonical.ply               dynamic Gaussians at their reference frame (INRIA 3DGS .ply)
so the anchorflow image can then run the GNN⊗SSM supervised/MDS training without
any MoSca deps. Time is integer-frame (MoSca has no physical dt) -> anchorflow uses dt=1.

    python exe/mosca_export.py --d_model logs/<run>/photometric_d_model_native_add3.pth \
        --out /workspace/mosca_out   [--ref_frame 0]
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d_model", required=True, help="photometric_d_model_*.pth")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref_frame", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    from lib_mosca.dynamic_gs import DynSCFGaussian
    d = DynSCFGaussian.load_from_ckpt(torch.load(args.d_model, map_location=dev), device=dev)
    scf = d.scf

    # --- anchor (scaffold node) trajectory ---
    node_traj = scf._node_xyz.detach().cpu().numpy()          # [T, M, 3]
    np.save(os.path.join(args.out, "node_traj.npy"), node_traj)
    T, M = node_traj.shape[:2]

    # --- canonical Gaussians at the reference frame -> INRIA .ply ---
    t = args.ref_frame
    with torch.no_grad():
        mu = d.get_xyz().detach()                             # world canonical [N,3]
        # opacity/scaling/rotation/SH at rest: use the leaf params (pre-activation)
        from plyfile import PlyData, PlyElement
        xyz = mu.cpu().numpy()
        opacity = d._opacity.detach().cpu().numpy()
        scaling = d._scaling.detach().cpu().numpy()
        rotation = d._rotation.detach().cpu().numpy()
        f_dc = d._features_dc.detach().reshape(xyz.shape[0], -1).cpu().numpy()
        rest = getattr(d, "_features_rest", None)
        f_rest = (rest.detach().reshape(xyz.shape[0], -1).cpu().numpy()
                  if rest is not None and rest.numel() else np.zeros((xyz.shape[0], 0)))

    fields = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
    fields += [f"f_rest_{i}" for i in range(f_rest.shape[1])]
    fields += ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    data = np.concatenate([xyz, np.zeros_like(xyz), f_dc, f_rest,
                           opacity.reshape(-1, 1) if opacity.ndim == 1 else opacity,
                           scaling, rotation], axis=1)
    elem = np.empty(xyz.shape[0], dtype=[(f, "f4") for f in fields])
    elem[:] = list(map(tuple, data))
    PlyData([PlyElement.describe(elem, "vertex")]).write(os.path.join(args.out, "canonical.ply"))

    print(f"[mosca_export] node_traj {node_traj.shape}, canonical.ply N={xyz.shape[0]} "
          f"-> {args.out}")


if __name__ == "__main__":
    main()

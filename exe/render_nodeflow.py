#!/usr/bin/env python
"""Render a trained NodeFlow checkpoint as a video.

Renders one or all views over time (t=0..T-1) and saves mp4s.

Usage:
    PYTHONPATH=/workspace/anchorflow/lib python exe/render_nodeflow.py \\
        --ply /workspace/wolf_views/wolf_aligned.ply \\
        --data /workspace/wolf_views \\
        --ckpt /workspace/wolf_nodeflow/ckpt_latest.pt \\
        --cfg  cfg/nodeflow_wolf.yaml \\
        --out  /workspace/wolf_render \\
        --fps  10
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import imageio
from omegaconf import OmegaConf
from PIL import Image

sys.path.append("gaussian-splatting")
from scene.gaussian_model import GaussianModel
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from plyfile import PlyData

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
from anchorflow.nodeflow import NodeFlow


class Cam:
    def __init__(self, meta):
        R = np.array(meta["R"], dtype=np.float32)
        T = np.array(meta["T"], dtype=np.float32)
        fovx, fovy = meta["fov_x"], meta["fov_y"]
        W, H = meta["W"], meta["H"]
        self.image_width, self.image_height = W, H
        self.FoVx, self.FoVy = fovx, fovy
        self.znear, self.zfar = 0.01, 100.0
        w2v = torch.tensor(getWorld2View2(R, T)).transpose(0, 1).cuda()
        proj = getProjectionMatrix(self.znear, self.zfar, fovx, fovy).transpose(0, 1).cuda()
        self.world_view_transform = w2v
        self.full_proj_transform = (w2v.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        self.tanfovx = math.tan(fovx * 0.5)
        self.tanfovy = math.tan(fovy * 0.5)


def render_frame(cam, xyz, opacities, scales, rotations, colors, bg):
    settings = GaussianRasterizationSettings(
        image_height=cam.image_height,
        image_width=cam.image_width,
        tanfovx=cam.tanfovx,
        tanfovy=cam.tanfovy,
        bg=bg,
        scale_modifier=1.0,
        viewmatrix=cam.world_view_transform,
        projmatrix=cam.full_proj_transform,
        sh_degree=0,
        campos=cam.camera_center,
        prefiltered=False,
        debug=False,
    )
    rasterizer = GaussianRasterizer(raster_settings=settings)
    means2D = torch.zeros_like(xyz)
    with torch.no_grad():
        rendered, _ = rasterizer(
            means3D=xyz,
            means2D=means2D,
            shs=None,
            colors_precomp=colors,
            opacities=opacities,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=None,
        )
    return (rendered.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


def load_gaussian_attrs(ply_path, device="cuda"):
    names = [p.name for p in PlyData.read(ply_path)["vertex"].properties
             if p.name.startswith("f_rest_")]
    sh = min(int(math.sqrt((len(names) + 3) // 3)) - 1 if names else 0, 3)
    g = GaussianModel(sh); g.load_ply(ply_path); g.active_sh_degree = sh
    SH_C0 = 0.28209479177387814
    f_dc = g._features_dc.detach()[:, 0, :]
    return {
        "xyz":       g.get_xyz.detach().to(device),
        "opacities": g.get_opacity.detach().to(device),
        "scales":    g.get_scaling.detach().to(device),
        "rotations": g.get_rotation.detach().to(device),
        "colors":    (SH_C0 * f_dc + 0.5).clamp(0, 1).to(device),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply",  required=True)
    ap.add_argument("--data", required=True, help="gen_views output dir (cameras.json)")
    ap.add_argument("--ckpt", required=True, help="checkpoint .pt file")
    ap.add_argument("--cfg",  required=True)
    ap.add_argument("--out",  required=True)
    ap.add_argument("--views", default=None,
                    help="comma-sep view indices to render (default: all)")
    ap.add_argument("--fps",  type=int, default=10)
    ap.add_argument("--white", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    cfg = OmegaConf.load(args.cfg)
    dev = "cuda"

    gauss = load_gaussian_attrs(args.ply, dev)
    canonical_xyz = gauss["xyz"]

    with open(os.path.join(args.data, "cameras.json")) as f:
        cam_metas = json.load(f)
    cameras = [Cam(m) for m in cam_metas]
    V = len(cameras)
    T = cfg.model.n_frames

    # build model
    model = NodeFlow(
        canonical_xyz=canonical_xyz,
        n_nodes=cfg.model.n_nodes,
        n_views=V,
        n_frames=T,
        hidden=cfg.model.hidden,
        n_gnn_layers=cfg.model.n_gnn_layers,
        k_node=cfg.model.k_node,
        k_gauss=cfg.model.k_gauss,
    ).to(dev)

    ckpt = torch.load(args.ckpt, map_location=dev)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state)
    model.eval()

    views = list(range(V))
    if args.views:
        views = [int(x) for x in args.views.split(",")]

    bg = torch.tensor([1., 1., 1.] if args.white else [0., 0., 0.], device=dev)

    for v in views:
        frames = []
        for t in range(T):
            disp = model(v, float(t))
            xyz_def = canonical_xyz + disp
            arr = render_frame(cameras[v], xyz_def,
                               gauss["opacities"], gauss["scales"],
                               gauss["rotations"], gauss["colors"], bg)
            frames.append(arr)
        out_path = os.path.join(args.out, f"view_{v:02d}.mp4")
        imageio.mimsave(out_path, frames, fps=args.fps, quality=8)
        print(f"[render] view {v} → {out_path}")

    # side-by-side grid video (all views, first row)
    if len(views) > 1:
        all_rows = []
        for t in range(T):
            row = []
            for v in views:
                disp = model(v, float(t))
                xyz_def = canonical_xyz + disp
                row.append(render_frame(cameras[v], xyz_def,
                                        gauss["opacities"], gauss["scales"],
                                        gauss["rotations"], gauss["colors"], bg))
            all_rows.append(np.concatenate(row, axis=1))
        grid_path = os.path.join(args.out, "grid.mp4")
        imageio.mimsave(grid_path, all_rows, fps=args.fps, quality=8)
        print(f"[render] grid → {grid_path}")


if __name__ == "__main__":
    main()

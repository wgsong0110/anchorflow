#!/usr/bin/env python
"""Diagnostic/standalone: render a canonical 3DGS .ply with the STANDARD
gaussian-splatting renderer (scales+rotations path), from an orbit around the
object's own bbox — no DreamPhysics transform2origin / precomputed-cov machinery.
Isolates whether a reconstruction .ply renders correctly at all.

    python exe/render_std.py --ply scgs_out/canonical.ply --out out --frames 12
"""
from __future__ import annotations
import argparse, os, sys, math
import numpy as np
import torch
import imageio

sys.path.append("gaussian-splatting")
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from plyfile import PlyData


class Pipe:
    convert_SHs_python = False
    compute_cov3D_python = False       # <- standard scales+rotations path
    debug = False


class Cam:
    """Minimal camera the rasterizer needs, built from a c2w + FoV."""
    def __init__(self, R, T, fovx, fovy, W, H):
        self.image_width, self.image_height = W, H
        self.FoVx, self.FoVy = fovx, fovy
        self.znear, self.zfar = 0.01, 100.0
        w2v = torch.tensor(getWorld2View2(R, T)).transpose(0, 1).cuda()
        proj = getProjectionMatrix(self.znear, self.zfar, fovx, fovy).transpose(0, 1).cuda()
        self.world_view_transform = w2v
        self.full_proj_transform = (w2v.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]


def look_at(eye, center, up):
    f = (center - eye); f = f / np.linalg.norm(f)
    s = np.cross(f, up); s = s / np.linalg.norm(s)
    u = np.cross(s, f)
    R = np.stack([s, u, -f], 1)          # camera-to-world rotation (cols)
    return R, eye


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--res", type=int, default=800)
    ap.add_argument("--fov", type=float, default=45.0)
    ap.add_argument("--radius_scale", type=float, default=2.2, help="radius = this * bbox_diag")
    ap.add_argument("--elev", type=float, default=15.0)
    ap.add_argument("--white", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    names = [p.name for p in PlyData.read(args.ply)["vertex"].properties
             if p.name.startswith("f_rest_")]
    sh = int(math.sqrt((len(names) + 3) // 3)) - 1 if names else 0
    g = GaussianModel(sh); g.load_ply(g_path := args.ply); g.active_sh_degree = sh
    pipe = Pipe()
    bg = torch.tensor([1., 1, 1] if args.white else [0, 0, 0], device="cuda")

    xyz = g.get_xyz.detach().cpu().numpy()
    center = (xyz.min(0) + xyz.max(0)) / 2.0
    diag = float(np.linalg.norm(xyz.max(0) - xyz.min(0)))
    radius = args.radius_scale * diag * 0.5
    fov = math.radians(args.fov)
    print(f"[render_std] N={xyz.shape[0]} sh={sh} center={center.round(2)} diag={diag:.2f} radius={radius:.2f}")

    up = np.array([0., 1., 0.])
    frames = []
    for t in range(args.frames):
        az = math.radians(360.0 * t / args.frames)
        el = math.radians(args.elev)
        eye = center + radius * np.array([math.cos(el) * math.sin(az),
                                          math.sin(el),
                                          math.cos(el) * math.cos(az)])
        Rc, _ = look_at(eye.astype(np.float32), center.astype(np.float32), up.astype(np.float32))
        # gaussian-splatting getWorld2View2 wants R (c2w rot) and T (w2c trans)
        T = -Rc.T @ eye
        cam = Cam(Rc.astype(np.float32), T.astype(np.float32), fov, fov, args.res, args.res)
        with torch.no_grad():
            out = render(cam, g, pipe, bg)["render"]
        arr = (out.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        frames.append(arr)
        imageio.imwrite(os.path.join(args.out, f"{t:05d}.png"), arr)
    imageio.mimsave(os.path.join(args.out, "orbit.mp4"), frames, fps=8, quality=8)
    print(f"[render_std] wrote {args.frames} frames -> {args.out}")


if __name__ == "__main__":
    main()

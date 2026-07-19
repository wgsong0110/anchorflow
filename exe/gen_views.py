#!/usr/bin/env python
"""Generate N single-view videos from a static 3DGS for NodeFlow training.

Pipeline (GPU instance only — NOT arm64 host):
  1. Load canonical 3DGS .ply
  2. For each of N cameras (spherical orbit):
       a. Render static view → PNG
       b. Run SVD img2vid → T video frames
       c. Save frames to  out/view_{i:02d}/
  3. Save out/cameras.json  (R, T, fov, W, H per view)

Usage:
    WS=/workspace/anchorflow
    python $WS/exe/gen_views.py \\
        --ply /workspace/wolf/wolf_aligned.ply \\
        --out /workspace/wolf_views \\
        --n_views 8 --elevation 15 --n_frames 21 --res 576 --seed 42

Cameras are equally spaced in azimuth (0,45,...,315 deg) at a fixed elevation.
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
from PIL import Image

sys.path.append("/workspace/SC-GS")
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render as _render_scgs

def render(cam, g, pipe, bg):
    zeros = torch.zeros_like(g.get_xyz)
    return _render_scgs(cam, g, pipe, bg, d_xyz=zeros, d_rotation=0.0, d_scaling=zeros)

from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from plyfile import PlyData


# ---------------------------------------------------------------------------
class Pipe:
    convert_SHs_python = False
    compute_cov3D_python = True
    debug = False


class Cam:
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
    f = (center - eye); f /= np.linalg.norm(f)
    s = np.cross(f, up); s /= np.linalg.norm(s)
    u = np.cross(s, f)
    R = np.stack([s, u, f], axis=1)
    return R.astype(np.float32), eye.astype(np.float32)


def make_camera(center, radius, az_deg, el_deg, fov_deg, W, H):
    az = math.radians(az_deg)
    el = math.radians(el_deg)
    eye = center + radius * np.array([
        math.cos(el) * math.sin(az),
        math.sin(el),
        math.cos(el) * math.cos(az),
    ], dtype=np.float32)
    up = np.array([0., 1., 0.], dtype=np.float32)
    Rc, _ = look_at(eye, center.astype(np.float32), up)
    T = -Rc.T @ eye
    fov = math.radians(fov_deg)
    cam = Cam(Rc, T, fov, fov, W, H)
    meta = {
        "R": Rc.tolist(),
        "T": T.tolist(),
        "fov_x": fov,
        "fov_y": fov,
        "W": W,
        "H": H,
        "azimuth_deg": az_deg,
        "elevation_deg": el_deg,
        "center": center.tolist(),
        "radius": float(radius),
    }
    return cam, meta


# ---------------------------------------------------------------------------
def load_gaussian(ply_path, sh_max=3):
    names = [p.name for p in PlyData.read(ply_path)["vertex"].properties
             if p.name.startswith("f_rest_")]
    sh = min(int(math.sqrt((len(names) + 3) // 3)) - 1 if names else 0, sh_max)
    g = GaussianModel(sh)
    g.load_ply(ply_path)
    g.active_sh_degree = sh
    return g


def render_view(g, cam, bg):
    with torch.no_grad():
        out = render(cam, g, Pipe(), bg)["render"]             # [3, H, W]
    arr = (out.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return arr


# ---------------------------------------------------------------------------
def run_svd(image_path, out_dir, n_frames, model_id, seed):
    """Run SVD img2vid and save frames to out_dir."""
    from diffusers import StableVideoDiffusionPipeline

    os.makedirs(out_dir, exist_ok=True)
    # check already done
    expected = os.path.join(out_dir, f"{n_frames - 1:05d}.png")
    if os.path.exists(expected):
        print(f"  [svd] already done: {out_dir}")
        return

    pipe = StableVideoDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16, variant="fp16")
    pipe.enable_model_cpu_offload()

    img = Image.open(image_path).convert("RGB").resize((1024, 576))
    gen = torch.Generator("cuda").manual_seed(seed)
    frames = pipe(img, num_frames=n_frames, decode_chunk_size=8,
                  generator=gen).frames[0]
    for i, f in enumerate(frames):
        # resize to match render resolution
        f = f.resize((576, 576))
        f.save(os.path.join(out_dir, f"{i:05d}.png"))
    print(f"  [svd] wrote {len(frames)} frames → {out_dir}")

    # free VRAM before next view
    del pipe
    import gc; gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True, help="canonical 3DGS .ply")
    ap.add_argument("--out", required=True, help="output dataset dir")
    ap.add_argument("--n_views", type=int, default=8)
    ap.add_argument("--elevation", type=float, default=15.0, help="degrees")
    ap.add_argument("--azimuths", default=None,
                    help="comma-sep azimuth degrees (overrides n_views)")
    ap.add_argument("--n_frames", type=int, default=21)
    ap.add_argument("--res", type=int, default=576)
    ap.add_argument("--fov", type=float, default=45.0, help="degrees")
    ap.add_argument("--radius_scale", type=float, default=2.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--white", action="store_true")
    ap.add_argument("--model", default="stabilityai/stable-video-diffusion-img2vid-xt")
    ap.add_argument("--render_only", action="store_true",
                    help="skip SVD, only render static views")
    ap.add_argument("--rot_x_deg", type=float, default=0.0,
                    help="rotate scene around X axis before rendering "
                         "(wolf needs +90 to stand upright: z-up->y-up)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # --- azimuths ---
    if args.azimuths:
        azimuths = [float(a) for a in args.azimuths.split(",")]
    else:
        azimuths = [360.0 * i / args.n_views for i in range(args.n_views)]
    n_views = len(azimuths)

    print(f"[gen_views] loading {args.ply}")
    g = load_gaussian(args.ply)
    if args.rot_x_deg != 0.0:
        ang = math.radians(args.rot_x_deg)
        c, s = math.cos(ang), math.sin(ang)
        Rx = torch.tensor([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=torch.float32)
        with torch.no_grad():
            g._xyz.data = g._xyz.data @ Rx.T.to(g._xyz.device)
        print(f"[gen_views] applied rot_x={args.rot_x_deg}°")
    bg = torch.tensor([1., 1, 1] if args.white else [0, 0, 0], device="cuda")

    # object center + radius from Gaussian positions
    xyz = g.get_xyz.detach().cpu().numpy()
    center = (xyz.min(0) + xyz.max(0)) / 2.0
    diag = float(np.linalg.norm(xyz.max(0) - xyz.min(0)))
    radius = args.radius_scale * diag * 0.5
    print(f"[gen_views] N={xyz.shape[0]} center={np.round(center,2)} "
          f"diag={diag:.2f} radius={radius:.2f}")

    # Save (possibly rotated) PLY so train/render scripts use correct geometry
    ply_out = os.path.join(args.out, "scene.ply")
    g.save_ply(ply_out)
    print(f"[gen_views] saved PLY → {ply_out}")

    cameras_meta = []
    for i, az in enumerate(azimuths):
        print(f"\n[gen_views] view {i}/{n_views}  az={az:.0f}°  el={args.elevation:.0f}°")
        cam, meta = make_camera(center, radius, az, args.elevation,
                                args.fov, args.res, args.res)
        cameras_meta.append(meta)

        # render static view
        frame_dir = os.path.join(args.out, f"view_{i:02d}")
        os.makedirs(frame_dir, exist_ok=True)
        static_png = os.path.join(frame_dir, "static.png")
        arr = render_view(g, cam, bg)
        imageio.imwrite(static_png, arr)
        print(f"  [render] → {static_png}")

        if args.render_only:
            continue

        # generate video from static render
        run_svd(static_png, frame_dir, args.n_frames, args.model, args.seed + i)

    # save camera parameters
    cam_path = os.path.join(args.out, "cameras.json")
    with open(cam_path, "w") as f:
        json.dump(cameras_meta, f, indent=2)
    print(f"\n[gen_views] cameras.json → {cam_path}")
    print(f"[gen_views] done — {n_views} views")


if __name__ == "__main__":
    main()

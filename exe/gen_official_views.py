#!/usr/bin/env python
"""Generate per-view target videos for --sup video training.

Renders the official 3DGS scene from the SAME cameras the trainer samples
(cameras.json, evenly spaced), then runs SVD img2vid on each render to produce a
target clip. These clips turn our generation problem into the paper's
reconstruction setting: the clip plays the role of the "input video".

    python exe/gen_official_views.py --model /workspace/gs_official/kitchen \
        --cfg cfg/anchorflow_kitchen.yaml --out /workspace/af_videos
"""
from __future__ import annotations

import argparse, json, os, sys

import numpy as np
import torch
import imageio.v2 as iio
from omegaconf import OmegaConf
from PIL import Image

sys.path.append("/workspace/SC-GS")
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render as _render_scgs
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, focal2fov


def render(cam, g, pipe, bg):
    zeros = torch.zeros_like(g.get_xyz)
    return _render_scgs(cam, g, pipe, bg, d_xyz=zeros, d_rotation=0.0, d_scaling=zeros)


class Cam:
    def __init__(self, R, T, fovx, fovy, Wd, Hd):
        self.image_width, self.image_height = Wd, Hd
        self.FoVx, self.FoVy = fovx, fovy
        self.znear, self.zfar = 0.01, 100.0
        w2v = torch.tensor(getWorld2View2(R, T)).T.cuda()
        proj = getProjectionMatrix(self.znear, self.zfar, fovx, fovy).T.cuda()
        self.world_view_transform = w2v
        self.full_proj_transform = (w2v.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0)
        self.camera_center = w2v.inverse()[3, :3]


class Pipe:
    convert_SHs_python = False
    compute_cov3D_python = True  # SC-GS rasterizer requires this
    debug = False
    antialiasing = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--iter", type=int, default=30000)
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--r2", default=None)
    ap.add_argument("--white_bg", action="store_true")
    ap.add_argument("--motion_bucket_id", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    cfg = OmegaConf.load(args.cfg)
    T = cfg.model.n_frames
    mbid = args.motion_bucket_id if args.motion_bucket_id is not None \
        else int(cfg.mds.motion_bucket_id)

    g = GaussianModel(3)
    g.load_ply(f"{args.model}/point_cloud/iteration_{args.iter}/point_cloud.ply")
    g.active_sh_degree = 3
    print(f"[gen] gaussians={g.get_xyz.shape[0]}")

    bg = torch.tensor([1., 1., 1.] if args.white_bg else [0., 0., 0.], device="cuda")

    # exactly the trainer's camera selection
    cams_json = json.load(open(f"{args.model}/cameras.json"))
    idx = np.linspace(0, len(cams_json) - 1, cfg.train.n_views).round().astype(int)
    long_side = int(cfg.model.res)

    from diffusers import StableVideoDiffusionPipeline
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt",
        torch_dtype=torch.float16, variant="fp16")
    pipe.enable_model_cpu_offload()

    for v, i in enumerate(idx):
        c = cams_json[int(i)]
        rot = np.array(c["rotation"], dtype=np.float32)
        pos = np.array(c["position"], dtype=np.float32)
        Wd, Hd = c["width"], c["height"]
        fovx, fovy = focal2fov(c["fx"], Wd), focal2fov(c["fy"], Hd)
        s = long_side / max(Wd, Hd)
        W8 = max(8, int(round(Wd * s / 8)) * 8)
        H8 = max(8, int(round(Hd * s / 8)) * 8)
        cam = Cam(rot, -rot.T @ pos, fovx, fovy, W8, H8)

        with torch.no_grad():
            img = render(cam, g, Pipe(), bg)["render"].clamp(0, 1)
        arr = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(arr).save(os.path.join(args.out, f"view_{v:02d}_frame0.png"))

        # SVD wants a reasonably sized image; generate then resize back to the
        # trainer's render resolution.
        cond = Image.fromarray(arr).resize((1024, 576), Image.LANCZOS)
        gen = torch.Generator("cuda").manual_seed(args.seed + v)
        frames = pipe(cond, decode_chunk_size=4, generator=gen,
                      num_frames=T, motion_bucket_id=mbid,
                      noise_aug_strength=0.02).frames[0]
        frames = [np.asarray(f.resize((W8, H8), Image.LANCZOS)) for f in frames[:T]]
        path = os.path.join(args.out, f"view_{v:02d}.mp4")
        iio.mimsave(path, frames, fps=8, quality=9)

        f0 = frames[0].astype(np.float32)
        d = np.abs(frames[-1].astype(np.float32) - f0)
        print(f"[gen] view_{v:02d} ({c['img_name']}) -> {path}  {W8}x{H8}x{len(frames)}  "
              f"motion(last vs first): meanAbsDiff={d.mean():.2f}")

        if args.r2:
            os.system(f"rclone copy {args.out} {args.r2} >/dev/null 2>&1")

    print(f"[gen] done -> {args.out}")


if __name__ == "__main__":
    main()

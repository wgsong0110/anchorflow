#!/usr/bin/env python
"""Render a video of the official kitchen 3DGS deformed by an UNTRAINED NodeFlow.

Sanity check for the deformation path before any MDS training: random z0
(initial velocity per node) -> physics rollout -> LBS -> official 3DGS render.
The whole scene deforms, not just the lego.

Uses the official INRIA pretrained model + its bundled cameras.json + the
official gaussian_renderer.render (SH degree 3, black background) — the settings
recorded in kitchen/cfg_args. No hand-tuned camera parameters.

    python exe/gen_random_gnn_video.py \
        --model /workspace/gs_official/kitchen \
        --out   /workspace/random_gnn.mp4 \
        --cam_id 0 --n_nodes 256 --frames 25 --scale 1080
"""
from __future__ import annotations

import argparse, json, math, os, sys

import numpy as np
import torch
import imageio.v2 as iio
from PIL import Image

sys.path.append("/workspace/gaussian-splatting")
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, focal2fov

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
from anchorflow.nodeflow import NodeFlow


class Cam:
    def __init__(self, R, T, fovx, fovy, W, H):
        self.image_width, self.image_height = W, H
        self.FoVx, self.FoVy = fovx, fovy
        self.znear, self.zfar = 0.01, 100.0
        w2v = torch.tensor(getWorld2View2(R, T)).T.cuda()
        proj = getProjectionMatrix(self.znear, self.zfar, fovx, fovy).T.cuda()
        self.world_view_transform = w2v
        self.full_proj_transform = (w2v.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0)
        self.camera_center = w2v.inverse()[3, :3]


class Pipe:
    convert_SHs_python = False
    compute_cov3D_python = False
    debug = False
    antialiasing = False


def cam_from_json(c, scale_long=None):
    """cameras.json entry -> Cam. 3DGS stores C2W rotation + camera center."""
    rot = np.array(c["rotation"], dtype=np.float32)
    pos = np.array(c["position"], dtype=np.float32)
    R, T = rot, -rot.T @ pos
    W, H = c["width"], c["height"]
    fovx, fovy = focal2fov(c["fx"], W), focal2fov(c["fy"], H)
    if scale_long:                      # FoV is resolution-independent
        s = scale_long / max(W, H)
        W, H = int(round(W * s)), int(round(H * s))
    return Cam(R, T, fovx, fovy, W, H)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",   required=True, help="gs_official/kitchen")
    ap.add_argument("--out",     required=True)
    ap.add_argument("--cam_id",  type=int, default=0)
    ap.add_argument("--n_nodes", type=int, default=256)
    ap.add_argument("--frames",  type=int, default=25)
    ap.add_argument("--dt",      type=float, default=0.1)
    ap.add_argument("--motion",  type=float, default=0.02,
                    help="max node travel over the clip, as a fraction of scene extent")
    ap.add_argument("--scale",   type=int, default=1080, help="render long side")
    ap.add_argument("--seed",    type=int, default=0)
    ap.add_argument("--fps",     type=int, default=8)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = "cuda"

    # ── official model ───────────────────────────────────────────────────────
    g = GaussianModel(3)
    g.load_ply(f"{args.model}/point_cloud/iteration_30000/point_cloud.ply")
    g.active_sh_degree = 3
    canon = g.get_xyz.detach().clone()
    G = canon.shape[0]
    print(f"[rand-gnn] gaussians={G}")

    # scene extent from the bulk of the scene (1-99 pct), robust to floaters.
    # All G gaussians, no subsampling.
    _q = canon.float()
    extent = float((torch.quantile(_q, 0.99, dim=0)
                    - torch.quantile(_q, 0.01, dim=0)).norm())
    del _q
    print(f"[rand-gnn] scene extent(1-99pct diag)={extent:.2f}")

    # ── untrained NodeFlow over the WHOLE scene ──────────────────────────────
    model = NodeFlow(
        canonical_xyz=canon, n_nodes=args.n_nodes, n_frames=args.frames,
        dt=args.dt,
    ).to(dev)
    K = model.n_nodes
    print(f"[rand-gnn] nodes={K} (FPS over whole scene)")

    # Untrained: accel_decoder is zero-init by design, so motion comes from z0
    # (initial velocity). Randomise the decoder head too so acceleration is
    # non-zero -> trajectories curve instead of being purely linear.
    with torch.no_grad():
        last = model.accel_decoder[-1]
        acc_std = args.motion * extent / (args.dt ** 2 * args.frames ** 2)
        torch.nn.init.normal_(last.weight, std=acc_std * 0.05)
        torch.nn.init.normal_(last.bias,   std=acc_std)

    # z0: initial velocity so that dt*|z0|*T ~= motion * extent
    z0_std = args.motion * extent / (args.dt * max(args.frames - 1, 1))
    z0 = torch.randn(K, 3, device=dev) * z0_std
    print(f"[rand-gnn] z0_std={z0_std:.4f}  acc_std={acc_std:.4f}  "
          f"(target travel ~{args.motion*100:.1f}% of extent)")

    with torch.no_grad():
        h = model.encode_scene()
        disps = model.rollout(h, z0)            # [T-1, G, 3]
    dmax = float(disps.norm(dim=-1).max())
    print(f"[rand-gnn] max gaussian displacement={dmax:.3f} "
          f"({dmax/extent*100:.2f}% of extent)")

    # ── official camera + official renderer ──────────────────────────────────
    cams = json.load(open(f"{args.model}/cameras.json"))
    c = cams[args.cam_id]
    cam = cam_from_json(c, scale_long=args.scale)
    print(f"[rand-gnn] camera[{args.cam_id}] {c['img_name']} -> "
          f"{cam.image_width}x{cam.image_height}")

    bg = torch.zeros(3, device=dev)             # white_background=False
    frames = []
    for t in range(args.frames):
        with torch.no_grad():
            g._xyz = canon if t == 0 else canon + disps[t - 1]
            img = render(cam, g, Pipe(), bg)["render"].clamp(0, 1)
        frames.append((img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
        if t % 6 == 0:
            print(f"  frame {t:2d}/{args.frames}  mean={frames[-1].mean():.1f}")

    iio.mimsave(args.out, frames, fps=args.fps, quality=9)
    print(f"[rand-gnn] wrote {args.out}")

    # per-frame diff vs t=0, to prove motion is actually present
    f0 = frames[0].astype(np.float32)
    for t in [args.frames // 2, args.frames - 1]:
        d = np.abs(frames[t].astype(np.float32) - f0)
        print(f"  t={t:2d}  meanAbsDiff={d.mean():.3f}  "
              f"px_changed(>2)={(d.max(2) > 2).sum()}")


if __name__ == "__main__":
    main()

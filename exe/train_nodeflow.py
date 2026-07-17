#!/usr/bin/env python
"""NodeFlow training — GNN control-node dynamics from multi-view rendering loss.

Architecture (From Tokens to Nodes-inspired):
  - canonical 3DGS (static, fixed) as Gaussian soup
  - K control nodes (FPS-sampled, fixed positions)
  - Shared trajectory [K, T-1, 3]: displacement of each node at each frame
  - Per-view initial offset [V, K, 3]: each generated video's physical initial state
  - GNN: K node displacements → G Gaussian displacements (message passing)
  - Rendered frames compared to SVD-generated single-view videos (L1 + SSIM loss)

Usage (on GPU instance, WS=/workspace):
    cd $WS/anchorflow
    PYTHONPATH=$WS/anchorflow/lib python exe/train_nodeflow.py \\
        --ply  $WS/wolf_views/wolf_aligned.ply \\
        --data $WS/wolf_views \\
        --cfg  cfg/nodeflow_wolf.yaml \\
        --out  $WS/wolf_nodeflow \\
        --resume
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn.functional as F
import imageio
from omegaconf import OmegaConf
from PIL import Image

sys.path.append("gaussian-splatting")
from scene.gaussian_model import GaussianModel
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings, GaussianRasterizer)
from plyfile import PlyData

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
from anchorflow.nodeflow import NodeFlow
from anchorflow.checkpoint import CheckpointManager, load_rng_state


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------
class Cam:
    def __init__(self, meta: dict):
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


# ---------------------------------------------------------------------------
# Differentiable renderer (rasterizer path, no GaussianModel dependency)
# ---------------------------------------------------------------------------
def render_gaussians(
    cam: Cam,
    xyz: torch.Tensor,          # [G,3] deformed positions (needs grad)
    opacities: torch.Tensor,    # [G,1]
    scales: torch.Tensor,       # [G,3]
    rotations: torch.Tensor,    # [G,4]
    colors: torch.Tensor,       # [G,3]  precomputed RGB
    bg: torch.Tensor,           # [3]
) -> torch.Tensor:
    """Returns [3, H, W] rendered image."""
    H, W = cam.image_height, cam.image_width
    settings = GaussianRasterizationSettings(
        image_height=H,
        image_width=W,
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
    means2D = torch.zeros_like(xyz, requires_grad=True)
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
    return rendered                                             # [3,H,W]


# ---------------------------------------------------------------------------
# SSIM loss (pure torch, no external deps)
# ---------------------------------------------------------------------------
def ssim(pred: torch.Tensor, target: torch.Tensor, C1=0.01**2, C2=0.03**2) -> torch.Tensor:
    """Simplified SSIM using box filter. pred/target: [3,H,W] in [0,1]."""
    p, t = pred.unsqueeze(0), target.unsqueeze(0)             # [1,3,H,W]
    k = 11
    pad = k // 2

    def pool(x):
        return F.avg_pool2d(x, k, stride=1, padding=pad)

    mu1, mu2 = pool(p), pool(t)
    mu1_sq, mu2_sq = mu1 ** 2, mu2 ** 2
    mu12 = mu1 * mu2
    s1 = pool(p * p) - mu1_sq
    s2 = pool(t * t) - mu2_sq
    s12 = pool(p * t) - mu12
    num = (2 * mu12 + C1) * (2 * s12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (s1 + s2 + C2)
    return (num / den.clamp(min=1e-8)).mean()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class ViewVideoDataset:
    """Loads pre-generated single-view videos from gen_views.py output."""

    def __init__(self, data_dir: str, n_frames: int, res: int):
        cam_path = os.path.join(data_dir, "cameras.json")
        with open(cam_path) as f:
            cam_metas = json.load(f)
        self.cameras = [Cam(m) for m in cam_metas]
        self.n_views = len(self.cameras)
        self.n_frames = n_frames
        self.res = res

        # load all frames [V, T, 3, H, W] float32 in [0,1]
        frames = []
        for i in range(self.n_views):
            view_dir = os.path.join(data_dir, f"view_{i:02d}")
            view_frames = []
            for t in range(n_frames):
                png = os.path.join(view_dir, f"{t:05d}.png")
                img = Image.open(png).convert("RGB").resize((res, res), Image.LANCZOS)
                arr = np.asarray(img, dtype=np.float32) / 255.0
                view_frames.append(torch.from_numpy(arr).permute(2, 0, 1))  # [3,H,W]
            frames.append(torch.stack(view_frames))                          # [T,3,H,W]
        self.frames = torch.stack(frames).cuda()                             # [V,T,3,H,W]
        print(f"[dataset] {self.n_views} views × {n_frames} frames  "
              f"res={res}  ({self.frames.shape})")

    def get(self, view_idx: int, frame_idx: int) -> torch.Tensor:
        return self.frames[view_idx, frame_idx]                # [3,H,W]


# ---------------------------------------------------------------------------
# Gaussian attributes from PLY (static, no grad)
# ---------------------------------------------------------------------------
def load_gaussian_attrs(ply_path: str, device="cuda"):
    sh_max = 3
    names = [p.name for p in PlyData.read(ply_path)["vertex"].properties
             if p.name.startswith("f_rest_")]
    sh = min(int(math.sqrt((len(names) + 3) // 3)) - 1 if names else 0, sh_max)
    g = GaussianModel(sh)
    g.load_ply(ply_path)
    g.active_sh_degree = sh

    SH_C0 = 0.28209479177387814
    xyz = g.get_xyz.detach()
    opacities = g.get_opacity.detach()                        # [G,1]
    scales    = g.get_scaling.detach()                        # [G,3]
    rotations = g.get_rotation.detach()                       # [G,4]
    # degree-0 SH → diffuse RGB (no view-dependent effects for simplicity)
    f_dc = g._features_dc.detach()[:, 0, :]                   # [G,3]
    colors = (SH_C0 * f_dc + 0.5).clamp(0, 1)                # [G,3]

    return {
        "xyz": xyz.to(device),
        "opacities": opacities.to(device),
        "scales": scales.to(device),
        "rotations": rotations.to(device),
        "colors": colors.to(device),
    }


# ---------------------------------------------------------------------------
def git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "nogit"


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply",  required=True, help="canonical 3DGS .ply")
    ap.add_argument("--data", required=True, help="gen_views.py output dir")
    ap.add_argument("--cfg",  required=True, help="OmegaConf yaml")
    ap.add_argument("--out",  required=True, help="checkpoint output dir")
    ap.add_argument("--r2",   default=None,  help="rclone R2 dest (optional)")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    cfg = OmegaConf.load(args.cfg)
    gh  = git_hash()
    dev = "cuda"

    # --- Gaussian attributes (fixed) ---
    gauss = load_gaussian_attrs(args.ply, dev)
    canonical_xyz = gauss["xyz"]                               # [G,3] no grad
    G = canonical_xyz.shape[0]
    print(f"[train] N={G} Gaussians  commit={gh}")

    # --- dataset ---
    dataset = ViewVideoDataset(args.data, cfg.model.n_frames, cfg.views.res)
    V = dataset.n_views
    T = dataset.n_frames

    bg = torch.tensor([1., 1., 1.] if cfg.get("white_bg", True) else [0., 0., 0.],
                      device=dev)

    # --- model ---
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
    print(f"[train] NodeFlow  K={model.n_nodes}  V={V}  T={T}  "
          f"hidden={cfg.model.hidden}  layers={cfg.model.n_gnn_layers}")

    # --- optimizer (three param groups for independent lr tuning) ---
    opt = torch.optim.Adam([
        {"params": list(model.node_encoder.parameters()) +
                   list(model.gnn_layers.parameters()) +
                   list(model.decoder.parameters()),
         "lr": cfg.train.lr_gnn},
        {"params": [model.node_traj],   "lr": cfg.train.lr_traj},
        {"params": [model.init_offset], "lr": cfg.train.lr_init},
    ])

    # --- resume ---
    ckpt_mgr = CheckpointManager(args.out)
    start = 0
    resume = ckpt_mgr.load() if args.resume else None
    if resume is not None:
        model.load_state_dict(resume["model"])
        opt.load_state_dict(resume["opt"])
        load_rng_state(resume.get("rng"))
        start = resume["step"] + 1
        print(f"[train] resumed from step {start}")

    # torch.compile the GNN forward for speed (model itself stays uncompiled so
    # state_dict / optimizer / resume keep clean, checkpoint-compatible keys).
    try:
        fwd = torch.compile(model)
        print("[train] torch.compile enabled")
    except Exception as e:
        fwd = model
        print(f"[train] torch.compile unavailable ({e}); running eager")

    def sync_r2():
        if args.r2:
            os.system(f"rclone copy {args.out} {args.r2} >/dev/null 2>&1")

    rng = torch.Generator(device="cpu")

    # --- training loop ---
    for step in range(start, cfg.train.iters):
        # sample random (view, frame)
        v = int(torch.randint(0, V, (1,), generator=rng).item())
        t = int(torch.randint(0, T, (1,), generator=rng).item())

        target = dataset.get(v, t)                             # [3,H,W]

        opt.zero_grad()

        # forward: GNN → Gaussian displacements
        gauss_disp = fwd(v, float(t))                         # [G,3]
        xyz_def = canonical_xyz + gauss_disp                   # [G,3] deformed

        # render
        pred = render_gaussians(
            cam=dataset.cameras[v],
            xyz=xyz_def,
            opacities=gauss["opacities"],
            scales=gauss["scales"],
            rotations=gauss["rotations"],
            colors=gauss["colors"],
            bg=bg,
        )                                                      # [3,H,W]

        # rendering loss
        l_l1   = F.l1_loss(pred, target)
        l_ssim = 1.0 - ssim(pred, target)
        loss   = l_l1 + cfg.train.lambda_ssim * l_ssim

        # regularization
        if cfg.train.lambda_arap > 0:
            loss = loss + cfg.train.lambda_arap * model.arap_loss(v, float(t))
        if cfg.train.lambda_smooth > 0:
            loss = loss + cfg.train.lambda_smooth * model.smooth_loss()
        if cfg.train.lambda_init > 0:
            loss = loss + cfg.train.lambda_init * model.traj_magnitude_loss()

        if not torch.isfinite(loss):
            print(f"[train {step}] non-finite loss, skip"); continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        opt.step()

        if step % cfg.train.log_every == 0:
            print(f"[{step}/{cfg.train.iters}] "
                  f"l1={float(l_l1):.4f}  ssim={float(l_ssim):.4f}  "
                  f"v={v}  t={t}")

        if step % cfg.train.ckpt_every == 0:
            ckpt_mgr.save(step, {
                "model": model.state_dict(),
                "opt":   opt.state_dict(),
                "step":  step,
            })
            sync_r2()

    # final checkpoint
    ckpt_mgr.save(cfg.train.iters - 1, {
        "model": model.state_dict(),
        "opt":   opt.state_dict(),
        "step":  cfg.train.iters - 1,
    })
    sync_r2()
    print(f"[train] done  commit={gh}  → {args.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""AnchorFlow: MDS-based dynamic 3DGS via semantic anchor nodes + z0-bank.

Canonical asset = the official INRIA 3DGS pretrained scene, rendered with its
own bundled cameras.json and the official gaussian_renderer.render (SH degree 3,
background per cfg_args). No hand-tuned camera parameters anywhere.

Pipeline:
  1. Anchor nodes over the WHOLE scene (FPS, or tokens_to_nodes for semantic)
  2. NodeFlow GNN: canonical node positions -> scene features h [K,H]
     Physics: vel[t] = z0 + dt*cumsum(acc), disp[t] = dt*cumsum(vel)
  3. z0_bank [B, K, 3]: learnable initial velocities, one sampled per step
  4. SVD MDS loss: grad = w*(eps(video) - eps(static_frame0))

Usage:
    python exe/train_anchorflow.py \
        --model /workspace/gs_official/kitchen \
        --cfg   cfg/anchorflow_kitchen.yaml \
        --out   /workspace/anchorflow_out --resume
"""
from __future__ import annotations

import argparse, json, math, os, random, subprocess, sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import imageio.v2 as iio
from torch.utils.checkpoint import checkpoint
from omegaconf import OmegaConf
from PIL import Image

sys.path.append("/workspace/gaussian-splatting")
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, focal2fov

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
from anchorflow.nodeflow import NodeFlow
from anchorflow.checkpoint import CheckpointManager, load_rng_state
from anchorflow.sds import SVDGuidance


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


def load_official_cameras(model_dir: str, n_views: int, long_side: int) -> list:
    """cameras.json -> Cam list, evenly spaced over the capture.

    3DGS stores the C2W rotation and the camera centre; FoV is
    resolution-independent so downscaling only changes W/H.
    """
    cams_json = json.load(open(f"{model_dir}/cameras.json"))
    idx = np.linspace(0, len(cams_json) - 1, n_views).round().astype(int)
    cams = []
    for i in idx:
        c = cams_json[int(i)]
        rot = np.array(c["rotation"], dtype=np.float32)
        pos = np.array(c["position"], dtype=np.float32)
        R, T = rot, -rot.T @ pos
        W, H = c["width"], c["height"]
        fovx, fovy = focal2fov(c["fx"], W), focal2fov(c["fy"], H)
        s = long_side / max(W, H)
        # VAE needs multiples of 8
        W8 = max(8, int(round(W * s / 8)) * 8)
        H8 = max(8, int(round(H * s / 8)) * 8)
        cams.append(Cam(R, T, fovx, fovy, W8, H8))
    print(f"[train] cameras={len(cams)} (official cameras.json) "
          f"{cams[0].image_width}x{cams[0].image_height}")
    return cams


def git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "nogit"


def save_video(frames, path, fps=8):
    arr = [(f.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
           for f in frames]
    iio.mimsave(path, arr, fps=fps, quality=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",  required=True,
                    help="official 3DGS model dir (has point_cloud/ + cameras.json)")
    ap.add_argument("--iter",   type=int, default=30000, help="pretrained iteration")
    ap.add_argument("--cfg",    required=True)
    ap.add_argument("--out",    required=True)
    ap.add_argument("--r2",     default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--white_bg", action="store_true",
                    help="match cfg_args white_background (kitchen: False)")
    ap.add_argument("--no-t2n", action="store_true", help="skip tokens_to_nodes, use FPS")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    cfg = OmegaConf.load(args.cfg)
    dev = "cuda"
    gh = git_hash()

    # ── official pretrained scene ────────────────────────────────────────────
    g = GaussianModel(3)
    g.load_ply(f"{args.model}/point_cloud/iteration_{args.iter}/point_cloud.ply")
    g.active_sh_degree = 3
    canonical_xyz = g.get_xyz.detach().clone()
    G = canonical_xyz.shape[0]
    print(f"[train] gaussians={G}  commit={gh}")

    bg = torch.tensor([1., 1., 1.] if args.white_bg else [0., 0., 0.], device=dev)

    T = cfg.model.n_frames
    cameras = load_official_cameras(args.model, cfg.train.n_views, cfg.model.res)
    V = len(cameras)

    def render_at(cam, xyz):
        """Official renderer with deformed positions. Gradients flow via xyz."""
        g._xyz = xyz
        return render(cam, g, Pipe(), bg)["render"]

    def render_ckpt(cam, xyz):
        """Checkpointed render: recompute in backward instead of storing the
        rasteriser's activations. At G=1.85M x T=25 the stored graph alone OOMs
        a 24GB card."""
        return checkpoint(lambda x: render_at(cam, x), xyz, use_reentrant=False)

    def render_canonical(cam):
        with torch.no_grad():
            return render_at(cam, canonical_xyz).clamp(0, 1)

    # sanity: cameras must actually see the scene
    img0 = render_canonical(cameras[0])
    cover = float((img0.max(0).values > 0.01).float().mean())
    print(f"[train] cam[0] coverage={cover*100:.1f}%  mean={float(img0.mean()):.3f}")
    if cover < 0.02:
        sys.exit("[train] ABORT: cameras do not see the scene")

    # ── scene extent (1-99 pct, robust to floaters) ─────────────────────────
    sub = canonical_xyz[torch.randperm(G, device=dev)[:200000]].float()
    extent = float((torch.quantile(sub, 0.99, dim=0)
                    - torch.quantile(sub, 0.01, dim=0)).norm())
    print(f"[train] scene extent={extent:.2f}")

    # ── anchor nodes over the whole scene ───────────────────────────────────
    node_pos = None
    if not args.no_t2n:
        try:
            from anchorflow.tokens_to_nodes import tokens_to_nodes
            import anchorflow.tokens_to_nodes as t2n_mod
            print("[train] building semantic nodes via tokens_to_nodes ...")
            node_pos = tokens_to_nodes(
                canonical_xyz, g.get_opacity.detach(),
                render_canonical, cameras[:cfg.get("t2n_views", 4)],
                n_nodes=cfg.model.n_nodes, device=dev,
            )
            if t2n_mod._dino_model is not None:
                del t2n_mod._dino_model
                t2n_mod._dino_model = None
            import gc; gc.collect(); torch.cuda.empty_cache()
            print("[train] DINOv2 freed")
        except Exception as e:
            print(f"[train] tokens_to_nodes failed ({e}) -> FPS")
            node_pos = None
            import gc; gc.collect(); torch.cuda.empty_cache()

    dt = float(cfg.model.get("dt", 0.1))
    model = NodeFlow(
        canonical_xyz=canonical_xyz, node_positions=node_pos,
        n_nodes=cfg.model.n_nodes, n_frames=T,
        hidden=cfg.model.hidden, n_gnn_layers=cfg.model.n_gnn_layers,
        k_node=cfg.model.k_node, k_gauss=cfg.model.k_gauss, dt=dt,
    ).to(dev)
    K = model.n_nodes
    print(f"[train] nodes={K}  hidden={cfg.model.hidden}  dt={dt}")

    # z0 scaled to the scene: dt*|z0|*T ~= z0_motion * extent
    z0_motion = float(cfg.train.get("z0_motion", 0.01))
    z0_std = z0_motion * extent / (dt * max(T - 1, 1))
    B = cfg.train.z0_bank_size
    z0_bank = torch.nn.Parameter(torch.randn(B, K, 3, device=dev) * z0_std)
    print(f"[train] z0_bank {list(z0_bank.shape)}  std={z0_std:.4f} "
          f"(~{z0_motion*100:.1f}% of extent per clip)")

    print("[train] loading SVD for MDS guidance ...")
    svd = SVDGuidance(
        sigma_min=cfg.mds.sigma_min, sigma_max=cfg.mds.sigma_max,
        guidance_scale=cfg.mds.guidance_scale,
        motion_bucket_id=cfg.mds.motion_bucket_id,
        grad_clip=cfg.mds.grad_clip, device=dev,
    )

    gnn_params = (list(model.node_encoder.parameters())
                  + list(model.gnn_layers.parameters())
                  + list(model.accel_decoder.parameters()))
    opt = torch.optim.Adam([
        {"params": gnn_params, "lr": cfg.train.lr_gnn},
        {"params": [z0_bank],  "lr": cfg.train.lr_z0},
    ])

    ckpt_mgr = CheckpointManager(args.out)
    start = 0
    if args.resume:
        ckpt = ckpt_mgr.load()
        if ckpt is not None:
            model.load_state_dict(ckpt["model"])
            opt.load_state_dict(ckpt["opt"])
            z0_bank.data.copy_(ckpt["z0_bank"])
            load_rng_state(ckpt.get("rng"))
            start = ckpt["step"] + 1
            print(f"[train] resumed from step {start}")

    torch.set_float32_matmul_precision("high")

    def sync_r2():
        if args.r2:
            os.system(f"rclone copy {args.out} {args.r2} >/dev/null 2>&1")

    rng = random.Random(42)

    for step in range(start, cfg.train.iters):
        k = rng.randint(0, B - 1)
        v = rng.randint(0, V - 1)
        z0, cam = z0_bank[k], cameras[v]

        frame0 = render_canonical(cam)

        h = model.encode_scene()
        # node-space rollout; LBS per frame (batched LBS would be ~2GB at G=1.85M)
        node_disps = model.rollout_nodes(h, z0)          # [T-1, K, 3]

        frames = [frame0]
        for i in range(T - 1):
            disp = model.lbs_frame(node_disps[i])        # [G, 3]
            frames.append(render_ckpt(cam, canonical_xyz + disp))
        frames_t = torch.stack(frames, dim=0)            # [T, 3, H, W]

        opt.zero_grad()
        loss = svd.mds_loss(frames_t, cond_image=frame0, w_power=cfg.mds.w_power)

        if cfg.train.lambda_arap > 0:
            t_reg = rng.randint(1, T - 1)
            loss = loss + cfg.train.lambda_arap * model.arap_loss(h, z0, t_reg)
        if cfg.train.lambda_z0 > 0:
            loss = loss + cfg.train.lambda_z0 * (z0_bank ** 2).mean()

        if not torch.isfinite(loss):
            print(f"[{step}] non-finite loss, skip")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(gnn_params + [z0_bank], cfg.train.grad_clip)
        opt.step()

        if step % cfg.train.log_every == 0:
            with torch.no_grad():
                travel = float(model.lbs_frame(node_disps[-1]).norm(dim=-1).max())
            print(f"[{step}/{cfg.train.iters}] loss={float(loss):.4f}  k={k} v={v}  "
                  f"z0_rms={float(z0_bank.detach().pow(2).mean().sqrt()):.4f}  "
                  f"travel={travel:.3f} ({travel/extent*100:.2f}%)")

        if step % cfg.train.ckpt_every == 0:
            ckpt_mgr.save(step, {
                "model": model.state_dict(), "opt": opt.state_dict(),
                "z0_bank": z0_bank.data, "step": step,
            })
            _save_rollout(step, model, z0_bank, cameras[0], canonical_xyz,
                          render_at, T, args.out)
            sync_r2()

    ckpt_mgr.save(cfg.train.iters - 1, {
        "model": model.state_dict(), "opt": opt.state_dict(),
        "z0_bank": z0_bank.data, "step": cfg.train.iters - 1,
    })
    sync_r2()
    print(f"[train] done  commit={gh} -> {args.out}")


@torch.no_grad()
def _save_rollout(step, model, z0_bank, cam, canon, render_at, T, out):
    h = model.encode_scene()
    node_disps = model.rollout_nodes(h, z0_bank[0])
    frames = [render_at(cam, canon).clamp(0, 1)]
    for i in range(T - 1):
        frames.append(render_at(cam, canon + model.lbs_frame(node_disps[i])).clamp(0, 1))
    path = os.path.join(out, f"rollout_step{step:06d}.mp4")
    save_video(frames, path)
    print(f"  saved rollout -> {path}")


if __name__ == "__main__":
    main()

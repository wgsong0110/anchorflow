#!/usr/bin/env python
"""SeqGen MDS training.

Non-autoregressive sequence generator trained with Motion Distillation Sampling.

Given initial velocities for K randomly-sampled conditioning nodes, SeqGen
generates the full trajectory [T, N, 3] for all anchor nodes in one forward
pass.  Supervision comes entirely from MDS (SVD-based video diffusion prior);
no GT trajectory is needed.

Usage:
    python exe/train_seqgen_mds.py \
        --model /workspace/scgs_jump_node \
        --ply_iter 80000 \
        --out /workspace/seqgen_jump \
        --cfg cfg/anchorflow_kitchen.yaml
"""
from __future__ import annotations

import argparse, json, math, os, random, subprocess, sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn as nn
import imageio.v2 as iio
from torch.utils.checkpoint import checkpoint
from omegaconf import OmegaConf

sys.path.append("/workspace/SC-GS")
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render as _render_scgs
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, focal2fov

def render(cam, g, pipe, bg):
    """Wrapper: SC-GS render with zero deformations (canonical / LBS-pre-applied)."""
    zeros = torch.zeros_like(g.get_xyz)
    return _render_scgs(cam, g, pipe, bg, d_xyz=zeros, d_rotation=0.0, d_scaling=zeros)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
from anchorflow.anchors import AnchorSet
from anchorflow.graph import knn_graph
from anchorflow import warp as W
from anchorflow.seqgen import SeqGen
from anchorflow.sds import SVDGuidance
from anchorflow.checkpoint import CheckpointManager


# ── camera / scene helpers (shared with train_anchorflow.py) ─────────────────

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
    compute_cov3D_python = True
    debug = False
    antialiasing = False


def load_cameras(model_dir, n_views, long_side):
    cams_json = json.load(open(f"{model_dir}/cameras.json"))
    idx = np.linspace(0, len(cams_json) - 1, n_views).round().astype(int)
    cams = []
    for i in idx:
        c = cams_json[int(i)]
        if "rotation" in c:
            rot = np.array(c["rotation"], dtype=np.float32)
            pos = np.array(c["position"], dtype=np.float32)
            Wd, Hd = c["width"], c["height"]
            fovx, fovy = focal2fov(c["fx"], Wd), focal2fov(c["fy"], Hd)
            T_vec = -rot.T @ pos
        else:
            rot = np.array(c["R"], dtype=np.float32)
            T_vec = np.array(c["T"], dtype=np.float32)
            Wd, Hd = c["W"], c["H"]
            fovx, fovy = c["fov_x"], c["fov_y"]
        s = long_side / max(Wd, Hd)
        W8 = max(8, int(round(Wd * s / 8)) * 8)
        H8 = max(8, int(round(Hd * s / 8)) * 8)
        cams.append(Cam(rot, T_vec, fovx, fovy, W8, H8))
    print(f"[train] cameras={len(cams)}  {cams[0].image_width}x{cams[0].image_height}")
    return cams


def save_video(frames, path, fps=8):
    arr = [(f.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
           for f in frames]
    iio.mimsave(path, arr, fps=fps, quality=8)


def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "nogit"


# ── velocity sampling ─────────────────────────────────────────────────────────

def sample_velocities(N: int, K: int, extent: float, device: str,
                      vel_scale: float = 0.05) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample K random conditioning nodes and unit-sphere velocities.

    Returns:
        cond_ids  [K] long
        cond_vel  [K, 3] float — random direction, magnitude ~ vel_scale * extent
    """
    cond_ids = torch.randperm(N, device=device)[:K]
    dirs = torch.randn(K, 3, device=device)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    mags = torch.rand(K, device=device) * vel_scale * extent
    cond_vel = dirs * mags[:, None]
    return cond_ids, cond_vel


# ── rollout → frames ──────────────────────────────────────────────────────────

def traj_to_frames(traj, canon_xyz, canon_cov6, anchors, g, bg, cam, pipe,
                   use_checkpoint=True):
    """Render [T, 3, H, W] from node trajectory [T, N, 3].

    Uses LBS warp: each Gaussian follows its K nearest anchor nodes.
    Checkpointing is applied per-frame to bound peak VRAM.
    """
    w_b, idx_b = anchors.cal_nn_weight(canon_xyz)

    def _render_frame(pt):
        pos, cov6, _ = W.lbs_warp(canon_xyz, canon_cov6, w_b, idx_b,
                                   anchors.canonical, pt)
        g._xyz = pos
        g.get_covariance = lambda scaling_modifier=1.0, **kw: cov6
        return render(cam, g, pipe, bg)["render"]

    frames = []
    for t in range(traj.shape[0]):
        pt = traj[t]
        if use_checkpoint and pt.requires_grad:
            frames.append(checkpoint(_render_frame, pt, use_reentrant=False))
        else:
            frames.append(_render_frame(pt))
    return torch.stack(frames, dim=0)   # [T, 3, H, W]


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",    required=True, help="SC-GS / 3DGS model dir")
    ap.add_argument("--ply_iter", type=int, default=30000,
                    help="point_cloud iteration to load")
    ap.add_argument("--out",      required=True)
    ap.add_argument("--cfg",      required=True)
    ap.add_argument("--iters",    type=int, default=None)
    ap.add_argument("--resume",   action="store_true")
    ap.add_argument("--white_bg", action="store_true")
    # SeqGen hyper-params (override cfg if provided)
    ap.add_argument("--n_nodes",  type=int, default=None)
    ap.add_argument("--n_frames", type=int, default=None)
    ap.add_argument("--d_model",  type=int, default=256)
    ap.add_argument("--n_layers", type=int, default=6)
    ap.add_argument("--n_heads",  type=int, default=8)
    # Conditioning
    ap.add_argument("--k_cond",   type=int, default=16,
                    help="number of conditioning nodes per step")
    ap.add_argument("--vel_scale", type=float, default=0.05,
                    help="velocity magnitude relative to scene extent")
    # MDS
    ap.add_argument("--w_power",  type=float, default=0.0)
    ap.add_argument("--lambda_arap", type=float, default=0.01)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    cfg = OmegaConf.load(args.cfg)
    if args.iters is not None:
        cfg.train.iters = args.iters
    if args.n_nodes is not None:
        cfg.model.n_nodes = args.n_nodes
    if args.n_frames is not None:
        cfg.model.n_frames = args.n_frames

    dev = "cuda"
    T = int(cfg.model.n_frames)
    gh = git_hash()

    # ── load canonical 3DGS ─────────────────────────────────────────────────
    ply_path = f"{args.model}/point_cloud/iteration_{args.ply_iter}/point_cloud.ply"
    # read hyper_dim from SC-GS cfg_args (saved alongside the model)
    _hyper_dim = 0
    _cfg_args_path = f"{args.model}/cfg_args"
    if os.path.exists(_cfg_args_path):
        import ast
        _ns = eval(open(_cfg_args_path).read().strip(), {"Namespace": lambda **kw: kw})
        _hyper_dim = _ns.get("hyper_dim", 0) if isinstance(_ns, dict) else 0
    g = GaussianModel(3, fea_dim=_hyper_dim)
    g.load_ply(ply_path)
    g.active_sh_degree = 3
    canon_xyz  = g.get_xyz.detach().clone()        # [G, 3]
    canon_cov6 = W.cov_from_scale_rot(
        g.get_scaling.detach(), g._rotation.detach()).detach()
    G_cnt = canon_xyz.shape[0]
    print(f"[train] gaussians={G_cnt}  T={T}  commit={gh}")

    bg = torch.tensor([1., 1., 1.] if args.white_bg else [0., 0., 0.], device=dev)
    pipe = Pipe()
    cameras = load_cameras(args.model, cfg.train.n_views, cfg.model.res)
    V = len(cameras)

    def render_canonical(cam):
        with torch.no_grad():
            g._xyz = canon_xyz
            g.get_covariance = lambda scaling_modifier=1.0, **kw: canon_cov6
            return render(cam, g, pipe, bg)["render"].clamp(0, 1)

    # sanity check
    img0 = render_canonical(cameras[0])
    cover = float((img0.max(0).values > 0.01).float().mean())
    print(f"[train] cam[0] coverage={cover*100:.1f}%")
    if cover < 0.02:
        sys.exit("[train] ABORT: cameras do not see the scene")

    extent = float((torch.quantile(canon_xyz, 0.99, dim=0)
                    - torch.quantile(canon_xyz, 0.01, dim=0)).norm())
    print(f"[train] scene extent={extent:.3f}")

    # ── anchor nodes (FPS) ──────────────────────────────────────────────────
    N_nodes = int(cfg.model.n_nodes)
    anchors, _ = AnchorSet.from_gaussians(
        canon_xyz, node_num=N_nodes, latent_dim=0, e_dim=0,
        K=int(cfg.model.k_gauss))
    anchors = anchors.to(dev)
    N = anchors.num
    print(f"[train] anchor nodes={N}")

    # ── SeqGen model ─────────────────────────────────────────────────────────
    model = SeqGen(
        canon_pos=anchors.canonical,
        n_frames=T,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
    ).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] SeqGen params={n_params/1e6:.2f}M")

    # ── ARAP edge graph (regulariser) ────────────────────────────────────────
    arap_k = min(6, N - 1)
    arap_edge = knn_graph(anchors.canonical.detach(), k=arap_k)

    # ── optimizer ────────────────────────────────────────────────────────────
    _lr = float(cfg.train.get("lr", 1e-4))
    opt = torch.optim.AdamW(model.parameters(), lr=_lr, weight_decay=1e-4)

    # ── SVD guidance (MDS) ───────────────────────────────────────────────────
    svd_model_id = cfg.get("svd_model", "stabilityai/stable-video-diffusion-img2vid-xt")
    svd = SVDGuidance(model_id=svd_model_id, device=dev)

    # Precompute per-camera frame-0 conditioning cache (canonical render)
    print("[train] precomputing MDS conditioning cache...")
    cond_cache = []
    frame0_cache = []
    for cam in cameras:
        f0 = render_canonical(cam)
        frame0_cache.append(f0)
        cond_cache.append(svd.precompute_cond(f0, T))
    print("[train] cache ready")

    # ── checkpoint manager ───────────────────────────────────────────────────
    ckpt_mgr = CheckpointManager(args.out, keep_last=3)
    start_step = 0
    if args.resume:
        ck = ckpt_mgr.load_latest()
        if ck is not None:
            model.load_state_dict(ck["model"])
            opt.load_state_dict(ck["opt"])
            start_step = ck.get("step", 0) + 1
            print(f"[train] resumed from step {start_step - 1}")

    # ── training loop ────────────────────────────────────────────────────────
    K_cond = args.k_cond
    vel_scale = args.vel_scale
    lambda_arap = args.lambda_arap
    w_power = args.w_power
    log_every  = int(cfg.train.log_every)
    ckpt_every = int(cfg.train.ckpt_every)

    for step in range(start_step, int(cfg.train.iters)):
        model.train()
        opt.zero_grad()

        # camera for this step
        v = step % V
        cam = cameras[v]

        # sample conditioning: K random nodes + random velocities
        cond_ids, cond_vel = sample_velocities(N, K_cond, extent, dev, vel_scale)
        cond_vel = cond_vel.requires_grad_(False)

        # SeqGen forward: [T, N, 3]
        traj = model(cond_ids, cond_vel)

        # render frames: [T, 3, H, W]
        frames_t = traj_to_frames(traj, canon_xyz, canon_cov6, anchors,
                                  g, bg, cam, pipe, use_checkpoint=True)

        # MDS loss
        loss = svd.mds_loss(
            frames_t,
            cond_image=frame0_cache[v],
            w_power=w_power,
            cond_cache=cond_cache[v],
            vae_checkpoint=False,
        )

        # ARAP regularisation on a random timestep
        if lambda_arap > 0:
            t_r = random.randint(1, T - 1)
            src, dst = arap_edge
            d_rest = (anchors.canonical[src] - anchors.canonical[dst]).norm(dim=-1)
            d_now  = (traj[t_r][src] - traj[t_r][dst]).norm(dim=-1)
            loss = loss + lambda_arap * ((d_now - d_rest) ** 2).mean()

        if not torch.isfinite(loss):
            print(f"[{step}] non-finite loss, skip")
            continue

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), float(cfg.train.grad_clip))
        opt.step()

        if step % log_every == 0:
            with torch.no_grad():
                travel = float((traj[-1] - anchors.canonical).norm(dim=-1).max())
            print(f"[{step}/{cfg.train.iters}] loss={float(loss):.4f} "
                  f"v={v} k={K_cond} travel={travel:.3f} "
                  f"({travel/extent*100:.1f}%)")

        if step % ckpt_every == 0 or step == cfg.train.iters - 1:
            ckpt_mgr.save(step, {"model": model.state_dict(),
                                 "opt": opt.state_dict(), "step": step})
            _save_rollout(step, model, anchors, N, T, extent, dev,
                          canon_xyz, canon_cov6, g, bg, cameras[0], pipe, args.out)

    print(f"[train] done commit={gh} -> {args.out}")


@torch.no_grad()
def _save_rollout(step, model, anchors, N, T, extent, dev,
                  canon_xyz, canon_cov6, g, bg, cam, pipe, out):
    model.eval()
    # fixed conditioning: a few nodes pushed in +x direction
    cond_ids = torch.arange(min(8, N), device=dev)
    cond_vel = torch.zeros(len(cond_ids), 3, device=dev)
    cond_vel[:, 0] = 0.05 * extent
    traj = model(cond_ids, cond_vel)
    w_b, idx_b = anchors.cal_nn_weight(canon_xyz)
    frames = []
    for t in range(T):
        pos, cov6, _ = W.lbs_warp(canon_xyz, canon_cov6, w_b, idx_b,
                                   anchors.canonical, traj[t])
        g._xyz = pos
        g.get_covariance = lambda scaling_modifier=1.0, **kw: cov6
        frames.append(render(cam, g, pipe, bg)["render"].clamp(0, 1))
    path = os.path.join(out, f"rollout_{step:06d}.mp4")
    save_video(frames, path)
    print(f"  [rollout] saved -> {path}")


if __name__ == "__main__":
    main()

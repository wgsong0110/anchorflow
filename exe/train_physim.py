#!/usr/bin/env python
"""Physics simulator (mass-spring) training with SVD MDS supervision.

Per-node spring stiffness and global damping are learned from video diffusion
prior (MDS).  External force f_ext is sampled randomly each step — random
direction and magnitude — so the model sees diverse motions during training.

Target objects: passive, externally-driven scenes (wind-blown plants, cloth, …).
Self-propelled objects (humans, animals) are out of scope.

Usage:
    python exe/train_physim.py \\
        --model /workspace/scgs_ficus_node \\
        --ply_iter 80000 \\
        --out /workspace/physim_ficus \\
        --cfg cfg/physim_ficus.yaml --resume
"""
from __future__ import annotations

import argparse, json, os, subprocess, sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn.functional as F
import imageio.v2 as iio
from torch.utils.checkpoint import checkpoint
from omegaconf import OmegaConf

sys.path.append("/workspace/SC-GS")
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render as _render_scgs
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, focal2fov

def render(cam, g, pipe, bg):
    zeros = torch.zeros_like(g.get_xyz)
    return _render_scgs(cam, g, pipe, bg, d_xyz=zeros, d_rotation=0.0, d_scaling=zeros)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
from anchorflow.anchors import AnchorSet
from anchorflow import warp as W
from anchorflow.warp import anchor_rotations_cache
from anchorflow.physim import SpringSim
from anchorflow.sds import SVDGuidance
from anchorflow.checkpoint import CheckpointManager


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
    print(f"[train] cameras={len(cams)}  {cams[0].image_width}x{cams[0].image_height}", flush=True)
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


def sample_force(extent: float, f_scale: float, device: str) -> torch.Tensor:
    """Random unit-sphere direction * uniform magnitude in [0, f_scale*extent]."""
    d = torch.randn(3, device=device)
    d = F.normalize(d, dim=0)
    mag = torch.rand(1, device=device).item() * f_scale * extent
    return d * mag


def traj_to_frames(traj, canon_xyz, canon_cov6, anchors, g, bg, cam, pipe,
                   use_checkpoint=True, _w_b=None, _idx_b=None,
                   _arot_idx=None, _arot_src=None):
    frames = []
    for t in range(traj.shape[0]):
        pt = traj[t]
        with torch.no_grad():
            anchor_R = W.anchor_rotations(anchors.canonical, pt,
                                          _idx=_arot_idx, _src=_arot_src)
        def _render_frame(pt, _R=anchor_R):
            pos, cov6, _ = W.lbs_warp(canon_xyz, canon_cov6, _w_b, _idx_b,
                                       anchors.canonical, pt, anchor_R=_R)
            g._xyz = pos
            g.get_covariance = lambda scaling_modifier=1.0, **kw: cov6
            return render(cam, g, pipe, bg)["render"]
        if use_checkpoint and pt.requires_grad:
            frames.append(checkpoint(_render_frame, pt, use_reentrant=False))
        else:
            frames.append(_render_frame(pt))
    return torch.stack(frames, dim=0)


@torch.no_grad()
def _save_rollout(step, sim, anchors, T, extent, dev,
                  canon_xyz, canon_cov6, g, bg, cam, pipe, out,
                  f_scale=0.05, n_dirs=4):
    sim.eval()
    _w_b, _idx_b = anchors.cal_nn_weight(canon_xyz)
    _arot_idx, _arot_src = anchor_rotations_cache(anchors.canonical)

    all_frames = []
    for i in range(n_dirs):
        # fixed cardinal-ish directions for reproducible rollout
        d = torch.zeros(3, device=dev)
        d[i % 3] = 1.0 if i < 3 else -1.0
        f_ext = d * f_scale * extent
        traj = sim(f_ext)
        frames = traj_to_frames(traj, canon_xyz, canon_cov6, anchors,
                                 g, bg, cam, pipe, use_checkpoint=False,
                                 _w_b=_w_b, _idx_b=_idx_b,
                                 _arot_idx=_arot_idx, _arot_src=_arot_src)
        all_frames.extend(list(frames))

    path = os.path.join(out, f"rollout_{step:06d}.mp4")
    save_video(all_frames, path)
    print(f"  [rollout] saved -> {path}", flush=True)
    sim.train()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",    required=True)
    ap.add_argument("--ply_iter", type=int, default=30000)
    ap.add_argument("--out",      required=True)
    ap.add_argument("--cfg",      required=True)
    ap.add_argument("--resume",   action="store_true")
    ap.add_argument("--white_bg", action="store_true")
    ap.add_argument("--n_nodes",  type=int, default=None)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cfg = OmegaConf.load(args.cfg)
    dev = "cuda"
    gh = git_hash()

    # ── 3DGS canonical scene ─────────────────────────────────────────────────
    g = GaussianModel(3)
    ply = os.path.join(args.model, "point_cloud",
                       f"iteration_{args.ply_iter}", "point_cloud.ply")
    g.load_ply(ply)
    g.cuda()
    canon_xyz  = g.get_xyz.detach()
    scales_act = g.get_scaling.detach()
    rots_act   = g.get_rotation.detach()
    canon_cov6 = W.cov_from_scale_rot(scales_act, rots_act).detach()
    bg = torch.tensor([1., 1., 1.] if (args.white_bg or cfg.get("white_bg")) else [0., 0., 0.],
                      device=dev)
    pipe = Pipe()
    print(f"[train] gaussians={len(canon_xyz)}", flush=True)

    # ── anchors ──────────────────────────────────────────────────────────────
    n_nodes = args.n_nodes or 512
    anchors, _ = AnchorSet.from_gaussians(canon_xyz, node_num=n_nodes)
    anchors = anchors.to(dev)
    N = anchors.canonical.shape[0]
    extent = float((anchors.canonical.max(0).values -
                    anchors.canonical.min(0).values).norm())
    print(f"[train] anchors={N}  extent={extent:.4f}", flush=True)

    # ── cameras ──────────────────────────────────────────────────────────────
    n_views = int(cfg.train.n_views)
    res     = int(cfg.train.res)
    cameras = load_cameras(args.model, n_views, res)
    V = len(cameras)

    # ── simulator ────────────────────────────────────────────────────────────
    T   = int(cfg.sim.T)
    sim = SpringSim(
        anchors.canonical,
        T=T,
        dt=float(cfg.sim.dt),
        K=int(cfg.sim.K),
        mass=float(cfg.sim.mass),
        stiffness_init=float(cfg.sim.stiffness_init),
        damping_init=float(cfg.sim.damping_init),
    ).to(dev)
    print(f"[train] SpringSim  T={T}  edges={sim.edge_index.shape[1]}", flush=True)

    opt = torch.optim.Adam(sim.parameters(), lr=float(cfg.train.lr))

    # ── SVD guidance ─────────────────────────────────────────────────────────
    svd_model_id = cfg.get("svd_model", "stabilityai/stable-video-diffusion-img2vid-xt")
    svd = SVDGuidance(model_id=svd_model_id, device=dev)

    def render_canonical(cam):
        with torch.no_grad():
            return render(cam, g, pipe, bg)["render"]

    print("[train] precomputing MDS conditioning cache...", flush=True)
    cond_cache  = []
    frame0_cache = []
    for cam in cameras:
        f0 = render_canonical(cam)
        frame0_cache.append(f0)
        cond_cache.append(svd.precompute_cond(f0, T))
    print("[train] cache ready", flush=True)

    # ── checkpoint ───────────────────────────────────────────────────────────
    ckpt_mgr = CheckpointManager(args.out, keep_last=3)
    start_step = 0
    if args.resume:
        ck = ckpt_mgr.load()
        if ck is not None:
            sim.load_state_dict(ck["sim"])
            opt.load_state_dict(ck["opt"])
            start_step = ck.get("step", 0) + 1
            print(f"[train] resumed from step {start_step - 1}", flush=True)

    sim = torch.compile(sim)

    # ── precompute constants ─────────────────────────────────────────────────
    with torch.no_grad():
        _w_b, _idx_b = anchors.cal_nn_weight(canon_xyz)
        _arot_idx, _arot_src = anchor_rotations_cache(anchors.canonical)

    f_scale    = float(cfg.train.f_scale)
    w_power    = float(cfg.train.w_power)
    grad_clip  = float(cfg.train.grad_clip)
    log_every  = int(cfg.train.log_every)
    ckpt_every = int(cfg.train.ckpt_every)
    iters      = int(cfg.train.iters)

    print(f"[train] start  commit={gh}  steps={iters}", flush=True)

    for step in range(start_step, iters):
        sim.train()
        opt.zero_grad()

        v = step % V
        cam = cameras[v]

        # sample random external force
        f_ext = sample_force(extent, f_scale, dev).requires_grad_(False)

        # simulate
        traj = sim(f_ext)   # [T, N, 3]

        # render
        frames_t = traj_to_frames(traj, canon_xyz, canon_cov6, anchors,
                                   g, bg, cam, pipe, use_checkpoint=True,
                                   _w_b=_w_b, _idx_b=_idx_b,
                                   _arot_idx=_arot_idx, _arot_src=_arot_src)

        # MDS loss
        loss = svd.mds_loss(
            frames_t,
            cond_image=frame0_cache[v],
            w_power=w_power,
            cond_cache=cond_cache[v],
            vae_checkpoint=False,
        )

        if not torch.isfinite(loss):
            print(f"[{step}] non-finite loss, skip", flush=True)
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(sim.parameters(), grad_clip)
        opt.step()

        if step % log_every == 0:
            with torch.no_grad():
                travel = float((traj[-1] - anchors.canonical).norm(dim=-1).max())
                k_mean = float(sim._orig_mod.stiffness.mean()
                               if hasattr(sim, '_orig_mod') else sim.stiffness.mean())
                d_val  = float(sim._orig_mod.damping
                               if hasattr(sim, '_orig_mod') else sim.damping)
            print(f"[{step}/{iters}] loss={float(loss):.4f}  v={v}"
                  f"  travel={travel:.4f} ({travel/extent*100:.1f}%)"
                  f"  k_mean={k_mean:.3f}  damping={d_val:.3f}", flush=True)

        if step % ckpt_every == 0 or step == iters - 1:
            raw_sim = sim._orig_mod if hasattr(sim, '_orig_mod') else sim
            ckpt_mgr.save(step, {"sim": raw_sim.state_dict(),
                                  "opt": opt.state_dict(), "step": step})
            _save_rollout(step, raw_sim, anchors, T, extent, dev,
                          canon_xyz, canon_cov6, g, bg, cameras[0], pipe, args.out,
                          f_scale=f_scale)

    print(f"[train] done  commit={gh}  -> {args.out}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""T2N reproduction: Cubic Hermite Spline + full loss (L_rgb + L_track + L_depth + L_mask + L_arap).

Follows arXiv:2510.02732 "From Tokens to Nodes":
  - Real N3DV monocular video (cam0) as supervision
  - Cubic Hermite Spline trajectory (K keyframes)
  - TAPIR 2D tracklets -> L_track + spline initialisation
  - DepthCrafter video depth -> L_depth
  - SAM2 foreground mask -> L_mask
  - ARAP temporal regularisation -> L_arap
  - MoDGS protocol: cam0 train, cam5/cam6 test

Usage:
  python exe/train_t2n_spline.py \\
      --model /workspace/gs_flame \\
      --cfg cfg/anchorflow_flame.yaml \\
      --frames /data/datasets/n3v/flame_steak/frames/cam00 \\
      --cameras /workspace/gs_flame/cameras.json \\
      --out /workspace/t2n_spline_flame \\
      --tracks /workspace/t2n_preprocess/flame_steak/tracks.npz \\
      --depths /workspace/t2n_preprocess/flame_steak/depths.npz \\
      --masks  /workspace/t2n_preprocess/flame_steak/masks.npz \\
      --eval_frames /data/datasets/n3v/flame_steak/frames/cam05,/data/datasets/n3v/flame_steak/frames/cam06 \\
      --eval_cam_idxs 5,6 \\
      --r2 r2:storage/result/anchorflow/t2n_spline_flame
"""
from __future__ import annotations

import argparse, json, math, os, random, subprocess, sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn as nn
import imageio.v2 as iio
from torch.utils.checkpoint import checkpoint
from omegaconf import OmegaConf

sys.path.append("/workspace/gaussian-splatting")
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, focal2fov

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
from anchorflow.anchors import AnchorSet
from anchorflow.graph import knn_graph
from anchorflow import warp as W
from anchorflow.checkpoint import CheckpointManager, load_rng_state
from anchorflow.spline import CubicHermiteTrajectory


# ─── Camera ───────────────────────────────────────────────────────────────────

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


def load_camera_by_index(cameras_json_path: str, cam_idx: int, long_side: int) -> Cam:
    cams_json = json.load(open(cameras_json_path))
    c = cams_json[cam_idx]
    if "rotation" in c:
        rot = np.array(c["rotation"], dtype=np.float32)
        pos = np.array(c["position"], dtype=np.float32)
        Wd, Hd = c["width"], c["height"]
        fovx, fovy = focal2fov(c["fx"], Wd), focal2fov(c["fy"], Hd)
        T_cw = -rot.T @ pos
    else:
        rot   = np.array(c["R"], dtype=np.float32)
        T_cw  = np.array(c["T"], dtype=np.float32)
        Wd, Hd = c["W"], c["H"]
        fovx, fovy = c["fov_x"], c["fov_y"]
    s   = long_side / max(Wd, Hd)
    W8  = max(8, int(round(Wd * s / 8)) * 8)
    H8  = max(8, int(round(Hd * s / 8)) * 8)
    return Cam(rot, T_cw, fovx, fovy, W8, H8)


# ─── Frame loading ────────────────────────────────────────────────────────────

def load_frames_cpu(frames_dir: str, T: int, cam: Cam) -> list[torch.Tensor]:
    """Load and resize frames on CPU. Returns list of [3,H,W] float32 in [0,1]."""
    H, W = cam.image_height, cam.image_width
    paths = sorted(Path(frames_dir).glob("*.jpg")) + sorted(Path(frames_dir).glob("*.png"))
    paths = paths[:T]
    result = []
    for p in paths:
        fr = torch.from_numpy(np.asarray(iio.imread(str(p)))).permute(2, 0, 1).float() / 255.
        if fr.shape[-2:] != (H, W):
            fr = nn.functional.interpolate(
                fr.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False
            ).squeeze(0)
        result.append(fr)
    return result


# ─── Helpers ──────────────────────────────────────────────────────────────────

def save_video(frames: list[torch.Tensor], path: str, fps: int = 8):
    arr = [(f.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
           for f in frames]
    iio.mimsave(path, arr, fps=fps, quality=8)


def psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
    mse = ((pred.clamp(0, 1) - gt) ** 2).mean().item()
    return -10 * math.log10(max(mse, 1e-10))


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "nogit"


# ─── Project node positions to 2D ─────────────────────────────────────────────

def project_nodes(pts_world: torch.Tensor, cam: Cam) -> torch.Tensor:
    """Project [M, 3] world-space node positions to [M, 2] pixel coordinates.

    Returns pixel (x, y) using the camera's full projection matrix.
    """
    M = pts_world.shape[0]
    ones = torch.ones(M, 1, device=pts_world.device, dtype=pts_world.dtype)
    pts_h = torch.cat([pts_world, ones], dim=-1)  # [M, 4]
    # full_proj_transform: world -> NDC (4x4)
    ndc = pts_h @ cam.full_proj_transform  # [M, 4]
    w   = ndc[:, 3:4].clamp(min=1e-8)
    xy  = ndc[:, :2] / w  # NDC in [-1, 1]
    # Convert to pixel coordinates
    px = (xy[:, 0] + 1) * 0.5 * cam.image_width
    py = (xy[:, 1] + 1) * 0.5 * cam.image_height
    return torch.stack([px, py], dim=-1)  # [M, 2]


# ─── L_track ──────────────────────────────────────────────────────────────────

def track_loss(node_pos: torch.Tensor, cam: Cam,
               tracks_xy: torch.Tensor, vis: torch.Tensor, t: int,
               sigma_px: float = 4.0) -> torch.Tensor:
    """Reprojection loss between projected node positions and 2D tracklets.

    Assigns each visible tracklet to its nearest projected node,
    then penalises the squared pixel distance.

    Args:
        node_pos: [M, 3] current node world positions
        cam:      camera for this frame
        tracks_xy [N, 2]: 2D tracklet positions at time t (pixel x,y)
        vis [N]:  visibility weight at time t (1=visible, 0=occluded)
        sigma_px: soft-assignment bandwidth (pixels)
    """
    M = node_pos.shape[0]
    proj = project_nodes(node_pos, cam)          # [M, 2]
    trk  = tracks_xy.to(proj.device)            # [N, 2]
    vw   = vis.to(proj.device)                  # [N]
    if vw.sum() < 1:
        return node_pos.sum() * 0.0

    # Nearest-node assignment
    dist2 = ((trk.unsqueeze(1) - proj.unsqueeze(0)) ** 2).sum(-1)  # [N, M]
    nn_idx = dist2.argmin(dim=1)                                     # [N]
    nn_dist2 = dist2[torch.arange(len(nn_idx)), nn_idx]             # [N]
    loss = (vw * nn_dist2).sum() / (vw.sum() + 1e-8)
    return loss / (sigma_px ** 2)


# ─── L_mask ───────────────────────────────────────────────────────────────────

def mask_loss(rendered: torch.Tensor, mask_gt: torch.Tensor) -> torch.Tensor:
    """BCE-style loss between rendered alpha (max over channels) and GT mask.

    rendered: [3, H, W] in [0,1]
    mask_gt:  [H, W] bool
    """
    alpha = rendered.mean(dim=0)  # [H, W] crude luminance proxy for alpha
    gt = mask_gt.float().to(alpha.device)
    return nn.functional.binary_cross_entropy(alpha.clamp(1e-4, 1 - 1e-4), gt)


# ─── L_depth ──────────────────────────────────────────────────────────────────

def depth_loss(rendered_depth: torch.Tensor | None,
               depth_gt: torch.Tensor) -> torch.Tensor | None:
    """Affine-invariant depth loss (scale-shift invariant L1).

    rendered_depth: [H, W] or None
    depth_gt:       [H, W] (DepthCrafter relative depth)
    """
    if rendered_depth is None:
        return None
    rd = rendered_depth.to(depth_gt.device)
    dg = depth_gt
    # Align scale/shift of rendered depth to match gt
    rd_flat = rd.reshape(-1)
    dg_flat = dg.reshape(-1)
    A = torch.stack([rd_flat, torch.ones_like(rd_flat)], dim=-1)  # [N, 2]
    params, _ = torch.linalg.lstsq(A, dg_flat.unsqueeze(-1))[:2]
    rd_aligned = params[0] * rd + params[1]
    return (rd_aligned - dg).abs().mean()


# ─── Rollout save ─────────────────────────────────────────────────────────────

def _save_rollout(step: int, spline: CubicHermiteTrajectory,
                  anchors: AnchorSet,
                  canon_xyz: torch.Tensor, canon_cov6: torch.Tensor,
                  render_fn, cam: Cam, T: int, out: str):
    frames = []
    with torch.no_grad():
        w_b, idx_b = anchors.cal_nn_weight(canon_xyz)
        for t in range(0, T, max(1, T // 30)):
            pt = spline(t)
            pos, cov6, _ = W.lbs_warp(canon_xyz, canon_cov6, w_b, idx_b,
                                       anchors.canonical, pt)
            frames.append(render_fn(cam, pos, cov6).clamp(0, 1))
    save_video(frames, os.path.join(out, f"rollout_{step:06d}.mp4"))


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",    required=True, help="GS model dir (with cameras.json)")
    ap.add_argument("--cfg",      required=True)
    ap.add_argument("--frames",   required=True, help="cam0 frames dir (000000.jpg ...)")
    ap.add_argument("--cameras",  default=None,  help="cameras.json (default: model/cameras.json)")
    ap.add_argument("--cam_idx",  type=int, default=0, help="camera index for train cam")
    ap.add_argument("--out",      required=True)
    ap.add_argument("--iter",     type=int, default=30000, help="GS PLY iteration")
    ap.add_argument("--iters",    type=int, default=None)
    ap.add_argument("--K",        type=int, default=20, help="number of Hermite keyframes")
    ap.add_argument("--res",      type=int, default=None, help="override cfg.model.res")
    # Optional preprocess data
    ap.add_argument("--tracks",   default=None, help="tracks.npz from TAPIR")
    ap.add_argument("--depths",   default=None, help="depths.npz from DepthCrafter")
    ap.add_argument("--masks",    default=None, help="masks.npz from SAM2")
    # Eval
    ap.add_argument("--eval_frames",   default=None,
                    help="comma-sep frame dirs for eval (e.g. .../cam05,.../cam06)")
    ap.add_argument("--eval_cam_idxs", default=None,
                    help="comma-sep camera indices matching eval_frames (e.g. 5,6)")
    # Loss weights (override cfg)
    ap.add_argument("--lambda_rgb",   type=float, default=None)
    ap.add_argument("--lambda_arap",  type=float, default=None)
    ap.add_argument("--lambda_track", type=float, default=50.0)
    ap.add_argument("--lambda_depth", type=float, default=0.1)
    ap.add_argument("--lambda_mask",  type=float, default=0.5)
    # Training
    ap.add_argument("--resume",    action="store_true")
    ap.add_argument("--white_bg",  action="store_true")
    ap.add_argument("--r2",        default=None)
    ap.add_argument("--no_t2n",    action="store_true", help="skip tokens_to_nodes (use FPS)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cfg = OmegaConf.load(args.cfg)
    if args.iters is not None:
        cfg.train.iters = args.iters
    if args.res is not None:
        cfg.model.res = args.res
    dev, gh = "cuda", git_hash()
    T = cfg.model.n_frames
    K = args.K
    long_side = cfg.model.res
    cameras_path = args.cameras or os.path.join(args.model, "cameras.json")

    # λ weights
    λ_rgb   = args.lambda_rgb   if args.lambda_rgb   is not None else float(cfg.train.get("lambda_rgb",  1.0))
    λ_arap  = args.lambda_arap  if args.lambda_arap  is not None else float(cfg.train.get("lambda_arap", 0.001))
    λ_track = args.lambda_track
    λ_depth = args.lambda_depth
    λ_mask  = args.lambda_mask

    print(f"[t2n-spline] T={T} K={K} res={long_side} λ_rgb={λ_rgb} λ_arap={λ_arap} "
          f"λ_track={λ_track} λ_depth={λ_depth} λ_mask={λ_mask} commit={gh}")

    # ── Load Gaussians ─────────────────────────────────────────────────────────
    g = GaussianModel(3)
    g.load_ply(f"{args.model}/point_cloud/iteration_{args.iter}/point_cloud.ply")
    g.active_sh_degree = 3
    canon_xyz  = g.get_xyz.detach().clone()
    canon_cov6 = W.cov_from_scale_rot(g.get_scaling.detach(), g._rotation.detach()).detach()
    G = canon_xyz.shape[0]
    print(f"[t2n-spline] gaussians={G}")
    for p in (g._features_dc, g._features_rest, g._opacity, g._scaling, g._rotation, g._xyz):
        p.requires_grad_(False)

    bg = torch.tensor([1., 1., 1.] if args.white_bg else [0., 0., 0.], device=dev)

    def render_with(cam, xyz, cov6):
        g._xyz = xyz
        g.get_covariance = lambda scaling_modifier=1.0: cov6
        return render(cam, g, Pipe(), bg)["render"]

    # ── Training camera ────────────────────────────────────────────────────────
    train_cam = load_camera_by_index(cameras_path, args.cam_idx, long_side)

    def render_canonical(cam):
        with torch.no_grad():
            return render_with(cam, canon_xyz, canon_cov6).clamp(0, 1)

    # ── Anchor nodes ───────────────────────────────────────────────────────────
    node_pos = None
    if not args.no_t2n:
        try:
            from anchorflow.tokens_to_nodes import tokens_to_nodes
            import anchorflow.tokens_to_nodes as t2n_mod
            print("[t2n-spline] tokens_to_nodes ...")
            node_pos = tokens_to_nodes(
                canon_xyz, g.get_opacity.detach(), render_canonical,
                [train_cam],
                n_nodes=cfg.model.n_nodes, device=dev)
            if t2n_mod._dino_model is not None:
                del t2n_mod._dino_model; t2n_mod._dino_model = None
            import gc; gc.collect(); torch.cuda.empty_cache()
        except Exception as e:
            print(f"[t2n-spline] tokens_to_nodes failed ({e}) -> FPS")
    z_dim = int(cfg.model.get("z_dim", 8))
    e_dim = int(cfg.model.get("e_dim", 8))
    kG    = int(cfg.model.k_gauss)
    if node_pos is not None:
        anchors = AnchorSet.from_trajectory(node_pos, latent_dim=z_dim,
                                            e_dim=e_dim, K=kG).to(dev)
    else:
        anchors, _ = AnchorSet.from_gaussians(canon_xyz, node_num=cfg.model.n_nodes,
                                              latent_dim=z_dim, e_dim=e_dim, K=kG)
        anchors = anchors.to(dev)
    M = anchors.num
    print(f"[t2n-spline] anchors={M}")

    # ── Spline trajectory ──────────────────────────────────────────────────────
    spline = CubicHermiteTrajectory(anchors.canonical.detach(), K=K, T=T).to(dev)
    print(f"[t2n-spline] spline K={K} params={sum(p.numel() for p in spline.parameters()):,}")

    # ── Load preprocess data ───────────────────────────────────────────────────
    tracks_xy_all, tracks_vis_all = None, None
    if args.tracks and os.path.exists(args.tracks):
        d = np.load(args.tracks)
        tracks_xy_all  = torch.from_numpy(d["points"]).float()     # [T, N, 2]
        tracks_vis_all = torch.from_numpy(d["visibility"]).float() # [T, N]
        print(f"[t2n-spline] tracks: {tuple(tracks_xy_all.shape)}")
        # Initialise spline from 3D tracklets if possible
        # (Requires depth data for back-projection; skip here)

    depths_all = None
    if args.depths and os.path.exists(args.depths):
        d = np.load(args.depths)
        depths_all = torch.from_numpy(d["depth"]).float()  # [T, H, W]
        depths_H, depths_W = depths_all.shape[1:]
        print(f"[t2n-spline] depths: {tuple(depths_all.shape)}")

    masks_all = None
    if args.masks and os.path.exists(args.masks):
        d = np.load(args.masks)
        masks_all = torch.from_numpy(d["mask"].astype(np.uint8)).bool()  # [T, H, W]
        print(f"[t2n-spline] masks: {tuple(masks_all.shape)}")

    # ── Load training frames on CPU ────────────────────────────────────────────
    print(f"[t2n-spline] loading {T} frames from {args.frames} ...")
    gt_frames_cpu = load_frames_cpu(args.frames, T, train_cam)
    print(f"[t2n-spline] frames loaded: {len(gt_frames_cpu)} x {tuple(gt_frames_cpu[0].shape)}")

    # ── Optimiser ─────────────────────────────────────────────────────────────
    lr_anchor = float(cfg.train.get("lr_anchor", 1e-3))
    lr_spline = float(cfg.train.get("lr_z0",     5e-4))
    opt = torch.optim.Adam([
        {"params": list(anchors.parameters()), "lr": lr_anchor},
        {"params": list(spline.parameters()),  "lr": lr_spline},
    ])
    arap_edge = knn_graph(anchors.canonical.detach(), k=min(6, M - 1))

    # ── Checkpoint ────────────────────────────────────────────────────────────
    ckpt_mgr = CheckpointManager(args.out)
    start = 0
    if args.resume:
        ck = ckpt_mgr.load()
        if ck is not None:
            anchors.load_state_dict(ck["anchors"])
            spline.load_state_dict(ck["spline"])
            opt.load_state_dict(ck["opt"])
            load_rng_state(ck.get("rng"))
            start = ck["step"] + 1
            print(f"[t2n-spline] resumed from step {start}")

    def sync_r2():
        if args.r2:
            os.system(f"rclone copy {args.out} {args.r2} --progress >/dev/null 2>&1")

    _q = canon_xyz.float()
    extent = float((torch.quantile(_q, 0.99, dim=0) - torch.quantile(_q, 0.01, dim=0)).norm())
    del _q
    rng = random.Random(42)
    torch.set_float32_matmul_precision("high")

    # ── Training loop ─────────────────────────────────────────────────────────
    for step in range(start, cfg.train.iters):
        t = rng.randint(0, T - 1)

        w_b, idx_b = anchors.cal_nn_weight(canon_xyz)
        pt = spline(t)  # [M, 3]

        def _f(pt_):
            pos, cov6, _ = W.lbs_warp(canon_xyz, canon_cov6, w_b, idx_b,
                                       anchors.canonical, pt_)
            return render_with(train_cam, pos, cov6)

        rendered = checkpoint(_f, pt, use_reentrant=False)

        gt_t = gt_frames_cpu[t].to(dev, non_blocking=True)
        loss = λ_rgb * (rendered - gt_t).abs().mean()

        # L_arap: penalise edge-length change vs canonical
        if λ_arap > 0:
            src, dst = arap_edge
            d_canon = (anchors.canonical[src] - anchors.canonical[dst]).norm(dim=-1).detach()
            d_now   = (pt[src] - pt[dst]).norm(dim=-1)
            loss = loss + λ_arap * ((d_now - d_canon) ** 2).mean()

        # L_track: reprojection vs TAPIR tracklets
        if λ_track > 0 and tracks_xy_all is not None:
            trk_xy = tracks_xy_all[t]   # [N, 2]
            trk_vis = tracks_vis_all[t] # [N]
            with torch.no_grad():
                w_b2, idx_b2 = anchors.cal_nn_weight(canon_xyz)
                pos_t, _, _ = W.lbs_warp(canon_xyz, canon_cov6, w_b2, idx_b2,
                                          anchors.canonical, pt.detach())
            loss = loss + λ_track * track_loss(pt, train_cam, trk_xy, trk_vis, t)

        # L_mask: rendered vs SAM2 foreground mask
        if λ_mask > 0 and masks_all is not None:
            mk = masks_all[t]
            if mk.shape != (train_cam.image_height, train_cam.image_width):
                mk = nn.functional.interpolate(
                    mk.float().unsqueeze(0).unsqueeze(0),
                    size=(train_cam.image_height, train_cam.image_width),
                    mode="nearest").squeeze().bool()
            loss = loss + λ_mask * mask_loss(rendered.detach(), mk)

        opt.zero_grad(set_to_none=True)
        if not torch.isfinite(loss):
            print(f"[{step}] non-finite loss, skip"); continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for g2 in opt.param_groups for p in g2["params"]],
            cfg.train.grad_clip)
        opt.step()

        if step % cfg.train.log_every == 0:
            with torch.no_grad():
                travel = float((spline(T - 1) - spline(0)).norm(dim=-1).max())
            print(f"[{step}/{cfg.train.iters}] loss={float(loss):.4f} t={t} "
                  f"travel={travel:.3f} ({travel/extent*100:.2f}%)")

        if step % cfg.train.ckpt_every == 0:
            ckpt_mgr.save(step, {"anchors": anchors.state_dict(),
                                  "spline": spline.state_dict(),
                                  "opt": opt.state_dict(), "step": step})
            _save_rollout(step, spline, anchors, canon_xyz, canon_cov6,
                          render_with, train_cam, T, args.out)
            sync_r2()

    ckpt_mgr.save(cfg.train.iters - 1,
                  {"anchors": anchors.state_dict(), "spline": spline.state_dict(),
                   "opt": opt.state_dict(), "step": cfg.train.iters - 1})
    sync_r2()
    print(f"[t2n-spline] training done. commit={gh}")

    # ── Eval on hold-out cameras ───────────────────────────────────────────────
    if args.eval_frames and args.eval_cam_idxs:
        eval_frame_dirs = args.eval_frames.split(",")
        eval_cam_idxs   = [int(x) for x in args.eval_cam_idxs.split(",")]
        assert len(eval_frame_dirs) == len(eval_cam_idxs)

        eval_results = {}
        for frames_dir, ci in zip(eval_frame_dirs, eval_cam_idxs):
            cam_e = load_camera_by_index(cameras_path, ci, long_side)
            gt_frames_e = load_frames_cpu(frames_dir, T, cam_e)
            psnrs = []
            with torch.no_grad():
                w_b, idx_b = anchors.cal_nn_weight(canon_xyz)
                for tt in range(T):
                    pt = spline(tt)
                    pos, cov6, _ = W.lbs_warp(canon_xyz, canon_cov6, w_b, idx_b,
                                               anchors.canonical, pt)
                    rendered = render_with(cam_e, pos, cov6).clamp(0, 1)
                    gt_t = gt_frames_e[tt].to(dev)
                    psnrs.append(psnr(rendered, gt_t))
            mean_psnr = np.mean(psnrs)
            eval_results[f"cam{ci:02d}"] = mean_psnr
            print(f"[eval] cam{ci:02d} mean PSNR = {mean_psnr:.2f} dB")

        with open(os.path.join(args.out, "eval_psnr.json"), "w") as f:
            json.dump(eval_results, f, indent=2)
        sync_r2()
        print("[eval] done →", eval_results)


if __name__ == "__main__":
    main()

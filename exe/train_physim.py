#!/usr/bin/env python
"""GNN physics simulator training with SVD MDS supervision.

Pipeline:
  1. SAM2 multi-view segmentation → per-anchor object IDs
  2. Build hierarchical graph (intra + inter object edges)
  3. Train GNNSim: impulse at t=0 + gravity → T-frame trajectory → MDS loss

Usage:
    python exe/train_physim.py \\
        --model /workspace/scgs_ficus_node \\
        --ply_iter 60000 \\
        --out /workspace/physim_ficus \\
        --cfg cfg/physim_ficus.yaml --resume
"""
from __future__ import annotations

import argparse, json, math, os, subprocess, sys

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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
from anchorflow.anchors import AnchorSet
from anchorflow import warp as W
from anchorflow.warp import anchor_rotations_cache
from anchorflow.physim import GNNSim
from anchorflow.graph import build_hierarchical_graph
from anchorflow.segment import segment_gaussians, assign_anchor_objects
from anchorflow.sds import SVDGuidance
from anchorflow.checkpoint import CheckpointManager


# ── camera helpers ────────────────────────────────────────────────────────── #

class Cam:
    def __init__(self, R, T, fovx, fovy, Wd, Hd):
        self.image_width, self.image_height = Wd, Hd
        self.FoVx, self.FoVy = fovx, fovy
        self.znear, self.zfar = 0.01, 100.0
        w2v  = torch.tensor(getWorld2View2(R, T)).T.cuda()
        proj = getProjectionMatrix(self.znear, self.zfar, fovx, fovy).T.cuda()
        self.world_view_transform = w2v
        self.full_proj_transform  = (w2v.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0)
        self.camera_center        = w2v.inverse()[3, :3]


class Pipe:
    convert_SHs_python = False
    compute_cov3D_python = True
    debug = False
    antialiasing = False


def _normalize(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-8)


def make_camera(pos, target=(0,0,0), up=(0,0,1), fov_deg=50, W=256, H=256) -> Cam:
    """Build a look-at camera with Z-up convention."""
    pos    = np.array(pos,    dtype=np.float32)
    target = np.array(target, dtype=np.float32)
    up     = np.array(up,     dtype=np.float32)
    fwd   = _normalize(target - pos)
    right = _normalize(np.cross(fwd, up))
    up2   = np.cross(right, fwd)
    rot   = np.stack([right, -up2, fwd], axis=1)   # C2W rotation
    T_vec = -(rot.T @ pos)
    fov   = math.radians(fov_deg)
    return Cam(rot, T_vec, fov, fov, W, H)


def load_cameras_json(model_dir: str, n_views: int, res: int) -> list:
    """Load cameras from SC-GS cameras.json (original training cameras)."""
    cams_json = json.load(open(f"{model_dir}/cameras.json"))
    idx = np.linspace(0, len(cams_json) - 1, n_views).round().astype(int)
    cams = []
    for i in idx:
        c    = cams_json[int(i)]
        rot  = np.array(c["rotation"], dtype=np.float32)
        pos  = np.array(c["position"], dtype=np.float32)
        Wd, Hd = c["width"], c["height"]
        fovx = focal2fov(c["fx"], Wd)
        fovy = focal2fov(c["fy"], Hd)
        T_vec = -rot.T @ pos
        s  = res / max(Wd, Hd)
        W8 = max(8, int(round(Wd * s / 8)) * 8)
        H8 = max(8, int(round(Hd * s / 8)) * 8)
        cams.append(Cam(rot, T_vec, fovx, fovy, W8, H8))
    return cams


def zup_cameras(n_views: int, radius: float, z: float,
                target=(0, 0, 0), fov_deg=50, res=256) -> list:
    """Evenly-spaced cameras in the XY plane looking at target, up=Z."""
    cams = []
    for i in range(n_views):
        theta = 2 * math.pi * i / n_views
        pos   = (radius * math.cos(theta), radius * math.sin(theta), z)
        cams.append(make_camera(pos, target=target, up=(0, 0, 1),
                                fov_deg=fov_deg, W=res, H=res))
    return cams


# ── rendering ─────────────────────────────────────────────────────────────── #

def render_gs(cam, g, pipe, bg) -> torch.Tensor:
    zeros = torch.zeros_like(g.get_xyz)
    return _render_scgs(cam, g, pipe, bg,
                        d_xyz=zeros, d_rotation=0.0, d_scaling=zeros)["render"]


def traj_to_frames(traj, canon_xyz, canon_cov6, anchors, g, bg, cam, pipe,
                   use_checkpoint=True, _w_b=None, _idx_b=None,
                   _arot_idx=None, _arot_src=None):
    frames = []
    for t in range(traj.shape[0]):
        pt = traj[t]
        with torch.no_grad():
            aR = W.anchor_rotations(anchors.canonical, pt,
                                    _idx=_arot_idx, _src=_arot_src)

        def _frame(pt, _R=aR):
            pos, cov6, _ = W.lbs_warp(canon_xyz, canon_cov6, _w_b, _idx_b,
                                       anchors.canonical, pt, anchor_R=_R)
            g._xyz = pos
            g.get_covariance = lambda sc=1.0, **kw: cov6
            return render_gs(cam, g, pipe, bg)

        if use_checkpoint and pt.requires_grad:
            frames.append(checkpoint(_frame, pt, use_reentrant=False))
        else:
            frames.append(_frame(pt))
    return torch.stack(frames, dim=0)


def save_video(frames, path, fps=8):
    arr = [(f.clamp(0,1).permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
           for f in frames]
    iio.mimsave(path, arr, fps=fps, quality=8)


def git_hash():
    try:
        return subprocess.check_output(["git","rev-parse","--short","HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "nogit"


def sample_impulse(extent: float, f_scale: float, device: str) -> torch.Tensor:
    """Random horizontal direction × random magnitude."""
    d = torch.randn(3, device=device)
    d[2] *= 0.2          # mostly horizontal (less vertical impulse)
    d = F.normalize(d, dim=0)
    mag = torch.rand(1, device=device).item() * f_scale * extent
    return d * mag


@torch.no_grad()
def _save_rollout(step, sim, anchors, T, extent, dev,
                  canon_xyz, canon_cov6, g, bg, rollout_cams, pipe, out,
                  f_scale=0.3):
    sim.eval()
    _w_b, _idx_b = anchors.cal_nn_weight(canon_xyz)
    _arot_idx, _arot_src = anchor_rotations_cache(anchors.canonical)
    all_frames = []
    # 4 cardinal impulse directions × first rollout camera
    for axis, sign in [(0, 1), (1, 1), (0, -1), (1, -1)]:
        d = torch.zeros(3, device=dev)
        d[axis] = sign
        f_ext = d * f_scale * extent
        traj  = sim(f_ext)
        frames = traj_to_frames(traj, canon_xyz, canon_cov6, anchors,
                                  g, bg, rollout_cams[0], pipe,
                                  use_checkpoint=False,
                                  _w_b=_w_b, _idx_b=_idx_b,
                                  _arot_idx=_arot_idx, _arot_src=_arot_src)
        all_frames.extend(list(frames))
    path = os.path.join(out, f"rollout_{step:06d}.mp4")
    save_video(all_frames, path)
    print(f"  [rollout] {path}", flush=True)
    sim.train()


# ── main ──────────────────────────────────────────────────────────────────── #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",    required=True)
    ap.add_argument("--ply_iter", type=int, default=60000)
    ap.add_argument("--out",      required=True)
    ap.add_argument("--cfg",      required=True)
    ap.add_argument("--resume",   action="store_true")
    ap.add_argument("--n_nodes",  type=int, default=512)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cfg = OmegaConf.load(args.cfg)
    dev = "cuda"
    gh  = git_hash()

    # ── 3DGS canonical ────────────────────────────────────────────────────── #
    _cfg_path  = os.path.join(args.model, "cfg_args")
    _hyper_dim = 0
    if os.path.exists(_cfg_path):
        _ns = eval(open(_cfg_path).read().strip(), {"Namespace": lambda **kw: kw})
        _hyper_dim = _ns.get("hyper_dim", 0) if isinstance(_ns, dict) else 0
    g = GaussianModel(3, fea_dim=_hyper_dim)
    ply = os.path.join(args.model, "point_cloud",
                       f"iteration_{args.ply_iter}", "point_cloud.ply")
    g.load_ply(ply)
    g.active_sh_degree = 3
    canon_xyz  = g.get_xyz.detach().clone()
    canon_cov6 = W.cov_from_scale_rot(
        g.get_scaling.detach(), g._rotation.detach()).detach()
    bg   = torch.tensor([0., 0., 0.], device=dev)
    pipe = Pipe()
    print(f"[train] gaussians={len(canon_xyz)}", flush=True)

    # SH0 albedo colour for segmentation
    gauss_colors = g.get_features[:, 0, :].detach()          # [N, 3] SH0 DC
    gauss_colors = (gauss_colors * 0.2820948 + 0.5).clamp(0, 1)  # approx RGB

    # ── anchors ───────────────────────────────────────────────────────────── #
    anchors, _ = AnchorSet.from_gaussians(canon_xyz, node_num=args.n_nodes)
    anchors     = anchors.to(dev)
    extent      = float((anchors.canonical.max(0).values -
                          anchors.canonical.min(0).values).norm())
    print(f"[train] anchors={anchors.canonical.shape[0]}  extent={extent:.4f}", flush=True)

    # ── training cameras (original SC-GS views) ───────────────────────────── #
    n_views = int(cfg.train.n_views)
    res     = int(cfg.train.res)
    train_cams = load_cameras_json(args.model, n_views, res)
    print(f"[train] train cameras={len(train_cams)}  "
          f"{train_cams[0].image_width}x{train_cams[0].image_height}", flush=True)

    # Z-up rollout cameras for visualisation
    z_center = float(anchors.canonical[:, 2].mean())
    rollout_cams = zup_cameras(8, radius=2.0, z=z_center + 0.3,
                               target=(0, 0, z_center), fov_deg=50, res=res)

    # ── segmentation ──────────────────────────────────────────────────────── #
    seg_path = os.path.join(args.out, "anchor_obj.pt")
    if os.path.exists(seg_path):
        anchor_obj = torch.load(seg_path, map_location=dev)
        print(f"[train] loaded segmentation from {seg_path}", flush=True)
    else:
        print("[train] running segmentation ...", flush=True)
        # Render canonical views for SAM2
        seg_renders = []
        with torch.no_grad():
            for cam in train_cams:
                seg_renders.append(render_gs(cam, g, pipe, bg))

        n_obj = int(cfg.sim.get("n_objects", 4))
        gauss_obj = segment_gaussians(
            gaussian_xyz=canon_xyz,
            cameras=train_cams,
            renders=seg_renders,
            n_objects=n_obj,
            gaussian_colors=gauss_colors,
            sam2_checkpoint=cfg.sim.get("sam2_checkpoint", None),
            sam2_cfg=cfg.sim.get("sam2_cfg", None),
            device=dev,
        )
        _w_b_seg, _idx_b_seg = anchors.cal_nn_weight(canon_xyz)
        anchor_obj = assign_anchor_objects(
            gauss_obj, _idx_b_seg, _w_b_seg,
            n_anchors=anchors.canonical.shape[0], n_objects=n_obj,
        ).to(dev)
        torch.save(anchor_obj, seg_path)
        print(f"[train] segmentation saved → {seg_path}", flush=True)
        for oi in range(int(anchor_obj.max()) + 1):
            print(f"  object {oi}: {int((anchor_obj == oi).sum())} anchors", flush=True)

    # ── hierarchical graph ────────────────────────────────────────────────── #
    graph_path = os.path.join(args.out, "graph.pt")
    if os.path.exists(graph_path):
        gd = torch.load(graph_path, map_location=dev)
        intra_ei, intra_r = gd["intra_ei"], gd["intra_r"]
        inter_ei, inter_r = gd["inter_ei"], gd["inter_r"]
        print(f"[train] loaded graph from {graph_path}", flush=True)
    else:
        intra_ei, intra_r, inter_ei, inter_r = build_hierarchical_graph(
            anchors.canonical,
            anchor_obj,
            k_intra=int(cfg.sim.get("k_intra", 12)),
            k_inter=int(cfg.sim.get("k_inter", 4)),
        )
        torch.save({"intra_ei": intra_ei, "intra_r": intra_r,
                    "inter_ei": inter_ei, "inter_r": inter_r}, graph_path)

    # anchor SH0 colour (for node features in GNN)
    _w_b, _idx_b = anchors.cal_nn_weight(canon_xyz)
    anchor_colors_sum = torch.zeros(anchors.canonical.shape[0], 3, device=dev)
    anchor_colors_cnt = torch.zeros(anchors.canonical.shape[0], device=dev)
    for k in range(_idx_b.shape[1]):
        anchor_colors_sum.scatter_add_(
            0, _idx_b[:, k:k+1].expand(-1, 3), gauss_colors.to(dev))
        anchor_colors_cnt.scatter_add_(
            0, _idx_b[:, k], torch.ones(len(canon_xyz), device=dev))
    anchor_colors = anchor_colors_sum / anchor_colors_cnt.unsqueeze(1).clamp(min=1)

    # ── GNNSim ────────────────────────────────────────────────────────────── #
    T   = int(cfg.sim.T)
    sim = GNNSim(
        canonical        = anchors.canonical,
        anchor_obj       = anchor_obj,
        anchor_colors    = anchor_colors,
        intra_edge_index = intra_ei,
        intra_rest       = intra_r,
        inter_edge_index = inter_ei,
        inter_rest       = inter_r,
        T                = T,
        dt               = float(cfg.sim.dt),
        hidden_dim       = int(cfg.sim.get("hidden_dim", 128)),
        gravity          = float(cfg.sim.get("gravity", 2.0)),
        gravity_axis     = int(cfg.sim.get("gravity_axis", 2)),
    ).to(dev)
    n_params = sum(p.numel() for p in sim.parameters())
    print(f"[train] GNNSim params={n_params:,}  T={T}", flush=True)

    opt = torch.optim.Adam(sim.parameters(), lr=float(cfg.train.lr))

    # ── SVD guidance ──────────────────────────────────────────────────────── #
    svd_model_id = cfg.get("svd_model", "stabilityai/stable-video-diffusion-img2vid-xt")
    svd = SVDGuidance(model_id=svd_model_id, device=dev)

    print("[train] precomputing MDS conditioning ...", flush=True)
    cond_cache   = []
    frame0_cache = []
    for cam in train_cams:
        with torch.no_grad():
            f0 = render_gs(cam, g, pipe, bg)
        frame0_cache.append(f0)
        cond_cache.append(svd.precompute_cond(f0, T))
    print("[train] cache ready", flush=True)

    # ── checkpoint ────────────────────────────────────────────────────────── #
    ckpt_mgr   = CheckpointManager(args.out, keep_last=3)
    start_step = 0
    if args.resume:
        ck = ckpt_mgr.load()
        if ck is not None:
            sim.load_state_dict(ck["sim"])
            opt.load_state_dict(ck["opt"])
            start_step = ck.get("step", 0) + 1
            print(f"[train] resumed from step {start_step - 1}", flush=True)

    sim = torch.compile(sim)

    # ── precompute LBS constants ───────────────────────────────────────────── #
    with torch.no_grad():
        _w_b, _idx_b = anchors.cal_nn_weight(canon_xyz)
        _arot_idx, _arot_src = anchor_rotations_cache(anchors.canonical)

    f_scale    = float(cfg.train.f_scale)
    grad_clip  = float(cfg.train.grad_clip)
    log_every  = int(cfg.train.log_every)
    ckpt_every = int(cfg.train.ckpt_every)
    iters      = int(cfg.train.iters)
    V          = len(train_cams)

    print(f"[train] start  commit={gh}  steps={iters}", flush=True)

    for step in range(start_step, iters):
        sim.train()
        opt.zero_grad()

        v_idx = step % V
        cam   = train_cams[v_idx]

        f_ext = sample_impulse(extent, f_scale, dev)
        traj  = sim(f_ext)                                        # [T, M, 3]

        frames_t = traj_to_frames(traj, canon_xyz, canon_cov6, anchors,
                                   g, bg, cam, pipe, use_checkpoint=True,
                                   _w_b=_w_b, _idx_b=_idx_b,
                                   _arot_idx=_arot_idx, _arot_src=_arot_src)

        loss = svd.mds_loss(
            frames_t,
            cond_image  = frame0_cache[v_idx],
            cond_cache  = cond_cache[v_idx],
            vae_checkpoint = False,
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
            print(f"[{step}/{iters}] loss={float(loss):.4f}  v={v_idx}"
                  f"  travel={travel:.4f} ({travel/extent*100:.1f}%)", flush=True)

        if step % ckpt_every == 0 or step == iters - 1:
            raw = sim._orig_mod if hasattr(sim, "_orig_mod") else sim
            ckpt_mgr.save(step, {"sim": raw.state_dict(),
                                  "opt": opt.state_dict(), "step": step})
            _save_rollout(step, raw, anchors, T, extent, dev,
                          canon_xyz, canon_cov6, g, bg, rollout_cams, pipe,
                          args.out, f_scale=f_scale)

    print(f"[train] done  commit={gh}", flush=True)


if __name__ == "__main__":
    main()

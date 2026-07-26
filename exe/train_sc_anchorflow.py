#!/usr/bin/env python
"""
SC-GS style training with GNN+SSM+integration (AnchorFlow) deformation.

Drop-in replacement for SC-GS's MLP DeformModel:
  - Canonical 3DGS: random init → optimized with SC-GS densification
  - Deformation: SSMDynamics (GNN + diagonal SSM + explicit integration)
  - Loss: photometric (L1 + SSIM) on ALL T frames per iteration (BPTT)
  - Data: NeRF-Blender format (ficus_ds_wind / transforms_train.json)

Usage:
  python exe/train_sc_anchorflow.py \\
      --data /workspace/ficus_ds_wind \\
      --out  /workspace/ficus_scgs_af \\
      --iters 60000
"""
from __future__ import annotations

import sys, os, math, json, random, argparse
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import trange

# ── SC-GS ────────────────────────────────────────────────────────────────────
sys.path.insert(0, "/workspace/SC-GS")
from scene.gaussian_model import GaussianModel
from scene.dataset_readers import BasicPointCloud
from gaussian_renderer import render as _gs_render
from utils.sh_utils import SH2RGB
from utils.loss_utils import l1_loss
from arguments import OptimizationParams
from argparse import ArgumentParser as _AP
from utils.graphics_utils import getWorld2View2, getProjectionMatrix

# ── AnchorFlow ───────────────────────────────────────────────────────────────
_lib = os.path.join(os.path.dirname(__file__), "..", "lib")
sys.path.insert(0, _lib)
from anchorflow.anchors import AnchorSet
from anchorflow.ssm_dynamics import SSMDynamics, ssm_rollout
from anchorflow import warp as W


# ── Camera wrapper ───────────────────────────────────────────────────────────
class Cam:
    gt_alpha_mask = None
    flow_dirs     = []

    def __init__(self, R, T, fovx, fovy, W, H):
        self.image_width  = W
        self.image_height = H
        self.FoVx, self.FoVy = fovx, fovy
        self.znear, self.zfar = 0.01, 100.0
        self.tanfovx = math.tan(fovx * 0.5)
        self.tanfovy = math.tan(fovy * 0.5)
        w2v  = torch.tensor(getWorld2View2(R, T, np.zeros(3), 1.0),
                             dtype=torch.float32, device="cuda").T
        proj = torch.tensor(getProjectionMatrix(0.01, 100.0, fovx, fovy),
                            dtype=torch.float32, device="cuda").T
        self.world_view_transform = w2v
        self.full_proj_transform  = (w2v.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0)
        self.camera_center = w2v.inverse()[3, :3]
        self.fid = None   # set per-frame during training


class ICNet(torch.nn.Module):
    """Infers per-anchor response magnitude from direction + anchor identity.

    User controls ic_dir (direction) at inference; ICNet infers ic_mag (magnitude)
    from ic_dir and the anchor's learned identity embedding e.
    """
    def __init__(self, e_dim=8, hidden=64):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(3 + e_dim, hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden, 1),
        )

    def forward(self, ic_dir, e):
        x = torch.cat([F.normalize(ic_dir, dim=-1), e], dim=-1)
        return F.softplus(self.net(x).squeeze(-1))   # [M], positive


class Pipe:
    convert_SHs_python  = False
    compute_cov3D_python = False
    debug               = False
    antialiasing        = False


# ── Data loading: NeRF-Blender format ────────────────────────────────────────
def load_blender_dataset(data_dir, res):
    """Load cameras + GT frames from transforms_train.json.

    Applies the SC-GS convention (same transform as readCamerasFromTransforms):
        matrix = inv(c2w)
        R = -matrix[:3,:3].T  with col0 negated
        T = -matrix[:3, 3]
    Returns:
        cams   [V] Cam
        frames [V][F] Tensor float32 [3,H,W] cuda (white bg composite)
    """
    from collections import defaultdict
    meta  = json.load(open(os.path.join(data_dir, "transforms_train.json")))
    fovx  = float(meta["camera_angle_x"])

    by_view = defaultdict(list)
    for f in meta["frames"]:
        view = f["file_path"].split("/")[-2]
        by_view[view].append(f)
    views = sorted(by_view.keys())

    cams, frames = [], []
    for view in views:
        sorted_f = sorted(by_view[view], key=lambda x: x["time"])
        c2w = np.array(sorted_f[0]["transform_matrix"], dtype=np.float32)
        matrix = np.linalg.inv(c2w)
        R = -matrix[:3, :3].T;  R[:, 0] = -R[:, 0]
        T = -matrix[:3, 3]

        img0 = Image.open(os.path.join(data_dir, sorted_f[0]["file_path"] + ".png"))
        Wd, Hd = img0.size
        fovy = 2 * math.atan(math.tan(fovx / 2) * Hd / Wd)
        s = res / max(Wd, Hd)
        Ws = max(8, int(round(Wd * s / 8)) * 8)
        Hs = max(8, int(round(Hd * s / 8)) * 8)
        cams.append(Cam(R, T, fovx, fovy, Ws, Hs))

        view_frames = []
        for fr in sorted_f:
            rgba = Image.open(os.path.join(data_dir, fr["file_path"] + ".png")).convert("RGBA")
            rgba = rgba.resize((Ws, Hs), Image.LANCZOS)
            arr  = np.array(rgba, dtype=np.float32) / 255.0
            rgb  = arr[:, :, :3] * arr[:, :, 3:4] + (1 - arr[:, :, 3:4])   # white bg
            t    = torch.from_numpy(rgb).permute(2, 0, 1).cuda()
            view_frames.append(t)
        frames.append(view_frames)

    V = len(views)
    F = len(frames[0])
    print(f"[data] {V} views × {F} frames  {Ws}×{Hs}  fovx={fovx:.3f}")
    return cams, frames


# ── GaussianModel random init (mirrors SC-GS Blender init) ───────────────────
def init_gaussians_random(n_pts=100_000, extent=1.3):
    xyz  = (np.random.random((n_pts, 3)) * 2 - 1) * extent
    shs  = np.random.random((n_pts, 3)) / 255.0
    pcd  = BasicPointCloud(
        points=xyz, colors=SH2RGB(shs), normals=np.zeros((n_pts, 3)))
    return pcd


# ── LBS warp: anchor displacement → per-Gaussian d_xyz ───────────────────────
def lbs_d_xyz(gauss_xyz, w, idx, anchor_canon, anchor_now):
    """Simple translation-only LBS (no rotation warp — faster, sufficient for init)."""
    d_anchor = anchor_now - anchor_canon             # [K, 3]
    d_per    = d_anchor[idx]                         # [N, K, 3]
    return (w[..., None] * d_per).sum(1)             # [N, 3]  weighted sum


# ── SSIM (simple window=11) ───────────────────────────────────────────────────
def _gaussian_kernel(window_size=11, sigma=1.5):
    x = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-x**2 / (2 * sigma**2))
    g /= g.sum()
    return g.outer(g)

_SSIM_WINDOW = None

def ssim_loss(img1, img2, window_size=11):
    global _SSIM_WINDOW
    if _SSIM_WINDOW is None or _SSIM_WINDOW.device != img1.device:
        k = _gaussian_kernel(window_size)
        _SSIM_WINDOW = k.expand(3, 1, window_size, window_size).to(img1.device)
    C1, C2 = 0.01**2, 0.03**2
    pad = window_size // 2
    mu1 = F.conv2d(img1.unsqueeze(0), _SSIM_WINDOW, padding=pad, groups=3)
    mu2 = F.conv2d(img2.unsqueeze(0), _SSIM_WINDOW, padding=pad, groups=3)
    mu1_sq, mu2_sq = mu1**2, mu2**2
    sigma1 = F.conv2d((img1*img1).unsqueeze(0), _SSIM_WINDOW, padding=pad, groups=3) - mu1_sq
    sigma2 = F.conv2d((img2*img2).unsqueeze(0), _SSIM_WINDOW, padding=pad, groups=3) - mu2_sq
    sigma12 = F.conv2d((img1*img2).unsqueeze(0), _SSIM_WINDOW, padding=pad, groups=3) - mu1*mu2
    ssim_map = ((2*mu1*mu2 + C1)*(2*sigma12 + C2)) / ((mu1_sq+mu2_sq+C1)*(sigma1+sigma2+C2))
    return 1 - ssim_map.mean()


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",  required=True,  help="NeRF-Blender dataset dir")
    ap.add_argument("--out",   required=True,  help="output dir")
    ap.add_argument("--iters", type=int,  default=60_000)
    ap.add_argument("--res",   type=int,  default=576)
    ap.add_argument("--n_anchors", type=int, default=256)
    ap.add_argument("--k_gauss",   type=int, default=4,   help="LBS K neighbors")
    ap.add_argument("--hidden",    type=int, default=128)
    ap.add_argument("--mp_steps",  type=int, default=6)
    ap.add_argument("--ssm_dim",   type=int, default=128)
    ap.add_argument("--e_dim",     type=int, default=8)
    ap.add_argument("--z_dim",     type=int, default=8)
    ap.add_argument("--lr_dyn",    type=float, default=3e-4,  help="SSM+AnchorSet lr")
    ap.add_argument("--dt",        type=float, default=0.05)
    ap.add_argument("--damping",   type=float, default=0.98)
    ap.add_argument("--accel_scale", type=float, default=0.01)
    ap.add_argument("--lambda_ssim",  type=float, default=0.2)
    ap.add_argument("--bptt_start", type=int, default=0,
                    help="detach rollout before this frame (0=full BPTT)")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    dev = "cuda"
    bg  = torch.tensor([1., 1., 1.], device=dev)

    # ── load data ─────────────────────────────────────────────────────────────
    cams, frames = load_blender_dataset(args.data, args.res)
    V = len(cams);  T = len(frames[0])
    print(f"[train] V={V} T={T}")

    # ── canonical Gaussians ───────────────────────────────────────────────────
    gaussians = GaussianModel(3)   # sh_degree=3, fea_dim=0
    pcd = init_gaussians_random(100_000, extent=1.3)
    gaussians.create_from_pcd(pcd, spatial_lr_scale=1.0)

    # SC-GS optimizer params
    _ap2 = _AP()
    opt_params = OptimizationParams(_ap2).extract(_ap2.parse_args([]))
    gaussians.training_setup(opt_params)
    print(f"[train] gaussians={gaussians.get_xyz.shape[0]} (random init)")

    # ── AnchorSet ─────────────────────────────────────────────────────────────
    anchors, _  = AnchorSet.from_gaussians(
        gaussians.get_xyz.detach(), node_num=args.n_anchors,
        latent_dim=args.z_dim, e_dim=args.e_dim, K=args.k_gauss)
    anchors = anchors.to(dev)
    M = anchors.num
    print(f"[train] anchors={M}")

    # ic_dir [M,3]: user-controllable initial condition (direction of initial acceleration)
    #   - learned during training to fit this video's actual IC
    #   - replaced by user input at inference (for some/all anchors)
    ic_dir = torch.nn.Parameter(torch.zeros(M, 3, device=dev))

    # ICNet: infers ic_mag from (ic_dir, anchor identity e) — NOT user-controlled
    #   - optimization target that learns "how strongly does each anchor respond to a given direction"
    ic_net = ICNet(e_dim=args.e_dim, hidden=64).to(dev)

    # ── SSMDynamics ───────────────────────────────────────────────────────────
    extent = 2 * 1.3   # initial estimate
    model  = SSMDynamics(
        hidden=args.hidden, mp_steps=args.mp_steps, ssm_dim=args.ssm_dim,
        e_dim=args.e_dim, z_dim=args.z_dim,
        accel_scale=args.accel_scale * extent).to(dev)
    graph_cfg = {"graph": "knn", "k": 8}
    print(f"[train] SSMDynamics hidden={args.hidden} mp={args.mp_steps} "
          f"ssm={args.ssm_dim} dt={args.dt} accel_scale={args.accel_scale*extent:.4f}")

    # ── optimizers ────────────────────────────────────────────────────────────
    dyn_opt = torch.optim.Adam([
        {"params": list(model.parameters())},
        {"params": list(anchors.parameters())},
        {"params": [ic_dir]},
        {"params": list(ic_net.parameters())},
    ], lr=args.lr_dyn)

    # ── LBS binding cache ─────────────────────────────────────────────────────
    _w_cache  = [None]
    _idx_cache = [None]

    def refresh_binding():
        with torch.no_grad():
            w, idx = anchors.cal_nn_weight(gaussians.get_xyz.detach())
        _w_cache[0]   = w
        _idx_cache[0] = idx

    refresh_binding()

    # ── rollout helper ────────────────────────────────────────────────────────
    def rollout():
        p0 = anchors.canonical
        # initial velocity derived from ic_dir: direction only, magnitude = accel_scale
        ic_mag = ic_net(ic_dir, anchors.e)
        v0 = F.normalize(ic_dir, dim=-1) * ic_mag.unsqueeze(-1)
        return ssm_rollout(
            model, p0, v0, anchors.e, anchors.z,
            init_vel=v0, init_pos=p0, steps=T - 1,
            bptt_start=args.bptt_start,
            cfg=graph_cfg, dt=args.dt, grad=True,
            damping=args.damping, vel_smooth=0.1)   # [T, M, 3]

    # ── SC-GS densification params (mirrors SC-GS defaults) ──────────────────
    densify_from  = 500
    densify_every = 100
    densify_until = 15_000
    opacity_reset_every = 3000
    max_grad      = 0.0002
    min_opacity   = 0.005
    max_screen    = 20

    pipe = Pipe()

    # ── resume ────────────────────────────────────────────────────────────────
    start = 0
    if args.resume:
        ckpt = os.path.join(args.out, "ckpt_last.pt")
        if os.path.exists(ckpt):
            state = torch.load(ckpt, map_location=dev)
            model.load_state_dict(state["model"])
            anchors.load_state_dict(state["anchors"])
            ic_dir.data.copy_(state["ic_dir"])
            ic_net.load_state_dict(state["ic_net"])
            dyn_opt.load_state_dict(state["dyn_opt"])
            # Gaussians: load latest PLY (highest iteration)
            import glob as _glob
            plys = sorted(_glob.glob(os.path.join(args.out, "point_cloud_*.ply")))
            if plys:
                gaussians.load_ply(plys[-1])
                gaussians.training_setup(opt_params)
            start = state["step"] + 1
            refresh_binding()
            print(f"[train] resumed from step {start}")
        else:
            print("[train] --resume: no checkpoint found, starting fresh")

    # ── training loop ─────────────────────────────────────────────────────────
    log_every  = 500
    save_every = 10_000
    pbar = trange(start, args.iters, desc="train")
    running_loss = 0.0

    for step in pbar:
        gaussians.update_learning_rate(step)

        # ── rollout: [T, M, 3] anchor positions ─────────────────────────────
        anchor_seq = rollout()

        # ── photometric loss over all T frames ───────────────────────────────
        total_loss = torch.tensor(0.0, device=dev)

        # recompute binding weights (anchors._radius, _node_weight can change)
        w   = _w_cache[0]
        idx = _idx_cache[0]

        gauss_xyz  = gaussians.get_xyz                      # [N, 3]

        # Accumulate grad stats for densification (need first frame)
        viewspace_pt = None
        visibility   = None

        for t in range(T):
            vi = random.randrange(V)                        # random view for this frame
            cam = cams[vi]
            cam.fid = torch.tensor([t / max(T - 1, 1)], device=dev)

            anchor_now = anchor_seq[t]                      # [M, 3]
            d_xyz = lbs_d_xyz(gauss_xyz, w, idx, anchors.canonical, anchor_now)

            pkg  = _gs_render(cam, gaussians, pipe, bg,
                              d_xyz, 0.0, torch.zeros_like(d_xyz),
                              d_rot_as_res=True)
            rendered = pkg["render"]
            gt       = frames[vi][t]

            loss = (1 - args.lambda_ssim) * l1_loss(rendered, gt) \
                 + args.lambda_ssim * ssim_loss(rendered, gt)
            total_loss = total_loss + loss

            # Collect densification stats from frame 0 (one view per step)
            if t == 0 and step < densify_until:
                viewspace_pt = pkg["viewspace_points"]
                visibility   = pkg["visibility_filter"]

        total_loss = total_loss / T

        # ── backward ─────────────────────────────────────────────────────────
        gaussians.optimizer.zero_grad(set_to_none=True)
        dyn_opt.zero_grad(set_to_none=True)
        total_loss.backward()

        # ── densification ────────────────────────────────────────────────────
        if step < densify_until and viewspace_pt is not None:
            gaussians.max_radii2D[visibility] = torch.max(
                gaussians.max_radii2D[visibility], pkg["radii"][visibility])
            gaussians.add_densification_stats(viewspace_pt, visibility)

            if step >= densify_from and step % densify_every == 0:
                size_thresh = max_screen if step > opacity_reset_every else None
                gaussians.densify_and_prune(max_grad, min_opacity, extent, size_thresh)
                refresh_binding()
                w   = _w_cache[0]
                idx = _idx_cache[0]
                print(f"[step {step}] densify → {gaussians.get_xyz.shape[0]} gaussians")

            if step % opacity_reset_every == 0 or (step == densify_from):
                gaussians.reset_opacity()

        # ── optimizer step ────────────────────────────────────────────────────
        gaussians.optimizer.step()
        dyn_opt.step()

        running_loss += float(total_loss)

        # ── logging ───────────────────────────────────────────────────────────
        if step % log_every == 0 and step > 0:
            avg = running_loss / log_every
            running_loss = 0.0
            pbar.set_postfix(loss=f"{avg:.4f}", n=gaussians.get_xyz.shape[0])

        # ── checkpoint + save ─────────────────────────────────────────────────
        if (step + 1) % save_every == 0 or step == args.iters - 1:
            ply_path = os.path.join(args.out, f"point_cloud_{step+1:06d}.ply")
            gaussians.save_ply(ply_path)
            torch.save({
                "step":      step,
                "model":     model.state_dict(),
                "anchors":   anchors.state_dict(),
                "ic_dir":    ic_dir.data,
                "ic_net":    ic_net.state_dict(),
                "dyn_opt":   dyn_opt.state_dict(),
            }, os.path.join(args.out, "ckpt_last.pt"))
            print(f"[step {step+1}] saved  gaussians={gaussians.get_xyz.shape[0]}")

    print("[train] done.")


if __name__ == "__main__":
    main()

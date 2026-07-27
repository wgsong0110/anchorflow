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
from anchorflow.ssm_dynamics import SSMDynamics, ssm_rollout, build_graph, HopDynamics, hop_rollout
from anchorflow import warp as W
import lbs as _lbs_cuda


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
    """GNN that propagates sparse ic_dir to v0 for ALL anchors.

    Specified anchors   (ic_dir ≠ 0): v0 = normalize(ic_dir) × GNN-predicted magnitude
    Unspecified anchors (ic_dir = 0): v0 = full GNN prediction (direction + magnitude)
                                       inferred from neighbouring specified anchors

    node input : [normalize(ic_dir)(3), is_specified(1), e(e_dim)]
    edge input : [rel_dir(3), dist(1)]
    output     : v0 [M, 3]
    """
    def __init__(self, e_dim=8, hidden=64, mp_steps=3):
        super().__init__()
        from anchorflow.dynamics import InteractionNetwork
        self.node_enc = torch.nn.Sequential(
            torch.nn.Linear(3 + 1 + e_dim, hidden), torch.nn.SiLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.LayerNorm(hidden),
        )
        self.edge_enc = torch.nn.Sequential(
            torch.nn.Linear(4, hidden), torch.nn.SiLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.LayerNorm(hidden),
        )
        self.processor = torch.nn.ModuleList(
            [InteractionNetwork(hidden) for _ in range(mp_steps)])
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(hidden, hidden), torch.nn.SiLU(),
            torch.nn.Linear(hidden, 3),
        )

    def forward(self, ic_dir, e, pos, edge_index):
        is_spec = (ic_dir.norm(dim=-1, keepdim=True) > 1e-6).float()   # [M,1]
        h = self.node_enc(torch.cat([
            F.normalize(ic_dir, dim=-1), is_spec, e], dim=-1))

        src, dst = edge_index
        rel  = pos[src] - pos[dst]
        dist = rel.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        ef   = self.edge_enc(torch.cat([rel / dist, dist], dim=-1))

        for layer in self.processor:
            h, ef = layer(h, edge_index, ef)

        v0_pred = self.decoder(h)                                        # [M,3]

        # specified: keep user direction, take magnitude from GNN
        mag = v0_pred.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        v0_spec = F.normalize(ic_dir, dim=-1) * mag
        return torch.where(is_spec.bool(), v0_spec, v0_pred)


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


# ── LBS warp: anchor displacement + rotation → per-Gaussian d_xyz/d_rot ──────
def lbs_d_xyz_rot(gauss_xyz, w, idx, anchor_canon, anchor_now, anchor_d_rot):
    """Translation + rotation LBS. anchor_d_rot [M,4] is the per-anchor residual
    quaternion straight from SSMDynamics.rot_decoder (zero-init -> identity at
    step 0). Directly predicted (no Procrustes/SVD), so it carries gradient to
    the dynamics model like the position branch. Matches SC-GS's own
    ControlNodeWarp.forward (d_rot_as_res branch): rotation = weighted sum of
    the per-node residual quaternions, passed to the renderer with
    d_rot_as_res=True (renderer normalizes [1,0,0,0] + d_rotation)."""
    d_anchor = anchor_now - anchor_canon              # [M, 3]
    d_xyz = (w[..., None] * d_anchor[idx]).sum(1)      # [N, 3]
    d_rot = (w[..., None] * anchor_d_rot[idx]).sum(1)  # [N, 4]
    return d_xyz, d_rot


def lbs_d_xyz_rot_batched(w, idx, anchor_canon, anchor_now_batch, anchor_d_rot_batch):
    """Same as lbs_d_xyz_rot but for a batch of B frames at once: one fused
    CUDA kernel call (lib/lbs rot_batch_forward/backward) instead of B
    separate per-frame python-loop gathers. Profiling showed the per-frame
    gather (IndexBackward0 / _index_put_impl_) is almost entirely
    CPU-dispatch-bound (many tiny kernel launches, not GPU-compute-bound) --
    this fuses gather + weighted-sum for both d_xyz and d_rot into a single
    kernel launch (falls back to the torch path if the CUDA ext isn't built).
    w, idx: [N, K]   anchor_canon: [M, 3]
    anchor_now_batch: [B, M, 3]   anchor_d_rot_batch: [B, M, 4]
    returns d_xyz [B, N, 3], d_rot [B, N, 4]"""
    return _lbs_cuda.lbs_blend_rot_batched(w, idx, anchor_canon, anchor_now_batch, anchor_d_rot_batch)


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
    ap.add_argument("--no_ssm", action="store_true",
                    help="ablation: disable SSM temporal recurrence (h=u every step, "
                         "memoryless per-step GNN decode) to isolate whether the SSM "
                         "itself is the bottleneck vs SC-GS")
    ap.add_argument("--predict_velocity", action="store_true",
                    help="ablation: decoder output is velocity directly (single "
                         "integration p'=p+v*dt) instead of acceleration (double "
                         "integration p'=p+v*dt, v'=v+a*dt)")
    ap.add_argument("--lambda_arap", type=float, default=1e-2,
                    help="ARAP regularizer weight on anchor motion (matches "
                         "SC-GS's own 1e-2 default; 0 disables)")
    ap.add_argument("--dynamics_mode", choices=["rollout", "hop"], default="rollout",
                    help="'rollout': fixed-dt sequential SSMDynamics rollout over all T "
                         "frames (original). 'hop': HopDynamics -- sequential but only "
                         "through the sampled frames, each hop's own (variable) Δt fed "
                         "explicitly, decoder outputs the position delta directly "
                         "(middle ground between full autoregression and one-shot "
                         "per-frame regression)")
    ap.add_argument("--n_time_freqs", type=int, default=6,
                    help="[hop mode] number of NeRF-style Δt positional-encoding bands")
    ap.add_argument("--lambda_ic", type=float, default=1.0,
                    help="[hop mode] weight on ||ic_dir - v_implied||^2, where v_implied "
                         "= (p(frame 1) - canonical) / dt is the actual initial velocity "
                         "of the generated path (frame 1 is always in frame_idxs) -- keeps "
                         "ic_dir a literal, user-steerable initial-velocity parameter even "
                         "though the decoder no longer integrates velocity explicitly")
    ap.add_argument("--lambda_consist", type=float, default=1.0,
                    help="[hop mode] weight on the fine-vs-coarse hop-chain position "
                         "consistency loss (both chains share the same h0/spatial embed; "
                         "enforces the flow-composition property Φ(Δt2)∘Φ(Δt1)=Φ(Δt1+Δt2))")
    ap.add_argument("--eval_every", type=int, default=1000,
                    help="full-T all-view PSNR/SSIM eval cadence (SC-GS-comparable; 0 disables)")
    ap.add_argument("--frames_per_step", type=int, default=6,
                    help="random subset of T frames to render+loss per step "
                         "(rollout still runs the full T frames; render/backward "
                         "is ~T/frames_per_step cheaper -- rasterizer backward is "
                         "78%% of step time by profiling). Last frame (t=T-1) is "
                         "always included since it's the hardest/most drift-prone.")
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

    # ── dynamics model ────────────────────────────────────────────────────────
    extent = 2 * 1.3   # initial estimate
    hop_mode = args.dynamics_mode == "hop"
    if hop_mode:
        # ICNet's raw [M,3] output (direction+magnitude propagated from ic_dir)
        # is repurposed as the control code z (replaces anchors.z, which is
        # unused in this mode) -- see --lambda_ic docstring.
        model = HopDynamics(
            hidden=args.hidden, mp_steps=args.mp_steps, ssm_dim=args.ssm_dim,
            e_dim=args.e_dim, z_dim=3, n_time_freqs=args.n_time_freqs).to(dev)
        print(f"[train] HopDynamics hidden={args.hidden} mp={args.mp_steps} "
              f"ssm={args.ssm_dim} dt={args.dt} n_time_freqs={args.n_time_freqs} "
              f"lambda_arap={args.lambda_arap} lambda_ic={args.lambda_ic} "
              f"lambda_consist={args.lambda_consist}")
    else:
        model  = SSMDynamics(
            hidden=args.hidden, mp_steps=args.mp_steps, ssm_dim=args.ssm_dim,
            e_dim=args.e_dim, z_dim=args.z_dim,
            accel_scale=args.accel_scale * extent,
            use_ssm=not args.no_ssm,
            predict_velocity=args.predict_velocity).to(dev)
        print(f"[train] SSMDynamics hidden={args.hidden} mp={args.mp_steps} "
              f"ssm={args.ssm_dim} dt={args.dt} accel_scale={args.accel_scale*extent:.4f} "
              f"use_ssm={not args.no_ssm} predict_velocity={args.predict_velocity} "
              f"lambda_arap={args.lambda_arap}")
    graph_cfg = {"graph": "knn", "k": 8}

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

    # ── ARAP loss cache (K-NN graph over the fixed canonical anchor positions --
    # geometric only, independent of anchors._radius/_node_weight, so this never
    # needs to be refreshed) ────────────────────────────────────────────────────
    _arap_idx, _arap_src = W.anchor_rotations_cache(anchors.canonical, K=8)

    # ── rollout helper(s) ─────────────────────────────────────────────────────
    def rollout(grad=True):
        """SSMDynamics mode: fixed-dt sequential rollout over all T frames."""
        p0 = anchors.canonical
        edge_index = build_graph(p0.detach(), graph_cfg)
        v0 = ic_net(ic_dir, anchors.e, p0, edge_index)
        return ssm_rollout(
            model, p0, v0, anchors.e, anchors.z,
            init_vel=v0, init_pos=p0, steps=T - 1,
            bptt_start=args.bptt_start,
            cfg=graph_cfg, dt=args.dt, grad=grad,
            damping=args.damping, vel_smooth=0.1,
            return_rotations=True)   # [T, M, 3], [T, M, 4]

    def hop_forward(frame_idxs, coarse_idxs=None, grad=True):
        """HopDynamics mode: sequential but only through frame_idxs (and,
        separately, coarse_idxs from the same shared h0/spatial embed, for
        the fine/coarse consistency loss). z = ic_net's propagated output
        (repurposes the old "initial velocity" role as a control code)."""
        p0 = anchors.canonical
        edge_index = build_graph(p0.detach(), graph_cfg)
        z = ic_net(ic_dir, anchors.e, p0, edge_index)          # [M,3], repurposed as z
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            spatial = model.spatial_embed(anchors.e, edge_index, p0)
            h0 = model.init_hidden(anchors.e, z)
        p_fine, rot_fine, _ = hop_rollout(
            model, p0, anchors.e, z, edge_index, frame_idxs, args.dt,
            grad=grad, h0=h0, spatial=spatial)
        p_coarse = rot_coarse = None
        if coarse_idxs:
            p_coarse, rot_coarse, _ = hop_rollout(
                model, p0, anchors.e, z, edge_index, coarse_idxs, args.dt,
                grad=grad, h0=h0, spatial=spatial)
        return p_fine, rot_fine, p_coarse, rot_coarse, z

    def hop_frame_idxs():
        """Fine subsample: like the rollout-mode sampler (always includes
        T-1), but also always includes frame 1 (needed for the initial-
        velocity-consistency loss -- the empirical v_implied is computed
        from canonical -> frame 1)."""
        k = min(args.frames_per_step, T)
        pool = list(range(2, T - 1))
        n_random = max(0, k - 2)
        chosen = set(random.sample(pool, min(n_random, len(pool))))
        chosen.add(1)
        chosen.add(T - 1)
        return sorted(chosen)

    def hop_coarse_idxs(frame_idxs):
        """Further subsample of frame_idxs (every other one) -- by construction
        at least one fine frame sits between consecutive coarse frames. T-1 is
        force-included so the consistency check also covers the longest hop."""
        if len(frame_idxs) < 3:
            return list(frame_idxs)
        coarse = sorted(set(frame_idxs[::2]) | {frame_idxs[-1]})
        return coarse

    best_psnr = {"value": -1.0, "step": -1}

    @torch.no_grad()
    def eval_psnr_ssim():
        """Full-T, all-view render under no_grad -- PSNR/SSIM (SC-GS-comparable,
        printed in the same 'Best PSNR=.. in Iteration ..' style)."""
        if hop_mode:
            p_full, rot_full, _, _, _ = hop_forward(list(range(1, T)), grad=False)
            anchor_seq = torch.stack([p_full[t] for t in range(T)], dim=0)
            anchor_rot_seq = torch.stack([rot_full[t] for t in range(T)], dim=0)
        else:
            anchor_seq, anchor_rot_seq = rollout(grad=False)
        w, idx = _w_cache[0], _idx_cache[0]
        gauss_xyz = gaussians.get_xyz
        d_xyz_all, d_rot_all = lbs_d_xyz_rot_batched(
            w, idx, anchors.canonical, anchor_seq, anchor_rot_seq)
        mse_sum, ssim_sum, n = 0.0, 0.0, 0
        for t in range(T):
            for vi in range(V):
                cam = cams[vi]
                cam.fid = torch.tensor([t / max(T - 1, 1)], device=dev)
                pkg = _gs_render(cam, gaussians, pipe, bg,
                                  d_xyz_all[t], d_rot_all[t], torch.zeros_like(d_xyz_all[t]),
                                  d_rot_as_res=True)
                rendered = pkg["render"]
                gt = frames[vi][t]
                mse_sum += F.mse_loss(rendered, gt).item()
                ssim_sum += 1 - ssim_loss(rendered, gt).item()
                n += 1
        mse = mse_sum / n
        psnr = -10 * math.log10(max(mse, 1e-10))
        ssim = ssim_sum / n
        if psnr > best_psnr["value"]:
            best_psnr["value"], best_psnr["step"] = psnr, step
        print(f"[step {step}] Eval: PSNR={psnr:.5f} SSIM={ssim:.5f} "
              f"Best PSNR={best_psnr['value']:.5f} in step {best_psnr['step']}", flush=True)

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
                # training_setup resets xyz_gradient_accum/denom to the loaded
                # point count but not max_radii2D -- it stays at the pre-load
                # size and desyncs densify_and_prune's boolean masking.
                gaussians.max_radii2D = torch.zeros(
                    (gaussians.get_xyz.shape[0],), device=dev)
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
    term_names = ["render", "arap", "ic", "consist"]
    running_terms = {k: 0.0 for k in term_names}
    loss_csv_path = os.path.join(args.out, "losses.csv")
    if not os.path.exists(loss_csv_path):
        with open(loss_csv_path, "w") as f:
            f.write("step,total," + ",".join(term_names) + "\n")

    for step in pbar:
        gaussians.update_learning_rate(step)

        # ── photometric loss over the sampled frames ─────────────────────────
        total_loss = torch.tensor(0.0, device=dev)
        render_loss_sum = torch.tensor(0.0, device=dev)

        # recompute binding weights (anchors._radius, _node_weight can change)
        w   = _w_cache[0]
        idx = _idx_cache[0]

        # Accumulate grad stats for densification (need first rendered frame)
        viewspace_pt = None
        visibility   = None

        if hop_mode:
            # sequential, but only through the sampled (fine) frames -- and,
            # separately, a coarser subsample sharing the same h0/spatial embed
            # (for the fine/coarse consistency loss). Frame 1 and T-1 are
            # always included (initial-velocity consistency / hardest-reach).
            frame_idxs = hop_frame_idxs()
            coarse_idxs = hop_coarse_idxs(frame_idxs)
            p_fine, rot_fine, p_coarse, rot_coarse, _z = hop_forward(frame_idxs, coarse_idxs)
            anchor_now_stack = torch.stack([p_fine[t] for t in frame_idxs], dim=0)
            anchor_rot_stack = torch.stack([rot_fine[t] for t in frame_idxs], dim=0)
        else:
            # rendering (esp. rasterizer backward) dominates step time -- rollout
            # itself still runs all T steps (needed for correct GNN+SSM state),
            # but we only render/loss a random subset. Last frame (T-1) is always
            # included since it's the most drift-prone / hardest to get right.
            anchor_seq, anchor_rot_seq = rollout()
            k = min(args.frames_per_step, T)
            frame_idxs = set(random.sample(range(T - 1), k - 1)) if k > 1 else set()
            frame_idxs.add(T - 1)
            frame_idxs = sorted(frame_idxs)
            anchor_now_stack = anchor_seq[frame_idxs]
            anchor_rot_stack = anchor_rot_seq[frame_idxs]

        # one batched gather for all k selected frames instead of k separate
        # small gathers -- profiling showed the per-frame gather is CPU
        # dispatch-bound (many tiny kernel launches), not GPU-compute-bound
        d_xyz_all, d_rotation_all = lbs_d_xyz_rot_batched(
            w, idx, anchors.canonical, anchor_now_stack, anchor_rot_stack)

        for i, t in enumerate(frame_idxs):
            vi = random.randrange(V)                        # random view for this frame
            cam = cams[vi]
            cam.fid = torch.tensor([t / max(T - 1, 1)], device=dev)

            d_xyz      = d_xyz_all[i]                       # [N, 3]
            d_rotation = d_rotation_all[i]                  # [N, 4]

            pkg  = _gs_render(cam, gaussians, pipe, bg,
                              d_xyz, d_rotation, torch.zeros_like(d_xyz),
                              d_rot_as_res=True)
            rendered = pkg["render"]
            gt       = frames[vi][t]

            loss = (1 - args.lambda_ssim) * l1_loss(rendered, gt) \
                 + args.lambda_ssim * ssim_loss(rendered, gt)
            total_loss = total_loss + loss
            render_loss_sum = render_loss_sum + loss

            # Collect densification stats from the first rendered frame this step
            if i == 0 and step < densify_until:
                viewspace_pt = pkg["viewspace_points"]
                visibility   = pkg["visibility_filter"]

        total_loss = total_loss / len(frame_idxs)
        render_loss_sum = render_loss_sum / len(frame_idxs)
        loss_ic = loss_consist = arap_loss = torch.tensor(0.0, device=dev)

        if hop_mode:
            # ── initial-velocity consistency: keep ic_dir a literal, user-
            # steerable initial velocity even though the decoder no longer
            # integrates velocity -- frame 1 is always in frame_idxs, so
            # v_implied is the actual empirical initial velocity of the path
            # the network just generated. ────────────────────────────────────
            v_implied = (p_fine[1] - anchors.canonical) / args.dt
            loss_ic = F.mse_loss(ic_dir, v_implied)
            total_loss = total_loss + args.lambda_ic * loss_ic

            # ── fine/coarse hop-chain consistency: both chains share the same
            # h0/spatial embed: Φ(Δt2)∘Φ(Δt1) should equal Φ(Δt1+Δt2). ──────────
            if p_coarse is not None and args.lambda_consist > 0:
                consist_terms = [F.mse_loss(p_fine[t], p_coarse[t])
                                  for t in coarse_idxs if t != 0]
                if consist_terms:
                    loss_consist = sum(consist_terms) / len(consist_terms)
                    total_loss = total_loss + args.lambda_consist * loss_consist

            # ── ARAP regularizer (SC-GS-style; same as rollout mode but sampled
            # from the frames we actually computed this step) ──────────────────
            if args.lambda_arap > 0:
                arap_ts = random.sample(frame_idxs, min(2, len(frame_idxs)))
                arap_loss = sum(
                    W.anchor_arap_loss(anchors.canonical, p_fine[t], K=8,
                                       _idx=_arap_idx, _src=_arap_src)
                    for t in arap_ts) / len(arap_ts)
                total_loss = total_loss + args.lambda_arap * arap_loss
        else:
            # ── ARAP regularizer (SC-GS-style: penalize local-rigidity violation
            # of anchor motion; ports SC-GS's utils/deform_utils.cal_arap_error
            # via the existing no_grad Procrustes rotation estimator) ──────────
            if args.lambda_arap > 0:
                arap_ts = random.sample(range(1, T), min(2, T - 1))  # skip t=0 (zero stretch)
                arap_loss = sum(
                    W.anchor_arap_loss(anchors.canonical, anchor_seq[t], K=8,
                                       _idx=_arap_idx, _src=_arap_src)
                    for t in arap_ts) / len(arap_ts)
                total_loss = total_loss + args.lambda_arap * arap_loss

        # ── backward ─────────────────────────────────────────────────────────
        gaussians.optimizer.zero_grad(set_to_none=True)
        dyn_opt.zero_grad(set_to_none=True)
        total_loss.backward()

        # ── PSNR/SSIM eval (SC-GS-comparable: measured *before* this step's
        # densify/opacity-reset, exactly mirroring SC-GS's train_gui.py, where
        # training_report() runs before the densify/reset block. Evaluating
        # after reset -- as we previously did -- captures the freshly-zeroed
        # opacity's worst-case render, which SC-GS's own logged curve never
        # shows since it logs the pre-reset state instead) ─────────────────────
        if args.eval_every > 0 and (step % args.eval_every == 0 or step == args.iters - 1):
            eval_psnr_ssim()

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
        running_terms["render"]  += float(render_loss_sum)
        running_terms["arap"]    += float(arap_loss)
        running_terms["ic"]      += float(loss_ic)
        running_terms["consist"] += float(loss_consist)

        # ── logging ───────────────────────────────────────────────────────────
        if step % log_every == 0 and step > 0:
            avg = running_loss / log_every
            term_avgs = {k: v / log_every for k, v in running_terms.items()}
            running_loss = 0.0
            running_terms = {k: 0.0 for k in term_names}
            pbar.set_postfix(loss=f"{avg:.4f}", n=gaussians.get_xyz.shape[0])
            with open(loss_csv_path, "a") as f:
                f.write(f"{step},{avg}," + ",".join(f"{term_avgs[k]}" for k in term_names) + "\n")

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

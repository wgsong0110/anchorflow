#!/usr/bin/env python
"""AnchorFlow: MDS-based dynamic 3DGS via semantic anchor nodes + z0-bank.

Pipeline:
  1. Build anchor nodes via tokens_to_nodes (DINOv2 semantic) or FPS fallback
  2. NodeFlow GNN: canonical_pos → scene features h [K,H]  (once per step)
     Physics integration: vel[t] = z0 + dt*cumsum(acc), disp[t] = dt*cumsum(vel)
  3. z0_bank [B, K, 3]: bank of learnable initial velocities
     - each MDS step samples one z0 → renders T-frame clip → MDS loss
     - both GNN weights and z0_bank are updated simultaneously
  4. SVD MDS loss: grad = w*(eps(video) - eps(static_frame0))
     - frame-0 is always canonical render (plausible SVD conditioning image)

Usage (on GPU instance):
    cd /workspace/anchorflow
    PYTHONPATH=/workspace/gaussian-splatting:/workspace/anchorflow/lib \\
    python exe/train_anchorflow.py \\
        --ply  /workspace/lego_canonical.ply \\
        --col  /workspace/kitchen_colmap \\
        --cfg  cfg/anchorflow_kitchen.yaml \\
        --out  /workspace/anchorflow_out \\
        --resume
"""
from __future__ import annotations

import argparse, json, math, os, subprocess, sys, random
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn.functional as F
import imageio.v2 as iio
from omegaconf import OmegaConf
from PIL import Image
from plyfile import PlyData

sys.path.append("gaussian-splatting")
from scene.gaussian_model import GaussianModel
from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary, qvec2rotmat
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
from anchorflow.nodeflow import NodeFlow
from anchorflow.checkpoint import CheckpointManager, load_rng_state
from anchorflow.sds import SVDGuidance


# ── Camera ───────────────────────────────────────────────────────────────────

class Cam:
    def __init__(self, R, T, fovx, fovy, W, H):
        self.image_width, self.image_height = W, H
        self.FoVx, self.FoVy = fovx, fovy
        self.znear, self.zfar = 0.01, 100.0
        w2v  = torch.tensor(getWorld2View2(R, T)).T.cuda()
        proj = getProjectionMatrix(self.znear, self.zfar, fovx, fovy).T.cuda()
        self.world_view_transform = w2v
        self.full_proj_transform  = (w2v.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0)
        self.camera_center        = w2v.inverse()[3, :3]
        self.tanfovx = math.tan(fovx * 0.5)
        self.tanfovy = math.tan(fovy * 0.5)


def look_at(eye, center, up):
    f = (center - eye); f /= np.linalg.norm(f)
    s = np.cross(f, up); s /= np.linalg.norm(s)
    u = np.cross(s, f)
    R = np.stack([s, u, f], axis=1)
    return R.astype(np.float32), eye.astype(np.float32)


def make_orbit_cameras(xyz, n_views, res, elevation=15.0,
                       fov_deg=50.0, radius_scale=2.5) -> list:
    """Spherical-orbit cameras derived from the Gaussian bbox (same convention
    as gen_views.py). Use for object-centric scenes with no matching COLMAP."""
    lo, hi = xyz.min(0).values.cpu().numpy(), xyz.max(0).values.cpu().numpy()
    center = ((lo + hi) * 0.5).astype(np.float32)
    diag   = float(np.linalg.norm(hi - lo))
    radius = radius_scale * diag * 0.5
    fov    = math.radians(fov_deg)
    el     = math.radians(elevation)
    cams   = []
    for i in range(n_views):
        az  = math.radians(360.0 * i / n_views)
        eye = center + radius * np.array([
            math.cos(el) * math.sin(az),
            math.sin(el),
            math.cos(el) * math.cos(az),
        ], dtype=np.float32)
        Rc, _ = look_at(eye, center, np.array([0., 1., 0.], dtype=np.float32))
        T = -Rc.T @ eye
        cams.append(Cam(Rc, T, fov, fov, res, res))
    print(f"[train] orbit cameras: center={center.round(3)} diag={diag:.2f} "
          f"radius={radius:.2f} elev={elevation}° fov={fov_deg}°")
    return cams


def load_colmap_cameras(col_dir: str, n_views: int, res: int) -> list:
    extr = read_extrinsics_binary(f"{col_dir}/sparse/0/images.bin")
    intr = read_intrinsics_binary(f"{col_dir}/sparse/0/cameras.bin")
    items = sorted(extr.values(), key=lambda im: im.name)
    idx   = np.linspace(0, len(items) - 1, n_views).round().astype(int)
    cams  = []
    for i in idx:
        im  = items[i]
        cam = intr[im.camera_id]
        f   = cam.params[0]
        W0, H0 = cam.width, cam.height
        S   = min(W0, H0)
        fov = 2 * math.atan(S / (2 * f))
        R   = qvec2rotmat(im.qvec).astype(np.float32)
        T   = np.array(im.tvec, dtype=np.float32)
        cams.append(Cam(R, T, fov, fov, res, res))
    return cams


# ── Rasterizer ───────────────────────────────────────────────────────────────

def render_gaussians(cam, xyz, opacities, scales, rotations, colors, bg):
    """Returns [3, H, W] in [0,1]."""
    H, W = cam.image_height, cam.image_width
    cfg  = GaussianRasterizationSettings(
        image_height=H, image_width=W,
        tanfovx=cam.tanfovx, tanfovy=cam.tanfovy,
        bg=bg, scale_modifier=1.0,
        viewmatrix=cam.world_view_transform,
        projmatrix=cam.full_proj_transform,
        sh_degree=0, campos=cam.camera_center,
        prefiltered=False, debug=False,
    )
    rast  = GaussianRasterizer(raster_settings=cfg)
    means2D = torch.zeros_like(xyz, requires_grad=True)
    img, _  = rast(means3D=xyz, means2D=means2D, shs=None,
                   colors_precomp=colors, opacities=opacities,
                   scales=scales, rotations=rotations, cov3D_precomp=None)
    return img


# ── Gaussian PLY loader ───────────────────────────────────────────────────────

def load_gaussian_attrs(ply_path: str, device="cuda"):
    names = [p.name for p in PlyData.read(ply_path)["vertex"].properties
             if p.name.startswith("f_rest_")]
    sh = min(int(math.sqrt((len(names) + 3) // 3)) - 1 if names else 0, 3)
    g  = GaussianModel(sh)
    g.load_ply(ply_path)
    g.active_sh_degree = sh
    SH_C0 = 0.28209479177387814
    return {
        "xyz":       g.get_xyz.detach().to(device),
        "opacities": g.get_opacity.detach().to(device),
        "scales":    g.get_scaling.detach().to(device),
        "rotations": g.get_rotation.detach().to(device),
        "colors":    (SH_C0 * g._features_dc.detach()[:, 0, :] + 0.5).clamp(0, 1).to(device),
    }


# ── misc ─────────────────────────────────────────────────────────────────────

def git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "nogit"


def save_video(frames_hw3: list, path: str, fps: int = 8):
    arr = [(f.clamp(0,1).permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
           for f in frames_hw3]
    iio.mimsave(path, arr, fps=fps, quality=8)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply",    required=True, help="canonical 3DGS .ply")
    ap.add_argument("--col",    default=None,
                    help="COLMAP directory (sparse/0/); required for --cams colmap")
    ap.add_argument("--cams",   choices=["orbit", "colmap"], default="orbit",
                    help="orbit = spherical cameras from the Gaussian bbox "
                         "(object-centric scenes); colmap = poses from --col. "
                         "Only use colmap when the PLY was reconstructed FROM that COLMAP.")
    ap.add_argument("--elevation",    type=float, default=15.0)
    ap.add_argument("--fov_deg",      type=float, default=50.0)
    ap.add_argument("--radius_scale", type=float, default=2.5)
    ap.add_argument("--cfg",    required=True)
    ap.add_argument("--out",    required=True)
    ap.add_argument("--r2",     default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no-t2n", action="store_true", help="skip tokens_to_nodes, use FPS")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    cfg = OmegaConf.load(args.cfg)
    dev = "cuda"
    gh  = git_hash()

    # ── Gaussian attrs ────────────────────────────────────────────────────────
    gauss        = load_gaussian_attrs(args.ply, dev)
    canonical_xyz = gauss["xyz"]
    G = canonical_xyz.shape[0]
    print(f"[train] Gaussians={G}  commit={gh}")

    bg = torch.tensor([1., 1., 1.] if cfg.get("white_bg", True) else [0., 0., 0.], device=dev)

    # ── cameras ───────────────────────────────────────────────────────────────
    if args.cams == "colmap":
        if not args.col:
            sys.exit("[train] --cams colmap requires --col")
        cameras = load_colmap_cameras(args.col, cfg.train.n_views, cfg.model.res)
    else:
        cameras = make_orbit_cameras(
            canonical_xyz, cfg.train.n_views, cfg.model.res,
            elevation=args.elevation, fov_deg=args.fov_deg,
            radius_scale=args.radius_scale,
        )
    V = len(cameras)
    T = cfg.model.n_frames
    print(f"[train] cameras={V} ({args.cams})  T={T}  res={cfg.model.res}")

    def render_fn(cam):
        with torch.no_grad():
            return render_gaussians(
                cam, canonical_xyz,
                gauss["opacities"], gauss["scales"], gauss["rotations"],
                gauss["colors"], bg,
            )

    # ── render sanity check ──────────────────────────────────────────────────
    # A near-empty render means the cameras do not see the Gaussians (e.g. a PLY
    # paired with a COLMAP from a different scene). Distilling on a blank image
    # silently produces zero motion, so fail loudly here instead.
    for ci, c in enumerate(cameras):
        img = render_fn(c).clamp(0, 1)
        cover = float((img.max(0).values > 0.01).float().mean())
        if ci == 0 or cover < 0.02:
            print(f"[train] cam[{ci}] coverage={cover*100:.2f}%  mean={float(img.mean()):.4f}")
        if cover < 0.02:
            sys.exit(
                f"[train] ABORT: cam[{ci}] renders {cover*100:.2f}% non-empty pixels.\n"
                f"        The cameras are not looking at the Gaussians.\n"
                f"        --cams={args.cams}"
                + (f" --col={args.col}\n        Is the PLY really reconstructed "
                   f"from this COLMAP? Try --cams orbit." if args.cams == "colmap" else
                   "\n        Try adjusting --radius_scale / --fov_deg / --elevation.")
            )

    # ── anchor nodes ─────────────────────────────────────────────────────────
    node_pos = None
    if not args.no_t2n:
        try:
            from anchorflow.tokens_to_nodes import tokens_to_nodes
            import anchorflow.tokens_to_nodes as t2n_mod
            print("[train] building semantic nodes via tokens_to_nodes ...")
            node_pos = tokens_to_nodes(
                canonical_xyz,
                gauss["opacities"],
                render_fn,
                cameras[:cfg.get("t2n_views", 4)],
                n_nodes=cfg.model.n_nodes,
                device=dev,
            )
            if t2n_mod._dino_model is not None:
                del t2n_mod._dino_model
                t2n_mod._dino_model = None
            import gc; gc.collect()
            torch.cuda.empty_cache()
            print("[train] DINOv2 freed")
        except Exception as e:
            print(f"[train] tokens_to_nodes failed ({e}), falling back to FPS")
            node_pos = None
            import gc; gc.collect()
            torch.cuda.empty_cache()

    # ── model ─────────────────────────────────────────────────────────────────
    dt = float(cfg.model.get("dt", 1.0))
    model = NodeFlow(
        canonical_xyz  = canonical_xyz,
        node_positions = node_pos,
        n_nodes        = cfg.model.n_nodes,
        n_frames       = T,
        hidden         = cfg.model.hidden,
        n_gnn_layers   = cfg.model.n_gnn_layers,
        k_node         = cfg.model.k_node,
        k_gauss        = cfg.model.k_gauss,
        dt             = dt,
    ).to(dev)
    K = model.n_nodes
    print(f"[train] nodes={K}  hidden={cfg.model.hidden}  dt={dt}")

    # z0_bank [B, K, 3]: initial velocities; non-zero scale ensures non-zero motion
    B = cfg.train.z0_bank_size
    z0_init_scale = float(cfg.train.get("z0_init_scale", 0.05))
    z0_bank = torch.nn.Parameter(
        torch.randn(B, K, 3, device=dev) * z0_init_scale
    )
    print(f"[train] z0_bank  shape={list(z0_bank.shape)}  init_scale={z0_init_scale}")

    # ── SVD MDS guidance ──────────────────────────────────────────────────────
    print("[train] loading SVD for MDS guidance ...")
    svd = SVDGuidance(
        sigma_min        = cfg.mds.sigma_min,
        sigma_max        = cfg.mds.sigma_max,
        guidance_scale   = cfg.mds.guidance_scale,
        motion_bucket_id = cfg.mds.motion_bucket_id,
        grad_clip        = cfg.mds.grad_clip,
        device           = dev,
    )

    # ── optimiser ────────────────────────────────────────────────────────────
    gnn_params = (list(model.node_encoder.parameters()) +
                  list(model.gnn_layers.parameters()) +
                  list(model.accel_decoder.parameters()))
    opt = torch.optim.Adam([
        {"params": gnn_params, "lr": cfg.train.lr_gnn},
        {"params": [z0_bank],  "lr": cfg.train.lr_z0},
    ])

    # ── resume ────────────────────────────────────────────────────────────────
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

    torch.set_float32_matmul_precision("high")   # TF32

    def sync_r2():
        if args.r2:
            os.system(f"rclone copy {args.out} {args.r2} >/dev/null 2>&1")

    rng = random.Random(42)

    # ── training loop ─────────────────────────────────────────────────────────
    for step in range(start, cfg.train.iters):
        k = rng.randint(0, B - 1)
        v = rng.randint(0, V - 1)
        z0  = z0_bank[k]          # [K, 3] — initial velocity for this trajectory
        cam = cameras[v]

        with torch.no_grad():
            frame0 = render_fn(cam).clamp(0, 1)   # [3, H, W]

        # scene encoding: canonical structure only, z0-independent
        h = model.encode_scene()                   # [K, H]

        # physics rollout: acc(t) → vel(t) → disp(t), fully vectorised
        all_disps = model.rollout(h, z0)           # [T-1, G, 3]

        # render T frames
        frames = [frame0]
        for i in range(T - 1):
            xyz_def = canonical_xyz + all_disps[i]
            img = render_gaussians(
                cam, xyz_def,
                gauss["opacities"], gauss["scales"],
                gauss["rotations"], gauss["colors"], bg,
            )
            frames.append(img)
        frames_t = torch.stack(frames, dim=0)      # [T, 3, H, W]

        opt.zero_grad()

        # MDS loss
        loss = svd.mds_loss(frames_t, cond_image=frame0, w_power=cfg.mds.w_power)

        # ARAP
        if cfg.train.lambda_arap > 0:
            t_reg = rng.randint(1, T - 1)
            loss  = loss + cfg.train.lambda_arap * model.arap_loss(h, z0, t_reg)

        # z0 magnitude regularisation
        if cfg.train.lambda_z0 > 0:
            loss = loss + cfg.train.lambda_z0 * (z0_bank ** 2).mean()

        if not torch.isfinite(loss):
            print(f"[{step}] non-finite loss, skip"); continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            gnn_params + [z0_bank], cfg.train.grad_clip
        )
        opt.step()

        if step % cfg.train.log_every == 0:
            z0_rms = float(z0_bank.detach().pow(2).mean().sqrt())
            print(f"[{step}/{cfg.train.iters}] loss={float(loss):.4f}  "
                  f"k={k}  v={v}  z0_rms={z0_rms:.4f}")

        if step % cfg.train.ckpt_every == 0:
            ckpt_mgr.save(step, {
                "model":   model.state_dict(),
                "opt":     opt.state_dict(),
                "z0_bank": z0_bank.data,
                "step":    step,
            })
            _save_rollout(step, model, z0_bank, cameras[0], canonical_xyz,
                          gauss, bg, T, args.out)
            sync_r2()

    ckpt_mgr.save(cfg.train.iters - 1, {
        "model":   model.state_dict(),
        "opt":     opt.state_dict(),
        "z0_bank": z0_bank.data,
        "step":    cfg.train.iters - 1,
    })
    sync_r2()
    print(f"[train] done  commit={gh}  → {args.out}")


# ── rollout preview ──────────────────────────────────────────────────────────

@torch.no_grad()
def _save_rollout(step, model, z0_bank, cam, canon_xyz, gauss, bg, T, out):
    H, W = cam.image_height, cam.image_width
    B    = z0_bank.shape[0]
    cols = min(B, 4)
    rows = math.ceil(B / cols)

    # scene features (shared across all bank entries)
    h = model.encode_scene()

    # bank preview at t=T//2
    t_mid  = T // 2
    canvas = np.zeros((rows * H, cols * W, 3), np.uint8)
    for bi in range(B):
        z0   = z0_bank[bi]
        disp = model(h, z0, t_mid)   # [G, 3]
        img  = render_gaussians(
            cam, canon_xyz + disp,
            gauss["opacities"], gauss["scales"], gauss["rotations"],
            gauss["colors"], bg,
        )
        arr = (img.clamp(0,1).permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
        r, c = divmod(bi, cols)
        canvas[r*H:(r+1)*H, c*W:(c+1)*W] = arr
    Image.fromarray(canvas).save(
        os.path.join(out, f"bank_t{t_mid:02d}_step{step:06d}.png"))

    # single rollout video for bank entry 0
    frames = []
    disps  = model.rollout(h, z0_bank[0])   # [T-1, G, 3]
    frame0 = render_gaussians(
        cam, canon_xyz,
        gauss["opacities"], gauss["scales"], gauss["rotations"],
        gauss["colors"], bg,
    )
    frames.append(frame0)
    for i in range(T - 1):
        img = render_gaussians(
            cam, canon_xyz + disps[i],
            gauss["opacities"], gauss["scales"], gauss["rotations"],
            gauss["colors"], bg,
        )
        frames.append(img)
    vid_path = os.path.join(out, f"rollout_step{step:06d}.mp4")
    save_video(frames, vid_path)
    print(f"  saved rollout → {vid_path}")


if __name__ == "__main__":
    main()

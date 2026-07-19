#!/usr/bin/env python
"""anchorflow per-scene training — GNN simulator distilled by SVD video prior.

A fork of DreamPhysics `svd_simulation.py`: everything (3DGS loading + normalize,
camera, differentiable rasterizer, SVDGuidance video-SDS) is reused; the MPM
simulator is REPLACED by our GNN + anchor LBS. Per scene we optimize
{GNN weights, actuation latents z_i, anchor radius/weight} by the same video-SDS.

Because the GNN + LBS path is plain differentiable torch, we do NOT need MPM's
warp/taichi tape — a standard loss.backward() + optimizer.step() suffices.

Run INSIDE the DreamPhysics fork (so its gaussian_renderer / utils / video_distillation
imports resolve), with anchorflow's lib on PYTHONPATH:

    python train_gen.py --model_path CANON.ply --cond horse.png \
        --config config/anchorflow_horse.yaml --out /data/.../out --resume

Crash-safe: checkpoints {gnn,anchors,optimizer,step,rng} atomically; --resume
continues from the latest. Records the git commit for reproducibility.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

import torch
from torch.utils.checkpoint import checkpoint
from omegaconf import OmegaConf
from PIL import Image

# INRIA gaussian-splatting (scene/, gaussian_renderer/) is cloned inside the
# DreamPhysics dir; add it to the path exactly as DreamPhysics's svd_simulation does.
sys.path.append("/workspace/SC-GS")

# --- DreamPhysics / 3DGS stack (reused verbatim) -------------------------- #
# NB: we deliberately avoid `utils.decode_param` (it imports warp + MPM at module
# level) — camera/preprocessing come from our yaml config instead, so the image
# needs no warp/taichi/MPM.
from utils.render_utils import load_params_from_gs, initialize_resterize
from utils.transformation_utils import (
    generate_rotation_matrices, apply_rotations, transform2origin,
    shift2center111, apply_cov_rotations, apply_inverse_rotations,
    undotransform2origin, undoshift2center111, apply_inverse_cov_rotations,
    get_center_view_worldspace_and_observant_coordinate,
)
from utils.camera_view_utils import get_camera_view
from utils.render_utils import convert_SH
from scene.gaussian_model import GaussianModel
from video_distillation.svd_guidance import SVDGuidance

# --- anchorflow ----------------------------------------------------------- #
from anchorflow.anchors import AnchorSet
from anchorflow.dynamics import GNSDynamics, rollout
from anchorflow import warp as W
from anchorflow import reg as R
from anchorflow.checkpoint import CheckpointManager, load_rng_state


class Pipe:
    convert_SHs_python = False
    compute_cov3D_python = True
    debug = False


def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "nogit"


def load_canonical(model_path):
    from plyfile import PlyData
    import math
    if os.path.isdir(model_path):            # DreamPhysics-style model dir
        import glob
        its = glob.glob(os.path.join(model_path, "point_cloud", "iteration_*"))
        it = max(int(p.split("iteration_")[-1]) for p in its)
        ply = os.path.join(model_path, "point_cloud", f"iteration_{it}", "point_cloud.ply")
    else:
        ply = model_path
    names = [p.name for p in PlyData.read(ply)["vertex"].properties
             if p.name.startswith("f_rest_")]
    sh = int(math.sqrt((len(names) + 3) // 3)) - 1
    g = GaussianModel(sh)
    g.load_ply(ply)
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True, help="canonical 3DGS .ply")
    ap.add_argument("--cond", required=True, help="SVD conditioning image (rest pose)")
    ap.add_argument("--config", required=True, help="scene/camera + train yaml")
    ap.add_argument("--guidance_config", default="./config/guidance/svd_guidance.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--r2_dest", default=None,
                    help="rclone dest (e.g. r2:storage/result/anchorflow/ball) — "
                         "outputs are auto-synced here at each checkpoint + at end")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--white_bg", type=bool, default=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda"
    gh = git_hash()

    cfg = OmegaConf.load(args.config)
    preprocessing_params, camera_params = cfg.preprocessing, cfg.camera

    # ---- load canonical Gaussians, mask, normalize (as DreamPhysics) ------ #
    gaussians = load_canonical(args.model_path)
    pipe = Pipe()
    background = torch.tensor([1., 1, 1] if args.white_bg else [0, 0, 0],
                              device=device)
    params = load_params_from_gs(gaussians, pipe)
    pos, cov = params["pos"], params["cov3D_precomp"]
    # we optimise only {GNN, z_i, anchor params}; the canonical Gaussians are
    # frozen -> detach their opacity/SH so they stay out of the autograd graph
    screen_pts = params["screen_points"]
    opacity, shs = params["opacity"].detach(), params["shs"].detach()
    keep = opacity[:, 0] > preprocessing_params["opacity_threshold"]
    pos, cov, opacity, screen_pts, shs = pos[keep], cov[keep], opacity[keep], \
        screen_pts[keep], shs[keep]
    # subsample dense assets (TRELLIS outputs ~1M gaussians) so the 14-frame
    # render graph fits in VRAM under retain_graph
    maxg = int(cfg.train.get("max_gaussians", 0))
    if maxg and pos.shape[0] > maxg:
        sel = torch.randperm(pos.shape[0], device=pos.device)[:maxg]
        pos, cov, opacity, screen_pts, shs = pos[sel], cov[sel], opacity[sel], \
            screen_pts[sel], shs[sel]
        print(f"[subsample] {keep.sum().item()} -> {maxg} gaussians")

    rot_mats = generate_rotation_matrices(
        torch.tensor(preprocessing_params["rotation_degree"]),
        preprocessing_params["rotation_axis"])
    pos = apply_rotations(pos, rot_mats)
    pos, scale_origin, mean_pos = transform2origin(pos)
    pos = shift2center111(pos)                              # normalized frame
    cov = apply_cov_rotations(cov, rot_mats)
    cov = scale_origin * scale_origin * cov
    canon_xyz = pos.detach().to(device)                    # [N,3] canonical (norm)
    canon_cov6 = cov.detach().to(device)                   # [N,6]

    # ---- camera (fixed view for SVD conditioning consistency) ------------- #
    center = torch.tensor(camera_params["mpm_space_viewpoint_center"]).reshape(1, 3).to(device)
    up = torch.tensor(camera_params["mpm_space_vertical_upward_axis"]).reshape(1, 3).to(device)
    view_center, observ = get_center_view_worldspace_and_observant_coordinate(
        center, up, rot_mats, scale_origin, mean_pos)

    # ---- anchors + GNN + actuation latents -------------------------------- #
    anchors, _ = AnchorSet.from_gaussians(
        canon_xyz, node_num=cfg.train.node_num, latent_dim=cfg.train.latent_dim,
        K=cfg.train.K, seed=cfg.train.seed)
    anchors = anchors.to(device)
    conn_idx, conn_w = R.connectivity(anchors.canonical, K=cfg.train.arap_k)
    fixed = torch.zeros(anchors.num, dtype=torch.bool, device=device)

    # ---- user-specifiable initial conditions (anchor position + velocity) --- #
    # config `train.init` (all optional; coords in the normalized frame):
    #   velocity:   [vx,vy,vz]   global initial velocity (per-step displacement)
    #   pos_offset: [ox,oy,oz]   global initial position offset
    #   regions:    [{box:[x0,x1,y0,y1,z0,z1], velocity:[...], pos_offset:[...]}]
    #   learnable:  bool         optimise the init offset/velocity under SDS too
    M, c = anchors.num, anchors.canonical
    init_offset = torch.zeros(M, 3, device=device)
    init_vel = torch.zeros(M, 3, device=device)
    icfg = dict(cfg.train.get("init", {}) or {})

    def _put(sel, spec):
        if spec.get("pos_offset") is not None:
            init_offset[sel] = torch.tensor(list(spec["pos_offset"]), device=device).float()
        if spec.get("velocity") is not None:
            init_vel[sel] = torch.tensor(list(spec["velocity"]), device=device).float()

    _put(slice(None), icfg)                                 # global
    for rgn in (icfg.get("regions") or []):                 # per-region overrides
        b = rgn["box"]
        sel = ((c[:, 0] >= b[0]) & (c[:, 0] <= b[1]) & (c[:, 1] >= b[2]) &
               (c[:, 1] <= b[3]) & (c[:, 2] >= b[4]) & (c[:, 2] <= b[5]))
        _put(sel, dict(rgn))
    init_params = []
    if icfg.get("learnable", False):
        init_offset = torch.nn.Parameter(init_offset)
        init_vel = torch.nn.Parameter(init_vel)
        init_params = [init_offset, init_vel]
    print(f"[init] |offset|={float(init_offset.detach().norm()):.3f} "
          f"|vel|={float(init_vel.detach().norm()):.3f} learnable={bool(init_params)}")

    gnn = GNSDynamics(hidden=cfg.train.hidden,
                      message_passing_steps=cfg.train.mp_steps,
                      latent_dim=cfg.train.latent_dim).to(device)
    if cfg.train.get("compile", False):      # off by default: its long CPU compile
        try:                                  # phase looks idle to the cost watchdog
            gnn = torch.compile(gnn, dynamic=True)
        except Exception as e:
            print(f"[warn] torch.compile off: {e}")

    opt = torch.optim.Adam(
        [{"params": gnn.parameters(), "lr": cfg.train.lr_gnn},
         {"params": anchors.parameters(), "lr": cfg.train.lr_anchor}]
        + ([{"params": init_params, "lr": cfg.train.lr_anchor}] if init_params else []))

    guidance = SVDGuidance(OmegaConf.load(args.guidance_config).guidance)
    pl = getattr(guidance, "pipe", None)                # quality-neutral SVD mem savings
    if pl is not None:
        try:
            pl.vae.enable_tiling(); pl.vae.enable_slicing()
        except Exception:
            pass
        try:
            pl.unet.enable_xformers_memory_efficient_attention()
        except Exception:
            try:
                pl.unet.set_attention_slice(1)
            except Exception:
                pass
    cond_image = Image.open(args.cond).convert("RGB")   # drop alpha (TRELLIS PNGs are RGBA)

    ckpt = CheckpointManager(args.out)
    start = 0
    if args.resume:
        st = ckpt.load(map_location=device)
        if st is not None:
            (gnn._orig_mod if hasattr(gnn, "_orig_mod") else gnn).load_state_dict(st["gnn"])
            anchors.load_state_dict(st["anchors"])
            opt.load_state_dict(st["opt"])
            load_rng_state(st.get("rng"))
            if st.get("init") and init_params:              # learnable init state
                with torch.no_grad():
                    init_offset.copy_(st["init"]["offset"].to(device))
                    init_vel.copy_(st["init"]["vel"].to(device))
            start = st["step"] + 1
            print(f"[resume] from step {start}")

    def collect():
        return {"gnn": (gnn._orig_mod if hasattr(gnn, "_orig_mod") else gnn).state_dict(),
                "anchors": anchors.state_dict(), "opt": opt.state_dict(),
                "init": {"offset": init_offset.detach().cpu(), "vel": init_vel.detach().cpu()}}

    def sync_r2():
        if args.r2_dest:
            os.system(f"rclone copy {args.out} {args.r2_dest} >/dev/null 2>&1")

    T = cfg.train.frames
    graph_cfg = {"graph": "knn", "k": cfg.train.K, "rebuild_graph": False}
    best = float("inf")

    def render_frame(node_now, t_int, w_bind, idx_bind):
        """Deform Gaussians by the anchors at this frame and rasterize -> [3,H,W].
        Wrapped in gradient checkpointing at the call site so only one frame's
        render graph is alive at a time (fits the full ~1.18M-gaussian asset in 24GB)."""
        R_k = W.anchor_rotations(anchors.canonical, node_now)
        p, c6, _ = W.lbs_warp(canon_xyz, canon_cov6, w_bind, idx_bind,
                              anchors.canonical, node_now, R_k)
        p_r = apply_inverse_rotations(
            undotransform2origin(undoshift2center111(p), scale_origin, mean_pos), rot_mats)
        c_r = apply_inverse_cov_rotations(c6 / (scale_origin * scale_origin), rot_mats)
        cam = get_camera_view(
            args.model_path, default_camera_index=camera_params["default_camera_index"],
            center_view_world_space=view_center, observant_coordinates=observ,
            show_hint=camera_params["show_hint"], init_azimuthm=camera_params["init_azimuthm"],
            init_elevation=camera_params["init_elevation"], init_radius=camera_params["init_radius"],
            move_camera=False, current_frame=int(t_int),
            delta_a=camera_params.get("delta_a", 0.0), delta_e=camera_params.get("delta_e", 0.0),
            delta_r=camera_params.get("delta_r", 0.0))
        rast = initialize_resterize(cam, gaussians, pipe, background)
        colors = convert_SH(shs, cam, gaussians, p_r, None)
        m2d = torch.zeros_like(p_r, requires_grad=True)
        img, _ = rast(means3D=p_r, means2D=m2d, shs=None, colors_precomp=colors,
                      opacities=opacity, scales=None, rotations=None, cov3D_precomp=c_r)
        return img
    ckpt.install_signal_handler(lambda: ckpt.save(step, collect(), rolling=False))
    print(f"[start] N={canon_xyz.shape[0]} anchors={anchors.num} T={T} commit={gh}")

    for step in range(start, cfg.train.steps):
        opt.zero_grad()
        # recompute RBF binding each step (weights depend on learnable radius/
        # node_weight) so the graph is fresh — no stale-graph reuse across steps
        w_bind, idx_bind = anchors.cal_nn_weight(canon_xyz)
        # autoregressive rollout of anchor state from rest, driven by z_i
        p0 = anchors.canonical + init_offset                # user/learned initial state
        node_seq = rollout(gnn, p0, p0 + init_vel, fixed,
                           steps=T - 2, cfg=graph_cfg, z=anchors.z, grad=True,
                           recenter=True)                  # [T,M,3], drift-free
        # gradient-checkpoint each frame's render -> only one frame's graph is
        # resident at a time, so the full (un-subsampled) asset fits in 24GB.
        img_list = torch.stack([
            checkpoint(render_frame, node_seq[t], t, w_bind, idx_bind, use_reentrant=False)
            for t in range(T)])                            # [T,3,H,W]

        out = guidance(img_list, cond_image, num_frames=T)
        loss = sum(v for k, v in out.items() if k.startswith("loss_")) * cfg.train.lambda_sds
        loss = loss + R.total(node_seq, conn_idx, conn_w, lambdas=tuple(cfg.train.reg))
        if not torch.isfinite(loss):
            print(f"[{step}] non-finite loss — skipping step")
            continue
        loss.backward(retain_graph=True)
        all_params = [p for grp in opt.param_groups for p in grp["params"]]
        torch.nn.utils.clip_grad_norm_(all_params, cfg.train.grad_clip)
        opt.step()

        lv = float(loss.item())
        if step % cfg.train.log_every == 0:
            print(f"[{step}] loss={lv:.4e}")
        if step % cfg.train.ckpt_every == 0:
            ckpt.save(step, collect(), metric=lv, is_best=(lv < best))
            best = min(best, lv)
            sync_r2()                            # crash-safe: push to R2 each ckpt

    ckpt.save(cfg.train.steps - 1, collect(), metric=best, is_best=False)
    print(f"[done] commit={gh} best={best:.4e} -> {args.out}")

    # --- render the learned self-actuated rollout to a video --------------- #
    import cv2
    from utils.save_video import save_video
    rf = cfg.train.get("render_frames", T)      # can exceed T (autonomous extrapolation)
    w_bind, idx_bind = anchors.cal_nn_weight(canon_xyz)
    with torch.no_grad():
        p0 = anchors.canonical + init_offset
        node_seq = rollout(gnn, p0, p0 + init_vel, fixed,
                           steps=rf - 2, cfg=graph_cfg, z=anchors.z, grad=False,
                           recenter=True)
        for t in range(rf):
            img = render_frame(node_seq[t], t, w_bind, idx_bind)
            arr = (255 * img.permute(1, 2, 0).clamp(0, 1).cpu().numpy()[..., ::-1]).astype("uint8")
            cv2.imwrite(os.path.join(args.out, f"{t:04d}.png".rjust(8, "0")), arr)
    save_video(args.out, os.path.join(args.out, "rollout.mp4"))
    print(f"[video] wrote rollout.mp4 ({rf} frames) -> {args.out}")
    sync_r2()
    if args.r2_dest:
        print(f"[r2] synced -> {args.r2_dest}")


if __name__ == "__main__":
    main()

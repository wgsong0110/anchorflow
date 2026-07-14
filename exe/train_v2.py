#!/usr/bin/env python
"""anchorflow v2 training — MoSca-grounded GNN⊗SSM simulator.

Stage 1 (supervised): fit the MoSca scaffold trajectory (one initial condition) —
    grounds the dynamics, no collapse.
Stage 2 (MDS refine): randomise the control (z_i + init conditions) and score
    rendered rollouts with the SVD video prior — generalise over initial conditions
    into a reusable, controllable simulator.

Runs in the anchorflow image (rasterizer + SVD), consuming MoSca's exported
{node_traj.npy [T,M,3], canonical.ply}. Time is frame-indexed -> dt = 1.

    python exe/train_v2.py --node_traj mosca_out/node_traj.npy \
        --canonical mosca_out/canonical.ply --cond subject.png \
        --config cfg/v2.yaml --out out --r2_dest r2:.../v2
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint
from omegaconf import OmegaConf
from PIL import Image

sys.path.append("gaussian-splatting")
from utils.render_utils import load_params_from_gs, initialize_resterize, convert_SH
from utils.transformation_utils import (
    transform2origin, shift2center111, undotransform2origin, undoshift2center111,
    generate_rotation_matrices, apply_rotations, apply_cov_rotations,
    apply_inverse_rotations, apply_inverse_cov_rotations,
    get_center_view_worldspace_and_observant_coordinate)
from utils.camera_view_utils import get_camera_view
from scene.gaussian_model import GaussianModel
from video_distillation.svd_guidance import SVDGuidance

from anchorflow.anchors import AnchorSet
from anchorflow.ssm_dynamics import SSMDynamics, ssm_rollout
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


def load_ply(path, sh=0):
    import math
    from plyfile import PlyData
    names = [p.name for p in PlyData.read(path)["vertex"].properties
             if p.name.startswith("f_rest_")]
    sh = int(math.sqrt((len(names) + 3) // 3)) - 1 if names else 0
    g = GaussianModel(sh)
    g.load_ply(path)
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--node_traj", required=True, help="MoSca node_traj.npy [T,M,3]")
    ap.add_argument("--canonical", required=True, help="MoSca canonical.ply")
    ap.add_argument("--cond", required=True, help="SVD cond image (subject)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--guidance_config", default="./config/guidance/svd_guidance.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--r2_dest", default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda"
    gh = git_hash()
    cfg = OmegaConf.load(args.config)
    dt = float(cfg.train.get("dt", 1.0))                    # MoSca = unit frame dt

    # ---- load canonical Gaussians + normalize ----------------------------- #
    gaussians = load_ply(args.canonical)
    pipe = Pipe()
    background = torch.tensor([1., 1, 1], device=device)
    params = load_params_from_gs(gaussians, pipe)
    pos, cov = params["pos"], params["cov3D_precomp"]
    screen, opacity, shs = params["screen_points"], params["opacity"].detach(), params["shs"].detach()
    keep = params["opacity"][:, 0] > cfg.preprocessing.get("opacity_threshold", 0.02)
    pos, cov, opacity, shs = pos[keep], cov[keep], opacity[keep], shs[keep]

    rot_mats = generate_rotation_matrices(
        torch.tensor(cfg.preprocessing.get("rotation_degree", [0])),
        cfg.preprocessing.get("rotation_axis", [0]))
    pos = apply_rotations(pos, rot_mats)
    pos, scale_origin, mean_pos = transform2origin(pos)
    pos = shift2center111(pos)
    cov = apply_cov_rotations(cov, rot_mats)
    cov = scale_origin * scale_origin * cov
    canon_xyz = pos.detach().to(device)
    canon_cov6 = cov.detach().to(device)

    # ---- MoSca node trajectory -> SAME normalized frame ------------------- #
    node_traj = torch.tensor(np.load(args.node_traj), dtype=torch.float32, device=device)  # [T,M,3]
    node_traj = apply_rotations(node_traj.reshape(-1, 3), rot_mats).reshape(node_traj.shape)
    node_traj = shift2center111((node_traj - mean_pos) / scale_origin)   # match transform2origin
    T_traj = node_traj.shape[0]

    # ---- anchors from MoSca canonical (frame 0), + GNN⊗SSM ---------------- #
    anchors = AnchorSet.from_trajectory(
        node_traj[0], latent_dim=cfg.train.latent_dim, e_dim=cfg.train.e_dim,
        K=cfg.train.K).to(device)
    w_bind, idx_bind = anchors.cal_nn_weight(canon_xyz)
    conn_idx, conn_w = R.connectivity(anchors.canonical, K=cfg.train.arap_k)
    fixed = torch.zeros(anchors.num, dtype=torch.bool, device=device)
    zero = torch.zeros(anchors.num, 3, device=device)

    model = SSMDynamics(hidden=cfg.train.hidden, mp_steps=cfg.train.mp_steps,
                        ssm_dim=cfg.train.ssm_dim, e_dim=cfg.train.e_dim,
                        z_dim=cfg.train.latent_dim,
                        accel_scale=cfg.train.get("accel_scale", 0.1)).to(device)
    opt = torch.optim.Adam(
        [{"params": model.parameters(), "lr": cfg.train.lr_gnn},
         {"params": anchors.parameters(), "lr": cfg.train.lr_anchor}])
    graph_cfg = {"graph": "knn", "k": cfg.train.K, "rebuild_graph": False}
    ckpt = CheckpointManager(args.out)

    def sync_r2():
        if args.r2_dest:
            os.system(f"rclone copy {args.out} {args.r2_dest} >/dev/null 2>&1")

    # ================= Stage 1: supervised on MoSca trajectory ============= #
    print(f"[v2] N={canon_xyz.shape[0]} anchors={anchors.num} T={T_traj} dt={dt} commit={gh}")
    p0 = node_traj[0]
    v0 = node_traj[1] - node_traj[0]                        # measured initial velocity
    for step in range(cfg.train.sup_steps):
        opt.zero_grad()
        seq = ssm_rollout(model, p0, v0, anchors.e, anchors.z, zero, zero,
                          steps=T_traj - 1, cfg=graph_cfg, dt=dt, grad=True,
                          recenter=False)                   # match actual trajectory
        loss = ((seq - node_traj) ** 2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for g in opt.param_groups for p in g["params"]], cfg.train.grad_clip)
        opt.step()
        if step % cfg.train.log_every == 0:
            print(f"[sup {step}] traj_mse={float(loss):.4e}")
    ckpt.save(0, {"model": model.state_dict(), "anchors": anchors.state_dict(),
                  "opt": opt.state_dict(), "stage": "sup"}, rolling=False)
    sync_r2()
    print("[v2] supervised pretrain done")

    # ================= Stage 2: MDS refine (IC generalization) ============ #
    center = torch.tensor(cfg.camera.mpm_space_viewpoint_center).reshape(1, 3).to(device)
    up = torch.tensor(cfg.camera.mpm_space_vertical_upward_axis).reshape(1, 3).to(device)
    view_center, observ = get_center_view_worldspace_and_observant_coordinate(
        center, up, rot_mats, scale_origin, mean_pos)
    guidance = SVDGuidance(OmegaConf.load(args.guidance_config).guidance)
    cond_image = Image.open(args.cond).convert("RGB")
    Tf = cfg.train.frames
    cam_p = cfg.camera

    def render_frame(node_now, t_int):
        Rk = W.anchor_rotations(anchors.canonical, node_now)
        p, c6, _ = W.lbs_warp(canon_xyz, canon_cov6, w_bind, idx_bind,
                              anchors.canonical, node_now, Rk)
        p_r = apply_inverse_rotations(
            undotransform2origin(undoshift2center111(p), scale_origin, mean_pos), rot_mats)
        c_r = apply_inverse_cov_rotations(c6 / (scale_origin * scale_origin), rot_mats)
        cam = get_camera_view(args.canonical, default_camera_index=cam_p.get("default_camera_index", -1),
                              center_view_world_space=view_center, observant_coordinates=observ,
                              show_hint=cam_p.get("show_hint", False),
                              init_azimuthm=cam_p.init_azimuthm, init_elevation=cam_p.init_elevation,
                              init_radius=cam_p.init_radius, move_camera=False, current_frame=int(t_int),
                              delta_a=cam_p.get("delta_a", 0.0), delta_e=cam_p.get("delta_e", 0.0),
                              delta_r=cam_p.get("delta_r", 0.0))
        rast = initialize_resterize(cam, gaussians, pipe, background)
        colors = convert_SH(shs, cam, gaussians, p_r, None)
        m2d = torch.zeros_like(p_r, requires_grad=True)
        img, _ = rast(means3D=p_r, means2D=m2d, shs=None, colors_precomp=colors,
                      opacities=opacity, scales=None, rotations=None, cov3D_precomp=c_r)
        return img

    for step in range(cfg.train.mds_steps):
        opt.zero_grad()
        # sample an initial condition: randomize control z + init velocity
        z = anchors.z + cfg.train.get("z_jitter", 0.1) * torch.randn_like(anchors.z)
        ivel = cfg.train.get("init_vel_std", 0.02) * torch.randn(anchors.num, 3, device=device)
        seq = ssm_rollout(model, anchors.canonical, ivel, anchors.e, z, ivel, zero,
                          steps=Tf - 1, cfg=graph_cfg, dt=dt, grad=True, recenter=True)
        img_list = torch.stack([
            checkpoint(render_frame, seq[t], t, use_reentrant=False) for t in range(Tf)])
        out = guidance(img_list, cond_image, num_frames=Tf)
        loss = sum(v for k, v in out.items() if k.startswith("loss_")) * cfg.train.lambda_sds
        loss = loss + R.total(seq, conn_idx, conn_w, lambdas=tuple(cfg.train.reg))
        if not torch.isfinite(loss):
            print(f"[mds {step}] non-finite, skip"); continue
        loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(
            [p for g in opt.param_groups for p in g["params"]], cfg.train.grad_clip)
        opt.step()
        if step % cfg.train.log_every == 0:
            print(f"[mds {step}] loss={float(loss):.4e}")
        if step % cfg.train.ckpt_every == 0:
            ckpt.save(step, {"model": model.state_dict(), "anchors": anchors.state_dict(),
                             "opt": opt.state_dict(), "stage": "mds"}, rolling=False)
            sync_r2()
    print(f"[v2] done commit={gh} -> {args.out}")
    sync_r2()


if __name__ == "__main__":
    main()

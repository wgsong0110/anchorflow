#!/usr/bin/env python
"""Render the trained v2 GNN⊗SSM **self-actuated** rollout to a video.

Loads a train_v2 checkpoint (model + anchors [+ comp]) and rolls the learned
dynamics out from the canonical rest pose driven by the learned actuation latent
z_i — no MoSca trajectory, no MDS: the reusable simulator running on its own.
Renders each frame through the SAME LBS-warp + rasterizer path as train_v2 so the
result is directly comparable. Runs in the anchorflow image.

    python exe/render_v2.py --canonical mosca_out/canonical.ply \
        --node_traj mosca_out/node_traj.npy --model_dir <dir with cameras.json> \
        --ckpt out_v2/ckpt_last.pt --config cfg/v2.yaml --out out_v2/render --frames 25
"""

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import imageio
from omegaconf import OmegaConf
from PIL import Image

sys.path.append("/workspace/SC-GS")
from utils.render_utils import load_params_from_gs, initialize_resterize, convert_SH
from utils.transformation_utils import (
    transform2origin, shift2center111, undotransform2origin, undoshift2center111,
    generate_rotation_matrices, apply_rotations, apply_cov_rotations,
    apply_inverse_rotations, apply_inverse_cov_rotations,
    get_center_view_worldspace_and_observant_coordinate)
from utils.camera_view_utils import get_camera_view
from scene.gaussian_model import GaussianModel

from anchorflow.anchors import AnchorSet
from anchorflow.ssm_dynamics import SSMDynamics, ssm_rollout
from anchorflow import warp as W


class Pipe:
    convert_SHs_python = False
    compute_cov3D_python = True
    debug = False


def load_ply(path):
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
    ap.add_argument("--canonical", required=True)
    ap.add_argument("--node_traj", required=True, help="only for anchor count/init")
    ap.add_argument("--model_dir", required=True, help="dir with cameras.json")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--damping", type=float, default=1.0, help="velocity friction γ (<1 bounds rollout)")
    ap.add_argument("--orbit", action="store_true", help="orbit camera during rollout")
    ap.add_argument("--static", action="store_true", help="diagnostic: render canonical (no rollout)")
    ap.add_argument("--radius", type=float, default=None,
                    help="absolute camera radius in ORIGINAL frame (overrides cfg/auto)")
    ap.add_argument("--no_warp", action="store_true",
                    help="diagnostic: render raw canonical (skip LBS warp)")
    ap.add_argument("--cov_scale", type=float, default=1.0,
                    help="diagnostic: multiply rendered covariance (probe gaussian size)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda"
    torch.manual_seed(args.seed)
    cfg = OmegaConf.load(args.config)
    dt = float(cfg.train.get("dt", 1.0))

    # ---- canonical Gaussians + normalize (identical to train_v2) ---------- #
    gaussians = load_ply(args.canonical)
    pipe = Pipe()
    background = torch.tensor([1., 1, 1], device=device)
    params = load_params_from_gs(gaussians, pipe)
    pos, cov = params["pos"], params["cov3D_precomp"]
    opacity, shs = params["opacity"].detach(), params["shs"].detach()
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

    node_traj = torch.tensor(np.load(args.node_traj), dtype=torch.float32, device=device)
    node_traj = apply_rotations(node_traj.reshape(-1, 3), rot_mats).reshape(node_traj.shape)
    node_traj = shift2center111((node_traj - mean_pos) / scale_origin).detach()

    # ---- anchors + model, load checkpoint --------------------------------- #
    anchors = AnchorSet.from_trajectory(
        node_traj[0], latent_dim=cfg.train.latent_dim, e_dim=cfg.train.e_dim,
        K=cfg.train.K).to(device)
    w_bind, idx_bind = anchors.cal_nn_weight(canon_xyz)
    model = SSMDynamics(hidden=cfg.train.hidden, mp_steps=cfg.train.mp_steps,
                        ssm_dim=cfg.train.ssm_dim, e_dim=cfg.train.e_dim,
                        z_dim=cfg.train.latent_dim,
                        accel_scale=cfg.train.get("accel_scale", 0.1)).to(device)
    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model"]); anchors.load_state_dict(state["anchors"])
    model.eval()
    print(f"[render_v2] loaded {args.ckpt} stage={state.get('stage')} step={state.get('step')} "
          f"N={canon_xyz.shape[0]} anchors={anchors.num}")

    graph_cfg = {"graph": "knn", "k": cfg.train.K, "rebuild_graph": False}
    zero = torch.zeros(anchors.num, 3, device=device)

    # ---- self-actuated rollout: rest + learned z, no external drive ------- #
    with torch.no_grad():
        if args.static:                                    # diagnostic: no motion
            seq = anchors.canonical[None].expand(args.frames, -1, -1)
        else:
            seq = ssm_rollout(model, anchors.canonical, zero, anchors.e, anchors.z,
                              zero, zero, steps=args.frames - 1, cfg=graph_cfg, dt=dt,
                              grad=False, recenter=True, damping=args.damping)

    # ---- camera (identical setup to train_v2) ----------------------------- #
    center = torch.tensor(cfg.camera.mpm_space_viewpoint_center).reshape(1, 3).to(device)
    up = torch.tensor(cfg.camera.mpm_space_vertical_upward_axis).reshape(1, 3).to(device)
    view_center, observ = get_center_view_worldspace_and_observant_coordinate(
        center, up, rot_mats, scale_origin, mean_pos)
    cam_p = cfg.camera
    # The scene is rendered in the ORIGINAL (reconstruction) frame, but cfg radius
    # is calibrated for the size-1.0 normalized object. Scale it by the object's
    # true size (1/scale_origin) so the camera sits at the right distance for any
    # reconstruction. (DreamPhysics assets had scale_origin~1, so this was a no-op.)
    render_radius = (args.radius if args.radius is not None
                     else float(cam_p.init_radius) / float(scale_origin))
    print(f"[render_v2] scale_origin={float(scale_origin):.3f} render_radius={render_radius:.3f}")

    frames = []
    for t in range(args.frames):
        node_now = seq[t]
        with torch.no_grad():
            if args.no_warp:
                p, c6 = canon_xyz, canon_cov6
            else:
                Rk = W.anchor_rotations(anchors.canonical, node_now)
                p, c6, _ = W.lbs_warp(canon_xyz, canon_cov6, w_bind, idx_bind,
                                      anchors.canonical, node_now, Rk)
            p_r = apply_inverse_rotations(
                undotransform2origin(undoshift2center111(p), scale_origin, mean_pos), rot_mats)
            c_r = apply_inverse_cov_rotations(c6 / (scale_origin * scale_origin), rot_mats)
            c_r = c_r * args.cov_scale
            da = cam_p.get("delta_a", 0.0) + (t * 2.0 if args.orbit else 0.0)
            cam = get_camera_view(
                args.model_dir, default_camera_index=cam_p.get("default_camera_index", -1),
                center_view_world_space=view_center, observant_coordinates=observ,
                show_hint=cam_p.get("show_hint", False),
                init_azimuthm=cam_p.init_azimuthm, init_elevation=cam_p.init_elevation,
                init_radius=render_radius, move_camera=False, current_frame=int(t),
                delta_a=da, delta_e=cam_p.get("delta_e", 0.0), delta_r=cam_p.get("delta_r", 0.0))
            rast = initialize_resterize(cam, gaussians, pipe, background)
            colors = convert_SH(shs, cam, gaussians, p_r, None)
            m2d = torch.zeros_like(p_r)
            img, _ = rast(means3D=p_r, means2D=m2d, shs=None, colors_precomp=colors,
                          opacities=opacity, scales=None, rotations=None, cov3D_precomp=c_r)
        arr = (img.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        frames.append(arr)
        Image.fromarray(arr).save(os.path.join(args.out, f"{t:05d}.png"))
    imageio.mimsave(os.path.join(args.out, "rollout.mp4"), frames, fps=8, quality=8)
    print(f"[render_v2] wrote {args.frames} frames + rollout.mp4 -> {args.out}")


if __name__ == "__main__":
    main()

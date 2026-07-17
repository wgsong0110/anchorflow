#!/usr/bin/env python
"""AnchorFlow: semantic anchor nodes + GNN⊗SSM dynamics on a canonical 3DGS.

Deformation = OUR method (unchanged):
    tokens_to_nodes -> AnchorSet -> SSMDynamics/ssm_rollout -> lbs_warp
    GNN (spatial) ⊗ per-node diagonal SSM (temporal) -> acceleration,
    explicit integration  p' = p + v·dt,  v' = γ(v + a·dt).

Node selection + learnable-parameter update follow "From Tokens to Nodes"
(arXiv:2510.02732):
    - semantic / dynamic-tendency node allocation (tokens_to_nodes)
    - RBF binding with LEARNABLE node radii:
        w_ij = exp(-|x_j - c_i|^2 / (2*rho_i^2)) / sum_k(...)        [AnchorSet]
    - Gaussian attributes optimised alongside the anchors (--lr_gaussian)
    - ARAP regularisation

Canonical asset + camera + renderer are the official INRIA release
(point_cloud.ply + bundled cameras.json + gaussian_renderer.render, SH3,
background per cfg_args). No hand-tuned camera parameters.

Supervision (--sup):
    mds   : SVD Motion Distillation Sampling (DreamPhysics)  [diffusion prior]
    video : per-view generated clips, direct photometric      [paper-style]

    python exe/train_anchorflow.py --model /workspace/gs_official/kitchen \
        --cfg cfg/anchorflow_kitchen.yaml --out /workspace/af_mds --sup mds
"""
from __future__ import annotations

import argparse, json, math, os, random, subprocess, sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import imageio.v2 as iio
from torch.utils.checkpoint import checkpoint
from omegaconf import OmegaConf

sys.path.append("/workspace/gaussian-splatting")
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, focal2fov

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
from anchorflow.anchors import AnchorSet
from anchorflow.ssm_dynamics import SSMDynamics, ssm_rollout
from anchorflow.graph import knn_graph
from anchorflow import warp as W
from anchorflow.checkpoint import CheckpointManager, load_rng_state


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
    compute_cov3D_python = True      # we supply the warped covariance
    debug = False
    antialiasing = False


def load_official_cameras(model_dir, n_views, long_side):
    cams_json = json.load(open(f"{model_dir}/cameras.json"))
    idx = np.linspace(0, len(cams_json) - 1, n_views).round().astype(int)
    cams = []
    for i in idx:
        c = cams_json[int(i)]
        rot = np.array(c["rotation"], dtype=np.float32)
        pos = np.array(c["position"], dtype=np.float32)
        Wd, Hd = c["width"], c["height"]
        fovx, fovy = focal2fov(c["fx"], Wd), focal2fov(c["fy"], Hd)
        s = long_side / max(Wd, Hd)
        W8 = max(8, int(round(Wd * s / 8)) * 8)
        H8 = max(8, int(round(Hd * s / 8)) * 8)
        cams.append(Cam(rot, -rot.T @ pos, fovx, fovy, W8, H8))
    print(f"[train] cameras={len(cams)} (official cameras.json) "
          f"{cams[0].image_width}x{cams[0].image_height}")
    return cams


def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "nogit"


def save_video(frames, path, fps=8):
    arr = [(f.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
           for f in frames]
    iio.mimsave(path, arr, fps=fps, quality=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--iter", type=int, default=30000)
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sup", choices=["mds", "video"], default="mds")
    ap.add_argument("--videos", default=None,
                    help="--sup video: dir with view_XX.mp4 target clips")
    ap.add_argument("--r2", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--white_bg", action="store_true")
    ap.add_argument("--no-t2n", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    cfg = OmegaConf.load(args.cfg)
    dev, gh = "cuda", git_hash()
    T = cfg.model.n_frames

    # ── official pretrained scene ────────────────────────────────────────────
    g = GaussianModel(3)
    g.load_ply(f"{args.model}/point_cloud/iteration_{args.iter}/point_cloud.ply")
    g.active_sh_degree = 3
    canon_xyz = g.get_xyz.detach().clone()
    G = canon_xyz.shape[0]
    print(f"[train] gaussians={G}  commit={gh}  sup={args.sup}")

    canon_cov6 = W.cov_from_scale_rot(g.get_scaling.detach(),
                                      g._rotation.detach()).detach()

    bg = torch.tensor([1., 1., 1.] if args.white_bg else [0., 0., 0.], device=dev)
    cameras = load_official_cameras(args.model, cfg.train.n_views, cfg.model.res)
    V = len(cameras)

    def render_with(cam, xyz, cov6):
        g._xyz = xyz
        g.get_covariance = lambda scaling_modifier=1.0: cov6
        return render(cam, g, Pipe(), bg)["render"]

    def render_canonical(cam):
        with torch.no_grad():
            return render_with(cam, canon_xyz, canon_cov6).clamp(0, 1)

    img0 = render_canonical(cameras[0])
    cover = float((img0.max(0).values > 0.01).float().mean())
    print(f"[train] cam[0] coverage={cover*100:.1f}%  mean={float(img0.mean()):.3f}")
    if cover < 0.02:
        sys.exit("[train] ABORT: cameras do not see the scene")

    _q = canon_xyz.float()
    extent = float((torch.quantile(_q, 0.99, dim=0)
                    - torch.quantile(_q, 0.01, dim=0)).norm())
    del _q
    print(f"[train] scene extent={extent:.2f}")

    # ── anchor nodes: paper's semantic / dynamic-tendency allocation ────────
    node_pos = None
    if not args.no_t2n:
        try:
            from anchorflow.tokens_to_nodes import tokens_to_nodes
            import anchorflow.tokens_to_nodes as t2n_mod
            print("[train] tokens_to_nodes (semantic + dynamic tendency) ...")
            node_pos = tokens_to_nodes(
                canon_xyz, g.get_opacity.detach(), render_canonical,
                cameras[:cfg.get("t2n_views", 4)],
                n_nodes=cfg.model.n_nodes, device=dev)
            if t2n_mod._dino_model is not None:
                del t2n_mod._dino_model
                t2n_mod._dino_model = None
            import gc; gc.collect(); torch.cuda.empty_cache()
            print("[train] DINOv2 freed")
        except Exception as e:
            print(f"[train] tokens_to_nodes failed ({e}) -> FPS")
            node_pos = None
            import gc; gc.collect(); torch.cuda.empty_cache()

    # ── AnchorSet: learnable rho (paper), node_weight, z (actuation), e (id) ─
    z_dim = int(cfg.model.get("z_dim", 8))
    e_dim = int(cfg.model.get("e_dim", 8))
    kG = int(cfg.model.k_gauss)
    if node_pos is not None:
        anchors = AnchorSet.from_trajectory(node_pos, latent_dim=z_dim,
                                            e_dim=e_dim, K=kG).to(dev)
    else:
        anchors, _ = AnchorSet.from_gaussians(canon_xyz, node_num=cfg.model.n_nodes,
                                              latent_dim=z_dim, e_dim=e_dim, K=kG)
        anchors = anchors.to(dev)
    M = anchors.num
    print(f"[train] anchors={M}  z_dim={z_dim} e_dim={e_dim} k_gauss={kG}")

    # ── OUR dynamics: GNN ⊗ SSM -> accel -> explicit integration ────────────
    dt = float(cfg.model.get("dt", 0.1))
    accel_scale = float(cfg.model.get("accel_scale", 0.01)) * extent
    model = SSMDynamics(hidden=cfg.model.hidden,
                        mp_steps=int(cfg.model.get("mp_steps", cfg.model.n_gnn_layers)),
                        ssm_dim=int(cfg.model.get("ssm_dim", cfg.model.hidden)),
                        e_dim=e_dim, z_dim=z_dim,
                        accel_scale=accel_scale).to(dev)
    graph_cfg = {"graph": "knn", "k": int(cfg.model.k_node)}
    damping = float(cfg.train.get("damping", 1.0))
    print(f"[train] SSMDynamics dt={dt} accel_scale={accel_scale:.4f} damping={damping}")

    # z = actuation, varied per initial condition (ssm_dynamics docstring)
    B = int(cfg.train.z0_bank_size)
    v0_motion = float(cfg.train.get("z0_motion", 0.01))
    v0_std = v0_motion * extent / (dt * max(T - 1, 1))
    z_bank = torch.nn.Parameter(0.01 * torch.randn(B, M, z_dim, device=dev))
    v0_bank = torch.nn.Parameter(torch.randn(B, M, 3, device=dev) * v0_std)
    print(f"[train] z_bank {list(z_bank.shape)}  v0_std={v0_std:.4f}")

    # ── learnable params (paper optimises gaussians + anchors) ──────────────
    for p in (g._features_dc, g._features_rest, g._opacity, g._scaling,
              g._rotation, g._xyz):
        p.requires_grad_(False)
    lr_g = float(cfg.train.get("lr_gaussian", 0.0))
    groups = [
        {"params": list(model.parameters()), "lr": float(cfg.train.lr_gnn)},
        {"params": list(anchors.parameters()),
         "lr": float(cfg.train.get("lr_anchor", 1e-3))},
        {"params": [z_bank, v0_bank], "lr": float(cfg.train.lr_z0)},
    ]
    if lr_g > 0:
        gp = [g._features_dc, g._features_rest, g._opacity, g._scaling, g._rotation]
        for p in gp:
            p.requires_grad_(True)
        groups.append({"params": gp, "lr": lr_g})
        print(f"[train] gaussian attrs optimised (lr={lr_g})")
    else:
        print("[train] gaussian attrs frozen (lr_gaussian=0)")
    opt = torch.optim.Adam(groups)

    # ── supervision ─────────────────────────────────────────────────────────
    svd = cond_cache = gt_videos = None
    frame0_cache = [render_canonical(c) for c in cameras]
    if args.sup == "mds":
        from anchorflow.sds import SVDGuidance
        print("[train] loading SVD for MDS ...")
        svd = SVDGuidance(sigma_min=cfg.mds.sigma_min, sigma_max=cfg.mds.sigma_max,
                          guidance_scale=cfg.mds.guidance_scale,
                          motion_bucket_id=cfg.mds.motion_bucket_id,
                          grad_clip=cfg.mds.grad_clip, device=dev)
        cond_cache = [svd.precompute_cond(f0, T) for f0 in frame0_cache]
        torch.cuda.empty_cache()
    else:
        if not args.videos:
            sys.exit("[train] --sup video requires --videos DIR")
        gt_videos = []
        for v in range(V):
            p = os.path.join(args.videos, f"view_{v:02d}.mp4")
            fr = [torch.from_numpy(np.asarray(f)).permute(2, 0, 1).float().cuda() / 255.
                  for f in iio.mimread(p, memtest=False)[:T]]
            clip = torch.stack(fr, 0)
            if clip.shape[-2:] != (cameras[v].image_height, cameras[v].image_width):
                clip = torch.nn.functional.interpolate(
                    clip, size=(cameras[v].image_height, cameras[v].image_width),
                    mode="bilinear", align_corners=False)
            gt_videos.append(clip)
        print(f"[train] target clips: {len(gt_videos)} x {tuple(gt_videos[0].shape)}")

    ckpt_mgr = CheckpointManager(args.out)
    start = 0
    if args.resume:
        ck = ckpt_mgr.load()
        if ck is not None:
            model.load_state_dict(ck["model"])
            anchors.load_state_dict(ck["anchors"])
            opt.load_state_dict(ck["opt"])
            z_bank.data.copy_(ck["z_bank"]); v0_bank.data.copy_(ck["v0_bank"])
            load_rng_state(ck.get("rng"))
            start = ck["step"] + 1
            print(f"[train] resumed from step {start}")

    torch.set_float32_matmul_precision("high")

    def sync_r2():
        if args.r2:
            os.system(f"rclone copy {args.out} {args.r2} >/dev/null 2>&1")

    def rollout_positions(k, grad=True):
        p0, v0 = anchors.canonical, v0_bank[k]
        return ssm_rollout(model, p0, v0, anchors.e, z_bank[k],
                           init_vel=v0, init_pos=p0, steps=T - 1,
                           cfg=graph_cfg, dt=dt, grad=grad, damping=damping)

    arap_edge = knn_graph(anchors.canonical.detach(), k=min(6, M - 1))
    rng = random.Random(42)

    for step in range(start, cfg.train.iters):
        k = rng.randint(0, B - 1)
        v = rng.randint(0, V - 1)
        cam = cameras[v]

        p_seq = rollout_positions(k)                       # [T, M, 3]
        # rho is learnable -> recompute the binding every step
        w_b, idx_b = anchors.cal_nn_weight(canon_xyz)

        frames = []
        for t in range(T):
            def _f(pt, wb=w_b, ib=idx_b):
                pos, cov6, _ = W.lbs_warp(canon_xyz, canon_cov6, wb, ib,
                                          anchors.canonical, pt)
                return render_with(cam, pos, cov6)
            frames.append(checkpoint(_f, p_seq[t], use_reentrant=False))
        frames_t = torch.stack(frames, 0)                  # [T,3,H,W]

        opt.zero_grad(set_to_none=True)
        if args.sup == "mds":
            loss = svd.mds_loss(frames_t, cond_image=frame0_cache[v],
                                w_power=cfg.mds.w_power, cond_cache=cond_cache[v],
                                vae_checkpoint=False)
        else:
            loss = float(cfg.train.get("lambda_rgb", 1.0)) * \
                (frames_t - gt_videos[v]).abs().mean()

        if cfg.train.lambda_arap > 0:
            t_r = rng.randint(1, T - 1)
            src, dst = arap_edge
            d_rest = (anchors.canonical[src] - anchors.canonical[dst]).norm(dim=-1)
            d_now = (p_seq[t_r][src] - p_seq[t_r][dst]).norm(dim=-1)
            loss = loss + cfg.train.lambda_arap * ((d_now - d_rest) ** 2).mean()
        if cfg.train.lambda_z0 > 0:
            loss = loss + cfg.train.lambda_z0 * (z_bank ** 2).mean()

        if not torch.isfinite(loss):
            print(f"[{step}] non-finite loss, skip")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for gr in opt.param_groups for p in gr["params"]],
            cfg.train.grad_clip)
        opt.step()

        if step % cfg.train.log_every == 0:
            with torch.no_grad():
                travel = float((p_seq[-1] - anchors.canonical).norm(dim=-1).max())
                rho = float(anchors.radius.mean())
            print(f"[{step}/{cfg.train.iters}] loss={float(loss):.4f} k={k} v={v} "
                  f"travel={travel:.3f} ({travel/extent*100:.2f}%) rho={rho:.3f}")

        if step % cfg.train.ckpt_every == 0:
            ckpt_mgr.save(step, {"model": model.state_dict(),
                                 "anchors": anchors.state_dict(),
                                 "opt": opt.state_dict(), "z_bank": z_bank.data,
                                 "v0_bank": v0_bank.data, "step": step})
            _save_rollout(step, rollout_positions, anchors, canon_xyz, canon_cov6,
                          render_with, cameras[0], T, args.out)
            sync_r2()

    ckpt_mgr.save(cfg.train.iters - 1,
                  {"model": model.state_dict(), "anchors": anchors.state_dict(),
                   "opt": opt.state_dict(), "z_bank": z_bank.data,
                   "v0_bank": v0_bank.data, "step": cfg.train.iters - 1})
    sync_r2()
    print(f"[train] done commit={gh} -> {args.out}")


@torch.no_grad()
def _save_rollout(step, rollout_positions, anchors, canon_xyz, canon_cov6,
                  render_with, cam, T, out):
    p_seq = rollout_positions(0, grad=False)
    w_b, idx_b = anchors.cal_nn_weight(canon_xyz)
    frames = []
    for t in range(T):
        pos, cov6, _ = W.lbs_warp(canon_xyz, canon_cov6, w_b, idx_b,
                                  anchors.canonical, p_seq[t])
        frames.append(render_with(cam, pos, cov6).clamp(0, 1))
    path = os.path.join(out, f"rollout_step{step:06d}.mp4")
    save_video(frames, path)
    print(f"  saved rollout -> {path}")


if __name__ == "__main__":
    main()

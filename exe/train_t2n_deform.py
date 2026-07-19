#!/usr/bin/env python
"""From Tokens to Nodes deformation baseline.

Ablation: replaces our GNN⊗SSM dynamics with DIRECT per-frame anchor position
optimisation (no dynamics model, no temporal structure). This is the deformation
approach the T2N paper uses when supervised by video:

    p_traj  [T, M, 3]  --  learnable per-frame anchor positions

Loss: photometric L1 vs target video + ARAP between consecutive frames.
Supervision: same SVD-generated multi-view clips as our run B (fair comparison).

    python exe/train_t2n_deform.py \
        --model /workspace/gs_official/kitchen \
        --cfg cfg/anchorflow_kitchen.yaml \
        --videos /workspace/af_videos \
        --out /workspace/af_t2n \
        --iters 10000 --r2 r2:storage/result/anchorflow/af_t2n
"""
from __future__ import annotations

import argparse, json, math, os, random, subprocess, sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import imageio.v2 as iio
from torch.utils.checkpoint import checkpoint
from omegaconf import OmegaConf

sys.path.append("/workspace/SC-GS")
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render as _render_scgs

def render(cam, g, pipe, bg):
    zeros = torch.zeros_like(g.get_xyz)
    return _render_scgs(cam, g, pipe, bg, d_xyz=zeros, d_rotation=0.0, d_scaling=zeros)

from utils.graphics_utils import getWorld2View2, getProjectionMatrix, focal2fov

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
from anchorflow.anchors import AnchorSet
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
    compute_cov3D_python = True
    debug = False
    antialiasing = False


def load_official_cameras(model_dir, n_views, long_side):
    cams_json = json.load(open(f"{model_dir}/cameras.json"))
    idx = np.linspace(0, len(cams_json) - 1, n_views).round().astype(int)
    cams = []
    for i in idx:
        c = cams_json[int(i)]
        if "rotation" in c:
            # INRIA format: rotation=R_wc, position=camera_center, fx, fy, width, height
            rot = np.array(c["rotation"], dtype=np.float32)
            pos = np.array(c["position"], dtype=np.float32)
            Wd, Hd = c["width"], c["height"]
            fovx, fovy = focal2fov(c["fx"], Wd), focal2fov(c["fy"], Hd)
            T = -rot.T @ pos  # t_cw
        else:
            # gen_views.py format: R=R_wc, T=t_cw, fov_x, fov_y, W, H
            rot = np.array(c["R"], dtype=np.float32)
            T = np.array(c["T"], dtype=np.float32)
            Wd, Hd = c["W"], c["H"]
            fovx, fovy = c["fov_x"], c["fov_y"]
        s = long_side / max(Wd, Hd)
        W8 = max(8, int(round(Wd * s / 8)) * 8)
        H8 = max(8, int(round(Hd * s / 8)) * 8)
        cams.append(Cam(rot, T, fovx, fovy, W8, H8))
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",  required=True)
    ap.add_argument("--cfg",    required=True)
    ap.add_argument("--videos", required=True, help="dir with view_XX.mp4 target clips")
    ap.add_argument("--out",    required=True)
    ap.add_argument("--iter",   type=int, default=30000)
    ap.add_argument("--iters",  type=int, default=None)
    ap.add_argument("--n_views", type=int, default=None,
                    help="override cfg.train.n_views")
    ap.add_argument("--r2",     default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--white_bg", action="store_true")
    ap.add_argument("--no-t2n", action="store_true")
    ap.add_argument("--eval_views", default=None,
                    help="comma-separated camera indices for PSNR eval (e.g. '5,6')")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    cfg = OmegaConf.load(args.cfg)
    if args.iters is not None:
        cfg.train.iters = args.iters
    if args.n_views is not None:
        cfg.train.n_views = args.n_views
    dev, gh = "cuda", git_hash()
    T = cfg.model.n_frames

    g = GaussianModel(3)
    g.load_ply(f"{args.model}/point_cloud/iteration_{args.iter}/point_cloud.ply")
    g.active_sh_degree = 3
    canon_xyz = g.get_xyz.detach().clone()
    G = canon_xyz.shape[0]
    print(f"[t2n] gaussians={G}  commit={gh}")

    canon_cov6 = W.cov_from_scale_rot(g.get_scaling.detach(),
                                      g._rotation.detach()).detach()

    for p in (g._features_dc, g._features_rest, g._opacity,
              g._scaling, g._rotation, g._xyz):
        p.requires_grad_(False)

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

    # ── anchor nodes ──────────────────────────────────────────────────────────
    node_pos = None
    if not args.no_t2n:
        try:
            from anchorflow.tokens_to_nodes import tokens_to_nodes
            import anchorflow.tokens_to_nodes as t2n_mod
            print("[t2n] tokens_to_nodes ...")
            node_pos = tokens_to_nodes(
                canon_xyz, g.get_opacity.detach(), render_canonical,
                cameras[:cfg.get("t2n_views", 4)],
                n_nodes=cfg.model.n_nodes, device=dev)
            if t2n_mod._dino_model is not None:
                del t2n_mod._dino_model; t2n_mod._dino_model = None
            import gc; gc.collect(); torch.cuda.empty_cache()
        except Exception as e:
            print(f"[t2n] tokens_to_nodes failed ({e}) -> FPS")
            node_pos = None

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
    print(f"[t2n] anchors={M}")

    # ── T2N DEFORMATION: direct per-frame anchor positions ───────────────────
    # p_traj[T, M, 3] — directly learnable, no dynamics model
    # Initialised at canonical; the model must LEARN motion from photometric loss.
    p_traj = torch.nn.Parameter(
        anchors.canonical.detach().unsqueeze(0).expand(T, -1, -1).clone())
    print(f"[t2n] p_traj {list(p_traj.shape)}  (direct per-frame optimisation)")

    # ── supervision ──────────────────────────────────────────────────────────
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
    print(f"[t2n] target clips: {len(gt_videos)} x {tuple(gt_videos[0].shape)}")

    # ── optimiser ────────────────────────────────────────────────────────────
    opt = torch.optim.Adam(
        [{"params": list(anchors.parameters()), "lr": float(cfg.train.get("lr_anchor", 1e-3))},
         {"params": [p_traj],                   "lr": float(cfg.train.lr_z0)}])

    arap_edge = knn_graph(anchors.canonical.detach(), k=min(6, M - 1))

    ckpt_mgr = CheckpointManager(args.out)
    start = 0
    if args.resume:
        ck = ckpt_mgr.load()
        if ck is not None:
            anchors.load_state_dict(ck["anchors"])
            opt.load_state_dict(ck["opt"])
            p_traj.data.copy_(ck["p_traj"])
            load_rng_state(ck.get("rng"))
            start = ck["step"] + 1
            print(f"[t2n] resumed from step {start}")

    def sync_r2():
        if args.r2:
            os.system(f"rclone copy {args.out} {args.r2} >/dev/null 2>&1")

    rng = random.Random(42)
    torch.set_float32_matmul_precision("high")

    _q = canon_xyz.float()
    extent = float((torch.quantile(_q, 0.99, dim=0)
                    - torch.quantile(_q, 0.01, dim=0)).norm())
    del _q

    for step in range(start, cfg.train.iters):
        t = rng.randint(0, T - 1)
        v = rng.randint(0, V - 1)
        cam = cameras[v]

        w_b, idx_b = anchors.cal_nn_weight(canon_xyz)

        def _f(pt):
            pos, cov6, _ = W.lbs_warp(canon_xyz, canon_cov6, w_b, idx_b,
                                       anchors.canonical, pt)
            return render_with(cam, pos, cov6)

        rendered = checkpoint(_f, p_traj[t], use_reentrant=False)
        loss = float(cfg.train.get("lambda_rgb", 1.0)) * \
               (rendered - gt_videos[v][t]).abs().mean()

        # ARAP between consecutive frames (temporal smoothness)
        if cfg.train.lambda_arap > 0 and t > 0:
            src, dst = arap_edge
            d_prev = (p_traj[t-1][src] - p_traj[t-1][dst]).norm(dim=-1)
            d_now  = (p_traj[t][src]   - p_traj[t][dst]).norm(dim=-1)
            loss = loss + cfg.train.lambda_arap * ((d_now - d_prev) ** 2).mean()

        opt.zero_grad(set_to_none=True)
        if not torch.isfinite(loss):
            print(f"[{step}] non-finite loss, skip"); continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for g in opt.param_groups for p in g["params"]],
            cfg.train.grad_clip)
        opt.step()

        if step % cfg.train.log_every == 0:
            with torch.no_grad():
                travel = float((p_traj[-1] - p_traj[0]).norm(dim=-1).max())
                rho = float(anchors.radius.mean())
            print(f"[{step}/{cfg.train.iters}] loss={float(loss):.4f} t={t} v={v} "
                  f"travel={travel:.3f} ({travel/extent*100:.2f}%) rho={rho:.3f}")

        if step % cfg.train.ckpt_every == 0:
            ckpt_mgr.save(step, {"anchors": anchors.state_dict(),
                                 "opt": opt.state_dict(), "p_traj": p_traj.data,
                                 "step": step})
            _save_rollout(step, p_traj, anchors, canon_xyz, canon_cov6,
                          render_with, cameras[0], T, args.out)
            sync_r2()

    ckpt_mgr.save(cfg.train.iters - 1,
                  {"anchors": anchors.state_dict(), "opt": opt.state_dict(),
                   "p_traj": p_traj.data, "step": cfg.train.iters - 1})
    sync_r2()
    print(f"[t2n] done commit={gh} -> {args.out}")

    # ── eval on hold-out cameras (e.g. cam5, cam6 per MoDGS protocol) ─────────
    if args.eval_views:
        eval_idxs = [int(x) for x in args.eval_views.split(",")]
        cams_json = json.load(open(f"{args.model}/cameras.json"))
        long_side = cfg.model.res
        eval_psnrs = {}
        for ei in eval_idxs:
            c = cams_json[ei]
            if "rotation" in c:
                rot = np.array(c["rotation"], dtype=np.float32)
                pos = np.array(c["position"], dtype=np.float32)
                Wd, Hd = c["width"], c["height"]
                fovx, fovy = focal2fov(c["fx"], Wd), focal2fov(c["fy"], Hd)
                T_cam = -rot.T @ pos
            else:
                rot = np.array(c["R"], dtype=np.float32)
                T_cam = np.array(c["T"], dtype=np.float32)
                Wd, Hd = c["W"], c["H"]
                fovx, fovy = c["fov_x"], c["fov_y"]
            s = long_side / max(Wd, Hd)
            W8 = max(8, int(round(Wd * s / 8)) * 8)
            H8 = max(8, int(round(Hd * s / 8)) * 8)
            eval_cam = Cam(rot, T_cam, fovx, fovy, W8, H8)

            # load GT video for this camera (cam05.mp4 / view_05.mp4 etc.)
            gt_path = os.path.join(args.videos, f"view_{ei:02d}.mp4")
            if not os.path.exists(gt_path):
                print(f"[eval] GT not found: {gt_path}, skip")
                continue
            # load + resize on CPU to avoid OOM (raw frames can be 2704x2028)
            gt_frames_cpu = []
            for f in iio.mimread(gt_path, memtest=False)[:T]:
                fr = torch.from_numpy(np.asarray(f)).permute(2, 0, 1).float() / 255.
                if fr.shape[-2:] != (H8, W8):
                    fr = torch.nn.functional.interpolate(
                        fr.unsqueeze(0), size=(H8, W8), mode="bilinear", align_corners=False
                    ).squeeze(0)
                gt_frames_cpu.append(fr)

            with torch.no_grad():
                w_b, idx_b = anchors.cal_nn_weight(canon_xyz)
                psnrs = []
                for t in range(min(T, len(gt_frames_cpu))):
                    pos_t, cov6_t, _ = W.lbs_warp(
                        canon_xyz, canon_cov6, w_b, idx_b, anchors.canonical, p_traj[t])
                    pred = render_with(eval_cam, pos_t, cov6_t).clamp(0, 1)
                    gt_t = gt_frames_cpu[t].cuda()
                    mse = (pred - gt_t).pow(2).mean()
                    if mse > 0:
                        psnrs.append(-10 * torch.log10(mse).item())
            avg = float(np.mean(psnrs)) if psnrs else 0.
            eval_psnrs[f"cam{ei:02d}"] = avg
            print(f"[eval] cam{ei:02d}: PSNR={avg:.2f} dB  ({len(psnrs)} frames)")
            _save_rollout(cfg.train.iters - 1, p_traj, anchors, canon_xyz, canon_cov6,
                          lambda cam, xyz, cov6: render_with(eval_cam, xyz, cov6),
                          eval_cam, T, args.out)
            os.rename(
                os.path.join(args.out, f"rollout_step{cfg.train.iters-1:06d}.mp4"),
                os.path.join(args.out, f"rollout_eval_cam{ei:02d}.mp4"))

        if eval_psnrs:
            mean_psnr = float(np.mean(list(eval_psnrs.values())))
            print(f"[eval] mean PSNR ({','.join(eval_psnrs.keys())}): {mean_psnr:.2f} dB")
        sync_r2()


@torch.no_grad()
def _save_rollout(step, p_traj, anchors, canon_xyz, canon_cov6,
                  render_with, cam, T, out):
    w_b, idx_b = anchors.cal_nn_weight(canon_xyz)
    frames = []
    for t in range(T):
        pos, cov6, _ = W.lbs_warp(canon_xyz, canon_cov6, w_b, idx_b,
                                   anchors.canonical, p_traj[t])
        frames.append(render_with(cam, pos, cov6).clamp(0, 1))
    path = os.path.join(out, f"rollout_step{step:06d}.mp4")
    iio.mimsave(path, [(f.permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
                       for f in frames], fps=8, quality=8)
    print(f"  saved rollout -> {path}")


if __name__ == "__main__":
    main()

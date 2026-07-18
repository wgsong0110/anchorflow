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

import argparse, glob, json, math, os, random, subprocess, sys

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
from anchorflow.ssm_dynamics import SSMDynamics, ssm_rollout, ssm_rollout_from
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
    print(f"[train] cameras={len(cams)} (official cameras.json) "
          f"{cams[0].image_width}x{cams[0].image_height}")
    return cams


def load_cam_by_idx(cam_data, long_side):
    if "rotation" in cam_data:
        rot = np.array(cam_data["rotation"], dtype=np.float32)
        pos = np.array(cam_data["position"], dtype=np.float32)
        Wd, Hd = cam_data["width"], cam_data["height"]
        fovx, fovy = focal2fov(cam_data["fx"], Wd), focal2fov(cam_data["fy"], Hd)
        T_vec = -rot.T @ pos
    else:
        rot = np.array(cam_data["R"], dtype=np.float32)
        T_vec = np.array(cam_data["T"], dtype=np.float32)
        Wd, Hd = cam_data["W"], cam_data["H"]
        fovx, fovy = cam_data["fov_x"], cam_data["fov_y"]
    s = long_side / max(Wd, Hd)
    W8 = max(8, int(round(Wd * s / 8)) * 8)
    H8 = max(8, int(round(Hd * s / 8)) * 8)
    return Cam(rot, T_vec, fovx, fovy, W8, H8)


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
    ap.add_argument("--sup", choices=["mds", "video", "frames"], default="mds")
    ap.add_argument("--videos", default=None,
                    help="--sup video: dir with view_XX.mp4 target clips")
    ap.add_argument("--frames", default=None,
                    help="--sup frames: dir with 000000.jpg ... (real N3DV cam0)")
    ap.add_argument("--cameras", default=None,
                    help="cameras.json path (default: model/cameras.json)")
    ap.add_argument("--cam_idx", type=int, default=0,
                    help="camera index for --sup frames training cam")
    ap.add_argument("--eval_frames", default=None,
                    help="comma-sep frame dirs for PSNR eval (e.g. .../cam05,.../cam06)")
    ap.add_argument("--eval_cam_idxs", default=None,
                    help="comma-sep camera indices matching eval_frames (e.g. 5,6)")
    ap.add_argument("--eval_max_frames", type=int, default=None,
                    help="evaluate only the first N frames (default: all T frames)")
    ap.add_argument("--eval_only", action="store_true",
                    help="skip training, load ckpt_last.pt and run eval only")
    ap.add_argument("--r2", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--white_bg", action="store_true")
    ap.add_argument("--no-t2n", action="store_true")
    ap.add_argument("--n_views", type=int, default=None,
                        help="override cfg.train.n_views (e.g. 1 for single-view)")
    ap.add_argument("--iters", type=int, default=None,
                    help="override cfg.train.iters (fit a wall-clock budget)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    cfg = OmegaConf.load(args.cfg)
    if args.iters is not None:
        cfg.train.iters = args.iters
    if args.n_views is not None:
        cfg.train.n_views = args.n_views
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
    gt_frames_cpu = None
    train_cam_single = None
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
    elif args.sup == "video":
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
    else:  # frames
        if not args.frames:
            sys.exit("[train] --sup frames requires --frames DIR")
        cam_json_path = args.cameras or f"{args.model}/cameras.json"
        cams_json_all = json.load(open(cam_json_path))
        train_cam_single = load_cam_by_idx(cams_json_all[args.cam_idx], cfg.model.res)
        flist = sorted(glob.glob(os.path.join(args.frames, "*.jpg")) +
                       glob.glob(os.path.join(args.frames, "*.png")))
        gt_frames_cpu = []
        for fpath in flist[:T]:
            img = iio.imread(fpath)
            img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.
            if img_t.shape[1:] != (train_cam_single.image_height, train_cam_single.image_width):
                img_t = torch.nn.functional.interpolate(
                    img_t.unsqueeze(0),
                    size=(train_cam_single.image_height, train_cam_single.image_width),
                    mode="bilinear", align_corners=False).squeeze(0)
            gt_frames_cpu.append(img_t)
        print(f"[train] frames: cam_idx={args.cam_idx} "
              f"{train_cam_single.image_width}x{train_cam_single.image_height} "
              f"n={len(gt_frames_cpu)}")

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

    def rollout_positions(k, steps=None, bptt_start=0, grad=True, return_states=False):
        p0, v0 = anchors.canonical, v0_bank[k]
        s = (T - 1) if steps is None else steps
        return ssm_rollout(model, p0, v0, anchors.e, z_bank[k],
                           init_vel=v0, init_pos=p0, steps=s,
                           bptt_start=bptt_start,
                           cfg=graph_cfg, dt=dt, grad=grad, damping=damping,
                           return_states=return_states)

    # ── state cache: avoid no-grad prefix rollout cost every step ────────────
    # _state_cache[k][t] = (p_t, v_t, h_t) on CPU tensors, refreshed every cache_every steps
    cache_every = int(cfg.train.get("cache_every", 200))
    _state_cache = [[None] * T for _ in range(B)]
    _cache_valid = [False]          # mutable one-element list so closure can mutate it

    def _refresh_state_cache():
        with torch.no_grad():
            for k_c in range(B):
                p0 = anchors.canonical.detach()
                v0 = v0_bank[k_c].detach()
                _, states = ssm_rollout(
                    model, p0, v0, anchors.e, z_bank[k_c].detach(),
                    init_vel=v0, init_pos=p0, steps=T - 1,
                    cfg=graph_cfg, dt=dt, grad=False, damping=damping,
                    return_states=True)
                _state_cache[k_c] = states      # list of T+1 (p_cpu, v_cpu, h_cpu)
        _cache_valid[0] = True
        print(f"[cache] refreshed B={B} x T={T}")

    # curriculum: bptt window grows from bptt_w0 to T-1 over bptt_warmup_frac of training
    bptt_w0   = int(cfg.train.get("bptt_window_start", 50))
    bptt_w1   = int(cfg.train.get("bptt_window_end",   T - 1))
    bptt_frac = float(cfg.train.get("bptt_warmup_frac", 0.8))

    def current_bptt_window(step):
        r = min(step / max(cfg.train.iters * bptt_frac, 1), 1.0)
        return int(bptt_w0 + (bptt_w1 - bptt_w0) * r)

    arap_edge = knn_graph(anchors.canonical.detach(), k=min(6, M - 1))
    rng = random.Random(42)

    rollout_cam0 = train_cam_single if args.sup == "frames" else cameras[0]

    if args.eval_only:
        if args.eval_frames and args.eval_cam_idxs:
            eval_T = args.eval_max_frames if args.eval_max_frames else T
            _do_eval(args, cfg, rollout_positions, anchors, canon_xyz, canon_cov6,
                     render_with, eval_T, dev, args.out, gh)
        else:
            print("[eval_only] --eval_frames and --eval_cam_idxs required")
        return

    for step in range(start, cfg.train.iters):
        k = rng.randint(0, B - 1)
        opt.zero_grad(set_to_none=True)

        if args.sup == "frames":
            # refresh state cache periodically
            if step % cache_every == 0 or not _cache_valid[0]:
                _refresh_state_cache()

            Tf = len(gt_frames_cpu)
            window = current_bptt_window(step)
            a = rng.randint(0, max(0, Tf - 1 - window))
            b = min(a + window, Tf - 1)
            cam = train_cam_single

            if a == 0 or _state_cache[k][a] is None:
                # no-grad prefix + grad suffix via bptt_start
                p_seq = rollout_positions(k, steps=b, bptt_start=a, grad=True)
                p_b = p_seq[b]
            else:
                # start from cached state at t=a (detached), window steps with grad
                p_a_cpu, v_a_cpu, h_a_cpu = _state_cache[k][a]
                p_seq = ssm_rollout_from(
                    model, p_a_cpu.to(dev), v_a_cpu.to(dev), h_a_cpu.to(dev),
                    anchors.e, z_bank[k], steps=b - a,
                    cfg=graph_cfg, dt=dt, grad=True, damping=damping)
                p_b = p_seq[-1]

            w_b, idx_b = anchors.cal_nn_weight(canon_xyz)
            def _f(pt, wb=w_b, ib=idx_b):
                pos, cov6, _ = W.lbs_warp(canon_xyz, canon_cov6, wb, ib,
                                          anchors.canonical, pt)
                return render_with(cam, pos, cov6)
            frame_pred = checkpoint(_f, p_b, use_reentrant=False)
            gt_f = gt_frames_cpu[b].to(dev)
            loss = float(cfg.train.get("lambda_rgb", 1.0)) * (frame_pred - gt_f).abs().mean()
            t_r = b
            arap_pt = p_b
        else:
            v = rng.randint(0, V - 1)
            cam = cameras[v]
            t_r = v
            p_seq = rollout_positions(k)                   # [T, M, 3]
            w_b, idx_b = anchors.cal_nn_weight(canon_xyz)
            frames = []
            for t in range(T):
                def _f(pt, wb=w_b, ib=idx_b):
                    pos, cov6, _ = W.lbs_warp(canon_xyz, canon_cov6, wb, ib,
                                              anchors.canonical, pt)
                    return render_with(cam, pos, cov6)
                frames.append(checkpoint(_f, p_seq[t], use_reentrant=False))
            frames_t = torch.stack(frames, 0)              # [T,3,H,W]
            if args.sup == "mds":
                loss = svd.mds_loss(frames_t, cond_image=frame0_cache[v],
                                    w_power=cfg.mds.w_power, cond_cache=cond_cache[v],
                                    vae_checkpoint=False)
            else:  # video
                loss = float(cfg.train.get("lambda_rgb", 1.0)) * \
                    (frames_t - gt_videos[v]).abs().mean()
            arap_pt = p_seq[rng.randint(1, T - 1)]

        if cfg.train.lambda_arap > 0:
            src, dst = arap_edge
            d_rest = (anchors.canonical[src] - anchors.canonical[dst]).norm(dim=-1)
            d_now = (arap_pt[src] - arap_pt[dst]).norm(dim=-1)
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
                p_last = p_seq[-1]
                travel = float((p_last - anchors.canonical).norm(dim=-1).max())
                rho = float(anchors.radius.mean())
            print(f"[{step}/{cfg.train.iters}] loss={float(loss):.4f} k={k} v={t_r} "
                  f"travel={travel:.3f} ({travel/extent*100:.2f}%) rho={rho:.3f}")

        if step % cfg.train.ckpt_every == 0:
            ckpt_mgr.save(step, {"model": model.state_dict(),
                                 "anchors": anchors.state_dict(),
                                 "opt": opt.state_dict(), "z_bank": z_bank.data,
                                 "v0_bank": v0_bank.data, "step": step})
            _save_rollout(step, rollout_positions, anchors, canon_xyz, canon_cov6,
                          render_with, rollout_cam0, T, args.out)
            sync_r2()

    ckpt_mgr.save(cfg.train.iters - 1,
                  {"model": model.state_dict(), "anchors": anchors.state_dict(),
                   "opt": opt.state_dict(), "z_bank": z_bank.data,
                   "v0_bank": v0_bank.data, "step": cfg.train.iters - 1})
    sync_r2()

    if args.eval_frames and args.eval_cam_idxs:
        eval_T = args.eval_max_frames if args.eval_max_frames else T
        _do_eval(args, cfg, rollout_positions, anchors, canon_xyz, canon_cov6,
                 render_with, eval_T, dev, args.out, gh)

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


@torch.no_grad()
def _do_eval(args, cfg, rollout_positions, anchors, canon_xyz, canon_cov6,
             render_with, T, dev, out, gh):
    cam_json_path = args.cameras or f"{args.model}/cameras.json"
    cams_json_all = json.load(open(cam_json_path))
    eval_dirs = [d.strip() for d in args.eval_frames.split(",")]
    eval_idxs = [int(i.strip()) for i in args.eval_cam_idxs.split(",")]

    p_seq = rollout_positions(0, grad=False)               # [T, M, 3]
    w_b, idx_b = anchors.cal_nn_weight(canon_xyz)

    psnrs = []
    for cam_idx, fdir in zip(eval_idxs, eval_dirs):
        cam = load_cam_by_idx(cams_json_all[cam_idx], cfg.model.res)
        flist = sorted(glob.glob(os.path.join(fdir, "*.jpg")) +
                       glob.glob(os.path.join(fdir, "*.png")))
        mse_sum = 0.0
        cnt = 0
        for t, fpath in enumerate(flist[:T]):
            gt = iio.imread(fpath)
            gt_t = torch.from_numpy(gt).permute(2, 0, 1).float().to(dev) / 255.
            if gt_t.shape[1:] != (cam.image_height, cam.image_width):
                gt_t = torch.nn.functional.interpolate(
                    gt_t.unsqueeze(0), size=(cam.image_height, cam.image_width),
                    mode="bilinear", align_corners=False).squeeze(0)
            pos, cov6, _ = W.lbs_warp(canon_xyz, canon_cov6, w_b, idx_b,
                                      anchors.canonical, p_seq[t])
            pred = render_with(cam, pos, cov6).clamp(0, 1)
            mse_sum += (pred - gt_t).pow(2).mean().item()
            cnt += 1
        psnr = -10 * math.log10(max(mse_sum / max(cnt, 1), 1e-10))
        psnrs.append(psnr)
        print(f"  [eval] cam{cam_idx:02d}: PSNR={psnr:.2f} dB  ({cnt} frames)")

    mean_psnr = sum(psnrs) / len(psnrs) if psnrs else 0.0
    print(f"[eval] mean PSNR={mean_psnr:.2f} dB  commit={gh}")
    with open(os.path.join(out, "eval_psnr.txt"), "w") as f:
        for ci, p in zip(eval_idxs, psnrs):
            f.write(f"cam{ci:02d}: {p:.4f}\n")
        f.write(f"mean: {mean_psnr:.4f}\n")


if __name__ == "__main__":
    main()

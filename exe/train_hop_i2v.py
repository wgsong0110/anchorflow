#!/usr/bin/env python
"""
HopDynamics trained on REAL generated videos from SVD image-to-video (actual
sampling, not the SDS/MDS diffusion-prior loss in train_hop_mds.py), with a
per-video trainable h0 (auto-decoder style latent code) instead of a single
shared/learned or randomly-resampled one.

Why: MDS (train_hop_mds.py) only ever gives a noisy, unnormalized gradient
signal (eps_dyn - eps_stat) -- expensive (a UNet+VAE forward every step) and,
per this project's own lr sweep, unstable at anything above lr=1e-4. Instead,
actually SAMPLE N concrete videos from SVD once (real img2vid generation, one
sample per random seed), then train with a plain, cheap, stable photometric
loss (L1+SSIM) against each -- SVD's own multi-step diffusion sampling is what
supplies motion diversity/plausibility, not a score-distillation gradient.

Why one h0 per video, network shared: each generated video is a different
concrete motion realization (different seed) -- there is no single h0 that
should explain all N of them. Instead, following the auto-decoder / latent
optimization pattern (DeepSDF, GRAF, etc.), each video i gets its own free
h0_bank[i] trained ONLY against that video's photometric loss, while
HopDynamics's GNN+SSM weights and the anchors (e/radius/node_weight) are
shared and see gradients from all N videos every epoch -- so the *network*
learns a general, consistent "h0 -> plausible motion" mapping, and each
h0_bank[i] just picks out which realization.

Usage:
  python exe/train_hop_i2v.py \\
      --model_dir /workspace/gs_official/bonsai --ply_iter 30000 \\
      --out /workspace/bonsai_hop_i2v --n_videos 8 --n_frames 25 \\
      --res 256 --iters 20000
"""
from __future__ import annotations

import sys, os, gc, random, argparse, subprocess
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HOME", "/data/huggingface")

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from tqdm import trange
import imageio.v2 as iio
import numpy as np
from PIL import Image

sys.path.insert(0, "/workspace/SC-GS")
from gaussian_renderer import render as _gs_render
from utils.loss_utils import l1_loss

_exe = os.path.dirname(__file__)
sys.path.insert(0, _exe)
_lib = os.path.join(_exe, "..", "lib")
sys.path.insert(0, _lib)

from train_sc_anchorflow import (
    GaussianModel, Pipe, load_blender_dataset, lbs_d_xyz_rot_batched, ssim_loss,
)
from train_seqgen_mds import load_cameras
from anchorflow.anchors import AnchorSet
from anchorflow.ssm_dynamics import HopDynamics, build_graph
from anchorflow import warp as W


def git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_exe, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "nogit"


def render_frames_checkpointed(cam, gaussians, pipe, bg, d_xyz_all, d_rot_all,
                                T, use_checkpoint=True):
    outs = []
    for t in range(T):
        d_xyz = d_xyz_all[t]
        d_rot = d_rot_all[t]

        def _render(dx, dr, _cam=cam, _t=t):
            _cam.fid = torch.tensor([_t / max(T - 1, 1)], device=dx.device)
            pkg = _gs_render(_cam, gaussians, pipe, bg, dx, dr,
                              torch.zeros_like(dx), d_rot_as_res=True)
            return pkg["render"]

        if use_checkpoint and d_xyz.requires_grad:
            outs.append(checkpoint(_render, d_xyz, d_rot, use_reentrant=False))
        else:
            outs.append(_render(d_xyz, d_rot))
    return torch.stack(outs, dim=0).clamp(0, 1)


def generate_videos(cam0, gaussians, pipe_render, bg, n_videos, n_frames,
                     svd_model, seed_base, out_dir, dev):
    """Actual SVD img2vid sampling (not the SDS wrapper) -- N real generated
    videos from the frozen canonical's frame-0 render, one per seed. Loaded,
    used, then fully freed before training starts (training itself needs no
    diffusion model resident)."""
    from diffusers import StableVideoDiffusionPipeline

    with torch.no_grad():
        N_g = gaussians.get_xyz.shape[0]
        zeros_xyz = torch.zeros(N_g, 3, device=dev)
        zeros_rot = torch.zeros(N_g, 4, device=dev)
        cam0.fid = torch.tensor([0.0], device=dev)
        frame0 = _gs_render(cam0, gaussians, pipe_render, bg, zeros_xyz, zeros_rot,
                             zeros_xyz, d_rot_as_res=True)["render"].clamp(0, 1)
    frame0_np = (frame0.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
    frame0_pil = Image.fromarray(frame0_np).resize((1024, 576))

    print("[gen] loading SVD img2vid pipeline ...", flush=True)
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        svd_model, torch_dtype=torch.float16, variant="fp16")
    pipe.enable_model_cpu_offload()

    H, W_ = cam0.image_height, cam0.image_width
    gt_videos = []
    for i in range(n_videos):
        gen = torch.Generator("cuda").manual_seed(seed_base + i)
        frames = pipe(frame0_pil, num_frames=n_frames, decode_chunk_size=8,
                      generator=gen).frames[0]
        video_path = os.path.join(out_dir, f"gt_video_{i:02d}.mp4")
        iio.mimsave(video_path, [f.resize((W_, H)) for f in frames], fps=8, quality=8)
        print(f"[gen] video {i}/{n_videos} -> {video_path}", flush=True)
        t_stack = torch.stack([
            torch.from_numpy(np.array(f.resize((W_, H)).convert("RGB")))
                 .permute(2, 0, 1).float() / 255.0
            for f in frames
        ], dim=0).to(dev)   # [T,3,H,W]
        gt_videos.append(t_stack)

    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    print("[gen] SVD pipeline freed, GPU clear for training", flush=True)
    return gt_videos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", default=None)
    ap.add_argument("--data", default=None)
    ap.add_argument("--model_dir", default=None)
    ap.add_argument("--ply_iter", type=int, default=30000)
    ap.add_argument("--n_frames", type=int, default=25,
                    help="SVD img2vid's native default; also T for hop_direct/render")
    ap.add_argument("--n_views", type=int, default=8,
                    help="[static-scene mode] cameras loaded from cameras.json "
                         "(only cams[0] is used as the i2v conditioning view, "
                         "the rest are unused -- kept for parity with train_hop_mds.py)")
    ap.add_argument("--white_bg", action="store_true")
    ap.add_argument("--out",   required=True)
    ap.add_argument("--iters", type=int, default=20_000)
    ap.add_argument("--res",   type=int, default=256)
    ap.add_argument("--n_videos", type=int, default=8,
                    help="number of SVD-generated videos; one trainable h0 per video")
    ap.add_argument("--seed_base", type=int, default=0)
    ap.add_argument("--svd_model", default="stabilityai/stable-video-diffusion-img2vid-xt")
    ap.add_argument("--n_anchors", type=int, default=256)
    ap.add_argument("--k_gauss",   type=int, default=4)
    ap.add_argument("--hidden",    type=int, default=128)
    ap.add_argument("--mp_steps",  type=int, default=6)
    ap.add_argument("--ssm_dim",   type=int, default=128)
    ap.add_argument("--e_dim",     type=int, default=8)
    ap.add_argument("--dt",        type=float, default=0.05)
    ap.add_argument("--n_time_freqs", type=int, default=6)
    ap.add_argument("--lr_dyn",    type=float, default=1e-4)
    ap.add_argument("--lambda_ssim", type=float, default=0.2)
    ap.add_argument("--lambda_arap",   type=float, default=1e-2)
    ap.add_argument("--lambda_smooth", type=float, default=1e-2)
    ap.add_argument("--grad_clip_norm", type=float, default=1.0)
    ap.add_argument("--ckpt_every", type=int, default=1000)
    ap.add_argument("--rollout_every", type=int, default=1000)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    dev = "cuda"
    static_mode = args.model_dir is not None
    if static_mode:
        assert args.ply is None and args.data is None
    else:
        assert args.ply is not None and args.data is not None
    bg = torch.tensor([0., 0., 0.] if (static_mode and not args.white_bg) else [1., 1., 1.],
                       device=dev)
    gh = git_hash()

    if static_mode:
        cams = load_cameras(args.model_dir, args.n_views, args.res)
        ply_path = f"{args.model_dir}/point_cloud/iteration_{args.ply_iter}/point_cloud.ply"
    else:
        cams, _frames_unused = load_blender_dataset(args.data, args.res)
        del _frames_unused
        ply_path = args.ply
    T = args.n_frames

    gaussians = GaussianModel(3)
    gaussians.load_ply(ply_path)
    gaussians.active_sh_degree = 3
    for p in (gaussians._xyz, gaussians._features_dc, gaussians._features_rest,
              gaussians._scaling, gaussians._rotation, gaussians._opacity):
        p.requires_grad_(False)
    print(f"[train] gaussians={gaussians.get_xyz.shape[0]} (frozen, from {ply_path})  "
          f"T={T} n_videos={args.n_videos}  commit={gh}", flush=True)

    anchors, _ = AnchorSet.from_gaussians(
        gaussians.get_xyz.detach(), node_num=args.n_anchors,
        latent_dim=0, e_dim=args.e_dim, K=args.k_gauss)
    anchors = anchors.to(dev)
    M = anchors.num
    print(f"[train] anchors={M}", flush=True)

    with torch.no_grad():
        w, idx = anchors.cal_nn_weight(gaussians.get_xyz.detach())
    _arap_idx, _arap_src = W.anchor_rotations_cache(anchors.canonical, K=8)

    pipe = Pipe()

    # ── generate N real videos via SVD img2vid (one-time, then freed) ────────
    gt_video_path = os.path.join(args.out, "gt_video_00.mp4")
    if os.path.exists(gt_video_path):
        print("[gen] GT videos already generated, loading cached frames from disk", flush=True)
        gt_videos = []
        for i in range(args.n_videos):
            reader = iio.mimread(os.path.join(args.out, f"gt_video_{i:02d}.mp4"), memtest=False)
            t_stack = torch.stack([
                torch.from_numpy(np.array(f)).permute(2, 0, 1).float() / 255.0
                for f in reader[:T]
            ], dim=0).to(dev)
            gt_videos.append(t_stack)
    else:
        gt_videos = generate_videos(cams[0], gaussians, pipe, bg, args.n_videos, T,
                                     args.svd_model, args.seed_base, args.out, dev)

    # ── HopDynamics (fresh) + per-video trainable h0 ─────────────────────────
    model = HopDynamics(hidden=args.hidden, mp_steps=args.mp_steps, ssm_dim=args.ssm_dim,
                         e_dim=args.e_dim, n_time_freqs=args.n_time_freqs).to(dev)
    h0_bank = torch.nn.Parameter(torch.randn(args.n_videos, M, args.ssm_dim, device=dev))
    print(f"[train] HopDynamics hidden={args.hidden} mp={args.mp_steps} ssm={args.ssm_dim} "
          f"dt={args.dt}  h0_bank={tuple(h0_bank.shape)} (one trainable h0 per video, "
          f"network+anchors shared)  lambda_arap={args.lambda_arap} "
          f"lambda_smooth={args.lambda_smooth}", flush=True)

    dyn_opt = torch.optim.Adam(
        [{"params": list(model.parameters())},
         {"params": list(anchors.parameters())},
         {"params": [h0_bank]}],
        lr=args.lr_dyn)

    graph_cfg = {"graph": "knn", "k": 8}

    def hop_spatial(grad=True):
        p0 = anchors.canonical
        edge_index = build_graph(p0.detach(), graph_cfg)
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            return model.spatial_embed(anchors.e, edge_index, p0)

    def hop_direct_full(h0, spatial):
        times = list(range(1, T))
        B = len(times)
        dt = torch.tensor([t * args.dt for t in times], device=dev)
        h0b = h0.unsqueeze(0).expand(B, M, -1)
        d_p, d_rot, _ = model.hop_batch(spatial, h0b, anchors.e, dt)
        canonical = anchors.canonical
        p_seq = torch.cat([canonical.unsqueeze(0), canonical.unsqueeze(0) + d_p], dim=0)
        rot0 = torch.zeros(1, M, 4, device=dev); rot0[..., 0] = 1.0
        rot_seq = torch.cat([rot0, d_rot], dim=0)
        return p_seq, rot_seq

    # ── resume ────────────────────────────────────────────────────────────────
    start = 0
    ckpt_path = os.path.join(args.out, "ckpt_last.pt")
    if args.resume and os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=dev)
        model.load_state_dict(state["model"])
        anchors.load_state_dict(state["anchors"])
        h0_bank.data.copy_(state["h0_bank"])
        dyn_opt.load_state_dict(state["dyn_opt"])
        start = state["step"] + 1
        print(f"[train] resumed from step {start}", flush=True)
    elif args.resume:
        print("[train] --resume: no checkpoint found, starting fresh", flush=True)

    loss_csv_path = os.path.join(args.out, "losses.csv")
    if not os.path.exists(loss_csv_path):
        with open(loss_csv_path, "w") as f:
            f.write("step,total,photo,arap,smooth,video_idx\n")

    pbar = trange(start, args.iters, desc="train")
    for step in pbar:
        i = step % args.n_videos
        gt = gt_videos[i]   # [T,3,H,W]

        spatial = hop_spatial()
        p_seq, rot_seq = hop_direct_full(h0_bank[i], spatial)
        d_xyz_all, d_rot_all = lbs_d_xyz_rot_batched(w, idx, anchors.canonical, p_seq, rot_seq)
        rendered = render_frames_checkpointed(cams[0], gaussians, pipe, bg,
                                               d_xyz_all, d_rot_all, T, use_checkpoint=True)

        loss_photo = (1 - args.lambda_ssim) * l1_loss(rendered, gt) \
                   + args.lambda_ssim * ssim_loss(rendered, gt)
        total_loss = loss_photo

        loss_arap = torch.tensor(0.0, device=dev)
        if args.lambda_arap > 0:
            t_r = random.randint(1, T - 1)
            loss_arap = W.anchor_arap_loss(anchors.canonical, p_seq[t_r], K=8,
                                            _idx=_arap_idx, _src=_arap_src)
            total_loss = total_loss + args.lambda_arap * loss_arap

        loss_smooth = torch.tensor(0.0, device=dev)
        if args.lambda_smooth > 0:
            vel = p_seq[1:] - p_seq[:-1]
            acc = vel[1:] - vel[:-1]
            loss_smooth = (acc ** 2).mean()
            total_loss = total_loss + args.lambda_smooth * loss_smooth

        if not torch.isfinite(total_loss):
            print(f"[step {step}] non-finite loss, skip", flush=True)
            continue

        dyn_opt.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
        dyn_opt.step()

        pbar.set_postfix(loss=float(total_loss), photo=float(loss_photo), vid=i)
        if step % 50 == 0:
            with open(loss_csv_path, "a") as f:
                f.write(f"{step},{float(total_loss)},{float(loss_photo)},"
                        f"{float(loss_arap)},{float(loss_smooth)},{i}\n")

        if step % args.ckpt_every == 0 or step == args.iters - 1:
            torch.save({"model": model.state_dict(), "anchors": anchors.state_dict(),
                        "h0_bank": h0_bank.data, "dyn_opt": dyn_opt.state_dict(),
                        "step": step}, ckpt_path)
            print(f"[step {step+1}] saved", flush=True)
            r2_dest = f"r2:storage/result/anchorflow/{os.path.basename(args.out)}/"
            for f in (ckpt_path, loss_csv_path):
                r = subprocess.run(["rclone", "copy", f, r2_dest],
                                    check=False, capture_output=True, timeout=120, text=True)
                if r.returncode != 0:
                    print(f"[step {step+1}] WARNING: R2 backup FAILED for {f}: {r.stderr.strip()}",
                          flush=True)

        if step % args.rollout_every == 0 or step == args.iters - 1:
            with torch.no_grad():
                spatial_eval = hop_spatial(grad=False)
                p_r, rot_r = hop_direct_full(h0_bank[0], spatial_eval)
                d_xyz_r, d_rot_r = lbs_d_xyz_rot_batched(w, idx, anchors.canonical, p_r, rot_r)
                frames_r = render_frames_checkpointed(
                    cams[0], gaussians, pipe, bg, d_xyz_r, d_rot_r, T, use_checkpoint=False)
            out_frames = [(f.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8") for f in frames_r]
            video_path = os.path.join(args.out, f"rollout_video00_{step:06d}.mp4")
            iio.mimsave(video_path, out_frames, fps=8, quality=8)
            print(f"  [rollout] saved -> {video_path} (h0_bank[0], compare against gt_video_00.mp4)",
                  flush=True)

    print(f"[train] done commit={gh} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
HopDynamics trained with Motion Distillation Sampling (MDS, DreamPhysics
AAAI'25 -- arxiv 2406.01476) instead of photometric loss, with h0 (the
initial SSM hidden state) freshly SAMPLED from N(0, I) each step instead of
being a learned free parameter.

Why canonical Gaussians must come pretrained/frozen: MDS only supplies a
"does this video move plausibly" signal (dynamic-vs-static-frame diffusion
prior comparison) -- it carries no information about what the object should
look like, so canonical shape/color needs a photometric source elsewhere.
This mirrors exe/train_seqgen_mds.py, which loads a pretrained SC-GS ply for
exactly the same reason; here the ply comes from a completed
train_sc_anchorflow.py run (e.g. the non-autoregressive-only exp008 run).

Why h0 is sampled, not learned: with a single learned init_h, HopDynamics
would only ever need to reproduce ONE specific trajectory (whatever this
video's actual motion was) -- but there's no GT trajectory here to anchor it
to. Sampling h0 ~ N(0, I) fresh every step instead trains the network to map
*any* random initial latent to a plausible, temporally-smooth, ARAP-rigid
motion (its role is analogous to SeqGen's random (cond_ids, cond_vel)
conditioning in train_seqgen_mds.py, just via an abstract latent instead of
literal per-node velocities).

Two input modes:
  1. NeRF-Blender dynamic dataset ply (--ply + --data): canonical Gaussians
     from a completed train_sc_anchorflow.py run on a Blender-format dataset
     (T = that dataset's own frame count).
  2. Publicly-released static real-scene reconstruction (--model_dir): a
     pretrained INRIA 3DGS checkpoint (cameras.json + point_cloud/iteration_N/
     point_cloud.ply, e.g. mip-NeRF360's "bonsai" or "kitchen" from
     https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/pretrained/models.zip)
     -- no dynamic dataset needed at all since MDS supplies its own motion
     signal; T is a free choice (--n_frames) since the scene has no inherent
     video length.

Usage:
  python exe/train_hop_mds.py \\
      --ply /workspace/ficus_sc_af_hop_coarseonly/point_cloud_060000.ply \\
      --data /workspace/ficus_ds_wind \\
      --out  /workspace/ficus_sc_af_hop_mds \\
      --iters 20000

  python exe/train_hop_mds.py \\
      --model_dir /workspace/gs_official/bonsai --ply_iter 30000 \\
      --out /workspace/bonsai_hop_mds --n_frames 14 --n_views 8 --iters 20000
"""
from __future__ import annotations

import sys, os, math, random, argparse, subprocess
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HOME", "/data/huggingface")

import torch
from torch.utils.checkpoint import checkpoint
from tqdm import trange
import imageio.v2 as iio

sys.path.insert(0, "/workspace/SC-GS")
from gaussian_renderer import render as _gs_render

_exe = os.path.dirname(__file__)
sys.path.insert(0, _exe)
_lib = os.path.join(_exe, "..", "lib")
sys.path.insert(0, _lib)

# Reuse module-level helpers from train_sc_anchorflow.py verbatim (Cam/Pipe/
# dataset loader/LBS gather) instead of re-implementing them -- none of this
# depends on that script's main(), which never runs on import.
from train_sc_anchorflow import (
    GaussianModel, Pipe, load_blender_dataset, lbs_d_xyz_rot_batched,
)
# COLMAP/cameras.json loader for real (INRIA-pretrained) static scenes --
# also module-level in train_seqgen_mds.py, never runs its main() on import.
from train_seqgen_mds import load_cameras
from anchorflow.anchors import AnchorSet
from anchorflow.ssm_dynamics import HopDynamics, build_graph
from anchorflow import warp as W
from anchorflow.sds import SVDGuidance


def git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_exe, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "nogit"


def render_frames_checkpointed(cam, gaussians, pipe, bg, d_xyz_all, d_rot_all,
                                T, use_checkpoint=True):
    """Render T frames for one camera from per-frame LBS deltas.
    Checkpointed per-frame (like train_seqgen_mds.py's traj_to_frames) to
    bound peak VRAM -- SVD's own VAE+UNet already occupy a lot of it."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", default=None,
                    help="[Blender mode] pretrained canonical point_cloud_*.ply -- frozen, "
                         "no further optimization (MDS gives no appearance signal)")
    ap.add_argument("--data",  default=None, help="[Blender mode] NeRF-Blender dataset dir (cameras only)")
    ap.add_argument("--model_dir", default=None,
                    help="[static-scene mode] INRIA pretrained checkpoint dir "
                         "(cameras.json + point_cloud/iteration_N/point_cloud.ply), "
                         "e.g. a mip-NeRF360 scene like bonsai/kitchen -- mutually "
                         "exclusive with --ply/--data")
    ap.add_argument("--ply_iter", type=int, default=30000,
                    help="[static-scene mode] which point_cloud/iteration_N to load")
    ap.add_argument("--n_frames", type=int, default=14,
                    help="[static-scene mode] T -- free choice, the scene has no "
                         "inherent video length (matches this project's other MDS "
                         "configs, cfg/anchorflow_*.yaml)")
    ap.add_argument("--n_views", type=int, default=8,
                    help="[static-scene mode] number of cameras to sample "
                         "(evenly spaced by index) from cameras.json")
    ap.add_argument("--white_bg", action="store_true",
                    help="[static-scene mode] both bonsai/kitchen use "
                         "white_background=false (per their own cfg_args) -- default black")
    ap.add_argument("--out",   required=True)
    ap.add_argument("--iters", type=int, default=20_000)
    ap.add_argument("--res",   type=int, default=256,
                    help="MDS/SVD render resolution -- matches this project's other "
                         "MDS configs (cfg/anchorflow_*.yaml all use 256, not the "
                         "576 used for photometric training; the SVD VAE+UNet "
                         "forward over T frames at once is the memory bottleneck, "
                         "not the rasterizer)")
    ap.add_argument("--n_anchors", type=int, default=256)
    ap.add_argument("--k_gauss",   type=int, default=4)
    ap.add_argument("--hidden",    type=int, default=128)
    ap.add_argument("--mp_steps",  type=int, default=6)
    ap.add_argument("--ssm_dim",   type=int, default=128)
    ap.add_argument("--e_dim",     type=int, default=8)
    ap.add_argument("--dt",        type=float, default=0.05)
    ap.add_argument("--n_time_freqs", type=int, default=6)
    ap.add_argument("--lr_dyn",    type=float, default=1e-4,
                    help="matches train_seqgen_mds.py's own MDS-training default "
                         "(1e-4), not train_sc_anchorflow.py's photometric default (3e-4)")
    ap.add_argument("--lambda_arap",   type=float, default=1e-2)
    ap.add_argument("--lambda_smooth", type=float, default=1e-2,
                    help="2nd-order (acceleration) temporal-smoothness penalty on "
                         "the anchor trajectory -- no consistency/photometric loss "
                         "to keep motion sane, so this replaces that role")
    ap.add_argument("--svd_model", default="stabilityai/stable-video-diffusion-img2vid-xt")
    ap.add_argument("--w_power",   type=float, default=0.0)
    ap.add_argument("--grad_clip_norm", type=float, default=1.0)
    ap.add_argument("--ckpt_every", type=int, default=1000)
    ap.add_argument("--rollout_every", type=int, default=1000)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    dev = "cuda"
    static_mode = args.model_dir is not None
    if static_mode:
        assert args.ply is None and args.data is None, \
            "--model_dir is mutually exclusive with --ply/--data"
    else:
        assert args.ply is not None and args.data is not None, \
            "Blender mode needs both --ply and --data"
    # Blender mode is always composited onto white in load_blender_dataset;
    # bonsai/kitchen's own cfg_args both say white_background=False -> black.
    bg = torch.tensor([0., 0., 0.] if (static_mode and not args.white_bg) else [1., 1., 1.],
                       device=dev)
    gh = git_hash()

    # ── cameras + frozen canonical Gaussians ──────────────────────────────────
    if static_mode:
        cams = load_cameras(args.model_dir, args.n_views, args.res)
        V = len(cams)
        T = args.n_frames
        ply_path = f"{args.model_dir}/point_cloud/iteration_{args.ply_iter}/point_cloud.ply"
    else:
        # GT frames loaded but unused -- MDS needs no ground truth
        cams, _frames_unused = load_blender_dataset(args.data, args.res)
        V = len(cams)
        T = len(_frames_unused[0])
        del _frames_unused
        ply_path = args.ply

    gaussians = GaussianModel(3)
    gaussians.load_ply(ply_path)
    gaussians.active_sh_degree = 3
    for p in (gaussians._xyz, gaussians._features_dc, gaussians._features_rest,
              gaussians._scaling, gaussians._rotation, gaussians._opacity):
        p.requires_grad_(False)
    print(f"[train] gaussians={gaussians.get_xyz.shape[0]} (frozen, from {ply_path})  "
          f"V={V} T={T}  commit={gh}", flush=True)

    # ── anchors (canonical buffer fixed; e/radius/node_weight trainable) ─────
    anchors, _ = AnchorSet.from_gaussians(
        gaussians.get_xyz.detach(), node_num=args.n_anchors,
        latent_dim=0, e_dim=args.e_dim, K=args.k_gauss)
    anchors = anchors.to(dev)
    M = anchors.num
    print(f"[train] anchors={M}", flush=True)

    with torch.no_grad():
        w, idx = anchors.cal_nn_weight(gaussians.get_xyz.detach())
    _arap_idx, _arap_src = W.anchor_rotations_cache(anchors.canonical, K=8)

    # ── HopDynamics (fresh -- trained from scratch via MDS) ──────────────────
    model = HopDynamics(hidden=args.hidden, mp_steps=args.mp_steps, ssm_dim=args.ssm_dim,
                         e_dim=args.e_dim, n_time_freqs=args.n_time_freqs).to(dev)
    print(f"[train] HopDynamics hidden={args.hidden} mp={args.mp_steps} "
          f"ssm={args.ssm_dim} dt={args.dt} h0=N(0,I) sampled (not learned) "
          f"lambda_arap={args.lambda_arap} lambda_smooth={args.lambda_smooth}", flush=True)

    dyn_opt = torch.optim.Adam(
        [{"params": list(model.parameters())},
         {"params": list(anchors.parameters())}],
        lr=args.lr_dyn)

    graph_cfg = {"graph": "knn", "k": 8}
    pipe = Pipe()

    def hop_spatial(grad=True):
        p0 = anchors.canonical
        edge_index = build_graph(p0.detach(), graph_cfg)
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            return model.spatial_embed(anchors.e, edge_index, p0)

    def hop_direct_full(h0, spatial):
        """Non-autoregressive full-T trajectory from a given h0 -- one batched
        hop per frame, straight from h0, exactly like train_sc_anchorflow.py's
        hop_direct(), just parameterized on h0 instead of closing over a
        learned init_h."""
        times = list(range(1, T))
        B = len(times)
        dt = torch.tensor([t * args.dt for t in times], device=dev)
        h0b = h0.unsqueeze(0).expand(B, M, -1)
        d_p, d_rot, _ = model.hop_batch(spatial, h0b, anchors.e, dt)
        canonical = anchors.canonical
        p_seq = torch.cat([canonical.unsqueeze(0), canonical.unsqueeze(0) + d_p], dim=0)
        rot0 = torch.zeros(1, M, 4, device=dev); rot0[..., 0] = 1.0
        rot_seq = torch.cat([rot0, d_rot], dim=0)
        return p_seq, rot_seq   # [T,M,3], [T,M,4]

    # ── SVD guidance (MDS) ────────────────────────────────────────────────────
    svd = SVDGuidance(model_id=args.svd_model, device=dev)

    print("[train] precomputing MDS conditioning cache (canonical render per camera)...",
          flush=True)
    with torch.no_grad():
        N_g = gaussians.get_xyz.shape[0]
        zeros_xyz = torch.zeros(N_g, 3, device=dev)
        zeros_rot = torch.zeros(N_g, 4, device=dev)   # additive residual (d_rot_as_res): 0 = no change
        frame0_cache, cond_cache = [], []
        for cam in cams:
            cam.fid = torch.tensor([0.0], device=dev)
            f0 = _gs_render(cam, gaussians, pipe, bg, zeros_xyz, zeros_rot,
                             zeros_xyz, d_rot_as_res=True)["render"].clamp(0, 1)
            frame0_cache.append(f0)
            cond_cache.append(svd.precompute_cond(f0, T))
    print("[train] cache ready", flush=True)

    # ── resume ────────────────────────────────────────────────────────────────
    start = 0
    ckpt_path = os.path.join(args.out, "ckpt_last.pt")
    if args.resume and os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=dev)
        model.load_state_dict(state["model"])
        anchors.load_state_dict(state["anchors"])
        dyn_opt.load_state_dict(state["dyn_opt"])
        start = state["step"] + 1
        print(f"[train] resumed from step {start}", flush=True)
    elif args.resume:
        print("[train] --resume: no checkpoint found, starting fresh", flush=True)

    fixed_h0 = torch.randn(M, args.ssm_dim, device=dev,
                            generator=torch.Generator(device=dev).manual_seed(0))

    loss_csv_path = os.path.join(args.out, "losses.csv")
    if not os.path.exists(loss_csv_path):
        with open(loss_csv_path, "w") as f:
            f.write("step,total,mds,arap,smooth\n")

    pbar = trange(start, args.iters, desc="train")
    for step in pbar:
        v = step % V
        cam = cams[v]

        h0 = torch.randn(M, args.ssm_dim, device=dev)
        spatial = hop_spatial()
        p_seq, rot_seq = hop_direct_full(h0, spatial)   # [T,M,3], [T,M,4]

        d_xyz_all, d_rot_all = lbs_d_xyz_rot_batched(w, idx, anchors.canonical, p_seq, rot_seq)
        frames_t = render_frames_checkpointed(cam, gaussians, pipe, bg,
                                               d_xyz_all, d_rot_all, T, use_checkpoint=True)

        loss_mds = svd.mds_loss(frames_t, cond_image=frame0_cache[v], w_power=args.w_power,
                                 cond_cache=cond_cache[v], vae_checkpoint=False)
        total_loss = loss_mds

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

        pbar.set_postfix(loss=float(total_loss), mds=float(loss_mds))
        if step % 50 == 0:
            with open(loss_csv_path, "a") as f:
                f.write(f"{step},{float(total_loss)},{float(loss_mds)},"
                        f"{float(loss_arap)},{float(loss_smooth)}\n")

        if step % args.ckpt_every == 0 or step == args.iters - 1:
            torch.save({"model": model.state_dict(), "anchors": anchors.state_dict(),
                        "dyn_opt": dyn_opt.state_dict(), "step": step}, ckpt_path)
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
                p_fixed, rot_fixed = hop_direct_full(fixed_h0, spatial_eval)
                d_xyz_f, d_rot_f = lbs_d_xyz_rot_batched(w, idx, anchors.canonical, p_fixed, rot_fixed)
                frames_fixed = render_frames_checkpointed(
                    cams[0], gaussians, pipe, bg, d_xyz_f, d_rot_f, T, use_checkpoint=False)
            out_frames = [(f.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8") for f in frames_fixed]
            video_path = os.path.join(args.out, f"rollout_{step:06d}.mp4")
            iio.mimsave(video_path, out_frames, fps=8, quality=8)
            print(f"  [rollout] saved -> {video_path}", flush=True)

    print(f"[train] done commit={gh} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()

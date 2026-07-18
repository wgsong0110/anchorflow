#!/usr/bin/env python3
"""T2N preprocessing: TAPIR tracking + DepthCrafter depth + SAM2 mask.

Outputs (all under --out):
  tracks.npz   -- 2D tracklets from TAPIR: {points [T,N,2], visibility [T,N]}
  depths.npz   -- video depth from DepthCrafter: {depth [T,H,W], scale, shift}
  masks.npz    -- foreground mask from SAM2: {mask [T,H,W] bool}

Run on GPU instance:
  python exe/preprocess_t2n.py \\
      --frames /data/datasets/n3v/flame_steak/frames/cam00 \\
      --cameras /workspace/gs_flame/cameras.json \\
      --cam_idx 0 \\
      --out /workspace/t2n_preprocess/flame_steak \\
      --r2 r2:storage/result/anchorflow/t2n_preprocess/flame_steak

Environment (must be installed on instance):
  pip install tapnet  (TAPIR)
  pip install depthcrafter
  pip install sam2
"""
from __future__ import annotations

import argparse
import os
import sys
import json
import subprocess
import numpy as np
from pathlib import Path


def load_frames(frames_dir: str, max_frames: int | None = None) -> np.ndarray:
    """Load JPEG frames sorted by name. Returns [T, H, W, 3] uint8."""
    import imageio.v2 as iio
    paths = sorted(Path(frames_dir).glob("*.jpg")) + sorted(Path(frames_dir).glob("*.png"))
    if max_frames:
        paths = paths[:max_frames]
    frames = [np.asarray(iio.imread(str(p))) for p in paths]
    return np.stack(frames, axis=0)


# ─── TAPIR ────────────────────────────────────────────────────────────────────

def run_tapir(frames_rgb: np.ndarray, out_path: str, grid_size: int = 20):
    """Extract 2D tracklets using TAPIR (tapnet).

    Samples a grid_size x grid_size grid of query points on the first frame.
    """
    import torch
    try:
        import tapnet.torch.tapir_model as tapir_model
    except ImportError:
        print("[preprocess] TAPIR (tapnet) not installed — skip track extraction")
        return

    T, H, W, _ = frames_rgb.shape
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Sample query points on a uniform grid of the first frame
    ys = np.linspace(H * 0.1, H * 0.9, grid_size, dtype=np.float32)
    xs = np.linspace(W * 0.1, W * 0.9, grid_size, dtype=np.float32)
    yy, xx = np.meshgrid(ys, xs)
    # TAPIR query format: [N, 3] = (time, y, x), all at t=0
    queries = np.stack([np.zeros_like(xx.ravel()), yy.ravel(), xx.ravel()], axis=-1)
    N = queries.shape[0]
    print(f"[tapir] {N} query points, {T} frames, {H}x{W}")

    # Load TAPIR checkpoint
    ckpt_path = "/data/huggingface/tapir/tapir_checkpoint_panning.pt"
    if not os.path.exists(ckpt_path):
        alt = os.path.expanduser("~/tapir_checkpoint_panning.pt")
        if os.path.exists(alt):
            ckpt_path = alt
        else:
            print(f"[tapir] checkpoint not found at {ckpt_path} — skip")
            return

    model = tapir_model.TAPIR(pyramid_level=1)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval().to(device)

    # Normalise frames to [-1, 1]
    video = torch.from_numpy(frames_rgb).float() / 255.0 * 2 - 1  # [T,H,W,3]
    video = video.permute(0, 3, 1, 2).unsqueeze(0).to(device)  # [1,T,3,H,W]
    q = torch.from_numpy(queries).unsqueeze(0).to(device)  # [1,N,3]

    with torch.no_grad():
        outputs = model(video, q)

    tracks_xy = outputs["tracks"][0].cpu().numpy()      # [T, N, 2] in pixel (x,y)
    visibility = outputs["occlusion"][0].cpu().numpy()  # [T, N]  logit
    visibility = (visibility < 0).astype(np.float32)   # 1 = visible

    np.savez_compressed(out_path,
                        points=tracks_xy.astype(np.float32),
                        visibility=visibility.astype(np.float32),
                        queries_yx=queries[:, 1:].astype(np.float32))
    print(f"[tapir] saved {out_path}  shape={tracks_xy.shape}")


# ─── DepthCrafter ─────────────────────────────────────────────────────────────

def run_depthcrafter(frames_rgb: np.ndarray, out_path: str):
    """Estimate per-frame depth using DepthCrafter."""
    import torch
    try:
        from depthcrafter.depth_crafter_ppl import DepthCrafterPipeline
        from depthcrafter.unet import DiffusersUNetSpatioTemporalConditionModelDepthCrafter
    except ImportError:
        print("[preprocess] DepthCrafter not installed — skip depth estimation")
        return

    T, H, W, _ = frames_rgb.shape
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_path = "/data/huggingface/hub/models--tencent--DepthCrafter"
    if not os.path.exists(model_path):
        model_path = "tencent/DepthCrafter"  # download from HF
    print(f"[depthcrafter] loading model from {model_path}")

    unet = DiffusersUNetSpatioTemporalConditionModelDepthCrafter.from_pretrained(
        model_path,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
    )
    pipe = DepthCrafterPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt",
        unet=unet,
        torch_dtype=torch.float16,
        variant="fp16",
    )
    pipe.to(device)

    # DepthCrafter expects frames normalised [0,1]
    import torch.nn.functional as F
    frames_t = torch.from_numpy(frames_rgb).float() / 255.0  # [T,H,W,3]
    # Process in chunks of 25 frames
    chunk = 25
    depths_list = []
    for start in range(0, T, chunk):
        clip = frames_t[start:start + chunk]
        with torch.no_grad():
            result = pipe(clip.to(device), height=H, width=W,
                          output_type="np", guidance_scale=1.2,
                          num_inference_steps=5, window_size=chunk)
        depths_list.append(result.frames[0])  # [chunk, H, W]
    depths = np.concatenate(depths_list, axis=0)[:T]  # [T, H, W] in [0,1]

    # Affine alignment: depth = scale * raw + shift (normalise to zero-mean unit-var)
    d_mean = depths.mean()
    d_std  = depths.std() + 1e-8
    scale  = 1.0 / d_std
    shift  = -d_mean / d_std

    np.savez_compressed(out_path,
                        depth=depths.astype(np.float32),
                        scale=np.float32(scale),
                        shift=np.float32(shift))
    print(f"[depthcrafter] saved {out_path}  shape={depths.shape}")


# ─── SAM2 / Track Anything mask ───────────────────────────────────────────────

def run_sam2_mask(frames_rgb: np.ndarray, out_path: str):
    """Generate foreground masks using SAM2 video predictor.

    Prompts with a bounding box or click in the centre of frame 0 to track
    the main foreground subject throughout the video.
    """
    import torch
    try:
        from sam2.build_sam import build_sam2_video_predictor
    except ImportError:
        print("[preprocess] SAM2 not installed — skip mask generation")
        return

    T, H, W, _ = frames_rgb.shape
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sam2_cfg  = "sam2_hiera_large.yaml"
    sam2_ckpt = "/data/huggingface/sam2/sam2_hiera_large.pt"
    if not os.path.exists(sam2_ckpt):
        sam2_ckpt = os.path.expanduser("~/sam2_hiera_large.pt")
    if not os.path.exists(sam2_ckpt):
        print(f"[sam2] checkpoint not found at {sam2_ckpt} — skip")
        return

    predictor = build_sam2_video_predictor(sam2_cfg, sam2_ckpt, device=device)

    import tempfile
    import imageio.v2 as iio

    with tempfile.TemporaryDirectory() as tmpdir:
        for i, fr in enumerate(frames_rgb):
            iio.imwrite(os.path.join(tmpdir, f"{i:06d}.jpg"), fr)

        with predictor.init_state(video_path=tmpdir) as state:
            # Prompt: positive click at image centre (foreground assumed centre)
            _, _, _ = predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=0,
                obj_id=1,
                points=np.array([[W / 2, H / 2]], dtype=np.float32),
                labels=np.array([1], dtype=np.int32),
            )
            masks_all = np.zeros((T, H, W), dtype=bool)
            for out_idx, obj_ids, masks in predictor.propagate_in_video(state):
                if obj_ids and len(masks):
                    masks_all[out_idx] = (masks[0][0].cpu().numpy() > 0)

    np.savez_compressed(out_path,
                        mask=masks_all.astype(np.uint8))
    print(f"[sam2] saved {out_path}  shape={masks_all.shape}")


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames",  required=True, help="dir with 000000.jpg ...")
    ap.add_argument("--out",     required=True, help="output directory")
    ap.add_argument("--max_frames", type=int, default=None)
    ap.add_argument("--tapir_grid", type=int, default=20,
                    help="grid size for TAPIR query points (NxN)")
    ap.add_argument("--skip_tapir",  action="store_true")
    ap.add_argument("--skip_depth",  action="store_true")
    ap.add_argument("--skip_mask",   action="store_true")
    ap.add_argument("--r2", default=None, help="rclone R2 destination")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f"[preprocess] loading frames from {args.frames} ...")
    frames = load_frames(args.frames, args.max_frames)
    T, H, W, _ = frames.shape
    print(f"[preprocess] {T} frames @ {H}x{W}")

    if not args.skip_tapir:
        run_tapir(frames, os.path.join(args.out, "tracks.npz"),
                  grid_size=args.tapir_grid)

    if not args.skip_depth:
        run_depthcrafter(frames, os.path.join(args.out, "depths.npz"))

    if not args.skip_mask:
        run_sam2_mask(frames, os.path.join(args.out, "masks.npz"))

    if args.r2:
        print(f"[preprocess] uploading to {args.r2} ...")
        os.system(f"rclone copy {args.out} {args.r2} --progress")

    print("[preprocess] done.")


if __name__ == "__main__":
    main()

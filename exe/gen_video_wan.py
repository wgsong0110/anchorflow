#!/usr/bin/env python
"""
Generate N videos from a single conditioning image using Wan2.1-I2V-14B
(diffusers WanImageToVideoPipeline), guided by an actual text prompt --
unlike SVD (used elsewhere in this project via gen_video.py/SVDGuidance),
which takes no text prompt at all and only exposes motion_bucket_id/
noise_aug_strength as numeric motion knobs.

Runs in the dedicated ghcr.io/wgsong0110/anchorflow-wan image (see
docker/Dockerfile.wan) -- needs a much newer diffusers than the main image's
0.29.2 pin. Uses group offloading (block/leaf-level CPU<->GPU) to fit the
14B model's ~56-80GB bf16 footprint onto a single 24GB GPU, per the official
diffusers memory-optimization pattern for Wan.

Usage:
  python exe/gen_video_wan.py \\
      --image /tmp/bonsai_frame0.png \\
      --prompt "wind gently blowing through the leaves and pink flowers of a potted bonsai plant, natural swaying motion" \\
      --out /workspace/bonsai_wan --n_videos 8 --seed_base 0
"""
from __future__ import annotations

import os, argparse, subprocess
os.environ.setdefault("HF_HOME", "/data/huggingface")

import torch
import numpy as np
from PIL import Image


DEFAULT_NEGATIVE = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, "
    "paintings, images, static, overall gray, worst quality, low quality, JPEG "
    "compression residue, ugly, incomplete, extra fingers, poorly drawn hands, "
    "poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, "
    "still picture, messy background"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="conditioning frame (canonical render)")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--negative_prompt", default=DEFAULT_NEGATIVE)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_videos", type=int, default=8)
    ap.add_argument("--seed_base", type=int, default=0)
    ap.add_argument("--n_frames", type=int, default=81, help="Wan2.1's own native default")
    ap.add_argument("--num_inference_steps", type=int, default=50)
    ap.add_argument("--guidance_scale", type=float, default=5.0)
    ap.add_argument("--model_id", default="Wan-AI/Wan2.1-I2V-14B-480P-Diffusers")
    ap.add_argument("--max_area", type=int, default=480 * 832)
    ap.add_argument("--offload_transformer", action="store_true",
                    help="leaf-level CPU<->GPU offload the transformer too (needed "
                         "on 24GB cards; leave off on 40GB+ where it fits resident "
                         "and offloading would only slow things down)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
    from diffusers.utils import export_to_video
    from transformers import CLIPVisionModel

    print("[gen_video_wan] loading image_encoder/vae/pipeline ...", flush=True)
    image_encoder = CLIPVisionModel.from_pretrained(
        args.model_id, subfolder="image_encoder", torch_dtype=torch.float32)
    vae = AutoencoderKLWan.from_pretrained(
        args.model_id, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanImageToVideoPipeline.from_pretrained(
        args.model_id, vae=vae, image_encoder=image_encoder, torch_dtype=torch.bfloat16)

    # Group offloading: the 14B transformer + UMT5 text encoder don't fit on a
    # single 24GB GPU in bf16 otherwise (~56-80GB unoptimized). Keeps whichever
    # blocks aren't actively computing on CPU, streamed in/out on demand --
    # official diffusers pattern for Wan on consumer GPUs.
    #
    # Leaf-level offloading of the *transformer* is what actually dominates
    # step time (it runs every single denoising step, streaming every leaf
    # submodule CPU<->GPU each time), unlike the text encoder (runs once per
    # video, so offloading it is essentially free). On a 40GB+ GPU the
    # transformer (~28GB bf16) fits resident on-device on its own -- only
    # offload it if --offload_transformer is passed (needed on 24GB cards).
    from diffusers.hooks.group_offloading import apply_group_offloading
    onload_device = torch.device("cuda")
    offload_device = torch.device("cpu")
    apply_group_offloading(pipe.text_encoder,
        onload_device=onload_device, offload_device=offload_device,
        offload_type="block_level", num_blocks_per_group=4)
    if args.offload_transformer:
        pipe.transformer.enable_group_offload(
            onload_device=onload_device, offload_device=offload_device,
            offload_type="leaf_level", use_stream=True)
    else:
        pipe.transformer.to(onload_device)
    pipe.vae.to(onload_device)
    pipe.image_encoder.to(onload_device)
    print(f"[gen_video_wan] pipeline ready (transformer "
          f"{'offloaded' if args.offload_transformer else 'resident on GPU'})", flush=True)

    image = Image.open(args.image).convert("RGB")
    aspect_ratio = image.height / image.width
    mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
    height = round(np.sqrt(args.max_area * aspect_ratio)) // mod_value * mod_value
    width = round(np.sqrt(args.max_area / aspect_ratio)) // mod_value * mod_value
    image = image.resize((width, height), Image.LANCZOS)
    print(f"[gen_video_wan] conditioning image resized to {width}x{height}", flush=True)

    for i in range(args.n_videos):
        gen = torch.Generator(device="cpu").manual_seed(args.seed_base + i)
        print(f"[gen_video_wan] generating video {i}/{args.n_videos} (seed={args.seed_base + i}) ...",
              flush=True)
        output = pipe(
            image=image,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            height=height, width=width,
            num_frames=args.n_frames,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=gen,
        ).frames[0]
        video_path = os.path.join(args.out, f"gt_video_{i:02d}.mp4")
        export_to_video(output, video_path, fps=16)
        print(f"[gen_video_wan] video {i}/{args.n_videos} -> {video_path}", flush=True)

        r2_dest = f"r2:storage/result/anchorflow/{os.path.basename(args.out)}/"
        r = subprocess.run(["rclone", "copy", video_path, r2_dest],
                            check=False, capture_output=True, timeout=120, text=True)
        if r.returncode != 0:
            print(f"[gen_video_wan] WARNING: R2 backup FAILED for {video_path}: {r.stderr.strip()}",
                  flush=True)

    print(f"[gen_video_wan] done -> {args.out}", flush=True)


if __name__ == "__main__":
    main()

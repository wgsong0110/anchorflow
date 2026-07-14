#!/usr/bin/env python
"""v3 img2vid stage — animate a single image into a monocular driving video with
**Wan 2.2 I2V** (upgrade over SVD-XT), written as square frames for the SV4D 2.0
multi-view stage (`gen_mv_video.py --video <frames_dir>`).

Runs in the Wan image (docker/Dockerfile.i2v: torch 2.4/cu121 + diffusers-main).
Kept SEPARATE from the SV4D/sgm env (torch 2.0-2.1) — the two stages hand off via
a frames folder on disk / R2.

    python exe/gen_i2v.py --image subject.png --out <ws>/mono_frames --n_frames 21
"""

from __future__ import annotations

import argparse
import os


def image_to_mono_frames(image_path, frames_dir, n_frames, size, seed,
                         prompt=("the object stays centered and animates with a "
                                 "subtle natural motion, static camera, plain "
                                 "background"),
                         neg=("camera pan, camera zoom, background clutter, "
                              "blurry, low quality, distorted")):
    import numpy as np
    import torch
    from PIL import Image
    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
    from transformers import CLIPVisionModel

    os.makedirs(frames_dir, exist_ok=True)
    model_id = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"

    image_encoder = CLIPVisionModel.from_pretrained(
        model_id, subfolder="image_encoder", torch_dtype=torch.float32)
    vae = AutoencoderKLWan.from_pretrained(
        model_id, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanImageToVideoPipeline.from_pretrained(
        model_id, vae=vae, image_encoder=image_encoder, torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()          # ~80GB native -> fits 48GB via offload

    # Wan wants square dims divisible by 16; 576 = 36*16.
    image = Image.open(image_path).convert("RGB").resize((size, size))
    gen = torch.Generator("cuda").manual_seed(seed)

    # Wan native I2V length is 81 frames @16fps; generate then subsample to
    # n_frames for SV4D (SV4D re-times internally — only frame COUNT matters).
    out = pipe(image=image, prompt=prompt, negative_prompt=neg,
               height=size, width=size, num_frames=81,
               guidance_scale=3.5, guidance_scale_2=3.5,
               num_inference_steps=40, generator=gen).frames[0]

    idx = np.linspace(0, len(out) - 1, n_frames).round().astype(int)
    for i, j in enumerate(idx):
        out[j].resize((size, size), Image.LANCZOS).save(
            os.path.join(frames_dir, f"{i:05d}.png"))
    print(f"[i2v] Wan2.2-I2V wrote {n_frames} frames -> {frames_dir}")
    return n_frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True, help="mono frames dir (SV4D --video input)")
    ap.add_argument("--n_frames", type=int, default=21)
    ap.add_argument("--size", type=int, default=576)
    ap.add_argument("--seed", type=int, default=23)
    args = ap.parse_args()
    image_to_mono_frames(args.image, args.out, args.n_frames, args.size, args.seed)


if __name__ == "__main__":
    main()

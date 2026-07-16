#!/usr/bin/env python
"""v3 img2vid stage — animate a single image into a monocular driving video with
**Wan 2.2 I2V** (upgrade over SVD-XT), written as square frames for the SV4D 2.0
multi-view stage (`gen_mv_video.py --video <frames_dir>`).

Model: Wan 2.2 **TI2V-5B** (single transformer, ~35GB, fits a normal disk). The
A14B I2V variant is ~140GB (two 14B experts) — too big for a normal instance disk.
TI2V-5B has NO image_encoder (unlike Wan 2.1) and no second expert, so we load the
pipeline directly (no CLIPVisionModel, no guidance_scale_2) and force output_type=pil.

ENV CAVEAT: the current docker/Dockerfile.i2v ships torch 2.4, but every Wan-capable
diffusers assumes torch >= 2.5 (custom_op PEP604 union schemas, torch.nn.attention.
flex_attention, SDPA enable_gqa). On the torch-2.4 image this needs runtime pins:
diffusers==0.35.1 + transformers==4.49.0, plus patching diffusers attention_dispatch
(force custom_op no-op for torch<2.5, drop enable_gqa from the native SDPA call).
PROPER FIX = rebuild Dockerfile.i2v on torch 2.5. Run with HF_HUB_OFFLINE=1 once the
model is cached (avoids HF HEAD-check timeouts).

    python exe/gen_i2v.py --image subject.png --out <ws>/mono_frames --n_frames 21
"""

from __future__ import annotations

import argparse
import os


def image_to_mono_frames(image_path, frames_dir, n_frames, size, seed,
                         prompt=("a strong gusty wind blows through a potted plant, its leaves and "
                                 "slender branches sway and bend dramatically back and forth, "
                                 "whole canopy swinging energetically in the wind, lively "
                                 "continuous motion, the plant moves on its own, static locked "
                                 "camera, plain white background, nobody present"),
                         neg=("hand, hands, arm, arms, person, people, human, holding, "
                              "grabbing, touching, fingers, still, frozen, motionless, "
                              "camera pan, camera zoom, zoom in, background clutter, blurry, "
                              "low quality, distorted")):
    import numpy as np
    import torch
    from PIL import Image
    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
    from transformers import CLIPVisionModel

    os.makedirs(frames_dir, exist_ok=True)
    model_id = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"

    vae = AutoencoderKLWan.from_pretrained(
        model_id, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanImageToVideoPipeline.from_pretrained(
        model_id, vae=vae, torch_dtype=torch.bfloat16)  # Wan 2.2 A14B: no image_encoder
    pipe.enable_model_cpu_offload()          # ~80GB native -> fits 48GB via offload

    # Wan wants square dims divisible by 16; 576 = 36*16.
    image = Image.open(image_path).convert("RGB").resize((size, size))
    gen = torch.Generator("cuda").manual_seed(seed)

    # Wan native I2V length is 81 frames @16fps; generate then subsample to
    # n_frames for SV4D (SV4D re-times internally — only frame COUNT matters).
    out = pipe(image=image, prompt=prompt, negative_prompt=neg,
               height=size, width=size, num_frames=81,
               guidance_scale=5.0,
               num_inference_steps=40, output_type="pil", generator=gen).frames[0]

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

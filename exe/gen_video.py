#!/usr/bin/env python
"""Generate a monocular video from a single image via SVD img2vid, saved as
frames for MoSca (`<ws>/images/*.png`). Runs in the anchorflow image (diffusers SVD).

    python exe/gen_video.py --image subject.png --out <ws>/images --frames 25
"""

from __future__ import annotations

import argparse
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True, help="frames dir (MoSca ws/images)")
    ap.add_argument("--frames", type=int, default=25)
    ap.add_argument("--model", default="stabilityai/stable-video-diffusion-img2vid-xt")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import torch
    from PIL import Image
    from diffusers import StableVideoDiffusionPipeline

    pipe = StableVideoDiffusionPipeline.from_pretrained(
        args.model, torch_dtype=torch.float16, variant="fp16")
    pipe.enable_model_cpu_offload()                          # fit 24GB

    image = Image.open(args.image).convert("RGB").resize((1024, 576))
    gen = torch.Generator("cuda").manual_seed(args.seed)
    frames = pipe(image, num_frames=args.frames, decode_chunk_size=8,
                  generator=gen).frames[0]
    for i, f in enumerate(frames):
        f.save(os.path.join(args.out, f"{i:05d}.png"))
    print(f"[gen_video] wrote {len(frames)} frames -> {args.out}")


if __name__ == "__main__":
    main()

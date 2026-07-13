#!/usr/bin/env python
"""Generate a canonical static 3DGS .ply from a single image via TRELLIS.

Runs on the GPU instance (needs >=16GB VRAM + TRELLIS deps). Produces the
canonical asset that anchorflow then animates. Same image is later reused as the
SVD frame-0 conditioning, so keep it (a neutral rest-pose quadruped).

    python exe/gen_canonical.py --image horse.png --out /data/.../horse.ply

Env (set before importing trellis):
    ATTN_BACKEND=xformers   SPCONV_ALGO=native   HF_HOME=/data/huggingface

TRELLIS export note: SH degree 0 (f_dc only), raw params, applies a Y-up->Z-up
axis transform by default; pass --no_transform to keep TRELLIS's native frame.
"""

from __future__ import annotations

import argparse
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--model", default="JeffreyXiang/TRELLIS-image-large")
    ap.add_argument("--no_transform", action="store_true")
    args = ap.parse_args()

    os.environ.setdefault("ATTN_BACKEND", "xformers")
    os.environ.setdefault("SPCONV_ALGO", "native")
    os.environ.setdefault("HF_HOME", "/data/huggingface")

    from PIL import Image
    from trellis.pipelines import TrellisImageTo3DPipeline

    pipe = TrellisImageTo3DPipeline.from_pretrained(args.model)
    pipe.cuda()

    image = Image.open(args.image)
    out = pipe.run(image, seed=args.seed, formats=["gaussian"],
                   preprocess_image=True)          # built-in rembg bg removal

    gaussian = out["gaussian"][0]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    if args.no_transform:
        gaussian.save_ply(args.out, transform=None)
    else:
        gaussian.save_ply(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

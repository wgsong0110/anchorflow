"""PNG 묶음을 mp4 로 묶는다. 노드에 ffmpeg 가 없어 파이썬으로 쓴다.

  python exe/pngs2mp4.py --dir <png 디렉토리> --out out.mp4 [--fps 30]
"""
from __future__ import annotations

import argparse
import glob
import os

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--fps", type=int, default=30)
a = ap.parse_args()

fs = sorted(glob.glob(os.path.join(a.dir, "*.png")))
if not fs:
    raise SystemExit(f"png 이 없다: {a.dir}")
try:
    import imageio.v2 as iio
    import imageio_ffmpeg                      # noqa: F401
    w = iio.get_writer(a.out, fps=a.fps, codec="libx264",
                       pixelformat="yuv420p", quality=8)
    for f in fs:
        w.append_data(iio.imread(f)[..., :3])
    w.close()
    print(f"[저장] {a.out}  {len(fs)} 프레임 (imageio-ffmpeg)")
except Exception as e:
    import cv2
    import numpy as np
    im = cv2.imread(fs[0])
    h, wd = im.shape[:2]
    vw = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps,
                         (wd, h))
    for f in fs:
        vw.write(cv2.imread(f))
    vw.release()
    print(f"[저장] {a.out}  {len(fs)} 프레임 (cv2, imageio 실패: "
          f"{type(e).__name__})")

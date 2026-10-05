"""Spring-Gaus 의 `camera.json` 을 만든다 (MPM 렌더로 만든 피팅 데이터용).

그쪽 로더(lib/datasets/mpm_synthetic.py)는 각 카메라마다
`{"camera": <폴더명>, "K": 3x3, "c2w": 4x4}` 를 기대하고, c2w 는 blender 규약
(y 위, z 뒤)이다 -- 읽은 뒤 `c2w[:3,1:3] *= -1` 로 COLMAP 으로 바꾼다.

  python exe/mk_sg_cameras.py --cams mvcams_mic.json --out sgdata/mic/camera.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--cams", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

out = []
for i, c in enumerate(json.load(open(a.cams))):
    R = np.array(c["rotation"], dtype=np.float64)        # camera-to-world (COLMAP)
    pos = np.array(c["position"], dtype=np.float64)
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = pos
    M[:3, 1] *= -1.0          # COLMAP -> blender
    M[:3, 2] *= -1.0
    K = [[c["fx"], 0.0, c["width"] / 2.0],
         [0.0, c["fy"], c["height"] / 2.0],
         [0.0, 0.0, 1.0]]
    out.append(dict(camera=f"cam_{i}", K=K, c2w=M.tolist(),
                    width=c["width"], height=c["height"]))
json.dump(out, open(a.out, "w"), indent=1)
print(f"[저장] {a.out}  카메라 {len(out)} 대  "
      f"{out[0]['width']}x{out[0]['height']}")

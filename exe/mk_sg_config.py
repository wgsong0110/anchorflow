"""Spring-Gaus 설정 파일을 만든다 (MPM 렌더 데이터로 피팅).

`sgdata/<형상>/sgmeta.json` 에 적힌 바닥점·중력축·바운딩박스를 그대로 쓴다.
그쪽 기본값(default.yaml)은 상속하고 DATA 절만 우리 장면으로 채운다.

  python exe/mk_sg_config.py --shape mic --out <경로>.yaml
"""
from __future__ import annotations

import argparse
import json

W = "/home/dkta/work"

ap = argparse.ArgumentParser()
ap.add_argument("--shape", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--n_frame", type=int, default=20, help="피팅에 쓰는 프레임 수")
ap.add_argument("--hit_frame", type=int, default=24, help="바닥에 닿는 프레임")
a = ap.parse_args()

meta = json.load(open(f"{W}/sgdata/{a.shape}/sgmeta.json"))
cam = json.load(open(f"{W}/sgdata/{a.shape}/camera.json"))[0]
lo, hi = meta["xyz_min"], meta["xyz_max"]
pad = 0.1 * max(hi[i] - lo[i] for i in range(3))
lo = [round(lo[i] - pad, 4) for i in range(3)]
hi = [round(hi[i] + pad, 4) for i in range(3)]
p0 = [round(q, 5) for q in meta["floor_point"]]
up = [round(q, 5) for q in meta["up"]]

y = f"""DEFAULT: config/mpm_synthetic/default.yaml
CHECKPOINTS_ROOT: checkpoints/{a.shape}

DATA:
  TYPE: MPM_Synthetic
  DATA_ROOT: {W}/sgdata
  OBJ_NAME: {a.shape}
  N_CAM: {len(json.load(open(f"{W}/sgdata/{a.shape}/camera.json")))}
  N_FRAME: {a.n_frame}
  FRAME_ALL: {meta["n_frames"]}
  HIT_FRAME: {a.hit_frame}
  H: {cam["height"]}
  W: {cam["width"]}
  WHITE_BKG: True
  XYZ_MIN: {lo}
  XYZ_MAX: {hi}
  BC: [[{p0}, {up}]]

  DT: {round(meta["frame_dt"], 6)}
  GLOBAL_M: 1
  GLOBAL_K: 1000
  GLOBAL_DAMP: 0.1

  EVAL_FREQ: 60
  EVAL_FRAME: {meta["n_frames"]}

MODEL:
  G: [0, 0, -9.8]
"""
open(a.out, "w").write(y)
print(f"[저장] {a.out}\n{y}")

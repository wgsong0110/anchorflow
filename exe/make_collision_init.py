"""Fracture-GS 두 물체 빠른 충돌 씬의 초기 상태(h5)와 설정(json)을 만든다 (repflow B).

같은 형상 두 벌(PG 공식 채우기 입자 = 가우시안 + 내부 채움)을 x 축으로 마주 보게 놓고
서로를 향해 ±v 로 보낸다. 시뮬 상자는 두 물체가 들어가게 grid_lim 4 (dx 0.02 = PG 의 2/100).
물성은 Fracture-GS 논문 표의 Teapot: E 5e5, ν 0.46, 밀도 5, NACC (α, β, ξ, M) = (0.98, 0.5, 1, 2.36).
  α 는 초기 Jp 로 읽어 alpha_0 = ln α, M 은 GF 의 M = 6 sinφ/(3 − sinφ) 로 마찰각을 거꾸로 푼다.
충돌 속도·간격·프레임 간격·중력은 논문에 수치가 없어 정한 값 (문서에 적는다).

  python exe/make_collision_init.py --shape lego --out /home/dkta/work/repip/col_lego
"""
import argparse
import json
import math
import os

import h5py
import numpy as np

W = "/home/dkta/work"
ap = argparse.ArgumentParser()
ap.add_argument("--shape", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--v", type=float, default=10.0, help="각 물체의 접근 속도 (상대 속도 2v)")
ap.add_argument("--gap", type=float, default=1.0,
                help="두 물체 표면 사이 처음 간격 -- 멀리서 달려와 부딪히게 (상대 속도 2v 로 약 15 프레임 뒤 충돌)")
ap.add_argument("--frames", type=int, default=60)
ap.add_argument("--frame_dt", type=float, default=1.0 / 300.0)
a = ap.parse_args()

X = np.load(f"{W}/pgfill_{a.shape}.npy").astype(np.float64)
c = X.mean(0)
ext = X.max(0) - X.min(0)
cx = 2.0
d = ext[0] / 2 + a.gap / 2
X0 = X - c + np.array([cx - d, 2.0, 2.0])
X1 = X - c + np.array([cx + d, 2.0, 2.0])
XA = np.concatenate([X0, X1])
VA = np.concatenate([np.tile([a.v, 0.0, 0.0], (len(X), 1)), np.tile([-a.v, 0.0, 0.0], (len(X), 1))])
OBJ = np.concatenate([np.zeros(len(X), np.int32), np.ones(len(X), np.int32)])
assert XA.min() > 0.1 and XA.max() < 3.9, (XA.min(), XA.max())
os.makedirs(a.out, exist_ok=True)
with h5py.File(f"{a.out}/init.h5", "w") as h:
    h.create_dataset("x", data=XA.T.astype(np.float32))
    h.create_dataset("v", data=VA.T.astype(np.float32))
    h.create_dataset("obj", data=OBJ)
M = 2.36
sphi = 3 * M / (6 + M)                             # M = 6 s / (3 - s)  ->  s = 3M / (6 + M)
cfg = dict(material="watermelon", E=5e5, nu=0.46, density=5.0,
           alpha_0=math.log(0.98), beta=0.5, xi=1.0, hardening=1.0,
           friction_angle=math.degrees(math.asin(sphi)),
           n_grid=200, grid_lim=4.0, flip_pic_ratio=0.7,
           substep_dt=1e-5, frame_dt=a.frame_dt, frame_num=a.frames,
           g=[0.0, 0.0, 0.0], init_velocity=[0.0, 0.0, 0.0],
           boundary_conditions=[{"type": "bounding_box"}],
           repflow_note=dict(shape=a.shape, v=a.v, gap=a.gap, n_each=int(len(X)),
                             center_shift=(-c).tolist()))
json.dump(cfg, open(f"{a.out}/config.json", "w"), indent=1)
print(f"[충돌 초기] {a.shape}: 물체당 {len(X)} 입자, 폭 {ext[0]:.3f}, 중심 x {cx - d:.3f} / {cx + d:.3f}, "
      f"±{a.v} -> {a.out}", flush=True)

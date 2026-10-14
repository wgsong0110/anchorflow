"""치마 바람 조건(a)의 초기 속도를 고정된 허리 띠(joint)에서부터 서서히 키운다 -- PG·GF·MPMAvatar 에 똑같이.

원래는 띠 바로 아래 자유 천에 처음부터 2 m/s, 띠는 0 이라 격자 한 칸 사이에 속도가 끊겨 PG·GF 에서 띠 아래가 찢어졌다
(2026-10-14 측정: 이웃 3 배 이탈 입자가 프레임 5 부터 높이 1.12~1.23 에 몰림). 속도 크기에 w = smoothstep(d / D) 를 곱한다
(d = 가장 가까운 고정 옷 입자까지 거리, 시뮬 좌표).

  python exe/skirt_vramp.py --pg_in pg/pg_in.pt --D 0.1 --out pg_ramp/pg_in.pt
"""
import argparse
import os

import numpy as np
import torch
from scipy.spatial import cKDTree

ap = argparse.ArgumentParser()
ap.add_argument("--pg_in", required=True)
ap.add_argument("--D", type=float, default=0.1)
ap.add_argument("--out", required=True)
a = ap.parse_args()
I = torch.load(a.pg_in, weights_only=False)
x = I["x"].numpy(); pin = I["pin"].numpy().astype(bool); cl = I["cloth"].numpy().astype(bool)
d, _ = cKDTree(x[cl & pin]).query(x)
t = np.clip(d / a.D, 0.0, 1.0); w = t * t * (3 - 2 * t)
w[pin] = 0.0
I["v"] = I["v"] * torch.as_tensor(w, dtype=I["v"].dtype)[:, None]
I["v_ramp_D"] = a.D
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
torch.save(I, a.out)
f = cl & ~pin
print(f"[속도 경사] D {a.D}  자유 옷 입자 {int(f.sum())} 중 w<1 {int((w[f] < 1).sum())} ({100 * (w[f] < 1).mean():.1f}%)", flush=True)

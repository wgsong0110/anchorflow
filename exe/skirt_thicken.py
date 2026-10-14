"""치마 천 입자에 두께를 준다 (PG·GF 공통 입력 pg_in.pt 를 바꾼다; 격자는 그대로).

격자 250 (칸 0.008) 에서 한 겹 천은 허리 띠 아래가 통째로 떨어져 나갔다 (10-10: E ×0.1~×100 전부, 격자 125 는 E×0.5 이상 유지;
10-14: 초기 속도 경사로는 변화 없음). 옷 입자(고정 띠 포함)마다 국소 PCA 법선을 구해 ±k·δ (k = 1..layers) 위치에 복제한다.
복제 입자는 원본의 고정 여부·속도·공분산·색을 그대로 받고, 렌더에서는 쓰지 않는다 (n_orig 이후). MPMAvatar 는 천을 메쉬
코디멘셔널 모델로 풀어 입자 두께 개념이 없으므로 그대로 둔다.

  python exe/skirt_thicken.py --pg_in pg/pg_in.pt --layers 1 --delta 0.008 --out thick/pg_in.pt
"""
import argparse
import os

import numpy as np
import torch
from scipy.spatial import cKDTree

ap = argparse.ArgumentParser()
ap.add_argument("--pg_in", required=True)
ap.add_argument("--layers", type=int, default=1)
ap.add_argument("--delta", type=float, default=0.008, help="층 간격 (시뮬 좌표; 격자 250 의 칸 = 0.008)")
ap.add_argument("--knn", type=int, default=16)
ap.add_argument("--out", required=True)
a = ap.parse_args()
I = torch.load(a.pg_in, weights_only=False)
x = I["x"].double().numpy(); cl = I["cloth"].numpy().astype(bool)
idx = np.nonzero(cl)[0]
P = x[idx]
_, nb = cKDTree(P).query(P, k=a.knn)
Q = P[nb] - P[nb].mean(1, keepdims=True)
C = np.einsum("nki,nkj->nij", Q, Q)
w, V = np.linalg.eigh(C)
nrm = V[:, :, 0]                                              # 가장 작은 고유값 방향 = 면 법선
n0 = len(x)
src, off = [], []
for k in range(1, a.layers + 1):
    for s in (1.0, -1.0):
        src.append(idx); off.append(s * k * a.delta * nrm)
src = np.concatenate(src); off = np.concatenate(off)
O = dict(I)
O["x"] = torch.cat([I["x"], torch.as_tensor(x[src] + off, dtype=I["x"].dtype)])
for key in ("v", "cov", "pin", "col", "op", "cloth"):
    O[key] = torch.cat([I[key], I[key][torch.as_tensor(src)]])
O["n_orig"] = n0
O["thicken"] = dict(layers=a.layers, delta=a.delta)
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
torch.save(O, a.out)
print(f"[두께] 옷 입자 {len(idx)} x {2 * a.layers} 복제 -> 전체 {len(O['x'])} (원본 {n0}), 두께 {2 * a.layers * a.delta:.4f}", flush=True)

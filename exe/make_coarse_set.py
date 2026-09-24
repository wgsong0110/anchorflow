"""PG 채움 결과를 **성기게** 다시 뽑고, 칸당 입자수가 목표 범위에 들도록 n_grid 를 정한다.

PhysGaussian 의 배경 격자는 `grid_lim / n_grid` 가 칸 크기다. 칸당 입자수 ppc 는
(dx / 입자간격)^3 로 정해지므로, 간격을 정하면 n_grid 가 따라온다.
"""
from __future__ import annotations
import argparse, json
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True, help="원본 채움 npy")
ap.add_argument("--out", required=True, help="성긴 입자셋 npy")
ap.add_argument("--cfg_in", required=True)
ap.add_argument("--cfg_out", required=True)
ap.add_argument("--target", type=int, default=20000, help="목표 입자 수")
ap.add_argument("--ppc", type=float, default=12.0, help="칸당 입자 수 목표")
ap.add_argument("--grid_lim", type=float, default=2.0)
a = ap.parse_args()

P = np.load(a.src).astype(np.float64)
print(f"[원본] {P.shape[0]} 점  범위 {P.min(0).round(3)} ~ {P.max(0).round(3)}")

# 목표 개수에 맞는 복셀 간격을 이분법으로 찾는다 (복셀당 한 점만 남긴다)
lo, hi = P.min(0), P.max(0)
ext = hi - lo
s_lo, s_hi = 1e-4, float(ext.max())
for _ in range(60):
    s = 0.5 * (s_lo + s_hi)
    k = np.floor((P - lo) / s).astype(np.int64)
    n = len(np.unique(k, axis=0))
    if n > a.target:
        s_lo = s
    else:
        s_hi = s
s = s_hi
k = np.floor((P - lo) / s).astype(np.int64)
_, idx = np.unique(k, axis=0, return_index=True)
Q = P[np.sort(idx)]
print(f"[성긴셋] {Q.shape[0]} 점, 복셀 간격 {s:.5f}")

# 실제 입자 간격(최근접 거리 중앙값)으로 ppc 를 맞춘다
sub = Q[np.random.default_rng(0).permutation(len(Q))[:3000]]
d = np.linalg.norm(sub[:, None, :] - sub[None, :, :], axis=-1)
np.fill_diagonal(d, np.inf)
spacing = float(np.median(d.min(1)))
dx = spacing * (a.ppc ** (1.0 / 3.0))
n_grid = int(round(a.grid_lim / dx))
dx_real = a.grid_lim / n_grid
ppc_real = (dx_real / spacing) ** 3
print(f"[격자] 입자간격 {spacing:.5f}  ->  n_grid {n_grid} (dx {dx_real:.5f}), "
      f"칸당 입자 {ppc_real:.1f}")

np.save(a.out, Q.astype(np.float32))
c = json.load(open(a.cfg_in))
c["n_grid"] = n_grid
if "particle_filling" in c and isinstance(c["particle_filling"], dict):
    c["particle_filling"]["n_grid"] = n_grid
json.dump(c, open(a.cfg_out, "w"), indent=1)
print(f"[저장] {a.out}  /  {a.cfg_out}  (n_grid {n_grid})")

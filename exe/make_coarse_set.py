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
ap.add_argument("--n_gauss", type=int, required=True,
                help="원본 채움 앞부분의 가우시안 개수 -- PhysGaussian 은 "
                     "mpm_init_pos[:gs_num] 이 가우시안이라고 가정한다")
ap.add_argument("--ply_out", default=None,
                help="솎아낸 가우시안 색인을 저장할 npy (렌더·시뮬이 같은 부분집합을 써야 한다)")
a = ap.parse_args()

P = np.load(a.src).astype(np.float64)
print(f"[원본] {P.shape[0]} 점  범위 {P.min(0).round(3)} ~ {P.max(0).round(3)}")
G, F = P[:a.n_gauss], P[a.n_gauss:]
print(f"[구성] 가우시안 {G.shape[0]} + 채움 {F.shape[0]}")

# 목표 개수에 맞는 복셀 간격을 이분법으로 찾는다 (복셀당 한 점만 남긴다).
# 가우시안과 채움점을 **따로** 솎아 순서를 유지해야 한다 -- PhysGaussian 이
# 앞쪽 gs_num 개를 가우시안으로 보기 때문이다.
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


def thin(X):
    kk = np.floor((X - lo) / s).astype(np.int64)
    _, ii = np.unique(kk, axis=0, return_index=True)
    return np.sort(ii)


gi = thin(G)
fi = thin(F)
Q = np.concatenate([G[gi], F[fi]], 0)
print(f"[성긴셋] 가우시안 {gi.size} + 채움 {fi.size} = {Q.shape[0]} 점, "
      f"복셀 간격 {s:.5f}")
if a.ply_out:
    np.save(a.ply_out, gi.astype(np.int64))
    print(f"[저장] 가우시안 색인 {a.ply_out}")

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

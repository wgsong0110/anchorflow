"""절 CM13.4: 단면 뷰어용 수평 층 자료 -- 프레임 t, 높이 z_k 의 얇은 층 (|z - z_k| < 두께/2) 입자의 (x, y, logJp).

- 층 두께 = 층 간격 = 정답 MPM 격자 칸 dx = grid_lim / n_grid = 2/300 (시뮬 공간 단위), z_k = 바닥 0.02 + (k + ½) dx.
- 정답 (다시 돌린 GF: h5 x + jp/sc_t) 과 우리 (pred.pt, 프레임 0 = 정답 0 프레임 x; sc_t) 를 같은 층으로.
- 위치가 비유한인 입자는 뺌. 층마다 표시용 부분표본 상한 --cap (고정 씨앗, 몇 개 중 몇 개인지 같이 저장) -- 시뮬·통계와 무관한 표시용.
"""
import argparse
import glob

import h5py
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--ours_pred", required=True)
ap.add_argument("--ours_sc", required=True)
ap.add_argument("--x0_h5", required=True)
ap.add_argument("--gt_h5", required=True)
ap.add_argument("--gt_sc", required=True)
ap.add_argument("--dx", type=float, default=2.0 / 300)
ap.add_argument("--z0", type=float, default=0.02)
ap.add_argument("--cap", type=int, default=1500)
ap.add_argument("--out", required=True)
a = ap.parse_args()


def h5x(p):
    with h5py.File(p, "r") as h:
        x = np.array(h["x"])
    return x.T if x.shape[0] == 3 else x


P = torch.load(a.ours_pred, map_location="cpu", weights_only=False, mmap=True)["pred"]
fg = sorted(glob.glob(f"{a.gt_h5}/sim_*.h5"))
T = min(len(fg), P.shape[0] + 1)
x0 = h5x(a.x0_h5)
rng = np.random.default_rng(0)
XY, V, IDX = [], [], []                 # IDX: (편 0 정답 / 1 우리, t, k, 시작, 표시 수, 전체 수)
off = 0
zmax = 0.0
for t in range(T):
    for side in (0, 1):
        if side == 0:
            x = h5x(fg[t]); v = np.load(f"{a.gt_sc}/sc_{t:04d}.npy")
        else:
            x = x0 if t == 0 else P[t - 1].numpy(); v = np.load(f"{a.ours_sc}/sc_{t:04d}.npy")
        ok = np.isfinite(x).all(1) & np.isfinite(v)
        x, v = x[ok], v[ok]
        k = np.floor((x[:, 2] - a.z0) / a.dx).astype(np.int64)
        good = k >= 0
        x, v, k = x[good], v[good], k[good]
        if k.size:
            zmax = max(zmax, float(x[:, 2].max()))
        o = np.argsort(k, kind="stable"); x, v, k = x[o], v[o], k[o]
        ks, st, cn = np.unique(k, return_index=True, return_counts=True)
        for kk, s0, c in zip(ks, st, cn):
            sel = np.arange(s0, s0 + c)
            if c > a.cap:
                sel = np.sort(rng.choice(sel, a.cap, replace=False))
            XY.append(x[sel, :2].astype(np.float32)); V.append(v[sel].astype(np.float32))
            IDX.append((side, t, int(kk), off, len(sel), int(c))); off += len(sel)
    print(f"[단면] 프레임 {t} 누적 점 {off}", flush=True)
np.savez_compressed(a.out, xy=np.concatenate(XY), v=np.concatenate(V), idx=np.array(IDX, np.int64), dx=a.dx, z0=a.z0, cap=a.cap, T=T)
print(f"[단면 저장] {a.out} 프레임 {T} 점 {off} 층 {len(IDX)} z 최대 {zmax:.3f}", flush=True)

"""초기 z 띠별 탄성 에너지 평균을 프레임에 따라 (우리 / PG 나란히).

입자를 **초기 z** 로 띠를 나누고 각 띠의 psi 평균을 프레임마다 잰다. 어느 높이의
물질이 언제 에너지를 지는지가 보인다. 두 칸은 같은 색 범위를 쓴다.
"""
from __future__ import annotations
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import os as _os
for _p in ("/home/wgsong/.fonts/NotoSansCJKkr-Regular.otf",
           _os.path.expanduser("~/.fonts/NotoSansCJKkr-Regular.otf")):
    try: fm.fontManager.addfont(_p)
    except Exception: pass
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Noto Sans CJK KR"
plt.rcParams["axes.unicode_minus"] = False

ap = argparse.ArgumentParser()
ap.add_argument("--dump", required=True)
ap.add_argument("--bands", type=int, default=24)
ap.add_argument("--mark", type=int, nargs="+", default=[])
ap.add_argument("--out", required=True)
a = ap.parse_args()


def L(p):
    try: return torch.load(p, map_location="cpu", weights_only=False)
    except TypeError: return torch.load(p, map_location="cpu")


D = L(a.dump)
X0 = np.asarray(D["x0"], dtype=np.float32)
fs = np.asarray(D["fscal"])[..., 4]                 # 우리 psi [T,N]
gf = D.get("gt_fscal")
gs = np.asarray(gf)[..., 4] if gf is not None else None
T = fs.shape[0]
z0 = X0[:, 2]
edges = np.linspace(z0.min(), z0.max(), a.bands + 1)
idx = np.clip(np.digitize(z0, edges) - 1, 0, a.bands - 1)
print(f"초기 z {z0.min():.3f} ~ {z0.max():.3f} 를 {a.bands} 띠로, "
      f"띠당 입자 {np.bincount(idx, minlength=a.bands).min()}~"
      f"{np.bincount(idx, minlength=a.bands).max()}", flush=True)


def grid(P):
    M = np.full((a.bands, T), np.nan)
    for b in range(a.bands):
        m = idx == b
        if m.any():
            M[b] = np.log10(np.maximum(P[:, m].mean(1), 1e-12))
    return M


Mo = grid(fs)
Mg = grid(gs) if gs is not None else None
_all = Mo if Mg is None else np.concatenate([Mo, Mg])
vlo, vhi = np.nanpercentile(_all, 2), np.nanpercentile(_all, 99)
n = 1 if Mg is None else 2
fig, ax = plt.subplots(1, n, figsize=(7.0 * n, 4.6), dpi=120, squeeze=False)
for k, (nm, M) in enumerate([("출력만 최적화", Mo)] +
                            ([("PG MPM", Mg)] if Mg is not None else [])):
    q = ax[0][k]
    im = q.imshow(M, aspect="auto", origin="lower", cmap="inferno",
                  vmin=vlo, vmax=vhi,
                  extent=[0, T, edges[0], edges[-1]])
    q.set_xlabel("프레임"); q.set_ylabel("초기 z")
    q.set_title(f"{nm}  띠별 평균 log10(psi)", fontsize=11)
    for m_ in a.mark:
        q.axvline(m_, color="w", ls=":", lw=1.2)
    fig.colorbar(im, ax=q)
fig.tight_layout(); fig.savefig(a.out)
print(f"[저장] {a.out}", flush=True)
for nm, M in [("우리", Mo)] + ([("PG", Mg)] if Mg is not None else []):
    print(f"  [{nm}] 마지막 프레임 띠별 log10(psi): 최저 {np.nanmin(M[:,-1]):.2f} "
          f"(초기 z {edges[np.nanargmin(M[:,-1])]:.2f}), 최고 "
          f"{np.nanmax(M[:,-1]):.2f} (초기 z {edges[np.nanargmax(M[:,-1])]:.2f})",
          flush=True)

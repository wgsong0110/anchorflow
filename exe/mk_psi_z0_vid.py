"""초기 z 띠별 평균 탄성 에너지를 **프레임마다** 그린 패널 영상.

입자 영상과 가로로 붙여 프레임 동기화해 보기 위한 것이다. 세로축이 초기 z,
가로축이 그 띠의 평균 log10(psi) 이고 우리와 PG 를 한 칸에 겹쳐 그린다.
"""
from __future__ import annotations
import argparse
import numpy as np
import torch
import imageio.v2 as imageio
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
from tqdm import tqdm

ap = argparse.ArgumentParser()
ap.add_argument("--dump", required=True)
ap.add_argument("--bands", type=int, default=24)
ap.add_argument("--t0", type=int, default=0, help="이 프레임부터만 그린다 (축 범위도 이 구간으로)")
ap.add_argument("--w", type=int, default=700)
ap.add_argument("--h", type=int, default=1100)
ap.add_argument("--fps", type=int, default=12)
ap.add_argument("--out", required=True)
a = ap.parse_args()


def L(p):
    try: return torch.load(p, map_location="cpu", weights_only=False)
    except TypeError: return torch.load(p, map_location="cpu")


D = L(a.dump)
z0 = np.asarray(D["x0"], dtype=np.float32)[:, 2]
fs = np.asarray(D["fscal"])[..., 4]
gf = D.get("gt_fscal")
gs = np.asarray(gf)[..., 4] if gf is not None else None
T = fs.shape[0]
edges = np.linspace(z0.min(), z0.max(), a.bands + 1)
cen = 0.5 * (edges[:-1] + edges[1:])
idx = np.clip(np.digitize(z0, edges) - 1, 0, a.bands - 1)
cnt = np.bincount(idx, minlength=a.bands)


def curves(P):
    M = np.full((T, a.bands), np.nan)
    for b in range(a.bands):
        m = idx == b
        if m.any():
            M[:, b] = np.log10(np.maximum(P[:, m].mean(1), 1e-12))
    return M


Co = curves(fs)
Cg = curves(gs) if gs is not None else None
_all = Co if Cg is None else np.concatenate([Co, Cg])
# 축 범위는 실제로 그리는 구간(t0 이후)만 보고 잡는다
_rng = _all[a.t0:] if Cg is None else np.concatenate([Co[a.t0:], Cg[a.t0:]])
xlo, xhi = np.nanmin(_rng), np.nanmax(_rng)
pad = 0.05 * (xhi - xlo)
xlo, xhi = xlo - pad, xhi + pad
print(f"초기 z {edges[0]:.3f}~{edges[-1]:.3f}, {a.bands} 띠 (띠당 입자 "
      f"{cnt.min()}~{cnt.max()}), log10(psi) 축 {xlo:.2f}~{xhi:.2f}, "
      f"프레임 {a.t0}~{T - 1} ({T - a.t0}장)", flush=True)

dpi = 100
fig, ax = plt.subplots(figsize=(a.w / dpi, a.h / dpi), dpi=dpi)
wr = imageio.get_writer(a.out, fps=a.fps, codec="libx264", quality=8,
                        macro_block_size=1)
for t in tqdm(range(a.t0, T), ncols=70):
    ax.clear()
    # 전 프레임 자취를 옅게 깔아 현재 프레임이 어디쯤인지 보이게 한다
    ax.plot(Co[a.t0:t + 1].T, cen, color="crimson", lw=0.4, alpha=0.12)
    if Cg is not None:
        ax.plot(Cg[a.t0:t + 1].T, cen, color="royalblue", lw=0.4, alpha=0.12)
    ax.plot(Co[t], cen, "-o", color="crimson", lw=2.2, ms=4,
            label="출력만 최적화")
    if Cg is not None:
        ax.plot(Cg[t], cen, "-o", color="royalblue", lw=2.2, ms=4,
                label="PG MPM")
    ax.set_xlim(xlo, xhi); ax.set_ylim(edges[0], edges[-1])
    ax.set_xlabel("띠별 평균 log10(psi)"); ax.set_ylabel("초기 z")
    ax.set_title(f"초기 z 띠별 탄성 에너지 평균   프레임 {t}", fontsize=12)
    ax.grid(alpha=0.25); ax.legend(loc="lower left", fontsize=10)
    fig.tight_layout()
    fig.canvas.draw()
    im = np.asarray(fig.canvas.buffer_rgba())[..., :3]
    wr.append_data(np.ascontiguousarray(im))
wr.close()
print(f"[저장] {a.out}  {T - a.t0} 프레임 (프레임 {a.t0}~{T - 1})", flush=True)

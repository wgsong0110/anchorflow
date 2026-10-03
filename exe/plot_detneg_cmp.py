"""여러 실행의 뒤집힌 셀/입자 비율을 프레임에 따라 겹쳐 그린다.

det 은 사면체 안에서 상수이므로 같은 프레임에서 det 값이 같은 입자를 한 셀로
묶어 센다.
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
ap.add_argument("--dumps", nargs="+", required=True, help="이름=경로")
ap.add_argument("--title", default="")
ap.add_argument("--out", required=True)
a = ap.parse_args()


def L(p):
    try: return torch.load(p, map_location="cpu", weights_only=False)
    except TypeError: return torch.load(p, map_location="cpu")


fig, ax = plt.subplots(1, 2, figsize=(13.5, 4.6), dpi=120)
for s in a.dumps:
    nm, path = s.split("=", 1)
    d = np.asarray(L(path)["fscal"])[..., 0]
    T, N = d.shape
    cells = np.zeros(T); parts = np.zeros(T)
    tot = len(np.unique(d[0]))
    for t in range(T):
        neg = d[t] < 0
        parts[t] = 100.0 * neg.mean()
        cells[t] = 100.0 * (len(np.unique(d[t][neg])) if neg.any() else 0) / tot
    ax[0].plot(np.arange(T), cells, lw=2, label=nm)
    ax[1].plot(np.arange(T), parts, lw=2, label=nm)
    print(f"[{nm}] 셀 {tot}, 뒤집힌 셀 최대 {cells.max():.2f}% "
          f"(프레임 {int(cells.argmax())}), 마지막 {cells[-1]:.2f}%, "
          f"입자 최대 {parts.max():.2f}%, 최소 det {d.min():.4f}", flush=True)
ax[0].set_ylabel("뒤집힌 셀 비율 (%)"); ax[0].set_title("셀 기준")
ax[1].set_ylabel("뒤집힌 셀 안 입자 비율 (%)"); ax[1].set_title("입자 기준")
for q in ax:
    q.set_xlabel("프레임"); q.grid(alpha=0.25); q.legend(fontsize=9)
fig.suptitle(a.title, fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.94]); fig.savefig(a.out)
print(f"[저장] {a.out}", flush=True)

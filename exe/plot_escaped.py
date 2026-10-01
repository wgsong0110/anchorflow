"""마지막 프레임에 **구를 벗어난 입자**의 초기 xy 를 흩뿌려 그린다.

"벗어났다" 는 기준은 PG MPM 자신의 공으로 잡는다: 마지막 프레임 PG 입자들의
중심에서 가장 먼 PG 입자까지의 거리를 그 공의 반지름으로 두고, 출력만 최적화의
입자가 그 밖에 있으면 벗어난 것이다. 색은 **마지막 프레임의 z**, 바탕에 원래
공의 테두리를 함께 그린다.
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
ap.add_argument("--dump", nargs="+", required=True)
ap.add_argument("--names", nargs="+", default=None,
                help="칸 제목. --dump 와 같은 개수 (이름에 '=' 가 들어가면 "
                     "'이름=경로' 꼴 파싱이 깨지므로 따로 받는다)")
ap.add_argument("--out", required=True)
ap.add_argument("--margin", type=float, default=1.0,
                help="PG 공 반지름의 몇 배 밖을 벗어난 것으로 볼지")
a = ap.parse_args()


def L(p):
    try: return torch.load(p, map_location="cpu", weights_only=False)
    except TypeError: return torch.load(p, map_location="cpu")


fig, ax = plt.subplots(1, len(a.dump), figsize=(5.6 * len(a.dump), 5.2),
                       dpi=120, squeeze=False)
for k, path in enumerate(a.dump):
    name = (a.names[k] if a.names and k < len(a.names)
            else _os.path.basename(path))
    D = L(path)
    P = D["pred"].float().numpy(); G = D["gt"].float().numpy()
    X0 = np.asarray(D["x0"], dtype=np.float32)
    cen = G[-1].mean(0)
    rad = float(np.linalg.norm(G[-1] - cen, axis=-1).max()) * a.margin
    dist = np.linalg.norm(P[-1] - cen, axis=-1)
    esc = dist > rad
    q = ax[0][k]
    # 원래 공의 테두리 (초기 xy 평면에서의 최대 반지름)
    c0 = X0.mean(0)
    r0 = float(np.linalg.norm(X0[:, :2] - c0[:2], axis=-1).max())
    q.add_patch(plt.Circle((c0[0], c0[1]), r0, fill=False, ec="0.4", lw=1.6,
                           ls="--"))
    if esc.any():
        sc = q.scatter(X0[esc, 0], X0[esc, 1], s=7.0, c=P[-1][esc, 2],
                       cmap="viridis", linewidths=0)
        cb = fig.colorbar(sc, ax=q); cb.set_label("마지막 프레임의 z")
    q.set_aspect("equal")
    q.set_xlabel("초기 x"); q.set_ylabel("초기 y")
    q.set_title(f"{name}\n벗어난 입자 {int(esc.sum())} / {len(esc)} "
                f"({100*esc.mean():.2f}%)   기준 반지름 {rad:.4f}", fontsize=10)
    print(f"[{name}] 벗어남 {int(esc.sum())}/{len(esc)} ({100*esc.mean():.2f}%), "
          f"PG 공 반지름 {rad:.4f}, 마지막 z 범위 "
          f"{P[-1][esc, 2].min():.3f}~{P[-1][esc, 2].max():.3f}"
          if esc.any() else f"[{name}] 벗어난 입자 없음", flush=True)
fig.tight_layout(); fig.savefig(a.out)
print(f"[저장] {a.out}", flush=True)

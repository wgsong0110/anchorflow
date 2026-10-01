"""마지막 프레임에 **구를 벗어난 입자**의 초기 xy 를 흩뿌려 그린다.

기준은 **마지막 프레임의 공**이다: 초기 중심에 가장 가까운 입자를 중심 입자로
잡고(그 입자를 계속 따라간다), 마지막 프레임에서 그 입자로부터 공 반지름보다
멀리 떨어진 입자를 벗어난 것으로 본다. 그중 **중심보다 높이 있는 것만** 그린다
(아래로 처진 것은 매달린 공에서 당연한 변형이라 구분이 안 된다).

색은 마지막 프레임의 z, 점선은 원래 공의 테두리다.
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
ap.add_argument("--r", type=float, default=0.0,
                help="공 반지름. 0 이면 초기 위치에서 중심 입자까지의 최대 "
                     "거리로 잡는다")
ap.add_argument("--margin", type=float, default=1.0,
                help="반지름의 몇 배 밖을 벗어난 것으로 볼지")
ap.add_argument("--above", action="store_true", default=True,
                help="중심보다 높이 있는 입자만 그린다 (기본)")
ap.add_argument("--all_dirs", dest="above", action="store_false")
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
    # 중심 **입자**를 정하고 (초기 중심에 가장 가까운 것) 그 입자를 따라간다
    c0 = X0.mean(0)
    cid = int(np.linalg.norm(X0 - c0, axis=-1).argmin())
    rad = (a.r if a.r > 0
           else float(np.linalg.norm(X0 - X0[cid], axis=-1).max())) * a.margin
    cen = P[-1][cid]                       # 마지막 프레임의 공 중심
    dist = np.linalg.norm(P[-1] - cen, axis=-1)
    esc = dist > rad
    if a.above:
        esc = esc & (P[-1][:, 2] > cen[2])
    q = ax[0][k]
    # 원래 공의 테두리 (초기 xy 평면에서의 최대 반지름)
    r0 = float(np.linalg.norm(X0[:, :2] - c0[:2], axis=-1).max())
    q.add_patch(plt.Circle((c0[0], c0[1]), r0, fill=False, ec="0.4", lw=1.6,
                           ls="--"))
    if esc.any():
        sc = q.scatter(X0[esc, 0], X0[esc, 1], s=7.0, c=P[-1][esc, 2],
                       cmap="viridis", linewidths=0)
        cb = fig.colorbar(sc, ax=q); cb.set_label("마지막 프레임의 z")
    q.set_aspect("equal")
    q.set_xlabel("초기 x"); q.set_ylabel("초기 y")
    q.set_title(f"{name}\n중심보다 위에서 벗어난 입자 {int(esc.sum())} / "
                f"{len(esc)} ({100*esc.mean():.2f}%)\n중심 입자 {cid}, "
                f"반지름 {rad:.4f}, 마지막 중심 z {cen[2]:.4f}", fontsize=9)
    print(f"[{name}] 벗어남 {int(esc.sum())}/{len(esc)} ({100*esc.mean():.2f}%), "
          f"반지름 {rad:.4f}, 중심 입자 {cid}, 마지막 z 범위 "
          f"{P[-1][esc, 2].min():.3f}~{P[-1][esc, 2].max():.3f}"
          if esc.any() else f"[{name}] 벗어난 입자 없음", flush=True)
fig.tight_layout(); fig.savefig(a.out)
print(f"[저장] {a.out}", flush=True)

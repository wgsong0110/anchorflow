"""train_deform.py 가 남긴 손실 이력을 그린다.

기록은 (위치, 야코비안, 정지 기준선, 앵커, 앵커 상대) 다섯 항이고 20 스텝마다
남는다. 위치 오차는 **정지 기준선과 같은 칸에** 그려야 뜻이 있다 -- 절대값이
내려가도 그 구간이 원래 안 움직이는 구간이면 배운 게 없기 때문이다.
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
import numpy as np                                                # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--json", nargs="+", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--smooth", type=int, default=9, help="이동 중앙값 창(기록 단위)")
a = ap.parse_args()


def smooth(v, w):
    if w <= 1 or len(v) < w:
        return v
    return np.array([np.median(v[max(0, i - w + 1):i + 1]) for i in range(len(v))])


runs = []
for p in a.json:
    d = json.load(open(p))
    h = np.array(d["loss_hist"], dtype=float)          # [T,5]
    runs.append((d.get("tag", os.path.basename(p)), d, h))

fig, ax = plt.subplots(1, 4, figsize=(18, 3.8))
COL = ["#d00000", "#0077b6", "#2a9d8f", "#9d4edd", "#f4a261"]
for i, (tag, d, h) in enumerate(runs):
    n = h.shape[0]
    step = np.arange(n) * 20
    c = COL[i % len(COL)]
    pos = 100 * np.sqrt(np.maximum(smooth(h[:, 0], a.smooth), 0))
    still = 100 * np.sqrt(np.maximum(smooth(h[:, 2], a.smooth), 0))
    ax[0].plot(step, pos, color=c, label=f"{tag}")
    ax[0].plot(step, still, color=c, ls=":", lw=1, label=f"{tag} still")
    ax[1].plot(step, smooth(h[:, 0], a.smooth) / np.maximum(
        smooth(h[:, 2], a.smooth), 1e-20) ** 1.0, color=c, label=tag)
    ax[2].plot(step, smooth(h[:, 1], a.smooth), color=c, label=f"{tag} J")
    ax[2].plot(step, smooth(h[:, 3], a.smooth), color=c, ls="--",
               label=f"{tag} anchor")
    # 실제로 밟는 목적함수: wx + lambda_J * wJ + lambda_anchor * wa
    ar = d.get("args", {})
    lJ = float(ar.get("lambda_J", 0.0)); la_ = float(ar.get("lambda_anchor", 0.0))
    tot = h[:, 0] + lJ * h[:, 1] + la_ * h[:, 3]
    ax[3].plot(step, smooth(tot, a.smooth), color=c,
               label=f"{tag}  (lJ={lJ:g}, la={la_:g})")

ax[0].set_ylabel("position RMSE (%)"); ax[0].set_yscale("log")
ax[0].set_title("position error vs standing-still", fontsize=10)
ax[1].axhline(1.0, color="0.5", lw=0.8)
ax[1].set_yscale("log"); ax[1].set_title("error / still  (<1 means it learned)", fontsize=10)
ax[2].set_yscale("log"); ax[2].set_title("jacobian loss and anchor loss", fontsize=10)
ax[3].set_yscale("log")
ax[3].set_title("total objective  x + lJ*J + la*anchor", fontsize=10)
for A in ax:
    A.set_xlabel("step"); A.grid(alpha=0.3); A.legend(fontsize=6.5)
fig.tight_layout()
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
fig.savefig(a.out, dpi=150)
print(f"[저장] {a.out}", flush=True)
for tag, d, h in runs:
    r = d.get("rollout", [])
    print(f"[{tag}] 마지막 위치 {100*h[-1,0]**0.5:.3f}% "
          f"(정지 {100*h[-1,2]**0.5:.3f}%), 기록 {h.shape[0]} 점", flush=True)
print("LC_OK", flush=True)

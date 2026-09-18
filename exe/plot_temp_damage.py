"""온도·반경·손상 변수를 프레임 축에 나란히 그린다.

probe_temp_damage.py 가 남긴 json 을 읽는다. 궤적을 여러 개 겹쳐 그리는 것이
핵심이다 -- 온도 곡선이 손상량과 무관하게 같은 모양이라는 것은 한 궤적만 보면
절대 안 보인다.
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--json", nargs="+", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

runs = []
for p in a.json:
    d = json.load(open(p))
    runs.append((d.get("traj", os.path.basename(p)), d["rows"]))

fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
COL = ["#d00000", "#0077b6", "#2a9d8f", "#9d4edd"]

for i, (tag, rows) in enumerate(runs):
    f = [r["f"] for r in rows]
    c = COL[i % len(COL)]
    ax[0].plot(f, [r["t_med"] for r in rows], color=c, label=tag)
    ax[1].plot(f, [r["r_med"] for r in rows], color=c, label=tag)
    ax[2].plot(f, [r["s1_p99"] for r in rows], color=c, label=f"{tag} sigma1 p99")
    ax[2].plot(f, [r["sep_p99"] for r in rows], color=c, ls="--",
               label=f"{tag} sep p99")

ax[0].set_yscale("log")
ax[0].set_title("anchor temperature $t_a$ (median)", fontsize=10)
ax[0].axhline(54.598, color="0.6", lw=0.8, ls=":")
ax[0].text(f[-1], 54.598, " clamp $e^4$", va="center", fontsize=7, color="0.4")
ax[1].set_title("anchor radius $r_a/h$ (median)", fontsize=10)
ax[2].set_title("damage (p99 over anchors)", fontsize=10)
for A in ax:
    A.set_xlabel("frame")
    A.grid(alpha=0.3)
    A.legend(fontsize=6.5)
fig.tight_layout()
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
fig.savefig(a.out, dpi=150)
print(f"[저장] {a.out}", flush=True)
print("PLOT_OK", flush=True)

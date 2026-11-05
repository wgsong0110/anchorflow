"""절 CM13.4: 프레임별 손상 입자 몫 곡선 (정답 GF 다시 돌림 · 우리 (2) SNH) -- wmd_stats.json -> png.

  python3 exe/wmd_curves.py --stats wmd_stats.json --out curve.png
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib import font_manager                  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--stats", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--font", default="/home/wgsong/.fonts/NotoSansCJKkr-Regular.otf")
a = ap.parse_args()
font_manager.fontManager.addfont(a.font)
plt.rcParams["font.family"] = font_manager.FontProperties(fname=a.font).get_name()
plt.rcParams["axes.unicode_minus"] = False
S = json.load(open(a.stats))
BL, OR = "#2a78d6", "#eb6834"
fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
for side, c, lab in (("gt", BL, "정답 (GF 다시 돌림)"), ("ours", OR, "우리 (2) Stable Neo-Hookean")):
    st = S[side]; t = list(range(len(st)))
    ax[0].plot(t, [100 * s["gt0"] for s in st], color=c, lw=2, label=f"{lab}: logJp > 0")
    ax[0].plot(t, [100 * s["gt04"] for s in st], color=c, lw=2, ls="--", label=f"{lab}: logJp > 0.4")
    ax[1].plot(t, [s["mean"] for s in st], color=c, lw=2, label=f"{lab}: 평균")
    ax[1].plot(t, [s["p99"] for s in st], color=c, lw=2, ls=":", label=f"{lab}: 99 백분위")
ax[0].set_title("손상 입자 몫 (위치 유한 입자 중)")
ax[0].set_ylabel("%"); ax[1].set_title("logJp 분포"); ax[1].set_ylabel("logJp")
for x in ax:
    x.set_xlabel("프레임"); x.grid(alpha=0.25, lw=0.6); x.legend(fontsize=8, frameon=False)
    for s in ("top", "right"):
        x.spines[s].set_visible(False)
fig.suptitle("수박 손상 변수 logJp (NACC 경화) -- logJp > 0: 응집력 p0 사라짐 (손상), > 0.4: GF 러너의 숨김 문턱", fontsize=11)
fig.tight_layout(); fig.savefig(a.out, dpi=130)
print("[곡선]", a.out)

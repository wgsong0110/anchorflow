"""exe/rollout_nn_dist.py 의 rollout_nn_dist.json -> 그림·표 (계산 없음).

  python exe/rollout_nn_plot.py --json DIR/rollout_nn_dist.json --png OUT.png --md OUT.md
"""
import argparse
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402

for f in fm.findSystemFonts() + glob.glob(os.path.expanduser("~/.fonts/*")):
    if "Noto" in f and ("CJK" in f or "KR" in f):
        fm.fontManager.addfont(f); plt.rcParams["font.family"] = fm.FontProperties(fname=f).get_name(); break
plt.rcParams["axes.unicode_minus"] = False

ap = argparse.ArgumentParser()
ap.add_argument("--json", required=True)
ap.add_argument("--png", required=True)
ap.add_argument("--md", required=True)
a = ap.parse_args()
J = json.load(open(a.json))
TRAJ = ["L000", "0000", "0001", "0002", "0003"]
STY = [("c18", "18채널 (8500 반복)", "#2a78d6", "o"), ("c51", "51채널 (7000 반복)", "#eb6834", "s"),
       ("old", "기존 다섯 궤적 (15000 반복)", "#1baf7a", "^"), ("gt", "정답 L000", "#222222", "D")]


def band(B, key):
    """학습 상태 프레임 i 별 5 궤적 중 최소·최대 (프레임 0 은 정답 시작이 다섯 궤적 공통이라 뺀다)."""
    lo, hi = np.full(80, np.nan), np.full(80, np.nan)
    for i in range(1, 80):
        v = [b[key] for b in B if b["frame"] == i]
        lo[i], hi[i] = min(v), max(v)
    return lo, hi


fig = plt.figure(figsize=(15, 11))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], hspace=0.28, wspace=0.18)
fr = np.arange(80)
for c, (key, ttl) in enumerate([("d", "위치 최근접 거리  mean‖X−P‖ / EXT"), ("d_com", "질량중심 맞춘 최근접 거리 / EXT")]):
    ax = fig.add_subplot(gs[0, c])
    lo, hi = band(J["base_other"], "d" if key == "d" else "d_com")
    ax.fill_between(fr[1:], lo[1:], hi[1:], color="#bbbbbb", alpha=0.55, lw=0, label="학습 상태 간: 다른 궤적만 (5 궤적 범위)")
    lo2, hi2 = band(J["base_self"], "d" if key == "d" else "d_com")
    ax.fill_between(fr[1:], lo2[1:], hi2[1:], color="#777777", alpha=0.45, lw=0, label="학습 상태 간: 자기 제외 전부 (이웃 프레임 포함)")
    for k, nm, col, mk in STY:
        R = J["res"][k]
        ax.plot([r["frame"] for r in R], [max(r[key], 1e-7) for r in R], color=col, lw=2, label=nm)
    ax.set_yscale("log"); ax.set_xlim(0, 81); ax.set_xlabel("롤아웃 프레임 (학습 상태는 프레임 번호 기준)")
    ax.set_ylabel("거리 (EXT 배)"); ax.set_title(ttl, fontsize=12)
    ax.grid(True, which="major", color="#e5e5e5", lw=0.8); ax.set_axisbelow(True)
    if c == 0:
        ax.legend(fontsize=9, loc="lower right")
ax = fig.add_subplot(gs[1, :])
for q, (k, nm, col, mk) in enumerate(STY):
    R = J["res"][k]
    y = [TRAJ.index(r["traj"]) * 80 + r["sframe"] + (q - 1.5) * 0.9 for r in R]
    ax.scatter([r["frame"] for r in R], y, s=22, color=col, marker=mk, label=nm, edgecolors="white", linewidths=0.4, zorder=3)
for j in range(1, 5):
    ax.axhline(j * 80 - 0.5, color="#999999", lw=0.8)
ax.plot([0, 80], [0, 80], color="#cccccc", lw=1, ls="--", zorder=1, label="L000 같은 프레임 (대각선)")
ax.set_yticks([j * 80 + 40 for j in range(5)]); ax.set_yticklabels([f"{t}\n(프레임 0–79)" for t in TRAJ])
ax.set_ylim(-2, 402); ax.set_xlim(0, 81)
ax.set_xlabel("롤아웃 프레임"); ax.set_ylabel("위치 최근접 학습 상태 (궤적·프레임)")
ax.set_title("최근접 학습 상태가 어느 궤적·프레임인가 (각 궤적 칸 안의 세로 위치 = 그 궤적의 프레임)", fontsize=12)
ax.grid(True, color="#eeeeee", lw=0.8); ax.set_axisbelow(True); ax.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=5, frameon=False)
fig.suptitle(f"lego L000 롤아웃 상태와 학습 상태 400 개(L000·0000~0003 출력만 최적화 덤프) 사이 최근접 거리  "
             f"(EXT = {J['EXT']:.4f}, 입자 {J['N']})", fontsize=13)
fig.savefig(a.png, dpi=130, bbox_inches="tight")

# 표
FR = [1, 10, 20, 40, 60, 80]


def pct(x):
    return "-" if x is None else f"{100 * x:.3f}%"


L = [f"# lego L000 롤아웃 상태 대 학습 상태 400 개 최근접 거리\n",
     f"거리 = mean_p ‖X_t,p − P_S,p‖ / EXT, EXT = {J['EXT']:.6f} (L000 프레임 0 바운딩박스 대각선, 고정). "
     f"질량중심 = traj_mass 질량 가중. 속도 = 위치 최근접 짝의 mean ‖v − v_S‖ / (EXT/h), h = 1/60. "
     f"학습 상태 = 궤적마다 프레임 0 정답 시작 + 출력만 최적화 덤프 프레임 1..79.\n",
     "| 실행 | 프레임 | 최근접 거리 | 최근접 상태 | 질량중심 맞춘 거리 | 그 최근접 상태 | 속도 차 | L000 같은 프레임 상태와 거리 |",
     "|---|---|---|---|---|---|---|---|"]
for k, nm, _, _ in STY:
    R = {r["frame"]: r for r in J["res"][k]}
    for f in FR:
        r = R.get(f)
        if r is None:
            continue
        dv = f"{100 * r['dv']:.3f}%" if r.get("dv") is not None else "-"
        L.append(f"| {nm} | {f} | {pct(r['d'])} | {r['traj']} #{r['sframe']} | {pct(r['d_com'])} | "
                 f"{r['traj_com']} #{r['sframe_com']} | {dv} | {pct(r.get('d_same_frame_L000'))} |")
L += ["", "## 기준선: 학습 상태끼리 최근접 거리 (프레임 1–79 상태, 5 궤적)", "",
      "| 기준 | 프레임 | 최소 | 중앙 | 최대 | 질량중심 맞춤 중앙 |", "|---|---|---|---|---|---|"]
for nm, B in (("다른 궤적만", J["base_other"]), ("자기 제외 전부 (이웃 포함)", J["base_self"])):
    for f in FR + ["전체"]:
        S = [b for b in B if (b["frame"] > 0 if f == "전체" else b["frame"] == min(f, 79))]
        v = np.array([b["d"] for b in S]); vc = np.array([b["d_com"] for b in S])
        L.append(f"| {nm} | {f if f != 80 else '79'} | {100 * v.min():.3f}% | {100 * np.median(v):.3f}% | {100 * v.max():.3f}% | {100 * np.median(vc):.3f}% |")
open(a.md, "w").write("\n".join(L) + "\n")
print("\n".join(L))

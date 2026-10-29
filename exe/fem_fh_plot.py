"""exe/fem_fh_stats.py 결과 (fh_stats.npz/json) 그림 (로컬, torch 없이).
사용: python exe/fem_fh_plot.py <출력 접두> <라벨>=<실행 디렉토리> ...
  <접두>_ts.png    : 실행마다 한 행 -- 뜨거운 셀 (폭주 노드 이웃, ‖P‖ > 정상 셀 99 %) 입자 대표 6 개의 프레임별 σ_min·σ_max·det F_p·이웃 벌어짐·자기 셀 ‖P‖·셀 입자 수
                     (실선 = 우리, 점선 = 정답 같은 입자), 정상 입자 1~99 % 띠 (파랑)·희소 정상 (입자 1~2 개 셀) 중앙 (회색), x = 발산 프레임 기준.
  <접두>_dist.png  : 발산 프레임 시작 시점 집단별 상자그림 (뜨거운 / 폭주 셀 전체 / 희소 정상 / 정상) -- 우리·정답 σ_min, det, 이웃 벌어짐 (우리·정답), 이웃 A 차 (발산 한 프레임 앞 끝), 바닥 높이.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                                       # noqa: E402
from matplotlib import font_manager                                                   # noqa: E402
import numpy as np                                                                    # noqa: E402

for f in ("/home/wgsong/.fonts/NotoSansCJKkr-Regular.otf", "/home/wgsong/.fonts/NotoSansCJKkr-Bold.otf"):
    if os.path.exists(f):
        font_manager.fontManager.addfont(f)
plt.rcParams["font.family"] = "Noto Sans CJK KR"
plt.rcParams["axes.unicode_minus"] = False
C_N, C_R, C_S = "#2a78d6", "#eb6834", "#8a8984"
pref = sys.argv[1]
RUNS = [q.split("=", 1) for q in sys.argv[2:]]

TS = [("s2", "gs2", "입자 F σ_min", False), ("s0", "gs0", "입자 F σ_max", False), ("detF", "gdet", "det F_p", False),
      ("mis", "gmis", "이웃 벌어짐 ‖F_p − 이웃 16 평균‖", True), ("PT", None, "자기 셀 ‖P(F̄)‖ (Pa)", True), ("npt", None, "자기 셀 입자 수", False)]
fig, axs = plt.subplots(len(RUNS), len(TS), figsize=(3.3 * len(TS), 2.9 * len(RUNS)), squeeze=False)
for ri, (lab, run) in enumerate(RUNS):
    Z = np.load(os.path.join(run, "fh_stats.npz"))
    fr = Z["frames"] - int(Z["fdiv"])
    nh = Z["hot_s2"].shape[0]
    pick = np.argsort(Z["hot_s2"][:, -1])[: min(6, nh)] if nh else []
    for mi, (k, gk, title, lg) in enumerate(TS):
        ax = axs[ri, mi]
        nv = Z[f"norm_{k}"]
        lo, md, hi = (np.nanpercentile(nv, q, axis=0) for q in (1, 50, 99))
        ax.fill_between(fr, lo, hi, color=C_N, alpha=0.18, lw=0, label="정상 입자 1~99 %")
        ax.plot(fr, md, color=C_N, lw=1.5, label="정상 중앙")
        ax.plot(fr, np.nanmedian(Z[f"sparse_{k}"], 0), color=C_S, lw=1.5, label="희소 정상 (셀 입자 1~2) 중앙")
        for j, p in enumerate(pick):
            ax.plot(fr, Z[f"hot_{k}"][p], color=C_R, lw=1.2, alpha=0.9, label="뜨거운 셀 입자 (우리)" if j == 0 else None)
            if gk is not None:
                ax.plot(fr, Z[f"hot_{gk}"][p], color=C_R, lw=1.0, ls=":", alpha=0.9, label="같은 입자 정답" if j == 0 else None)
        if gk is not None:
            ax.plot(fr, np.nanmedian(Z[f"norm_{gk}"], 0), color=C_N, lw=1.0, ls=":", label="정상 정답 중앙")
        if lg:
            ax.set_yscale("log")
        ax.set_title(f"{lab}: {title}", fontsize=9)
        ax.grid(alpha=0.25, lw=0.5)
        if ri == len(RUNS) - 1:
            ax.set_xlabel("발산 프레임 기준 프레임")
        if ri == 0 and mi == 0:
            ax.legend(fontsize=6.5, loc="lower left")
fig.tight_layout()
fig.savefig(pref + "_ts.png", dpi=130)
plt.close(fig)

GRP = [("hot", "뜨거운 셀", C_R), ("run", "폭주 셀 전체", "#f3a37e"), ("sparse", "희소 정상", C_S), ("norm", "정상", C_N)]
DM = [("s2", "σ_min 우리", False), ("gs2", "σ_min 정답", False), ("detF", "det F 우리", False), ("gdet", "det F 정답", False),
      ("mis", "이웃 벌어짐 우리", True), ("gmis", "이웃 벌어짐 정답", True), ("Amis", "이웃 A 차 (발산 −1 프레임 끝)", True), ("height", "바닥 높이 (m)", False)]
fig, axs = plt.subplots(len(RUNS), len(DM), figsize=(2.6 * len(DM), 2.8 * len(RUNS)), squeeze=False)
for ri, (lab, run) in enumerate(RUNS):
    Z = np.load(os.path.join(run, "fh_stats.npz"))
    for mi, (k, title, lg) in enumerate(DM):
        ax = axs[ri, mi]
        data = []
        for g, _, _ in GRP:
            v = Z[f"{g}_{k}"][:, -2 if k == "Amis" else -1]
            v = v[np.isfinite(v)]
            if lg:
                v = v[v > 0]
            data.append(v if v.size else np.array([np.nan]))
        bp = ax.boxplot(data, widths=0.6, patch_artist=True, showfliers=False)
        for b, (_, _, c) in zip(bp["boxes"], GRP):
            b.set_facecolor(c)
            b.set_alpha(0.7)
        ax.set_xticks(range(1, len(GRP) + 1), [g[1] for g in GRP], fontsize=7, rotation=30)
        if lg and any(np.nanmax(q) > 0 for q in data if np.isfinite(q).any()):
            ax.set_yscale("log")
        ax.set_title(f"{lab}: {title}", fontsize=9)
        ax.grid(alpha=0.25, lw=0.5, axis="y")
fig.tight_layout()
fig.savefig(pref + "_dist.png", dpi=130)
plt.close(fig)
print("저장", pref + "_ts.png", pref + "_dist.png")

"""절 CM14.3 그래프 (로컬, numpy + matplotlib): 프레임별 장벽 에너지 합 (로그축) 과 장벽 걸린 칸 수, lego L000 · wolf 한 열씩.
입력 = 묶음 barnode_e 의 <판>_<씬>/fem.jsonl 의 be (exe/fem_an_rt_sim.py --bar_ecell): Eend = 프레임 끝 (마지막 걸음 끝) 전체 장벽 에너지 (표본점 + 노드),
Emax = 프레임 안 걸음 끝 (K = 2) 전체 에너지의 최대, ncl = 프레임 끝 에너지 > 0 칸 수, ncm = 프레임 안 걸음 끝 어디서든 에너지 > 0 칸 수. 에너지 0 인 프레임은 로그축에서 빈칸.
사용: python exe/fem_bener_curve.py <barnode_e 디렉토리> <출력 png>"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import numpy as np

B, OUT = sys.argv[1:3]
for f in ("NotoSansCJKkr-Regular.otf", "NotoSansCJKkr-Bold.otf"):
    p = os.path.expanduser(f"~/.fonts/{f}")
    if os.path.exists(p):
        fm.fontManager.addfont(p)
plt.rcParams.update({"font.family": "Noto Sans CJK KR", "axes.unicode_minus": False, "font.size": 10})
VS = [("o", "(B) 장벽 (표본점)", "#2a78d6"), ("na5", "(가) 표본점 + 노드, 두께 ½dx", "#eb6834"), ("na7", "(가) 표본점 + 노드, 두께 ¾dx", "#1baf7a"),
      ("no5", "(나) 노드만, 두께 ½dx", "#eda100"), ("no7", "(나) 노드만, 두께 ¾dx", "#e87ba4")]
SC = [("L000", "lego L000 (80 프레임)"), ("wolf", "wolf (50 프레임)")]
fig, ax = plt.subplots(2, 2, figsize=(13, 8.5), sharex="col", gridspec_kw=dict(height_ratios=[3, 2]))
for j, (t, tt) in enumerate(SC):
    for v, nm, col in VS:
        p = f"{B}/{v}_{t}/fem.jsonl"
        if not os.path.exists(p):
            continue
        R = [json.loads(q) for q in open(p)]
        R = [r for r in R if "be" in r]
        fr = np.array([r["t"] + 1 for r in R])
        ee = np.array([r["be"]["Eend"] for r in R]); em = np.array([r["be"]["Emax"] for r in R])
        ax[0, j].plot(fr, np.where(ee > 0, ee, np.nan), "-", color=col, lw=2, label=nm)
        ax[0, j].plot(fr, np.where(em > 0, em, np.nan), "--", color=col, lw=1.4)
        ax[1, j].plot(fr, [r["be"]["ncl"] for r in R], "-", color=col, lw=2)
        ax[1, j].plot(fr, [r["be"]["ncm"] for r in R], "--", color=col, lw=1.4)
    ax[0, j].set_yscale("log")
    ax[0, j].set_title(tt, fontsize=12, loc="left")
    ax[0, j].set_ylabel("장벽 에너지 합 (시뮬 단위, 로그축)")
    ax[1, j].set_ylabel("장벽 걸린 칸 수")
    ax[1, j].set_xlabel("프레임")
    for a_ in ax[:, j]:
        a_.grid(True, color="#e4e4e0", lw=0.8); a_.set_axisbelow(True)
        for s_ in ("top", "right"):
            a_.spines[s_].set_visible(False)
h, l = ax[0, 0].get_legend_handles_labels()
h += [plt.Line2D([], [], color="#555", lw=2, ls="-"), plt.Line2D([], [], color="#555", lw=1.4, ls="--")]
l += ["실선 = 프레임 끝 (마지막 걸음 끝) 값", "점선 = 프레임 안 걸음 끝 (K = 2) 최대"]
fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=10)
fig.suptitle("절 CM14.3 프레임별 장벽 에너지 (표본점 몫 + 노드 몫, 바닥 거리 0 < d < 장벽 두께 인 활성 점·노드) 와 장벽 걸린 칸 수 -- 같은 명령 + --bar_ecell (결과 불변); wolf 는 (가) = (나) 비트 같음 (선 겹침)", fontsize=11, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0.08, 1, 0.96))
fig.savefig(OUT, dpi=150)
print("[저장]", OUT)

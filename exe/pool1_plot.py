"""pool1 (lego 한 궤적 풀 학습) 학습 곡선 비교 그림 (한국어).
색 = 몸통 (파랑 일반 GNN, 주황 Graph U-Net), 선 = 학습 방식 (실선 한 궤적 풀, 파선 한 궤적 상태 적합 sf1, 점선 다섯 궤적 상태 적합).
여섯 실행 모두 관성 예측 없음, 바닥 softplus 0.01 + 미끄러짐 방지, 15000 반복.
왼쪽: L000 정답 시작 80 프레임 롤아웃 위치 오차 % (풀 = 500 반복마다 저장한 망으로 따로 굴림, 상태 적합 = 학습 중 검증)
가운데: 풀 학습 목적함수 (100 반복 이동평균) | 오른쪽: 풀 검증 증분 포텐셜 (홀드아웃 4 궤적 80 프레임)
사용: python3 exe/pool1_plot.py <pool1_gnn curves.json> <pool1_gunet curves.json> <sf1_gnn curves.json> <sf1_gunet curves.json> out.png"""
import json
import subprocess
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

*srcs, out = sys.argv[1:]
F = subprocess.run("fc-list :lang=ko file | head -1 | cut -d: -f1", shell=True, capture_output=True, text=True).stdout.strip()
if F:
    fm.fontManager.addfont(F)
    plt.rcParams["font.family"] = fm.FontProperties(fname=F).get_name()
plt.rcParams["axes.unicode_minus"] = False
M = {}
for s in srcs:
    M.update(json.load(open(s)))
BLUE, ORANGE = "#2a78d6", "#eb6834"
# (실행, 이름, 색, 선, L000 오차 키, 반복 축 +1 여부)
RUNS = [("pool1_gnn", "일반 GNN · 한 궤적 풀", BLUE, "-", "L000롤아웃_위치오차%", 0),
        ("sf1_gnn", "일반 GNN · 한 궤적 상태 적합", BLUE, "--", "상태적합_롤아웃/위치오차_평균%", 1),
        ("noinertial", "일반 GNN · 다섯 궤적 상태 적합", BLUE, ":", "다궤적적합_롤아웃_위치오차%/L000", 1),
        ("pool1_gunet", "Graph U-Net · 한 궤적 풀", ORANGE, "-", "L000롤아웃_위치오차%", 0),
        ("sf1_gunet", "Graph U-Net · 한 궤적 상태 적합", ORANGE, "--", "상태적합_롤아웃/위치오차_평균%", 1),
        ("gunet5_near", "Graph U-Net · 다섯 궤적 상태 적합", ORANGE, ":", "다궤적적합_롤아웃_위치오차%/L000", 1),
        ("pool1s32_gunet", "Graph U-Net · 한 궤적 풀, 무작위 폐기 끔·풀 32", "#1baf7a", "-", "L000롤아웃_위치오차%", 0),
        ("pool1s32g3_gunet", "Graph U-Net · 한 궤적 풀, 무작위 폐기 끔·풀 32·문턱 1e-3", "#e87ba4", "-", "L000롤아웃_위치오차%", 0),
        ("pool1s32r_gunet", "Graph U-Net · 낙하 분포 시작 풀, 무작위 폐기 끔·풀 32", "#1baf7a", "-.", "L000롤아웃_위치오차%", 0),
        ("pool1s32g3r_gunet", "Graph U-Net · 낙하 분포 시작 풀, 무작위 폐기 끔·풀 32·문턱 1e-3", "#e87ba4", "-.", "L000롤아웃_위치오차%", 0)]
RUNS = [r for r in RUNS if r[0] in M]


def ser(run, key):
    s = sorted(M.get(run, {}).get(key, []))
    return np.array([p[0] for p in s]), np.array([p[1] for p in s])


def smooth(y, k=100):
    if len(y) < k:
        return y
    c = np.cumsum(np.insert(y, 0, 0.0))
    return np.concatenate([c[1:k] / np.arange(1, k), (c[k:] - c[:-k]) / k])


fig, axs = plt.subplots(1, 3, figsize=(26, 8.5), gridspec_kw={"width_ratios": [1.4, 1, 1]})
ax = axs[0]
txt = []
for run, lab, col, ls, key, p1 in RUNS:
    x, y = ser(run, key)
    if len(x):
        ax.plot(x + p1, y, ls, marker="o", ms=5, color=col, lw=2, label=lab)
        txt.append(f"{lab}: 마지막 {y[-1]:.2f}%, 최저 {y.min():.2f}% (반복 {int(x[np.argmin(y)]) + p1})")
ax.set_title("L000 정답 시작 80 프레임 롤아웃 위치 오차 %", fontsize=13)
ax.set_xlabel("반복"); ax.grid(alpha=0.3); ax.legend(fontsize=10, loc="upper right")
for ax, key, tt, sm in ((axs[1], "학습/목적함수", "풀 학습 목적함수 (100 반복 이동평균)", True),
                        (axs[2], "검증/증분포텐셜", "풀 검증 증분 포텐셜 (홀드아웃 4 궤적, 80 프레임)", False)):
    for run, lab, col, ls, _, _ in RUNS:
        if not run.startswith("pool1"):  # 풀 학습 실행만
            continue
        x, y = ser(run, key)
        if len(x):
            ax.plot(x, smooth(y) if sm else y, ls, marker=None if sm else "o", ms=5, color=col, lw=2, label=lab)
    ax.set_title(tt, fontsize=13); ax.set_xlabel("반복"); ax.grid(alpha=0.3); ax.legend(fontsize=10); ax.set_yscale("log")
fig.suptitle("lego 학습 곡선: 한 궤적 풀 학습 vs 한 궤적·다섯 궤적 상태 적합 (관성 예측 없음, 바닥 softplus + 미끄러짐 방지)", fontsize=15)
fig.text(0.01, 0.005, "\n".join("  |  ".join(txt[i:i + 3]) for i in range(0, len(txt), 3)), fontsize=9)
fig.tight_layout(rect=(0, 0.07, 1, 0.97))
fig.savefig(out, dpi=110)
print("그림", out)

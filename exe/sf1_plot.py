"""sf1 (lego L000 한 궤적 학습) 학습 곡선 비교 그림 (한국어).
색 = 몸통 (파랑 일반 GNN 메시지 패싱 8 층 128, 주황 Graph U-Net h·2h·4h), 선 = 학습 데이터 (실선 L000 한 궤적, 점선 다섯 궤적).
네 실행 모두 관성 예측 없음, 바닥 softplus 0.01 + 미끄러짐 방지, 손실 증분 포텐셜, 배치 8, 15000 반복.
위: 검증(정답 시작 80 프레임 롤아웃) L000 위치 오차 % | 학습 표본 증분 포텐셜 E | 학습 표본 출력 오차 %
아래: 학습 손실 | 기울기 노름 | 마지막 값 글
사용: python3 exe/sf1_plot.py <sf1_gnn curves.json> <sf1_gunet curves.json> out.png"""
import json
import subprocess
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

g_src, u_src, out = sys.argv[1:4]
F = subprocess.run("fc-list :lang=ko file | head -1 | cut -d: -f1", shell=True, capture_output=True, text=True).stdout.strip()
if F:
    fm.fontManager.addfont(F)
    plt.rcParams["font.family"] = fm.FontProperties(fname=F).get_name()
plt.rcParams["axes.unicode_minus"] = False
M = {}
for s in (g_src, u_src):
    M.update(json.load(open(s)))
BLUE, ORANGE = "#2a78d6", "#eb6834"
# (실행, 이름, 색, 선, 접두: 한 궤적 = 상태적합, 다섯 궤적 = 다궤적적합)
RUNS = [("sf1_gnn", "일반 GNN · L000 한 궤적", BLUE, "-", "상태적합"),
        ("noinertial", "일반 GNN · 다섯 궤적", BLUE, "--", "다궤적적합"),
        ("sf1_gunet", "Graph U-Net · L000 한 궤적", ORANGE, "-", "상태적합"),
        ("gunet5_near", "Graph U-Net · 다섯 궤적", ORANGE, "--", "다궤적적합")]
RUNS = [r for r in RUNS if r[0] in M]


def ser(run, key):
    s = M.get(run, {}).get(key, [])
    return np.array([p[0] for p in s]), np.array([p[1] for p in s])


def val_key(pre):
    return "상태적합_롤아웃/위치오차_평균%" if pre == "상태적합" else "다궤적적합_롤아웃_위치오차%/L000"


def smooth(y, k=100):
    if len(y) < k:
        return y
    c = np.cumsum(np.insert(y, 0, 0.0))
    return np.concatenate([c[1:k] / np.arange(1, k), (c[k:] - c[:-k]) / k])


fig, axs = plt.subplots(2, 3, figsize=(22, 11))
ax = axs[0, 0]
for run, lab, col, ls, pre in RUNS:
    x, y = ser(run, val_key(pre))
    if len(x):
        ax.plot(x + 1, y, ls, marker="o", ms=8, color=col, lw=2, label=lab)
ax.set_title("검증: L000 정답 시작 80 프레임 롤아웃 위치 오차 %", fontsize=12)
ax.set_xlabel("반복"); ax.grid(alpha=0.3); ax.legend(fontsize=10)
for ax, suf, tt, lg in ((axs[0, 1], "증분포텐셜", "학습 표본 증분 포텐셜 E (100 반복 이동평균)", True),
                        (axs[0, 2], "출력오차%_평균", "학습 표본 출력 오차 % (출력만 최적화 다음 상태 대비, 이동평균)", True),
                        (axs[1, 0], "손실", "학습 손실 (100 반복 이동평균)", True),
                        (axs[1, 1], "기울기노름", "기울기 노름 (자르기 전, 100 반복 이동평균)", True)):
    for run, lab, col, ls, pre in RUNS:
        x, y = ser(run, f"{pre}/{suf}")
        if len(x):
            ax.plot(x, smooth(y), ls, color=col, lw=2, label=lab)
    ax.set_title(tt, fontsize=12); ax.set_xlabel("반복"); ax.grid(alpha=0.3); ax.legend(fontsize=10)
    if lg:
        ax.set_yscale("log")
axs[1, 2].axis("off")
txt = []
for run, lab, _, _, pre in RUNS:
    x, y = ser(run, val_key(pre))
    _, e = ser(run, f"{pre}/증분포텐셜")
    txt.append(f"{lab}\n  마지막 검증 L000 {y[-1]:.2f}% (반복 {int(x[-1]) + 1})" if len(y) else f"{lab}\n  검증 없음")
    if len(y):
        txt[-1] += f", 최저 {y.min():.2f}% (반복 {int(x[np.argmin(y)]) + 1})"
axs[1, 2].text(0.0, 0.5, "\n\n".join(txt), fontsize=12, va="center")
fig.suptitle("lego 학습 곡선: L000 한 궤적 학습 vs 다섯 궤적 학습 (일반 GNN · Graph U-Net, 관성 예측 없음, 바닥 softplus + 미끄러짐 방지)",
             fontsize=14)
fig.tight_layout()
fig.savefig(out, dpi=110)
print("그림", out)

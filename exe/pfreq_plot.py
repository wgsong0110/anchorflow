"""pfreq curves.json -> 학습 곡선·검증 비교 (한국어). 기존 spstick 다섯 궤적(파랑) vs 입자 주파수 인코딩 입력 --pfreq(주황), 같은 반복축.
위: 학습 증분 포텐셜 E (100 반복 이동평균) | 학습 표본 출력오차% | 학습 손실 | 기울기 노름 | 마지막 검증 글
아래: 검증 롤아웃 위치오차% 궤적별 (L000·0000~0003). 손실·배치 분포가 같아 직접 비교 가능하다.
사용: python3 exe/pfreq_plot.py curves.json out.png [K]"""
import json
import subprocess
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

src, out = sys.argv[1], sys.argv[2]
KF = sys.argv[3] if len(sys.argv) > 3 else "6"
F = subprocess.run("fc-list :lang=ko file | head -1 | cut -d: -f1", shell=True, capture_output=True, text=True).stdout.strip()
if F:
    fm.fontManager.addfont(F)
    plt.rcParams["font.family"] = fm.FontProperties(fname=F).get_name()
plt.rcParams["axes.unicode_minus"] = False
M = json.load(open(src))
RUNS = [("spstick", "기존 (입자 모멘트 29 + 바닥 4 = 33 채널)", "#2a78d6"),
        ("pfreq", f"이번 (입자 주파수 인코딩 15×(1+2·{KF}) + 바닥 4 채널)", "#eb6834")]
TAGS = ["L000", "0000", "0001", "0002", "0003"]


def ser(run, key):
    s = M.get(run, {}).get(key, [])
    return np.array([p[0] for p in s]), np.array([p[1] for p in s])


def smooth(y, k=100):
    if len(y) < k:
        return y
    c = np.cumsum(np.insert(y, 0, 0.0))
    return np.concatenate([c[1:k] / np.arange(1, k), (c[k:] - c[:-k]) / k])


fig, axs = plt.subplots(2, 5, figsize=(28, 10))
for ax, key, tt in ((axs[0, 0], "다궤적적합/증분포텐셜", "학습 표본 증분 포텐셜 E (100 반복 이동평균)"),
                    (axs[0, 1], "다궤적적합/출력오차%_평균", "학습 표본 출력 오차 % (출력만 최적화 다음 상태 대비, 이동평균)"),
                    (axs[0, 2], "다궤적적합/손실", "학습 손실 (100 반복 이동평균)"),
                    (axs[0, 3], "다궤적적합/기울기노름", "기울기 노름 (자르기 전, 100 반복 이동평균)")):
    for run, lab, col in RUNS:
        x, y = ser(run, key)
        if len(x):
            ax.plot(x, smooth(y), color=col, lw=2, label=lab)
    ax.set_title(tt, fontsize=11); ax.set_xlabel("반복"); ax.grid(alpha=0.3); ax.legend(fontsize=9); ax.set_yscale("log")
axs[0, 4].axis("off")
fin = []
for run, lab, _ in RUNS:
    v = []
    for t in TAGS:
        x, y = ser(run, f"다궤적적합_롤아웃_위치오차%/{t}")
        v.append(f"{t} {y[-1]:.2f}%" if len(y) else f"{t} -")
    fin.append(f"{lab}\n마지막 검증: " + ", ".join(v[:3]) + "\n" + ", ".join(v[3:]))
axs[0, 4].text(0.0, 0.5, "\n\n".join(fin), fontsize=11, va="center", wrap=True)
for c, t in enumerate(TAGS):
    ax = axs[1, c]
    for run, lab, col in RUNS:
        x, y = ser(run, f"다궤적적합_롤아웃_위치오차%/{t}")
        if len(x):
            ax.plot(x, y, "-o", ms=8, color=col, lw=2, label=lab)
    ax.set_title(f"검증: 정답 시작 80프레임 롤아웃 위치 오차 % · {t}", fontsize=11); ax.set_xlabel("반복"); ax.grid(alpha=0.3)
    if c == 0:
        ax.legend(fontsize=9)
fig.suptitle(f"lego 다섯 궤적 학습: 기존 spstick vs 노드 입력 = 입자 물리량 주파수 인코딩 (--pfreq {KF}) -- 반복별 학습 곡선과 검증", fontsize=14)
fig.tight_layout()
fig.savefig(out, dpi=110)
print("그림", out)

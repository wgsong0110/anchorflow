"""pool1 변형(무작위 폐기 끔·풀 32, 문턱 1e-3) 풀 관찰 그림 (한국어, 사용자 10-26).
기준 pool1_gunet (풀 128, 무작위 폐기 0.01, 문턱 1e-4) 과 변형 실행의 풀 상태 통계를 같은 반복축에 그린다:
전진 프레임 평균·중앙·최대 | 정체 횟수 평균·중앙·최대 | 정체 100 이상 비율 | 풀 안 상태 교체율(100 반복) | 새 상태 입장·거부 누적 |
전진 통과율 | 상태별 pgrad 판정값 1e-4·1e-3 이하 비율 (기록이 있는 실행만)
사용: python3 exe/pool1v_poolplot.py out.png <curves.json> [<curves.json> ...]   (json 의 최상위 키 = 실행 이름)"""
import json
import subprocess
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

out, *srcs = sys.argv[1:]
F = subprocess.run("fc-list :lang=ko file | head -1 | cut -d: -f1", shell=True, capture_output=True, text=True).stdout.strip()
if F:
    fm.fontManager.addfont(F)
    plt.rcParams["font.family"] = fm.FontProperties(fname=F).get_name()
plt.rcParams["axes.unicode_minus"] = False
M = {}
for s in srcs:
    M.update(json.load(open(s)))
STY = {"pool1_gunet": ("기준: 풀 128 · 무작위 폐기 0.01 · 문턱 1e-4", "#eb6834"),
       "pool1s32_gunet": ("풀 32 · 무작위 폐기 끔 · 문턱 1e-4", "#1baf7a"),
       "pool1s32g3_gunet": ("풀 32 · 무작위 폐기 끔 · 문턱 1e-3", "#e87ba4")}
RUNS = [r for r in STY if r in M]


def ser(run, key):
    s = sorted(M[run].get(key, []))
    return np.array([p[0] for p in s]), np.array([p[1] for p in s])


def smooth(y, k=25):
    if len(y) < k:
        return y
    c = np.cumsum(np.insert(y, 0, 0.0))
    return np.concatenate([c[1:k] / np.arange(1, k), (c[k:] - c[:-k]) / k])


PANELS = [("전진 프레임 (평균 실선 · 중앙 파선 · 최대 점선)", [("풀/전진프레임_평균", "-"), ("풀/전진프레임_중앙", "--"), ("풀/전진프레임_최대", ":")], False),
          ("정체 횟수 (평균 실선 · 중앙 파선 · 최대 점선)", [("풀/정체_평균", "-"), ("풀/정체_중앙", "--"), ("풀/정체_최대", ":")], True),
          ("정체 100 회 이상인 풀 상태 비율", [("풀/정체100이상_비율", "-")], False),
          ("풀 안 상태 교체율 (100 반복 사이 바뀐 비율)", [("풀/교체율_100반복", "-")], False),
          ("새 상태 누적 (입장 실선 · 풀이 차서 거부 파선)", [("풀/신규입장누적", "-"), ("풀/신규거부누적", "--")], True),
          ("전진 통과율 (문턱 이하로 전진한 방문 비율)", [("게이트/통과율", "-")], False),
          ("상태별 pgrad 판정값 비율 (1e-4 이하 실선 · 1e-3 이하 파선)", [("게이트/pgrad_1e-4이하_비율", "-"), ("게이트/pgrad_1e-3이하_비율", "--")], False),
          ("상태별 pgrad 판정값 (중앙 실선 · 하위 10% 파선 · 상위 10% 점선)", [("게이트/pgrad_중앙", "-"), ("게이트/pgrad_하위10%", "--"), ("게이트/pgrad_상위10%", ":")], True)]
fig, axs = plt.subplots(2, 4, figsize=(30, 12))
for ax, (tt, keys, lg) in zip(axs.ravel(), PANELS):
    for run in RUNS:
        lab, col = STY[run]
        for i, (k, ls) in enumerate(keys):
            x, y = ser(run, k)
            if len(x):
                ax.plot(x, smooth(y) if "누적" not in k else y, ls, color=col, lw=2, label=lab if i == 0 else None)
    ax.set_title(tt, fontsize=12); ax.set_xlabel("반복"); ax.grid(alpha=0.3); ax.legend(fontsize=9)
    if lg:
        ax.set_yscale("symlog" if "누적" in tt or "정체" in tt else "log")
fig.suptitle("lego 한 궤적 풀 학습 (Graph U-Net): 풀 상태 관찰 -- 무작위 폐기 끔·풀 32 변형 vs 기준", fontsize=15)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(out, dpi=100)
print("그림", out)

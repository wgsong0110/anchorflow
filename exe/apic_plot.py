"""APIC 비교 그림 (사용자 10-24): 프레임별 위치 오차·E/E* -- 기준판 noin (모멘트 입력), B = P2G + near 바닥, A = P2G + APIC + 응력 + near 바닥.
위치 오차 칸에는 출력만 최적화 (records 의 ref_err) 도 같이. 축 하나씩 (두 칸), 한국어.
사용: python3 exe/apic_plot.py <noin records.json> <B records.json> <A records.json> <out.png>"""
import json
import subprocess
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

pn, pb, pa, out = sys.argv[1:5]
F = subprocess.run("fc-list :lang=ko file | head -1 | cut -d: -f1", shell=True, capture_output=True, text=True).stdout.strip()
if F:
    fm.fontManager.addfont(F)
    plt.rcParams["font.family"] = fm.FontProperties(fname=F).get_name()
plt.rcParams["axes.unicode_minus"] = False
R = [(json.load(open(pn)), "기준판 (입자 모멘트 입력, 원래 바닥)", "#2a78d6"),
     (json.load(open(pb)), "B: P2G + 바닥 near", "#eb6834"),
     (json.load(open(pa)), "A: P2G + APIC C + 응력 + 바닥 near", "#1baf7a")]
fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
ov = sorted((q["t"] + 1, 100 * q["ref_err"]) for q in R[2][0] if q.get("ref_err") is not None)
if ov:
    ax[0].plot(*zip(*ov), color="#52514e", lw=2, ls="--", label=f"출력만 최적화 (평균 {np.mean([e for _, e in ov]):.2f}%)")
for rr, lab, c in R:
    t = [q["t"] + 1 for q in rr]
    pe = [100 * q["pos_err_model"] for q in rr]
    ax[0].plot(t, pe, color=c, lw=2, label=f"{lab} (평균 {np.mean(pe):.2f}%)")
    er = [(q["t"] + 1, q["ratio_E_Estar"]) for q in rr if q.get("ratio_E_Estar") is not None]
    ax[1].plot(*zip(*er), color=c, lw=2, label=f"{lab} (중앙 {np.median([e for _, e in er]):.3f})")
ax[0].set_title("프레임별 위치 오차 (정답 대비, %)")
ax[0].set_xlabel("프레임"); ax[0].set_ylabel("위치 오차 %")
ax[1].set_title("프레임별 E / E* (E* = 같은 상태 출력만 최적화 L-BFGS 500)")
ax[1].set_xlabel("프레임"); ax[1].set_ylabel("E / E*")
for a_ in ax:
    a_.grid(alpha=0.25); a_.legend(fontsize=9, frameon=False)
    for s in ("top", "right"):
        a_.spines[s].set_visible(False)
fig.suptitle("L000 80 프레임 · 증분 포텐셜 프레임별 망 최적화 (Adam 2000/프레임, 관성 예측 없음, sgnn 8 층)", fontsize=12)
fig.tight_layout()
fig.savefig(out, dpi=130)
print(out)

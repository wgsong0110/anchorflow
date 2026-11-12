"""절 CM115 그림 (로컬, 작은 json 만): startpt3_<판>.json -> startpt3.png. 사용: python exe/startpt3_tab.py <json 디렉토리> <출력 디렉토리> [H9판 H5판]"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

jd, od = sys.argv[1:3]
R9, R5 = (sys.argv[3:5] if len(sys.argv) >= 5 else ("r9_1", "r5_1"))
for f in fm.findSystemFonts():
    if any(k in os.path.basename(f).lower() for k in ("nanum", "notosanscjk", "malgun")):
        fm.fontManager.addfont(f)
for nm in ("NanumGothic", "Noto Sans CJK KR", "Malgun Gothic"):
    if any(x.name == nm for x in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = nm; break
plt.rcParams["axes.unicode_minus"] = False
A, B = (json.load(open(os.path.join(jd, f"startpt3_{k}.json"))) for k in (R9, R5))
fr = sorted(int(f) for f in A)
S = [("H9 · 기본 정의 (e^-x, 덮어쓰기 전)", [A[str(f)]["KBS"]["kb_기본"]["corner_partial"] for f in fr], "#6250d6"),
     ("H9 · 최종 (S_구속 덮어쓴 값)", [A[str(f)]["KBS"]["kb_최종"]["corner_partial"] for f in fr], "#2a78d6"),
     ("H5 · 최종 (기본 정의)", [B[str(f)]["KBS"]["kb_최종"]["corner_partial"] for f in fr], "#eb6834")]
fig, ax = plt.subplots(1, 2, figsize=(16, 5.5))
x = np.arange(len(fr)); w = 0.27
for i, (nm, v, c) in enumerate(S):
    ax[0].bar(x + (i - 1) * w, v, w * 0.92, color=c, label=nm)
ax[0].set_xticks(x); ax[0].set_xticklabels([f"f{f}" for f in fr]); ax[0].set_ylabel("부분 구속 칸 꼭짓점 수 (0 < k_구속 < 1)")
ax[0].set_title("부분 구속 꼭짓점 수 -- 같은 H9 상태에서 정의만 바꾸면 H5 와 같아짐")
for nm_, key, c in (("방출 앞면 노드 (받침 입자 일부만 미방출)", "front", "#1baf7a"),):
    ax[1].plot(fr, [A[str(f)][key] for f in fr], "-o", color=c, lw=2, label=f"H9 {nm_}")
    ax[1].plot(fr, [B[str(f)][key] for f in fr], "--s", color=c, lw=1.5, alpha=0.7, label=f"H5 {nm_}")
ax[1].plot(fr, [A[str(f)]["KBS"]["kb_최종"]["partial"] for f in fr], "-o", color="#2a78d6", lw=2, label="H9 부분 구속 노드 (최종)")
ax[1].plot(fr, [A[str(f)]["KBS"]["kb_최종"]["partial_front"] for f in fr], ":o", color="#2a78d6", lw=1.5, label="  그중 앞면 노드")
ax[1].plot(fr, [A[str(f)]["KBS"]["kb_최종"]["partial_ext"] for f in fr], "-.o", color="#e87ba4", lw=1.5, label="  그중 확장 외부 노드")
ax[1].plot(fr, [B[str(f)]["KBS"]["kb_최종"]["partial"] for f in fr], "--s", color="#eb6834", lw=2, label="H5 부분 구속 노드")
ax[1].set_title("노드 수 (활성 약 3.46 만)"); ax[1].set_xlabel("프레임")
for a in ax:
    a.grid(axis="y", color="#e5e4df", lw=0.6); a.set_axisbelow(True)
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
    a.legend(fontsize=8, frameon=False)
fig.suptitle("H9 의 부분 구속 꼭짓점이 많은 이유 (절 CM115)", fontsize=13)
fig.tight_layout(); fig.savefig(os.path.join(od, "startpt3.png"), dpi=110); print("끝")

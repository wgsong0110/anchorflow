"""절 CM114 그림 (로컬, 작은 json 만): startpt2_<판>.json -> startpt2.png.
사용: python exe/startpt2_tab.py <json 디렉토리> <출력 디렉토리> [H9판 H5판]"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

jd, od = sys.argv[1:3]
R9, R5 = (sys.argv[3:5] if len(sys.argv) >= 5 else ("q9_1", "q5_1"))
for f in fm.findSystemFonts():
    if any(k in os.path.basename(f).lower() for k in ("nanum", "notosanscjk", "malgun")):
        fm.fontManager.addfont(f)
for nm in ("NanumGothic", "Noto Sans CJK KR", "Malgun Gothic"):
    if any(x.name == nm for x in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = nm; break
plt.rcParams["axes.unicode_minus"] = False
D = {k: json.load(open(os.path.join(jd, f"startpt2_{k}.json"))) for k in (R9, R5)}
LAB = {R9: "H9 wolf", R5: "H5 wolf (대조)"}
fig, ax = plt.subplots(1, 3, figsize=(19, 5.5))
for k, c, ls in ((R9, "#2a78d6", "-"), (R5, "#eb6834", "--")):
    fb = D[k]["fb"]; fr = sorted((int(f) for f in fb if "pU_given_prev" in fb[f] and fb[f]["pU_given_prev"] is not None))
    ax[0].plot(fr, [fb[str(f)]["pU_given_prev"] for f in fr], ls, marker="o", color=c, lw=2, label=f"{LAB[k]}: 앞 프레임 끝 눌린 칸")
    ax[0].plot(fr, [fb[str(f)]["pU_given_notprev"] for f in fr], ls, marker="s", color=c, lw=1, alpha=0.6, label=f"{LAB[k]}: 앞 프레임 끝 안 눌린 칸")
    ax[1].plot(fr, [fb[str(f)]["prev_among_U"] for f in fr], ls, marker="o", color=c, lw=2, label=LAB[k])
    fr2 = sorted(int(f) for f in fb if fb[f].get("vrate_compU") is not None)
    ax[2].plot(fr2, [fb[str(f)]["vrate_compU"] for f in fr2], ls, marker="o", color=c, lw=2, label=LAB[k])
ax[0].set_title("다음 걸음 관성 예측 Ũ 에서 눌릴 (r<0.98) 확률"); ax[0].set_ylabel("몫"); ax[0].set_xlabel("프레임")
ax[1].set_title("Ũ 눌린 칸 중 앞 프레임 끝에도 눌렸던 몫"); ax[1].set_xlabel("프레임")
ax[2].set_title("Ũ 눌린 칸의 걸음 시작 속도장 세로 압축률 중앙 (1/s)"); ax[2].set_xlabel("프레임")
for a in ax:
    a.grid(color="#e5e4df", lw=0.6)
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
    a.legend(fontsize=8, frameon=False)
fig.suptitle("되먹임: 앞 걸음의 덜 풀린 압축 -> 걸음 시작 속도장 -> 다음 관성 예측 압축 (절 CM114)", fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(od, "startpt2.png"), dpi=110)
print("끝")

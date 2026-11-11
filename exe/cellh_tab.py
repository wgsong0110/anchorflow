"""절 CM112 표·그림 (로컬, 작은 json 만): exe/cellh_an.py 의 cellh_<판>.json -> cellh.md · cellh.png · cellh_cum.png.
사용: python exe/cellh_tab.py <json 디렉토리> <출력 디렉토리>   (판 c9_1 c9_2 c5_1)"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

jd, od = sys.argv[1:3]
for f in fm.findSystemFonts():
    if any(k in os.path.basename(f).lower() for k in ("nanum", "notosanscjk", "malgun")):
        fm.fontManager.addfont(f)
for nm in ("NanumGothic", "Noto Sans CJK KR", "Malgun Gothic"):
    if any(x.name == nm for x in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = nm; break
plt.rcParams["axes.unicode_minus"] = False
RUNS = [("c9_1", "H9 wolf 판 1"), ("c9_2", "H9 wolf 판 2"), ("c5_1", "H5 wolf (대조)")]
LAY = list(range(6))
D = {k: json.load(open(os.path.join(jd, f"cellh_{k}.json"))) for k, _ in RUNS}
col = plt.cm.viridis(np.linspace(0.0, 0.9, len(LAY)))


def series(d, L, st, key="layers"):
    return np.array([fr[key].get(str(L), {}).get(st, np.nan) for fr in d["frames"]], dtype=float)


F = np.array([fr["f"] for fr in D["c9_1"]["frames"]])
STAT = [("med", "중앙값"), ("p10", "10 백분위"), ("min", "최소")]
fig, ax = plt.subplots(3, 3, figsize=(17, 11), sharex=True)
for j, (k, t) in enumerate(RUNS):
    for i, (st, sn) in enumerate(STAT):
        a = ax[i, j]
        a.axvspan(8, 9, color="0.85", zorder=0)
        a.axvline(5, color="r", ls="--", lw=1)
        for L in LAY:
            a.plot(F, series(D[k], L, st), color=col[L], lw=1.4, label=f"{L} 층")
        a.axhline(1, color="k", lw=0.6)
        a.set_ylim(0.2 if st != "med" else 0.5, 1.15)
        a.set_title(f"{t} -- 프레임 걸음 r = h끝/h시작 {sn}")
        if j == 0: a.set_ylabel("칸 두께 비율 r")
        if i == 2: a.set_xlabel("프레임")
ax[0, 0].legend(ncol=3, fontsize=8, title="칸 바닥 높이 층 (칸 크기 단위)")
ax[0, 1].text(5.3, 0.55, "f5 첫 압축", color="r", fontsize=9); ax[0, 1].text(9.3, 0.52, "f8~9 바닥 접촉", fontsize=9)
fig.suptitle("wolf 활성 칸의 세로 두께 변화율 (프레임 시작 -> 끝), 층별")
fig.tight_layout(); fig.savefig(os.path.join(od, "cellh.png"), dpi=110)
fig, ax = plt.subplots(1, 3, figsize=(17, 4.6), sharey=True)
for j, (k, t) in enumerate(RUNS):
    a = ax[j]; a.axvspan(8, 9, color="0.85", zorder=0); a.axvline(5, color="r", ls="--", lw=1)
    for L in LAY:
        a.plot(F, series(D[k], L, "med", "cum"), color=col[L], lw=1.4, label=f"{L} 층")
    a.axhline(1, color="k", lw=0.6); a.set_title(f"{t} -- 누적 비율 h_t/h_0 중앙값 (입자별, 첫 프레임 층)"); a.set_xlabel("프레임")
    if j == 0: a.set_ylabel("누적 비율"); a.legend(ncol=3, fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(od, "cellh_cum.png"), dpi=110)

o = []
o.append("# wolf 칸 세로 두께 변화 (절 CM112, 묶음 cellh)\n")
o.append("- 칸 = 입자가 든 이차 B-스플라인 칸 (꼭짓점 노드 8 개). 두께 h = 꼭짓점 높이 최대 - 최소 (바닥 법선 z). 프레임 f = 걸음 f-1 -> f. r = h끝/h시작.")
o.append("- 격자는 [111] 이 z 인 기울어진 정육면체라 시작 두께 = √3 dx (dx = 0.02 m) 로 모든 칸이 같음 (확인). 층 = floor(칸 바닥 시작 높이 / dx), 0 미만은 0.")
o.append("- 누적 = 입자별 r (그 입자가 든 칸) 의 프레임 곱 (격자는 프레임마다 새로 깔려 칸 이름표가 없음), 층 = 그 입자의 f1 층. 칸 수는 프레임별 고유 활성 칸.\n")
for k, t in RUNS:
    d = D[k]
    for st, sn, key in [("med", "r 중앙값", "layers"), ("p10", "r 10 백분위", "layers"), ("min", "r 최소", "layers"), ("med", "누적 비율 중앙값", "cum")]:
        o.append(f"\n## {t} ({k}) -- {sn}\n")
        o.append("| 프레임 | 칸 수 | " + " | ".join(f"{L} 층" for L in LAY) + " |")
        o.append("|---|---|" + "---|" * len(LAY))
        for fr in d["frames"]:
            v = []
            for L in LAY:
                x = fr[key].get(str(L))
                v.append("-" if x is None else f"{x[st]:.3f}")
            tag = " (첫 압축)" if fr["f"] == 5 else (" (바닥 접촉)" if fr["f"] in (8, 9) else "")
            o.append(f"| {fr['f']}{tag} | {fr['n_cells']} | " + " | ".join(v) + " |")
open(os.path.join(od, "cellh.md"), "w").write("\n".join(o) + "\n")
print("ok")

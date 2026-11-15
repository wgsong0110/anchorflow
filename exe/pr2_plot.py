"""[paper_remeasure2] 파레토 산점도 (선 없음): python exe/pr2_plot.py <grid.json> <출력 png 접두> -- 로그판(<접두>.png)과 선형판(<접두>_lin.png).
점 = 설정마다 3 회 반복의 중앙값, 가는 막대 = 반복 최소~최대. 세로 점선 하나 = 60 fps. 축 범위는 모든 점이 보이게 자동."""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                                       # noqa: E402
from matplotlib import font_manager as fm                                            # noqa: E402

f = "/home/wgsong/.fonts/NotoSansCJKkr-Regular.otf"
try:
    fm.fontManager.addfont(f); plt.rcParams["font.family"] = fm.FontProperties(fname=f).get_name()
except Exception:
    pass
G = json.load(open(sys.argv[1]))
ST = dict(ours=("우리 (현재 기본 판)", "#d62728", "o"), mpm=("정답 MPM (PhysGaussian)", "#1f77b4", "s"), vrgs=("VR-GS (XPBD 케이지)", "#2ca02c", "^"),
          bnv=("명시적 B-스플라인", "#9467bd", "v"), phystwin=("PhysTwin", "#ff7f0e", "D"), simplicits=("Simplicits", "#8c564b", "P"),
          springgaus=("Spring-Gaus", "#17becf", "X"), gf=("GaussianFluent", "#e377c2", "*"))
SC = [(s, n) for s, n in (("ficus", "ficus (탄성)"), ("wolf", "wolf (모래)"), ("bread", "bread (찢어짐)"), ("vasedeck", "vasedeck (탄성, 회전 구동)"), ("plane", "plane (금속 소성)"), ("pillow2sofa", "pillow2sofa (충돌)"), ("sand_castle", "sand castle"), ("toast", "toast"),
                          ("bullet", "jelly + 총알"), ("fruitninja", "fruitninja")) if any(g["scene"] == s for g in G)]
for lin in (False, True):
    nc = 2
    nr = (len(SC) + 1) // 2
    fig, ax = plt.subplots(nr, nc, figsize=(11, 4.2 * nr), squeeze=False)
    for k, (sc, nm) in enumerate(SC):
        a = ax[k // nc][k % nc]
        for m, (lab, c, mk) in ST.items():
            q = [g for g in G if g["scene"] == sc and g["method"] == m and g["err"] is not None and g["ms"]]
            if not q:
                continue
            YF = 1e-3                                                           # 로그판: 오차가 이 값 아래 (정답 MPM 공식 서브스텝 = 자기 자신) 는 바닥에 속이 빈 표시로
            xs, ys = [g["ms"] for g in q], [max(g["err"], YF) if not lin else g["err"] for g in q]
            a.errorbar(xs, ys, xerr=[[g["ms"] - min(g["ms_all"]) for g in q], [max(g["ms_all"]) - g["ms"] for g in q]],
                       yerr=[[g["err"] - min(g["err_all"]) for g in q], [max(g["err_all"]) - g["err"] for g in q]], fmt="none", ecolor=c, elinewidth=0.6, alpha=0.6)
            a.scatter(xs, ys, c=c, marker=mk, s=40, label=lab, alpha=0.9, zorder=3)
            if not lin:
                lo = [(x_, y_) for x_, y_, g in zip(xs, ys, q) if g["err"] < YF]
                if lo:
                    a.scatter([x_ for x_, _ in lo], [y_ for _, y_ in lo], facecolors="white", edgecolors=c, marker=mk, s=40, zorder=4)
        a.axvline(1000 / 60, c="gray", ls="--", lw=0.8)
        if not lin:
            a.set_xscale("log"); a.set_yscale("log")
        else:
            a.set_xlim(left=0); a.set_ylim(bottom=0)
        a.margins(0.08)
        a.set_xlabel("프레임 시간 (ms, A6000 단독)"); a.set_ylabel("롤아웃 평균 오차 (%)")
        a.set_title(f"{nm} (점선 = 60 fps)"); a.grid(alpha=0.3, which="both")
    for j in range(len(SC), nr * nc):
        ax[j // nc][j % nc].axis('off')
    h, l = {}, {}
    for r_ in ax:
        for a in r_:
            for hh, ll in zip(*a.get_legend_handles_labels()):
                h[ll] = hh
    ax[0][0].legend(h.values(), h.keys(), fontsize=7)
    fig.tight_layout()
    out = sys.argv[2] + ("_lin" if lin else "") + ".png"
    fig.savefig(out, dpi=170); print("저장", out)

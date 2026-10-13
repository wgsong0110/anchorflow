"""서브스텝 안정성 지표 그림 (exe/stab_metrics.py 의 json -> png). 로컬에서 돈다.

행 = 지표, 열 = 형상 (lego·mic·ficus), x = 서브스텝 수 N (로그), 선 = 방법 (PG·GF·i-PG·우리=기본 실행 조합 ovs).
60 프레임 끝낸 실행만 찍고 발산 점은 뺀다. 발산 방지 보정판(*_safe)은 같은 선에 빈 점으로 잇는다 (todo A2).
계산 시간은 그리지 않는다.

  python exe/stab_plot.py --metrics metrics_20261014.json --out stab_metrics.png
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                              # noqa: E402
from matplotlib import font_manager                          # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--metrics", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

font_manager.fontManager.addfont("/home/wgsong/.fonts/NotoSansCJKkr-Regular.otf")
plt.rcParams["font.family"] = "Noto Sans CJK KR"
plt.rcParams["axes.unicode_minus"] = False

R = json.load(open(a.metrics))
SHAPES = [s for s in ("lego", "mic", "ficus") if s in R]
METH = [("pg", "PG", "#2a78d6", "o"), ("gf", "GF", "#eb6834", "s"), ("ipg", "i-PG", "#1baf7a", "^"),
        ("ovs", "우리", "#eda100", "D")]
MET = [("e_drift_max", "총에너지 변화 최대 (E_t−E_0)/E_0"), ("detneg_max", "det F ≤ 0 비율 최대"),
       ("jdev_max", "|J−1| 평균 최대"), ("pen_ratio_max", "바닥 아래 입자 비율 최대"),
       ("pen_depth_max", "바닥 관통 최대 깊이 / L"), ("jitter_d2", "떨림: 위치 2차 차분 평균 / L"),
       ("jitter_hf", "떨림: 속도 고주파 에너지 비율"), ("freefall_mom_err", "자유낙하 운동량 상대 오차 최대"),
       ("cd_mpm", "PG N1024 대비 Chamfer / L")]

fig, axs = plt.subplots(len(MET), len(SHAPES), figsize=(4.2 * len(SHAPES), 2.5 * len(MET)), squeeze=False)
for i, (mk, mlabel) in enumerate(MET):
    for j, S in enumerate(SHAPES):
        ax = axs[i][j]
        for m, name, col, mar in METH:
            pts = []                                         # (N, 값, 보정판?)
            for src, safe in ((m, False), (m + "_safe", True)):
                for n, d in R[S].get(src, {}).items():
                    v = d.get(mk)
                    if d.get("status") != "끝" or v is None:
                        continue
                    if not safe and any(str(n) == str(k) for k in R[S].get(m + "_safe", {})):
                        continue
                    pts.append((int(n), float(v), safe))
            pts.sort()
            if not pts:
                continue
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=col, lw=2, label=name, zorder=2)
            for n, v, safe in pts:
                ax.plot(n, v, marker=mar, ms=8, color=col, mfc="white" if safe else col, mec=col, mew=1.5, zorder=3)
        ax.set_xscale("log", base=2)
        # 보정판·발산 직전 값이 1e13 대까지 커서 선형 축이면 나머지가 0 에 붙는다 -> 대칭 로그 (0 근처는 선형)
        ax.set_yscale("symlog", linthresh=1e-4)
        ys = [l_.get_ydata() for l_ in ax.get_lines()]
        if ys and min(min(y) for y in ys) >= 0:
            ax.set_ylim(bottom=0)
        ax.grid(alpha=0.25, lw=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        if i == 0:
            ax.set_title(S, fontsize=13)
        if j == 0:
            ax.set_ylabel(mlabel, fontsize=9)
        if i == len(MET) - 1:
            ax.set_xlabel("서브스텝 수 N")
h, l = axs[0][0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=4, frameon=False, fontsize=11, bbox_to_anchor=(0.5, 1.0))
fig.text(0.5, 0.985 - 0.012, "빈 점 = 발산 방지 보정판, 발산한 실행은 뺌, 60 프레임 끝낸 실행만, 세로축 대칭 로그 (|값| < 1e-4 는 선형)", ha="center", fontsize=9,
         color="#52514e")
fig.tight_layout(rect=(0, 0, 1, 0.975))
fig.savefig(a.out, dpi=110)
print("[그림]", a.out)

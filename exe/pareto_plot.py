"""파레토 산점도 (로컬): python exe/pareto_plot.py <points.json> <출력 png> -- 씬마다 한 칸, x = 프레임 시간 (ms, 로그), y = 오차 % (로그), 60 fps 선."""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                                       # noqa: E402
from matplotlib import font_manager as fm                                            # noqa: E402

for f in ("/home/wgsong/.fonts/NotoSansCJKkr-Regular.otf",):
    try:
        fm.fontManager.addfont(f); plt.rcParams["font.family"] = fm.FontProperties(fname=f).get_name()
    except Exception:
        pass
P = json.load(open(sys.argv[1]))
ST = dict(ours=("우리", "#d62728", "o"), mpm=("MPM (정답 솔버)", "#1f77b4", "s"), vrgs=("VR-GS (XPBD 케이지)", "#2ca02c", "^"))
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for k, sc in enumerate(("L000", "ficus")):
    a = ax[k]
    for mth, (lab, c, mk) in ST.items():
        q = [p for p in P if p["scene"] == sc and p["method"] == mth and p["err"] is not None and p["ms"]]
        a.scatter([p["ms"] for p in q], [p["err"] for p in q], c=c, marker=mk, s=28, label=lab, alpha=0.85)
        # 파레토 앞선
        q = sorted(q, key=lambda p: p["ms"]); fr, best = [], 1e9
        for p in q:
            if p["err"] < best:
                fr.append(p); best = p["err"]
        if fr:
            a.step([p["ms"] for p in fr], [p["err"] for p in fr], where="post", c=c, lw=1, alpha=0.6)
    a.axvline(1000 / 60, c="gray", ls="--", lw=0.8)
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlabel("프레임 시간 (ms, A6000 단독)"); a.set_ylabel("롤아웃 평균 오차 (%)")
    a.set_title(f"{'lego L000' if sc == 'L000' else 'ficus'} (점선 = 60 fps)")
    a.grid(alpha=0.3, which="both")
ax[0].legend(fontsize=8)
fig.tight_layout()
fig.savefig(sys.argv[2], dpi=180)
print("저장", sys.argv[2])

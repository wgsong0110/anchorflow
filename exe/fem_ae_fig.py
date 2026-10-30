"""절 AE ficus 진동 주기 그림 (로컬): fem_ae_curve.py 의 json -> png. 사용: python exe/fem_ae_fig.py <curve.json> <out.png> [이름=주기 표시...]"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402

for f in ("/home/wgsong/.fonts/NotoSansCJKkr-Regular.otf",):
    fm.fontManager.addfont(f)
plt.rcParams["font.family"] = "Noto Sans CJK KR"
plt.rcParams["axes.unicode_minus"] = False
R = json.load(open(sys.argv[1]))
lab = dict(q.split("=", 1) for q in sys.argv[3:])
t = R["t"]
fig, ax = plt.subplots(figsize=(10, 4.2), dpi=150)
ax.plot(t, [1000 * u for u in R["gt"]], color="black", lw=2.2, label=lab.get("정답", "정답 (PG)"))
cols = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
for c, (nm, u) in zip(cols, R["pred"].items()):
    ax.plot(t, [1000 * v for v in u], color=c, lw=1.4, label=lab.get(nm, nm))
ax.set_xlabel("시간 (초)")
ax.set_ylabel("잎·가지 끝 (z 상위 5 % 입자) 평균 변위 (mm, 정답 주 방향)")
ax.set_title("ficus 125 프레임 진동: 정답 vs 절 AB · AD · AE 최고 판")
ax.grid(alpha=0.3)
ax.legend(fontsize=8, loc="upper right")
fig.tight_layout()
fig.savefig(sys.argv[2])
print("[저장]", sys.argv[2])

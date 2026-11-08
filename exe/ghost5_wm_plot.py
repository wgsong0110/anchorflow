"""절 CM55 수박 (b) 칸 덤프 그림: 방식 0 벌점 면 중 3x3x3 열림 (방식 1) 이 빼는 면 (빨강) · 남는 면 (하늘) · 입자 (회색, 손상 logJp > 0 은 주황) -- 옆 (x-z)·위 (x-y).
사용: python exe/ghost5_wm_plot.py <출력.png> <g4cells_f*.npz>...  (판마다 한 줄)"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

for f_ in ("/home/wgsong/.fonts/NotoSansCJKkr-Regular.otf", "/home/dkta/.fonts/NotoSansCJKkr-Regular.otf"):
    try:
        font_manager.fontManager.addfont(f_)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=f_).get_name()
        break
    except Exception:
        pass
plt.rcParams["axes.unicode_minus"] = False
out, ps = sys.argv[1], sys.argv[2:]
fig, ax = plt.subplots(len(ps), 2, figsize=(14, 4.2 * len(ps)), squeeze=False)
rng = np.random.default_rng(0)
for r, p in enumerate(ps):
    d = np.load(p)
    IP, cb, A = d["IP"], d["cb"].astype(np.int64), d["A"].astype(np.float64)
    n = IP[6:9] - 2
    key = lambda q: (q[:, 0] * n[1] + q[:, 1]) * n[2] + q[:, 2]
    k1 = set()
    fm0, fm1 = d["fm0"].astype(np.int64), d["fm1"].astype(np.int64)
    cen = {}
    rem, kep = [], []
    for a in range(3):
        c = np.where((fm0 >> a) & 1)[0]
        e = np.zeros(3); e[a] = 0.5
        pos = (cb[c] + IP[0:3][None] + 1.0 + e[None]) @ A.T
        isr = ((fm1[c] >> a) & 1) == 0
        rem.append(pos[isr]); kep.append(pos[~isr])
    rem, kep = np.concatenate(rem), np.concatenate(kep)
    x = d["x"]
    sel = rng.choice(len(x), min(len(x), 60000), replace=False)
    lj = d["ljp"][sel] if d["ljp"].size else np.zeros(len(sel))
    fr = p.split("_f")[-1].split(".")[0]
    for j, (i0, i1, nm) in enumerate(((0, 2, "옆 (x-z)"), (0, 1, "위 (x-y)"))):
        a_ = ax[r, j]
        a_.scatter(x[sel, i0], x[sel, i1], s=0.2, c=np.where(lj > 0, "#f0a030", "#bbbbbb"), rasterized=True)
        a_.scatter(kep[:, i0], kep[:, i1], s=0.6, c="#5ab4e8", alpha=0.5, rasterized=True, label=f"남는 면 {len(kep)}")
        a_.scatter(rem[:, i0], rem[:, i1], s=4, c="#d62728", rasterized=True, label=f"빼는 면 {len(rem)}")
        a_.set_aspect("equal"); a_.set_title(f"f{fr} {nm}"); a_.legend(loc="upper right", fontsize=8, markerscale=4)
plt.tight_layout()
plt.savefig(out, dpi=110)
print("저장", out)

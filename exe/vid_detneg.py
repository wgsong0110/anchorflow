"""우리 결과만: 입자는 반투명 회색, det(F)<0 셀의 입자는 빨강. 옆+위 한 그림."""
from __future__ import annotations
import argparse
import numpy as np
import torch
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import os as _os
for _p in ("/home/wgsong/.fonts/NotoSansCJKkr-Regular.otf",
           _os.path.expanduser("~/.fonts/NotoSansCJKkr-Regular.otf")):
    try: fm.fontManager.addfont(_p)
    except Exception: pass
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Noto Sans CJK KR"
plt.rcParams["axes.unicode_minus"] = False
from tqdm import tqdm

ap = argparse.ArgumentParser()
ap.add_argument("--dump", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--fps", type=int, default=12)
ap.add_argument("--s", type=float, default=1.2, help="회색 점 크기")
ap.add_argument("--neg_s", type=float, default=4.0, help="빨간 점 크기")
ap.add_argument("--alpha", type=float, default=0.25)
ap.add_argument("--label", default="")
a = ap.parse_args()


def L(p):
    try: return torch.load(p, map_location="cpu", weights_only=False)
    except TypeError: return torch.load(p, map_location="cpu")


D = L(a.dump)
P = np.asarray(D["pred"], dtype=np.float32)
NEG = np.asarray(D["fscal"])[..., 0] < 0
T, N = P.shape[0], P.shape[1]
R = float(np.asarray(D["ctrl_R"]).reshape(-1)[0]) if "ctrl_R" in D else 0.0
cid = int(np.asarray(D["ctrl_id"]).reshape(-1)[0]) if "ctrl_id" in D else None
print(f"입자 {N}, 프레임 {T}, det<0 최대 {int(NEG.sum(1).max())} "
      f"({100*NEG.sum(1).max()/N:.2f}%), 처음 생기는 프레임 "
      f"{int(np.nonzero(NEG.any(1))[0][0]) if NEG.any() else -1}", flush=True)

lo = P.reshape(-1, 3).min(0) - 0.02
hi = P.reshape(-1, 3).max(0) + 0.02
th = np.linspace(0, 2 * np.pi, 200)
fig, axs = plt.subplots(1, 2, figsize=(11.0, 6.4), dpi=110)
wr = imageio.get_writer(a.out, fps=a.fps, codec="libx264", quality=8,
                        macro_block_size=1)
for t in tqdm(range(T), ncols=70):
    n = NEG[min(t, NEG.shape[0] - 1)]
    for k, ((i, j), nm) in enumerate((((0, 2), "옆에서 (xz)"),
                                      ((0, 1), "위에서 (xy)"))):
        q = axs[k]; q.clear()
        q.scatter(P[t][~n, i], P[t][~n, j], s=a.s, c="0.45",
                  alpha=a.alpha, linewidths=0)
        if n.any():
            q.scatter(P[t][n, i], P[t][n, j], s=a.neg_s, c="red", linewidths=0)
        if cid is not None and R > 0:
            c = P[t, cid]
            q.plot(c[i] + R * np.cos(th), c[j] + R * np.sin(th),
                   color="deepskyblue", lw=1.5)
        q.set_xlim(lo[i], hi[i]); q.set_ylim(lo[j], hi[j])
        q.set_aspect("equal"); q.grid(alpha=0.15)
        q.set_title(nm, fontsize=11)
    fig.suptitle(f"{a.label}   프레임 {t}   det(F)<0 입자 {int(n.sum())} / {N} "
                 f"({100*n.sum()/N:.2f}%)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.canvas.draw()
    wr.append_data(np.ascontiguousarray(
        np.asarray(fig.canvas.buffer_rgba())[..., :3]))
wr.close()
print(f"[저장] {a.out}  {T} 프레임", flush=True)

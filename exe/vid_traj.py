"""궤적 .pt 하나를 옆(xz)·위(xy) 두 칸으로 렌더한다 (기준 궤적 확인용)."""
from __future__ import annotations
import argparse, os
import numpy as np
import torch
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
for _p in ("/home/wgsong/.fonts/NotoSansCJKkr-Regular.otf",
           os.path.expanduser("~/.fonts/NotoSansCJKkr-Regular.otf")):
    try: fm.fontManager.addfont(_p)
    except Exception: pass
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Noto Sans CJK KR"
plt.rcParams["axes.unicode_minus"] = False
from tqdm import tqdm

ap = argparse.ArgumentParser()
ap.add_argument("--traj", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--key", default="x")
ap.add_argument("--sub", type=int, default=200000)
ap.add_argument("--s", type=float, default=0.6)
ap.add_argument("--fps", type=int, default=12)
ap.add_argument("--label", default="")
a = ap.parse_args()

try: D = torch.load(a.traj, map_location="cpu", weights_only=False)
except TypeError: D = torch.load(a.traj, map_location="cpu")
X = D[a.key]
T, N = X.shape[0], X.shape[1]
g = torch.Generator().manual_seed(0)
sel = (torch.randperm(N, generator=g)[:a.sub].sort().values
       if N > a.sub else torch.arange(N))
X = X[:, sel].float().numpy()
c0 = X[0][:, 2]
lo, hi = X.reshape(-1, 3).min(0) - 0.02, X.reshape(-1, 3).max(0) + 0.02
print(f"궤적 {T} 프레임 x {N} 입자 (표시 {len(sel)}), z {lo[2]:.3f}~{hi[2]:.3f}",
      flush=True)
fig, axs = plt.subplots(1, 2, figsize=(11, 6), dpi=110)
wr = imageio.get_writer(a.out, fps=a.fps, codec="libx264", quality=8,
                        macro_block_size=1)
for t in tqdm(range(T), ncols=70):
    for k, ((i, j), nm) in enumerate((((0, 2), "옆에서 (xz)"),
                                      ((0, 1), "위에서 (xy)"))):
        q = axs[k]; q.clear()
        q.scatter(X[t][:, i], X[t][:, j], s=a.s, c=c0, cmap="turbo", lw=0)
        q.set_xlim(lo[i], hi[i]); q.set_ylim(lo[j], hi[j])
        q.set_aspect("equal"); q.grid(alpha=0.15); q.set_title(nm, fontsize=11)
    fig.suptitle(f"{a.label}   프레임 {t} / {T - 1}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94]); fig.canvas.draw()
    wr.append_data(np.ascontiguousarray(
        np.asarray(fig.canvas.buffer_rgba())[..., :3]))
wr.close()
print(f"[저장] {a.out}  {T} 프레임", flush=True)

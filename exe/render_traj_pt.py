"""학습 궤적 .pt 하나를 입자 구름 영상으로 낸다 (손잡이 표시 포함).

두 시점(정면·측면)을 나란히 두고, 색은 **첫 프레임 대비 누적 변위**다.
손잡이는 빨간 점 + 영향 반경 원으로 그린다.
"""
from __future__ import annotations
import argparse
import numpy as np
import torch
from tqdm import tqdm
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
import imageio.v2 as imageio

ap = argparse.ArgumentParser()
ap.add_argument("--traj", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--sub", type=int, default=6000, help="그릴 입자 수")
ap.add_argument("--fps", type=int, default=12)
a = ap.parse_args()

d = torch.load(a.traj, map_location="cpu", weights_only=False)
X = d["x"].float().numpy()
T, N, _ = X.shape
rng = np.random.default_rng(0)
sel = rng.permutation(N)[:min(a.sub, N)]
P = X[:, sel]
C = d["ctrl_pos"].float().numpy() if "ctrl_pos" in d else None
R = d["ctrl_R"].float().numpy() if "ctrl_R" in d else None
disp = np.linalg.norm(P - P[0], axis=-1)
vmax = float(np.percentile(disp, 99.5)) or 1.0
lo, hi = X.reshape(-1, 3).min(0), X.reshape(-1, 3).max(0)
pad = 0.06 * float(np.linalg.norm(hi - lo))
EXT = float(np.linalg.norm(hi - lo))
print(f"[궤적] {d.get('tag')} seed {d.get('seed')}  {T} 프레임, 입자 {N} "
      f"(그리는 건 {len(sel)}), 최대 변위 {100*disp.max()/EXT:.2f}% EXT", flush=True)

AXES = [((0, 2), (1,), "정면 (xz)"), ((1, 2), (0,), "측면 (yz)")]
frames = []
for t in tqdm(range(T), desc="렌더", ncols=80):
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 5.2), dpi=110)
    for k, ((i, j), _, ttl) in enumerate(AXES):
        s = ax[k]
        s.scatter(P[t][:, i], P[t][:, j], s=1.1, c=disp[t], cmap="viridis",
                  vmin=0, vmax=vmax, linewidths=0)
        if C is not None:
            for h in range(C.shape[1]):
                c = C[min(t, C.shape[0] - 1), h]
                s.add_patch(plt.Circle((c[i], c[j]),
                                       float(R[min(t, len(R) - 1)]) if R is not None else 0.1,
                                       fill=False, color="red", lw=1.4, alpha=0.85))
                s.plot([c[i]], [c[j]], "o", color="red", ms=5)
        s.set_xlim(lo[i] - pad, hi[i] + pad); s.set_ylim(lo[j] - pad, hi[j] + pad)
        s.set_aspect("equal"); s.set_title(ttl, fontsize=10)
        s.set_xticks([]); s.set_yticks([])
    fig.suptitle(f"{d.get('tag')}  seed {d.get('seed')}   프레임 {t:3d}/{T-1}   "
                 f"색 = 첫 프레임 대비 누적 변위 (빨간 원 = 손잡이 영향 반경)",
                 fontsize=11)
    fig.tight_layout()
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), np.uint8).reshape(h, w, 4)[..., :3]
    if buf.shape[0] % 2 or buf.shape[1] % 2:     # yuv420p 는 짝수 크기만 받는다
        buf = buf[:buf.shape[0] // 2 * 2, :buf.shape[1] // 2 * 2]
    frames.append(buf.copy())
    plt.close(fig)

imageio.mimsave(a.out, frames, fps=a.fps, quality=8, macro_block_size=1)
print(f"[저장] {a.out}  {len(frames)} 프레임", flush=True)

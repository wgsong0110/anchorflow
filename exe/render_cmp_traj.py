"""두 궤적을 **같은 조건**으로 나란히 그린다.

render_traj_pt.py 는 궤적마다 (1) 입자를 자기 전체 개수에서 따로 뽑고
(2) 색 스케일을 자기 99.5 백분위로 정규화하고 (3) 축 범위도 따로 잡는다.
그래서 두 영상을 나란히 놓고 "안쪽 움직임이 다르다" 고 판단할 수 없다.
여기서는 **같은 입자 색인·같은 색 스케일·같은 축**으로 그려 비교를 성립시킨다.
손잡이 원은 각 궤적 **자신의** ctrl_pos 를 쓴다.
"""
from __future__ import annotations
import argparse, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

ap = argparse.ArgumentParser()
ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
ap.add_argument("--la", default="A"); ap.add_argument("--lb", default="B")
ap.add_argument("--out", required=True)
ap.add_argument("--sub", type=int, default=6000)
ap.add_argument("--fps", type=int, default=12)
ap.add_argument("--axis", default="xz", choices=("xz", "yz"))
a = ap.parse_args()


def L(p):
    try: return torch.load(p, map_location="cpu", weights_only=False)
    except TypeError: return torch.load(p, map_location="cpu")


da, db = L(a.a), L(a.b)
# 같은 입자를 본다: 둘 중 작은 쪽의 sel 을 기준으로 맞춘다
sa = da.get("sel"); sb = db.get("sel")
na, nb = da["x"].shape[1], db["x"].shape[1]
if na == nb:
    ia = ib = np.arange(na)
elif na < nb:                      # a 가 부분표본, b 가 전체
    ia = np.arange(na); ib = sa.numpy()
else:
    ib = np.arange(nb); ia = sb.numpy()
rng = np.random.default_rng(0)
pick = rng.permutation(len(ia))[:min(a.sub, len(ia))]
ia, ib = ia[pick], ib[pick]
XA = da["x"].float().numpy()[:, ia]
XB = db["x"].float().numpy()[:, ib]
T = min(XA.shape[0], XB.shape[0])
XA, XB = XA[:T], XB[:T]
dA = np.linalg.norm(XA - XA[0], axis=-1)
dB = np.linalg.norm(XB - XB[0], axis=-1)
vmax = float(np.percentile(np.concatenate([dA, dB]), 99.5)) or 1.0   # 공통 색
allp = np.concatenate([XA.reshape(-1, 3), XB.reshape(-1, 3)])
lo, hi = allp.min(0), allp.max(0)
pad = 0.06 * float(np.linalg.norm(hi - lo))
i, j = (0, 2) if a.axis == "xz" else (1, 2)
print(f"[비교] {a.la} {na} 입자 / {a.lb} {nb} 입자 -> 공통 {len(ia)} 개, "
      f"{T} 프레임, 색 상한 {vmax:.4f}", flush=True)


def circ(d, t):
    if "ctrl_pos" not in d or d["ctrl_pos"] is None:
        return []
    C = d["ctrl_pos"].float().numpy()
    R = d["ctrl_R"].float().numpy() if "ctrl_R" in d else np.array([0.15])
    return [(C[min(t, C.shape[0] - 1), h], float(R[min(t, len(R) - 1)]))
            for h in range(C.shape[1])]


frames = []
for t in tqdm(range(T), desc="렌더", ncols=80):
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 5.2), dpi=110)
    for k, (X, dd, d_, lab) in enumerate(((XA, dA, da, a.la),
                                          (XB, dB, db, a.lb))):
        s = ax[k]
        s.scatter(X[t][:, i], X[t][:, j], s=1.1, c=dd[t], cmap="viridis",
                  vmin=0, vmax=vmax, linewidths=0)
        for c, r in circ(d_, t):
            s.add_patch(plt.Circle((c[i], c[j]), r, fill=False, color="red",
                                   lw=1.4, alpha=0.85))
        s.set_xlim(lo[i] - pad, hi[i] + pad)
        s.set_ylim(lo[j] - pad, hi[j] + pad)
        s.set_aspect("equal"); s.set_xticks([]); s.set_yticks([])
        s.set_title(f"{lab}  t={t}", fontsize=11)
    fig.tight_layout()
    fig.canvas.draw()
    frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    plt.close(fig)
import imageio
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
imageio.mimsave(a.out, frames, fps=a.fps, macro_block_size=1)
print(f"[저장] {a.out}  {len(frames)} 프레임", flush=True)

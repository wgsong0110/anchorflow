"""롤아웃 덤프를 **정답 대비 나란히** 보여주는 영상으로 만든다.

왼쪽 교사 MPM, 가운데 학생 예측, 오른쪽은 둘을 겹쳐 놓은 것이다.
색은 두 번째 칸에서 **정답과의 거리**라, 어디서 틀리는지가 바로 보인다.
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
ap.add_argument("--dump", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--sub", type=int, default=6000)
ap.add_argument("--fps", type=int, default=8)
a = ap.parse_args()

D = torch.load(a.dump, map_location="cpu", weights_only=False)
P, G = D["pred"].float().numpy(), D["gt"].float().numpy()
EXT = float(D["EXT"])
T, N, _ = P.shape
rng = np.random.default_rng(0)
sel = rng.permutation(N)[:min(a.sub, N)]
P, G = P[:, sel], G[:, sel]
err = np.linalg.norm(P - G, axis=-1) / EXT * 100
lo = np.minimum(P.reshape(-1, 3).min(0), G.reshape(-1, 3).min(0))
hi = np.maximum(P.reshape(-1, 3).max(0), G.reshape(-1, 3).max(0))
pad = 0.06 * float(np.linalg.norm(hi - lo))
vmax = float(np.percentile(err, 99)) or 1.0
print(f"[덤프] {D['tag']} t0={D['t0']}  {T} 프레임, 입자 {N} "
      f"(그리는 건 {len(sel)})  오차 중앙 {np.median(err):.3f}% 최대 {err.max():.3f}%",
      flush=True)

# 손잡이 위치와 반경 (없으면 그리지 않는다)
_CP = D.get("ctrl_pos")
if _CP is not None:
    _CP = np.asarray(_CP, dtype=np.float32)
_R_CTRL = float(D.get("ctrl_R", 0.15))

frames = []
for t in tqdm(range(T), desc="렌더", ncols=80):
    fig, ax = plt.subplots(1, 3, figsize=(14.4, 5.0), dpi=110)
    i, j = 0, 2                                          # xz 평면
    ax[0].scatter(G[t][:, i], G[t][:, j], s=1.1, c="0.25", linewidths=0)
    ax[0].set_title("교사 MPM (정답)", fontsize=11)
    s = ax[1].scatter(P[t][:, i], P[t][:, j], s=1.1, c=err[t], cmap="inferno",
                      vmin=0, vmax=vmax, linewidths=0)
    ax[1].set_title(f"학생 예측 (색 = 정답과의 거리, 평균 {err[t].mean():.3f}% EXT)",
                    fontsize=11)
    ax[2].scatter(G[t][:, i], G[t][:, j], s=1.1, c="0.55", linewidths=0,
                  label="정답")
    ax[2].scatter(P[t][:, i], P[t][:, j], s=1.1, c="crimson", linewidths=0,
                  alpha=.55, label="예측")
    ax[2].set_title("겹쳐 보기", fontsize=11)
    # 손잡이를 그린다. 이게 없으면 구동이 들어갔는지 눈으로 확인할 수 없어
    # "손잡이가 없는 것 같다" 는 오해를 부른다 (덤프에는 늘 들어 있다).
    if _CP is not None:
        _ti = min(int(D["t0"]) + t, _CP.shape[0] - 1)
        for _k in range(_CP.shape[1]):
            _c = _CP[_ti, _k]
            for _q in ax:
                _q.add_patch(plt.Circle((_c[i], _c[j]), _R_CTRL,
                                        fill=False, ec="deepskyblue", lw=1.6,
                                        alpha=.9))
                _q.plot([_c[i]], [_c[j]], marker="x", ms=7, mew=2.0,
                        color="deepskyblue")
    ax[2].legend(fontsize=9, markerscale=6, loc="upper right")
    for q in ax:
        q.set_xlim(lo[i] - pad, hi[i] + pad); q.set_ylim(lo[j] - pad, hi[j] + pad)
        q.set_aspect("equal"); q.set_xticks([]); q.set_yticks([])
    fig.suptitle(f"{D['tag']}  (학습에 쓰지 않은 시드)   "
                 f"자기회귀 {t + 1}/{T} 프레임", fontsize=12)
    fig.tight_layout()
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), np.uint8).reshape(h, w, 4)[..., :3]
    if buf.shape[0] % 2 or buf.shape[1] % 2:
        buf = buf[:buf.shape[0] // 2 * 2, :buf.shape[1] // 2 * 2]
    frames.append(buf.copy())
    plt.close(fig)

imageio.mimsave(a.out, frames, fps=a.fps, quality=8, macro_block_size=1)
print(f"[저장] {a.out}  {len(frames)} 프레임", flush=True)

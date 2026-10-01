"""롤아웃 덤프를 **정답 대비 나란히** 보여주는 영상으로 만든다.

왼쪽 PG MPM(기준), 오른쪽은 --label 로 받는다. (겹쳐 보기 칸은 요청으로 제거했다.)
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
ap.add_argument("--label", default="출력만 최적화", help="오른쪽 칸 이름")
ap.add_argument("--color", choices=["none", "err", "z0", "r0"], default="none",
                help="입자 색. none=검정, err=정답과의 거리(오른쪽 칸만), "
                     "z0=초기 높이, r0=초기 중심에서의 거리. z0/r0 은 **두 칸에 "
                     "같은 색**을 입혀 어느 부분이 어디로 갔는지 맞대 볼 수 있다")
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
print(f"[덤프] {D['tag']} t0={D['t0']}  {T} 프레임, 입자 {N} "
      f"(그리는 건 {len(sel)})  오차 중앙 {np.median(err):.3f}% 최대 {err.max():.3f}%",
      flush=True)

# 손잡이 위치와 반경 (없으면 그리지 않는다)
_CP = D.get("ctrl_pos")
if _CP is not None:
    _CP = np.asarray(_CP, dtype=np.float32)
_R_CTRL = float(D.get("ctrl_R", 0.15))

# 입자별 고정 색 (z0/r0). 두 칸이 같은 값을 쓰므로 대응이 보인다.
CVAL, CLAB, CMAP = None, "", "viridis"
if a.color in ("z0", "r0"):
    X0 = np.asarray(D["x0"], dtype=np.float32)[sel]
    if a.color == "z0":
        CVAL, CLAB, CMAP = X0[:, 2], "초기 높이 z", "viridis"
    else:
        cen = X0.mean(0)
        CVAL = np.linalg.norm(X0 - cen, axis=-1)
        CLAB, CMAP = "초기 중심에서의 거리", "plasma"
    print(f"[색] {CLAB}  {CVAL.min():.4f} ~ {CVAL.max():.4f}", flush=True)

frames = []
for t in tqdm(range(T), desc="렌더", ncols=80):
    fig, ax = plt.subplots(1, 2, figsize=(10.2, 5.0), dpi=110)
    i, j = 0, 2                                          # xz 평면
    # 손잡이 반경 안에 든 입자를 **빨갛게** 칠한다. 기준 칸은 PG 위치로,
    # 오른쪽 칸은 그 칸의 위치로 각각 판정한다 (같은 중심·같은 반경).
    mG = mP = None
    if _CP is not None:
        _ti0 = min(int(D["t0"]) + t, _CP.shape[0] - 1)
        mG = np.zeros(len(sel), bool); mP = np.zeros(len(sel), bool)
        for _k in range(_CP.shape[1]):
            _c = _CP[_ti0, _k]
            mG |= np.linalg.norm(G[t] - _c, axis=-1) < _R_CTRL
            mP |= np.linalg.norm(P[t] - _c, axis=-1) < _R_CTRL

    _oG = slice(None) if mG is None else ~mG
    _cG = "0.25" if CVAL is None else CVAL[_oG]
    ax[0].scatter(G[t][_oG, i], G[t][_oG, j], s=1.1, c=_cG, cmap=CMAP,
                  vmin=None if CVAL is None else CVAL.min(),
                  vmax=None if CVAL is None else CVAL.max(), linewidths=0)
    if mG is not None and mG.any():
        ax[0].scatter(G[t][mG, i], G[t][mG, j], s=1.6, c="red", linewidths=0)
    ax[0].set_title(f"PG MPM (기준)   손잡이 안 {0 if mG is None else int(mG.sum())}",
                    fontsize=11)
    _oP = slice(None) if mP is None else ~mP
    # 손잡이 밖은 기준 칸과 같은 검은색이다. 구속이 들어간 자리만 빨강으로
    # 떠야 하니 오차 색칠을 걷어냈다 (요청).
    if a.color == "err":
        ax[1].scatter(P[t][_oP, i], P[t][_oP, j], s=1.1, c=err[t][_oP],
                      cmap="inferno", vmin=0,
                      vmax=float(np.percentile(err, 99)) or 1.0, linewidths=0)
    else:
        ax[1].scatter(P[t][_oP, i], P[t][_oP, j], s=1.1,
                      c="0.25" if CVAL is None else CVAL[_oP], cmap=CMAP,
                      vmin=None if CVAL is None else CVAL.min(),
                      vmax=None if CVAL is None else CVAL.max(), linewidths=0)
    if mP is not None and mP.any():
        ax[1].scatter(P[t][mP, i], P[t][mP, j], s=1.6, c="red", linewidths=0)
    ax[1].set_title(f"{a.label} (평균 오차 {err[t].mean():.3f}% EXT)"
                    f"   손잡이 안 {0 if mP is None else int(mP.sum())}",
                    fontsize=11)
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

    for q in ax:
        q.set_xlim(lo[i] - pad, hi[i] + pad); q.set_ylim(lo[j] - pad, hi[j] + pad)
        q.set_aspect("equal"); q.set_xticks([]); q.set_yticks([])
    if CLAB:
        ax[0].set_title(ax[0].get_title() + f"   색 = {CLAB}", fontsize=10)
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

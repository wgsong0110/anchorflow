"""손잡이 3R 안 입자의 t -> t+1 위치 (xz) 와, t+1 에서 손잡이 위로 올라간 입자의 출신.

색은 **t 프레임의 z** 로 고정해 두 그림에서 같은 입자가 같은 색을 갖는다.
"""
from __future__ import annotations
import argparse
import numpy as np
import torch
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

ap = argparse.ArgumentParser()
ap.add_argument("--dump", required=True)
ap.add_argument("--t", type=int, default=27)
ap.add_argument("--rmul", type=float, default=3.0)
ap.add_argument("--out", required=True)
a = ap.parse_args()


def L(p):
    try: return torch.load(p, map_location="cpu", weights_only=False)
    except TypeError: return torch.load(p, map_location="cpu")


D = L(a.dump)
P = np.asarray(D["pred"], dtype=np.float64)
G = np.asarray(D["gt"], dtype=np.float64)
X0 = np.asarray(D["x0"], dtype=np.float64)
R = float(np.asarray(D["ctrl_R"]).reshape(-1)[0])
cid = int(np.asarray(D["ctrl_id"]).reshape(-1)[0])
t, t1 = a.t, a.t + 1
c0h = X0[cid]
in0 = np.linalg.norm(X0 - c0h, axis=1) <= R      # 처음부터 손잡이 안이던 입자

fig, axs = plt.subplots(2, 2, figsize=(13.5, 11.0), dpi=120)
th = np.linspace(0, 2 * np.pi, 200)
for row, (nm, A) in enumerate((("출력만 최적화", P), ("PG MPM", G))):
    c_t, c_t1 = A[t, cid], A[t1, cid]
    sel = np.linalg.norm(A[t] - c_t, axis=1) <= a.rmul * R   # t 에서 3R 안
    z27 = A[t, sel, 2]
    vlo, vhi = z27.min(), z27.max()
    # t+1 에서 손잡이 위 (중심 z + R 보다 위)
    zcut = c_t1[2] + R
    up = A[t1, sel, 2] > zcut
    print(f"--- {nm}  3R 안 입자 {int(sel.sum())}  손잡이중심 t {np.round(c_t,4)} "
          f"-> t+1 {np.round(c_t1,4)}", flush=True)
    print(f"    t+1 에서 z > {zcut:.4f} 인 입자 {int(up.sum())} 개", flush=True)
    if up.any():
        idx = np.nonzero(sel)[0][up]
        d_t = np.linalg.norm(A[t, idx] - c_t, axis=1)
        d_0 = np.linalg.norm(X0[idx] - c0h, axis=1)
        dz = A[t1, idx, 2] - A[t, idx, 2]
        print(f"    출신: t 에서 z {A[t,idx,2].min():.4f}~{A[t,idx,2].max():.4f} "
              f"(손잡이 중심 z {c_t[2]:.4f}), 중심거리/R {d_t.min()/R:.2f}~"
              f"{d_t.max()/R:.2f}, 그중 손잡이 안(<=R) {int((d_t<=R).sum())}",
              flush=True)
        print(f"    초기: z0 {X0[idx,2].min():.4f}~{X0[idx,2].max():.4f}, "
              f"초기 중심거리/R {d_0.min()/R:.2f}~{d_0.max()/R:.2f}, "
              f"처음부터 손잡이 안이던 것 {int(in0[idx].sum())} / {len(idx)}",
              flush=True)
        print(f"    한 프레임 z 이동 {dz.min():+.4f}~{dz.max():+.4f} "
              f"(손잡이 z 이동 {c_t1[2]-c_t[2]:+.4f})", flush=True)
    for col, (tt, cc) in enumerate(((t, c_t), (t1, c_t1))):
        q = axs[row][col]
        s = q.scatter(A[tt, sel, 0], A[tt, sel, 2], c=z27, cmap="turbo",
                      vmin=vlo, vmax=vhi, s=14, lw=0)
        if up.any() and col == 1:
            q.scatter(A[tt, sel][up, 0], A[tt, sel][up, 2], s=60,
                      facecolors="none", edgecolors="k", lw=1.0,
                      label=f"손잡이 위 {int(up.sum())}개")
        if up.any() and col == 0:
            q.scatter(A[tt, sel][up, 0], A[tt, sel][up, 2], s=60,
                      facecolors="none", edgecolors="k", lw=1.0,
                      label="같은 입자의 t 위치")
        q.plot(cc[0] + R * np.cos(th), cc[2] + R * np.sin(th), "k--", lw=1.3)
        q.plot(cc[0] + a.rmul * R * np.cos(th), cc[2] + a.rmul * R * np.sin(th),
               color="gray", ls=":", lw=1.1)
        q.axhline(c_t1[2] + R, color="red", ls="-", lw=1.0, alpha=0.7)
        q.set_aspect("equal"); q.grid(alpha=0.2)
        q.set_xlabel("x"); q.set_ylabel("z")
        q.set_title(f"{nm}  프레임 {tt}" + ("  (색 = 프레임 %d 의 z)" % t),
                    fontsize=11)
        if up.any(): q.legend(fontsize=9, loc="upper left")
        fig.colorbar(s, ax=q, shrink=0.85)
    lim = a.rmul * R * 1.6
    for col, cc in enumerate((c_t, c_t1)):
        axs[row][col].set_xlim(c_t[0] - lim, c_t[0] + lim)
        axs[row][col].set_ylim(c_t[2] - lim, c_t[2] + lim)
fig.suptitle(f"프레임 {t} 에서 손잡이 {a.rmul:g}R 안이던 입자의 {t} -> {t1} "
             f"(점선 원 = 손잡이 R, 빨간 선 = 프레임 {t1} 의 손잡이 중심 z + R)",
             fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(a.out)
print(f"[저장] {a.out}", flush=True)

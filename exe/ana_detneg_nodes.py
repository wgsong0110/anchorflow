"""프레임별 det(F)<0 셀 수, 그리고 특정 프레임의 손잡이 주변 격자 변위 벡터.

det 은 사면체 안에서 상수이므로, 한 프레임에서 det 값이 같은 입자들을 한 셀로
묶어 센다 (부동소수 값이 정확히 일치하는 것끼리 묶인다).
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
ap.add_argument("--frame", type=int, default=27)
ap.add_argument("--rmul", type=float, default=3.0, help="손잡이 반경의 몇 배 안")
ap.add_argument("--out_det", required=True)
ap.add_argument("--out_q", required=True)
a = ap.parse_args()


def L(p):
    try: return torch.load(p, map_location="cpu", weights_only=False)
    except TypeError: return torch.load(p, map_location="cpu")


D = L(a.dump)
fs = np.asarray(D["fscal"])[..., 0]          # [T,N] det F (우리)
gs = np.asarray(D["gt_fscal"])[..., 0]       # PG
T, N = fs.shape
R = float(np.asarray(D["ctrl_R"]).reshape(-1)[0])
cid = int(np.asarray(D["ctrl_id"]).reshape(-1)[0])
P = np.asarray(D["pred"], dtype=np.float64)

# ---- det<0 세기 ----
def count(M):
    cells, parts = [], []
    for t in range(T):
        neg = M[t] < 0
        parts.append(int(neg.sum()))
        cells.append(int(len(np.unique(M[t][neg]))) if neg.any() else 0)
    return np.array(cells), np.array(parts)


co, po = count(fs)
cg, pg = count(gs)
tot_cell = np.array([len(np.unique(fs[t])) for t in range(T)])
print(f"입자 {N}, 프레임 {T}, 셀(= det 상수 덩어리) 평균 {tot_cell.mean():.0f} 개",
      flush=True)
print(f"[우리] det<0 셀 최대 {co.max()} (프레임 {int(co.argmax())}), 마지막 {co[-1]}, "
      f"det<0 입자 최대 {po.max()}, 최소 det {fs.min():.4f}", flush=True)
print(f"[PG]   det<0 셀 최대 {cg.max()}, det<0 입자 최대 {pg.max()}, "
      f"최소 det {gs.min():.4f}", flush=True)
nz = np.nonzero(co)[0]
if len(nz):
    print(f"       우리 det<0 이 처음 생기는 프레임 {int(nz[0])}", flush=True)
for t in range(0, T, max(1, T // 12)):
    print(f"   프레임 {t:3d}: 우리 셀 {co[t]:4d} / 입자 {po[t]:5d}  |  "
          f"PG 셀 {cg[t]:4d} / 입자 {pg[t]:5d}", flush=True)

fig, ax = plt.subplots(1, 2, figsize=(13, 4.2), dpi=120)
fr = np.arange(T)
ax[0].plot(fr, co, color="crimson", lw=2, label="출력만 최적화")
ax[0].plot(fr, cg, color="royalblue", lw=2, label="PG MPM")
ax[0].set_ylabel("det<0 셀 수"); ax[0].set_title("뒤집힌 셀")
ax[1].plot(fr, po, color="crimson", lw=2, label="출력만 최적화")
ax[1].plot(fr, pg, color="royalblue", lw=2, label="PG MPM")
ax[1].set_ylabel("det<0 입자 수"); ax[1].set_title("뒤집힌 셀 안의 입자")
for q in ax:
    q.set_xlabel("프레임"); q.grid(alpha=0.25); q.legend()
    q.axvline(a.frame, color="gray", ls=":", lw=1)
fig.tight_layout(); fig.savefig(a.out_det)
print(f"[저장] {a.out_det}", flush=True)

# ---- 손잡이 주변 격자 변위 ----
t = a.frame
npos, dpn, sup, cat = D["nodes"][t]
npos = np.asarray(npos, dtype=np.float64)
dpn = np.asarray(dpn, dtype=np.float64)
c = P[t, cid]
d = np.linalg.norm(npos - c, axis=1)
m = d <= a.rmul * R
mag = np.linalg.norm(dpn[m], axis=1)
print(f"프레임 {t}: 손잡이 중심 {np.round(c, 4)}, {a.rmul}R={a.rmul * R:.4f} 안의 "
      f"격자점 {int(m.sum())} 개 (전체 {len(npos)}), 변위 크기 평균 {mag.mean():.5f} "
      f"최대 {mag.max():.5f}, R 안 {int((d <= R).sum())} 개", flush=True)

PRJ = [((0, 2), "x", "z"), ((0, 1), "x", "y"), ((1, 2), "y", "z")]
lim = a.rmul * R * 1.15
th = np.linspace(0, 2 * np.pi, 200)
fig2, axs = plt.subplots(1, 3, figsize=(16.5, 5.6), dpi=120)
for k, ((u, v), lu, lv) in enumerate(PRJ):
    q = axs[k]
    q.plot(R * np.cos(th), R * np.sin(th), "k--", lw=1.3, label="손잡이 R")
    q.plot(a.rmul * R * np.cos(th), a.rmul * R * np.sin(th), color="gray",
           ls=":", lw=1.1, label=f"{a.rmul:g}R")
    rel = npos[m] - c
    q.quiver(rel[:, u], rel[:, v], dpn[m][:, u], dpn[m][:, v],
             mag, cmap="viridis", angles="xy", scale_units="xy", scale=1.0,
             width=0.004)
    q.scatter(rel[:, u], rel[:, v], s=4, c="k", alpha=0.35, lw=0)
    q.set_xlim(-lim, lim); q.set_ylim(-lim, lim); q.set_aspect("equal")
    q.set_xlabel(lu); q.set_ylabel(lv); q.grid(alpha=0.2)
    q.set_title(f"{lu}{lv} 평면 (화살표는 실제 크기)", fontsize=11)
axs[0].legend(fontsize=9, loc="upper left")
sm = plt.cm.ScalarMappable(cmap="viridis",
                           norm=plt.Normalize(mag.min(), mag.max()))
fig2.colorbar(sm, ax=axs, shrink=0.85, label="변위 크기")
fig2.suptitle(f"프레임 {t} · 손잡이 중심에서 {a.rmul:g}R 안의 격자점 "
              f"{int(m.sum())}개의 격자 변위", fontsize=13)
fig2.savefig(a.out_q, bbox_inches="tight")
print(f"[저장] {a.out_q}", flush=True)

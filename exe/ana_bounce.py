"""튕기는 씬 분석: 손잡이 위로 솟는 입자의 출신, det<0, 손잡이 주변 격자 변위.

손잡이 중심은 덤프에 ctrl_id 가 있으면 그 입자의 현재 위치, 없으면 명령 궤적
ctrl_pos[t] 를 쓴다.
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
ap.add_argument("--t", type=int, default=27, help="이 프레임 -> t+1 을 본다")
ap.add_argument("--rmul", type=float, default=3.0)
ap.add_argument("--tag", default="np64s")
ap.add_argument("--outdir", required=True)
a = ap.parse_args()


def L(p):
    try: return torch.load(p, map_location="cpu", weights_only=False)
    except TypeError: return torch.load(p, map_location="cpu")


D = L(a.dump)
P = np.asarray(D["pred"], dtype=np.float64)
G = np.asarray(D["gt"], dtype=np.float64)
X0 = np.asarray(D["x0"], dtype=np.float64)
R = float(np.asarray(D["ctrl_R"]).reshape(-1)[0])
CP = np.asarray(D["ctrl_pos"], dtype=np.float64)[:, 0, :]
cid = int(np.asarray(D["ctrl_id"]).reshape(-1)[0]) if "ctrl_id" in D else None
T, N = P.shape[0], P.shape[1]
t, t1 = a.t, a.t + 1
print(f"{a.tag}: 입자 {N}, 프레임 {T}, R {R:.4f}, 중심 = "
      f"{'ctrl_id 입자' if cid is not None else 'ctrl_pos 명령'}", flush=True)


def cen(A, tt):
    return A[tt, cid] if cid is not None else CP[tt]


c0 = cen(P, 0)
in0 = np.linalg.norm(X0 - c0, axis=1) <= R

# ---- 1. t+1 에서 손잡이 위로 솟은 입자의 출신 ----
for nm, A in (("출력만 최적화", P), ("PG MPM", G)):
    c, c1 = cen(A, t), cen(A, t1)
    zcut = c1[2] + R
    up = A[t1, :, 2] > zcut
    print(f"[{nm}] 손잡이 중심 z {c[2]:.4f} -> {c1[2]:.4f}, 기준 z>{zcut:.4f}, "
          f"프레임 {t1} 에서 위에 있는 입자 {int(up.sum())} / {N} "
          f"(전체 최고 z {A[t1,:,2].max():.4f})", flush=True)
    if up.any():
        i = np.nonzero(up)[0]
        d0 = np.linalg.norm(X0[i] - c0, axis=1)
        dt = np.linalg.norm(A[t, i] - c, axis=1)
        print(f"    프레임 {t} 위치: z {A[t,i,2].min():.4f}~{A[t,i,2].max():.4f} "
              f"(손잡이 중심 z {c[2]:.4f}), 중심거리/R {dt.min()/R:.2f}~"
              f"{dt.max()/R:.2f}, 그중 손잡이 안 {int((dt<=R).sum())}, "
              f"3R 안 {int((dt<=3*R).sum())}", flush=True)
        print(f"    초기 위치: z0 {X0[i,2].min():.4f}~{X0[i,2].max():.4f}, "
              f"초기 중심거리/R {d0.min()/R:.2f}~{d0.max()/R:.2f}, "
              f"처음부터 손잡이 안 {int(in0[i].sum())} / {len(i)}", flush=True)
        dz = A[t1, i, 2] - A[t, i, 2]
        print(f"    {t}->{t1} z 이동 5/50/95% "
              f"{np.round(np.percentile(dz,[5,50,95]),4)} "
              f"(손잡이 {c1[2]-c[2]:+.4f})", flush=True)

# ---- 2. t -> t+1 상대 위치 그림 (xz, 3R 안) ----
fig, axs = plt.subplots(2, 2, figsize=(13.5, 11.0), dpi=120)
th = np.linspace(0, 2 * np.pi, 200)
for row, (nm, A) in enumerate((("출력만 최적화", P), ("PG MPM", G))):
    c, c1 = cen(A, t), cen(A, t1)
    sel = np.linalg.norm(A[t] - c, axis=1) <= a.rmul * R
    zt = A[t, sel, 2]
    up = A[t1, sel, 2] > c1[2] + R
    for col, (tt, cc) in enumerate(((t, c), (t1, c1))):
        q = axs[row][col]
        s = q.scatter(A[tt, sel, 0], A[tt, sel, 2], c=zt, cmap="turbo",
                      s=10, lw=0)
        if up.any():
            q.scatter(A[tt, sel][up, 0], A[tt, sel][up, 2], s=45,
                      facecolors="none", edgecolors="k", lw=0.8,
                      label=f"프레임 {t1} 에 손잡이 위 {int(up.sum())}개")
            q.legend(fontsize=9, loc="upper left")
        q.plot(cc[0] + R * np.cos(th), cc[2] + R * np.sin(th), "k--", lw=1.3)
        q.plot(cc[0] + a.rmul * R * np.cos(th), cc[2] + a.rmul * R * np.sin(th),
               color="gray", ls=":", lw=1.1)
        q.axhline(c1[2] + R, color="red", lw=1.0, alpha=0.8)
        q.set_aspect("equal"); q.grid(alpha=0.2)
        q.set_xlabel("x"); q.set_ylabel("z")
        q.set_title(f"{nm}  프레임 {tt}  (색 = 프레임 {t} 의 z)", fontsize=11)
        fig.colorbar(s, ax=q, shrink=0.85)
    lim = a.rmul * R * 1.7
    for col in (0, 1):
        axs[row][col].set_xlim(c[0] - lim, c[0] + lim)
        axs[row][col].set_ylim(c[2] - lim, c[2] + lim)
fig.suptitle(f"[{a.tag}] 프레임 {t} 에 손잡이 {a.rmul:g}R 안이던 입자의 {t}->{t1}  "
             f"(점선 원 = 손잡이 R, 빨간 선 = 프레임 {t1} 중심 z + R)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
f1 = f"{a.outdir}/{a.tag}_f{t}{t1}.png"
fig.savefig(f1); print(f"[저장] {f1}", flush=True)

# ---- 3. det<0 셀 수와 입자 비율 ----
if "fscal" in D:
    fs = np.asarray(D["fscal"])[..., 0]
    gf = D.get("gt_fscal")
    gs = np.asarray(gf)[..., 0] if gf is not None else None
    cells = np.zeros(T, int); parts = np.zeros(T, int)
    frac = np.zeros((T, 3))
    for tt in range(T):
        neg = fs[tt] < 0
        parts[tt] = neg.sum()
        cells[tt] = len(np.unique(fs[tt][neg])) if neg.any() else 0
        d = np.linalg.norm(P[tt] - cen(P, tt), axis=1)
        i1, i3 = d <= R, d <= a.rmul * R
        frac[tt] = (neg.mean() * 100,
                    neg[i1].mean() * 100 if i1.any() else np.nan,
                    neg[i3].mean() * 100 if i3.any() else np.nan)
    tot = len(np.unique(fs[0]))
    print(f"[det<0] 셀 총 {tot} 개 중 최대 {cells.max()} (프레임 "
          f"{int(cells.argmax())}), 입자 최대 {parts.max()}, 최소 det "
          f"{fs.min():.4f}, 처음 생기는 프레임 "
          f"{int(np.nonzero(cells)[0][0]) if cells.any() else -1}", flush=True)
    if gs is not None:
        print(f"        PG det<0 입자 최대 {int((gs<0).sum(1).max())}", flush=True)
    print(f"[비율 %] 전체 최대 {np.nanmax(frac[:,0]):.2f}, R 안 최대 "
          f"{np.nanmax(frac[:,1]):.2f}, {a.rmul:g}R 안 최대 "
          f"{np.nanmax(frac[:,2]):.2f}", flush=True)
    for tt in [t - 1, t, t1, t1 + 1]:
        if 0 <= tt < T:
            print(f"   프레임 {tt}: 셀 {cells[tt]}, 전체 {frac[tt,0]:.2f}%, "
                  f"R 안 {frac[tt,1]:.2f}%, {a.rmul:g}R 안 {frac[tt,2]:.2f}%",
                  flush=True)
    f2, x2 = plt.subplots(1, 2, figsize=(13, 4.2), dpi=120)
    x2[0].plot(np.arange(T), cells, color="crimson", lw=2)
    x2[0].set_ylabel("det<0 셀 수"); x2[0].set_title(f"뒤집힌 셀 (총 {tot})")
    for j, (lab, col) in enumerate((("전체", "k"), ("손잡이 R 안", "crimson"),
                                    (f"손잡이 {a.rmul:g}R 안", "darkorange"))):
        x2[1].plot(np.arange(T), frac[:, j], color=col, lw=2, label=lab)
    x2[1].set_ylabel("det<0 입자 비율 (%)"); x2[1].set_title("뒤집힌 셀 안 입자 비율")
    x2[1].legend()
    for q in x2:
        q.set_xlabel("프레임"); q.grid(alpha=0.25)
        q.axvline(t, color="gray", ls=":", lw=1)
    f2.tight_layout()
    fp = f"{a.outdir}/{a.tag}_detneg.png"
    f2.savefig(fp); print(f"[저장] {fp}", flush=True)

# ---- 4. 손잡이 주변 격자 변위 ----
if "nodes" in D:
    npos, dpn = [np.asarray(x, dtype=np.float64) for x in D["nodes"][t][:2]]
    c = cen(P, t)
    d = np.linalg.norm(npos - c, axis=1)
    m = d <= a.rmul * R
    mag = np.linalg.norm(dpn[m], axis=1)
    print(f"[격자] 프레임 {t}: {a.rmul:g}R 안 격자점 {int(m.sum())} / {len(npos)} "
          f"(R 안 {int((d<=R).sum())}), 변위 크기 평균 {mag.mean():.5f} 최대 "
          f"{mag.max():.5f}, z 성분 평균 {dpn[m][:,2].mean():+.5f}", flush=True)
    PRJ = [((0, 2), "x", "z"), ((0, 1), "x", "y"), ((1, 2), "y", "z")]
    lim = a.rmul * R * 1.15
    f3, x3 = plt.subplots(1, 3, figsize=(16.5, 5.6), dpi=120)
    for k, ((u, v), lu, lv) in enumerate(PRJ):
        q = x3[k]
        q.plot(R * np.cos(th), R * np.sin(th), "k--", lw=1.3, label="손잡이 R")
        q.plot(a.rmul * R * np.cos(th), a.rmul * R * np.sin(th), color="gray",
               ls=":", lw=1.1, label=f"{a.rmul:g}R")
        rel = npos[m] - c
        q.quiver(rel[:, u], rel[:, v], dpn[m][:, u], dpn[m][:, v], mag,
                 cmap="viridis", angles="xy", scale_units="xy", scale=1.0,
                 width=0.004)
        q.scatter(rel[:, u], rel[:, v], s=4, c="k", alpha=0.35, lw=0)
        q.set_xlim(-lim, lim); q.set_ylim(-lim, lim); q.set_aspect("equal")
        q.set_xlabel(lu); q.set_ylabel(lv); q.grid(alpha=0.2)
        q.set_title(f"{lu}{lv} 평면 (화살표 실제 크기)", fontsize=11)
    x3[0].legend(fontsize=9, loc="upper left")
    sm = plt.cm.ScalarMappable(cmap="viridis",
                               norm=plt.Normalize(mag.min(), mag.max()))
    f3.colorbar(sm, ax=x3, shrink=0.85, label="변위 크기")
    f3.suptitle(f"[{a.tag}] 프레임 {t} · 손잡이 {a.rmul:g}R 안 격자점 "
                f"{int(m.sum())}개의 격자 변위", fontsize=13)
    fq = f"{a.outdir}/{a.tag}_nodeq{t}.png"
    f3.savefig(fq, bbox_inches="tight"); print(f"[저장] {fq}", flush=True)

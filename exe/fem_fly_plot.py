"""사용자 10-29: lego lag 판 발산 프레임에서 '날아가는' 노드의 전후 위치 그림 (exe/fem_fly.sh --lagdump 결과의 lagdump.npz, torch 없이).
사용: python exe/fem_fly_plot.py <실행 디렉토리> <출력 접두> [--top 5]
  그림 1 <접두>_before_after.png : 정면·측면 x (발산 프레임 시작 | 마지막 유한 서브스텝). 입자 회색, 메시 노드 질량 색 (모든 칸 같은 범위), 바닥,
                                    날아간 노드 ○ -> × 화살표, 그 노드에 걸친 사면체 모서리 (det A_T < 0 빨강).
  그림 2 <접두>_timeseries.png   : 그 노드들의 서브스텝별 높이·속도 크기·이웃 사면체 det 최솟값 (덤프된 프레임 전부, 노드는 격자 좌표로 프레임 사이 대응), 접촉 (α > 0) 표시.
  <접두>_summary.json             : 수치 요약."""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                                       # noqa: E402
from matplotlib import font_manager                                                   # noqa: E402
from matplotlib.colors import LogNorm                                                 # noqa: E402
import numpy as np                                                                    # noqa: E402

for f in ("/home/wgsong/.fonts/NotoSansCJKkr-Regular.otf", "/home/wgsong/.fonts/NotoSansCJKkr-Bold.otf"):
    if os.path.exists(f):
        font_manager.fontManager.addfont(f)
plt.rcParams["font.family"] = "Noto Sans CJK KR"
plt.rcParams["axes.unicode_minus"] = False

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("prefix")
ap.add_argument("--top", type=int, default=5)
ap.add_argument("--k", type=int, default=-1, help="그림 1 대상 = 덤프 안 프레임 순번 (-1 = 마지막 = 발산 프레임 t*, -2 = t*-1 ...)")
ap.add_argument("--s", type=int, default=-1, help="그림 1 '이후' 상태 서브스텝 (-1 = 마지막 유한 상태, -2 = 그 프레임 max|U| 최대 서브스텝)")
a = ap.parse_args()

Z = np.load(os.path.join(a.run, "lagdump.npz"))
meta = json.loads(str(Z["meta"]))
K = len(meta)
G = lambda k, n: Z[f"f{k}_{n}"]
cmdline = open(os.path.join(a.run, "cmd.txt")).read().strip()
kL = K + a.k if a.k < 0 else a.k                                                     # 그림 1 대상 프레임
tstar = meta[-1]["t"] + 1                                                             # 발산 프레임
mt = meta[kL]
pt, nr = np.array(mt["plane"][0]), np.array(mt["plane"][1])
off = float(pt @ nr)
npos, mn, occn = G(kL, "npos"), G(kL, "mn"), G(kL, "occn")
tets, U, V, dA, al = G(kL, "tets"), G(kL, "U"), G(kL, "V"), G(kL, "detA"), G(kL, "alpha")
N = U.shape[0] - 1
fin = np.array([np.isfinite(U[s]).all() and np.isfinite(V[s]).all() for s in range(N + 1)])
live = mn > 0
if a.s == -1:
    s_end = int(np.nonzero(fin)[0].max())                                             # 마지막 유한 상태 (0 = 프레임 시작, s = s 번째 서브스텝 뒤)
elif a.s == -2:
    s_end = int(np.nanargmax([np.nanmax(np.linalg.norm(U[q][live], axis=-1)) if fin[q] else -1 for q in range(N + 1)]))
else:
    s_end = a.s
med = float(np.median(mn[live]))
disp = np.linalg.norm(U[s_end], axis=-1)
disp[~live] = -1
top = np.argsort(-disp)[:a.top]
# 서브스텝별 노드 이웃 사면체 det 최솟값
M, T = npos.shape[0], tets.shape[0]


def nbr_tets(k_, nodes):
    t_ = G(k_, "tets")
    return [np.nonzero((t_ == n).any(1))[0] for n in nodes]


nb = nbr_tets(kL, top)
x0 = npos[top]
x1 = npos[top] + U[s_end][top]
# 날아간 노드 이웃 셀이 처음 뒤집힌 서브스텝 (그 프레임 안)
first_inv = []
for i_, n in enumerate(top):
    dmin = np.nanmin(dA[:, nb[i_]], axis=1)                                          # [N+1]
    bad = np.nonzero(dmin <= 0)[0]
    s0 = int(bad[0]) if bad.size else None
    first_inv.append((s0, float(np.linalg.norm(V[s0][n])) if s0 is not None and np.isfinite(V[s0][n]).all() else None,
                      float(dmin[s0]) if s0 is not None else None))
# 전체 사면체 중 처음 뒤집힌 서브스텝 (그 프레임)
any_inv = np.nonzero((np.nan_to_num(dA, nan=1.0) <= 0).any(1))[0]

# ---------------------------------------------------------------- 그림 1
xp = G(kL, "xp")
rows, lamb = G(kL, "rows").astype(np.int64), G(kL, "lamb")
xp1 = xp + (lamb[..., None] * U[s_end][rows]).sum(1)
up = nr / np.linalg.norm(nr)
e1 = np.cross(up, [0, 1, 0] if abs(up[1]) < 0.9 else [1, 0, 0]); e1 /= np.linalg.norm(e1)
e2 = np.cross(up, e1)
views = [("정면", e1), ("측면", e2)]
EDGES = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
mnorm = LogNorm(vmin=1e-3, vmax=3.0)
cm = plt.get_cmap("viridis")
cols = plt.get_cmap("tab10")
fig, axs = plt.subplots(2, 3, figsize=(25, 14), layout="constrained")
FL = "프레임 %d (발산 프레임 %d%s)" % (mt["t"] + 1, tstar, "" if mt["t"] + 1 == tstar else " 의 %d 프레임 앞" % (tstar - mt["t"] - 1))
states = [(0, "%s 시작 (서브스텝 0)" % FL), (s_end, "서브스텝 %d / %d 뒤" % (s_end, N))]
for r_, (vn, eh) in enumerate(views):
    for c_, (s, ttl) in enumerate(states + [(None, "확대: 시작 (점선·○) 과 서브스텝 %d (실선·×) 겹침" % s_end)]):
        ax = axs[r_, c_]
        zoom = s is None
        P = xp1 if (zoom or s != 0) else xp
        ok = np.isfinite(P).all(1)
        ax.scatter(P[ok] @ eh, P[ok] @ up - off, s=(0.6 if zoom else 0.05), c="0.7", alpha=0.3, lw=0, rasterized=True, label="입자")
        Q = npos + U[s_end if zoom else s]
        okn = live & np.isfinite(Q).all(1)
        sc = ax.scatter(Q[okn] @ eh, Q[okn] @ up - off, s=(6 if zoom else 1.2), c=mn[okn] / med, cmap=cm, norm=mnorm, lw=0, rasterized=True)
        # 이웃 사면체 모서리 (확대 칸: 시작 점선 + 이후 실선)
        for ss_, ls_ in ([(0, ":"), (s_end, "-")] if zoom else [(s, "-")]):
            Qs = npos + U[ss_]
            for i_, n in enumerate(top):
                for t_ in nb[i_]:
                    pts = Qs[tets[t_]]
                    neg = dA[ss_, t_] <= 0
                    for e0, e1_ in EDGES:
                        ax.plot([pts[e0] @ eh, pts[e1_] @ eh], [pts[e0] @ up - off, pts[e1_] @ up - off], ls=ls_,
                                color=("red" if neg else cols(i_)), lw=(1.4 if neg else 0.5), alpha=(0.95 if neg else 0.5))
        for i_, n in enumerate(top):
            ax.plot(x0[i_] @ eh, x0[i_] @ up - off, "o", mfc="none", mec=cols(i_), ms=10, mew=1.8)
            ax.plot(x1[i_] @ eh, x1[i_] @ up - off, "x", color=cols(i_), ms=11, mew=2.2)
            ax.annotate("", xy=(x1[i_] @ eh, x1[i_] @ up - off), xytext=(x0[i_] @ eh, x0[i_] @ up - off),
                        arrowprops=dict(arrowstyle="->", color=cols(i_), lw=1.5))
            ax.annotate(f"#{n}", (x1[i_] @ eh, x1[i_] @ up - off), xytext=(5, 5), textcoords="offset points", fontsize=9, color=cols(i_),
                        weight="bold", annotation_clip=True)
            if zoom:
                ax.text(0.01, 0.99 - 0.045 * i_, f"#{n}  질량 {100 * mn[n] / med:.3g}% (중앙 대비)  점유율 {occn[n]:.2f}  이동 {disp[n]:.3g} m",
                        transform=ax.transAxes, fontsize=9.5, color=cols(i_), weight="bold", va="top",
                        bbox=dict(fc="white", ec="none", alpha=0.75, pad=1))
        ax.axhline(0, color="saddlebrown", lw=1.2)
        if not zoom:
            ax.text(0.01, 0.01, "바닥 (높이 0)", transform=ax.get_yaxis_transform(), color="saddlebrown", fontsize=9)
        ax.set_aspect("equal", adjustable="datalim" if zoom else "box")
        ax.set_title(f"{vn} -- {ttl}", fontsize=11)
        ax.set_xlabel(f"{vn} 가로 축 (m)")
        ax.set_ylabel("바닥 위 높이 (m)")
        if zoom:
            zp = np.concatenate([x0] + [npos[tets[nb[i_]].reshape(-1)] for i_ in range(len(top))])      # 시작 상태 이웃 셀
            c0, r0 = zp.mean(0), np.abs(zp - zp.mean(0)).max()
            far = np.linalg.norm(x1 - c0, axis=-1) < r0 + 0.4                       # 0.4 m 넘게 날아간 끝점은 칸 밖 (전체 칸에서 보인다)
            zp = np.concatenate([zp, x1[far]] + [(npos + U[s_end])[tets[nb[i_]].reshape(-1)] for i_ in range(len(top)) if far[i_]])
            allp = zp[np.isfinite(zp).all(1)]
            pad = 0.03
        else:
            allp = np.concatenate([xp, x0, x1])
            allp = allp[np.isfinite(allp).all(1)]
            pad = 0.05
        lo_, hi_ = np.percentile(allp @ eh, [0, 100]), np.percentile(allp @ up - off, [0, 100])
        ax.set_xlim(lo_[0] - pad, lo_[1] + pad)
        ax.set_ylim((hi_[0] if zoom else min(hi_[0], 0)) - pad, hi_[1] + pad)
cb = fig.colorbar(sc, ax=axs, shrink=0.5, pad=0.02)
cb.set_label("노드 질량 / 중앙값 (로그, 모든 칸 같은 범위)")
fig.suptitle(f"lego lag 판 {FL}: 이동이 큰 노드 {a.top} 개 (○ 시작 → × 서브스텝 {s_end})  |  모서리: 그 노드에 걸친 사면체 (빨강 = det A_T ≤ 0)\n"
             f"{os.path.basename(a.run)}  {cmdline.split('fem_ex_sim.py')[-1].strip()[:150]}", fontsize=11)
fig.savefig(a.prefix + "_before_after.png", dpi=110)
plt.close(fig)

# ---------------------------------------------------------------- 그림 2 (덤프 프레임 전부, 격자 좌표로 대응)
key = lambda P: [tuple(q) for q in np.round(P / 1e-5).astype(np.int64)]
kt = key(npos[top])
fig, axs = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
xoff = 0
series = {int(n): [] for n in top}
hseen = []
for k in range(K):
    mk = meta[k]
    npk, Uk, Vk, dAk, alk, tk = G(k, "npos"), G(k, "U"), G(k, "V"), G(k, "detA"), G(k, "alpha"), G(k, "tets")
    Nk = Uk.shape[0] - 1
    def idx_get(i_):
        d_ = np.linalg.norm(npk - npos[top[i_]], axis=-1)
        j_ = int(np.argmin(d_))
        return j_ if d_[j_] < 1e-3 else None
    ss = np.arange(Nk + 1) + xoff
    for i_, n in enumerate(top):
        j = idx_get(i_)
        if j is None:
            continue
        nbk = np.nonzero((tk == j).any(1))[0]
        hgt = (npk[j] + Uk[:, j]) @ up - off
        spd = np.linalg.norm(Vk[:, j], axis=-1)
        dmn = np.nanmin(dAk[:, nbk], axis=1) if nbk.size else np.full(Nk + 1, np.nan)
        if mk["first_bad_sub"] is not None:                                          # 비유한 뒤 값은 그리지 않는다
            spd, hgt, dmn = [q.astype(float).copy() for q in (spd, hgt, dmn)]
            for q in (spd, hgt, dmn):
                q[mk["first_bad_sub"] + 1:] = np.nan
        lab = f"#{n}" if k == kL else None
        hseen.append(hgt[:s_end + 1] if k == kL else hgt)
        axs[0].plot(ss, hgt, color=cols(i_), lw=1, label=lab)
        axs[1].plot(ss, spd, color=cols(i_), lw=1, label=lab)
        axs[2].plot(ss, dmn, color=cols(i_), lw=1, label=lab)
        con = np.nonzero(alk[:, j] > 0)[0]
        if con.size:
            axs[0].plot(ss[con], hgt[con], "|", color=cols(i_), ms=6)
            axs[1].plot(ss[con], spd[con], "|", color=cols(i_), ms=6)
        series[int(n)].append(dict(frame=mk["t"] + 1, node_idx=int(j), m_pct=float(100 * G(k, "mn")[j] / mk["m_med"]), occ=float(G(k, "occn")[j]),
                                   contact_substeps=int(con.size), first_contact=(int(con[0]) if con.size else None),
                                   detmin=float(np.nanmin(dmn)) if np.isfinite(dmn).any() else None,
                                   first_inv=(int(np.nonzero(dmn <= 0)[0][0]) if (dmn <= 0).any() else None), vmax=float(np.nanmax(spd))))
    # 전체 접촉 노드 수 (배경)
    ncon = (alk > 0).sum(1)
    for ax in axs:
        ax.axvline(xoff, color="k", lw=0.8, ls="--")
        if mk["first_bad_sub"] is not None:
            ax.axvspan(xoff + mk["first_bad_sub"] + 1, xoff + Nk, color="red", alpha=0.08)
    axs[0].text(xoff + 1, 1.0, f"프레임 {mk['t'] + 1} 시작", transform=axs[0].get_xaxis_transform(), fontsize=9, va="top")
    ax2 = axs[1].twinx() if k == 0 else ax2
    ax2.plot(ss[:-1], ncon, color="0.6", lw=0.8)
    xoff += Nk + 1
ax2.set_ylabel("접촉 노드 수 (α > 0, 회색)", color="0.4")
axs[0].axhline(0, color="saddlebrown", lw=1)
hs_ = np.concatenate(hseen)
hs_ = hs_[np.isfinite(hs_)]
axs[0].set_ylim(min(-0.02, hs_.min() - 0.03), hs_.max() + 0.05)                       # 그림 1 의 '이후' 서브스텝까지 범위 (그 뒤 폭주 값은 축 밖)
axs[0].text(0.005, 0.03, f"세로 축 = 그림 1 의 서브스텝 {s_end} 까지 범위 (그 뒤 폭주는 축 밖)", transform=axs[0].transAxes, fontsize=9)
axs[0].set_ylabel("노드 높이 (바닥 위, m)")
axs[1].set_ylabel("노드 속도 크기 |v| (m/s)")
axs[1].set_yscale("log")
axs[2].set_ylabel("이웃 사면체 det A_T 최솟값")
axs[2].axhline(0, color="red", lw=1)
axs[2].set_ylim(-3, 1.1)
axs[2].text(0.005, 0.03, "세로 축 −3 아래는 잘림; 붉은 띠 = 비유한 값이 나온 뒤 (그리지 않음)", transform=axs[2].transAxes, fontsize=9)
axs[2].set_xlabel("서브스텝 (덤프 프레임 이어 붙임; 프레임 시작마다 노드는 격자 자리 U = 0 으로 돌아간다)")
axs[0].legend(loc="upper right", fontsize=9)
for ax in axs:
    ax.grid(alpha=0.3)
fig.suptitle(f"lego lag 판: {FL} 서브스텝 {s_end} 에서 이동이 큰 노드의 서브스텝별 시계열 (세로 막대 | = 그 노드 접촉 α > 0)\n{os.path.basename(a.run)}", fontsize=11)
fig.tight_layout()
fig.savefig(a.prefix + "_timeseries.png", dpi=110)
plt.close(fig)

summ = dict(run=a.run, cmd=cmdline, frame=mt["t"] + 1, t_star=tstar, N=N, s_end=s_end, first_bad_sub=mt["first_bad_sub"], m_med=med,
            first_inv_any=(int(any_inv[0]) if any_inv.size else None), n_inv_end=int((dA[s_end] <= 0).sum()),
            nodes=[dict(node=int(n), m_pct=float(100 * mn[n] / med), occ=float(occn[n]), x0=x0[i_].tolist(), x1=x1[i_].tolist(),
                        h0=float(x0[i_] @ up - off), h1=float(x1[i_] @ up - off), dist=float(disp[n]),
                        v_end=float(np.linalg.norm(V[s_end][n])), first_inv_sub=first_inv[i_][0], v_at_first_inv=first_inv[i_][1],
                        det_at_first_inv=first_inv[i_][2], contact_substeps=int((al[:, n] > 0).sum()), n_nbr_tets=int(len(nb[i_])),
                        frames=series[int(n)]) for i_, n in enumerate(top)])
json.dump(summ, open(a.prefix + "_summary.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(summ, ensure_ascii=False, indent=1))

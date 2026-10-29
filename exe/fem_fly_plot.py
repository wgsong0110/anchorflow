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
a = ap.parse_args()

Z = np.load(os.path.join(a.run, "lagdump.npz"))
meta = json.loads(str(Z["meta"]))
K = len(meta)
G = lambda k, n: Z[f"f{k}_{n}"]
cmdline = open(os.path.join(a.run, "cmd.txt")).read().strip()
kL = K - 1                                                                            # 발산 프레임 = 덤프 마지막
mt = meta[kL]
pt, nr = np.array(mt["plane"][0]), np.array(mt["plane"][1])
off = float(pt @ nr)
npos, mn, occn = G(kL, "npos"), G(kL, "mn"), G(kL, "occn")
tets, U, V, dA, al = G(kL, "tets"), G(kL, "U"), G(kL, "V"), G(kL, "detA"), G(kL, "alpha")
N = U.shape[0] - 1
fin = np.array([np.isfinite(U[s]).all() and np.isfinite(V[s]).all() for s in range(N + 1)])
s_end = int(np.nonzero(fin)[0].max())                                                 # 마지막 유한 상태 (0 = 프레임 시작, s = s 번째 서브스텝 뒤)
live = mn > 0
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
fig, axs = plt.subplots(2, 2, figsize=(17, 15))
states = [(0, "발산 프레임 %d 시작 (서브스텝 0)" % (mt["t"] + 1)), (s_end, "마지막 유한 서브스텝 %d / %d" % (s_end, N))]
for r_, (vn, eh) in enumerate(views):
    for c_, (s, ttl) in enumerate(states):
        ax = axs[r_, c_]
        P = xp if s == 0 else xp1
        ok = np.isfinite(P).all(1)
        ax.scatter(P[ok] @ eh, P[ok] @ up - off, s=0.05, c="0.7", alpha=0.25, lw=0, rasterized=True, label="입자")
        Q = npos + U[s]
        okn = live & np.isfinite(Q).all(1)
        sc = ax.scatter(Q[okn] @ eh, Q[okn] @ up - off, s=1.2, c=mn[okn] / med, cmap=cm, norm=mnorm, lw=0, rasterized=True)
        # 이웃 사면체 모서리
        for i_, n in enumerate(top):
            for t_ in nb[i_]:
                pts = Q[tets[t_]]
                neg = dA[s, t_] <= 0
                for e0, e1_ in EDGES:
                    ax.plot([pts[e0] @ eh, pts[e1_] @ eh], [pts[e0] @ up - off, pts[e1_] @ up - off],
                            color=("red" if neg else cols(i_)), lw=(1.4 if neg else 0.4), alpha=(0.95 if neg else 0.45))
        for i_, n in enumerate(top):
            ax.plot(x0[i_] @ eh, x0[i_] @ up - off, "o", mfc="none", mec=cols(i_), ms=10, mew=1.8)
            ax.plot(x1[i_] @ eh, x1[i_] @ up - off, "x", color=cols(i_), ms=11, mew=2.2)
            ax.annotate("", xy=(x1[i_] @ eh, x1[i_] @ up - off), xytext=(x0[i_] @ eh, x0[i_] @ up - off),
                        arrowprops=dict(arrowstyle="->", color=cols(i_), lw=1.5))
            ax.annotate(f"#{n}  질량 {100 * mn[n] / med:.2g}%  점유 {100 * occn[n]:.2g}%  {disp[n]:.3g} m", (x1[i_] @ eh, x1[i_] @ up - off),
                        xytext=(6, 6 - 13 * i_), textcoords="offset points", fontsize=8, color=cols(i_))
        ax.axhline(0, color="saddlebrown", lw=1.2)
        ax.text(0.01, 0.01, "바닥 (높이 0)", transform=ax.get_yaxis_transform(), color="saddlebrown", fontsize=9)
        ax.set_aspect("equal")
        ax.set_title(f"{vn} -- {ttl}", fontsize=11)
        ax.set_xlabel(f"{vn} 가로 축 (m)")
        ax.set_ylabel("바닥 위 높이 (m)")
        allp = np.concatenate([xp, x0, x1])
        allp = allp[np.isfinite(allp).all(1)]
        lo_, hi_ = np.percentile(allp @ eh, [0, 100]), np.percentile(allp @ up - off, [0, 100])
        pad = 0.05
        ax.set_xlim(lo_[0] - pad, lo_[1] + pad)
        ax.set_ylim(min(hi_[0], 0) - pad, hi_[1] + pad)
cb = fig.colorbar(sc, ax=axs, shrink=0.5, pad=0.02)
cb.set_label("노드 질량 / 중앙값 (로그, 모든 칸 같은 범위)")
fig.suptitle(f"lego lag 판 발산 프레임 {mt['t'] + 1}: 이동이 큰 노드 {a.top} 개 (○ 시작 → × 서브스텝 {s_end})  |  모서리: 그 노드에 걸친 사면체 (빨강 = det A_T ≤ 0)\n"
             f"{os.path.basename(a.run)}  {cmdline.split('fem_ex_sim.py')[-1].strip()[:150]}", fontsize=11)
fig.savefig(a.prefix + "_before_after.png", dpi=110)
plt.close(fig)

# ---------------------------------------------------------------- 그림 2 (덤프 프레임 전부, 격자 좌표로 대응)
key = lambda P: [tuple(q) for q in np.round(P / 1e-5).astype(np.int64)]
kt = key(npos[top])
fig, axs = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
xoff = 0
series = {int(n): [] for n in top}
for k in range(K):
    mk = meta[k]
    npk, Uk, Vk, dAk, alk, tk = G(k, "npos"), G(k, "U"), G(k, "V"), G(k, "detA"), G(k, "alpha"), G(k, "tets")
    Nk = Uk.shape[0] - 1
    idx = {q: j for j, q in enumerate(key(npk))}
    ss = np.arange(Nk + 1) + xoff
    for i_, n in enumerate(top):
        j = idx.get(kt[i_])
        if j is None:
            continue
        nbk = np.nonzero((tk == j).any(1))[0]
        hgt = (npk[j] + Uk[:, j]) @ up - off
        spd = np.linalg.norm(Vk[:, j], axis=-1)
        dmn = np.nanmin(dAk[:, nbk], axis=1) if nbk.size else np.full(Nk + 1, np.nan)
        lab = f"#{n}" if k == kL else None
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
    axs[0].text(xoff + 1, 1.0, f"프레임 {mk['t'] + 1} 시작", transform=axs[0].get_xaxis_transform(), fontsize=9, va="top")
    ax2 = axs[1].twinx() if k == 0 else ax2
    ax2.plot(ss[:-1], ncon, color="0.6", lw=0.8)
    xoff += Nk + 1
ax2.set_ylabel("접촉 노드 수 (α > 0, 회색)", color="0.4")
axs[0].axhline(0, color="saddlebrown", lw=1)
axs[0].set_ylabel("노드 높이 (바닥 위, m)")
axs[1].set_ylabel("노드 속도 크기 |v| (m/s)")
axs[1].set_yscale("log")
axs[2].set_ylabel("이웃 사면체 det A_T 최솟값")
axs[2].axhline(0, color="red", lw=1)
axs[2].set_xlabel("서브스텝 (덤프 프레임 이어 붙임; 프레임 시작마다 노드는 격자 자리 U = 0 으로 돌아간다)")
axs[0].legend(loc="upper right", fontsize=9)
for ax in axs:
    ax.grid(alpha=0.3)
fig.suptitle(f"lego lag 판: 발산 프레임 {mt['t'] + 1} 에서 이동이 큰 노드의 서브스텝별 시계열 (세로 막대 | = 그 노드 접촉 α > 0)\n{os.path.basename(a.run)}", fontsize=11)
fig.tight_layout()
fig.savefig(a.prefix + "_timeseries.png", dpi=110)
plt.close(fig)

summ = dict(run=a.run, cmd=cmdline, frame=mt["t"] + 1, N=N, s_end=s_end, first_bad_sub=mt["first_bad_sub"], m_med=med,
            first_inv_any=(int(any_inv[0]) if any_inv.size else None), n_inv_end=int((dA[s_end] <= 0).sum()),
            nodes=[dict(node=int(n), m_pct=float(100 * mn[n] / med), occ=float(occn[n]), x0=x0[i_].tolist(), x1=x1[i_].tolist(),
                        h0=float(x0[i_] @ up - off), h1=float(x1[i_] @ up - off), dist=float(disp[n]),
                        v_end=float(np.linalg.norm(V[s_end][n])), first_inv_sub=first_inv[i_][0], v_at_first_inv=first_inv[i_][1],
                        det_at_first_inv=first_inv[i_][2], contact_substeps=int((al[:, n] > 0).sum()), n_nbr_tets=int(len(nb[i_])),
                        frames=series[int(n)]) for i_, n in enumerate(top)])
json.dump(summ, open(a.prefix + "_summary.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(summ, ensure_ascii=False, indent=1))

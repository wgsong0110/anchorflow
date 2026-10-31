"""수박 발산 프레임 서브스텝별 노드 속도 그림 (exe/wm_divnode_stats.py 의 npz -> PNG 4 장).

1 원래 축: 서브스텝 1..N, |v| (m/s), 대표 노드별 선, 발산 서브스텝 (처음 발산 판정 속도를 넘은 서브스텝) 점·세로 점선, 발산 판정 속도 가로선
2 정규화: 노드마다 가로 s / (2 s_div), 세로 |v| / (2 v_thr) -- 발산 지점이 (0.5, 0.5)
3 평균 ± 표준편차: (가) 정규화 축 끝까지 넘어 있는 노드 전부 (가로 0..1 을 같은 칸으로 나눠 보간), (나) 원래 축 같은 노드들 vs 한 번도 안 넘은 점유 노드
4 원인: 같은 대표 노드의 서브스텝별 내부력 |f|, |f|/m, 노드 질량 분포
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
for _p in ("/home/wgsong/.fonts/NotoSansCJKkr-Regular.otf", os.path.expanduser("~/.fonts/NotoSansCJKkr-Regular.otf")):
    try:
        fm.fontManager.addfont(_p)
    except Exception:
        pass
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Noto Sans CJK KR"
plt.rcParams["axes.unicode_minus"] = False

ap = argparse.ArgumentParser()
ap.add_argument("--npz", required=True)
ap.add_argument("--out", required=True, help="출력 접두 (예 .../wm_divnode)")
ap.add_argument("--bins", type=int, default=100)
a = ap.parse_args()
Z = np.load(a.npz, allow_pickle=True)
N, vthr, mmed = int(Z["N"]), float(Z["vthr"]), float(Z["m_med"])
E, R, selE, selR = Z["E"], Z["R"], Z["selE"], Z["selR"]
vE, fE, mE, sE = Z["vE"], Z["fE"], Z["mE"], Z["sE"]
vR, fR, mR, sR = Z["vR"], Z["fR"], Z["mR"], Z["sR"]
def vpick(idx, sv, n):                                                           # 발산 서브스텝 값이 고르게 퍼지도록: 최소..최대를 n 등분한 목표에 가장 가까운 노드 (중복 없이)
    out = []
    for tg in np.linspace(sv.min(), sv.max(), n):
        o = np.argsort(np.abs(sv - tg), kind="stable")
        for q in o:
            if int(idx[q]) not in out:
                out.append(int(idx[q])); break
    return np.array(out)


selE = vpick(E, sE, len(selE))
selR = vpick(R, sR, len(selR))
posE = {int(n): k for k, n in enumerate(E)}
posR = {int(n): k for k, n in enumerate(R)}
sub = np.arange(1, N + 1)
cmapE = plt.get_cmap("viridis")
cols = [cmapE(x) for x in np.linspace(0.0, 0.85, len(selE))]
colsR = ["#d62728", "#ff7f0e", "#e377c2"][:len(selR)]
rows = [("E", int(n), posE[int(n)], c) for n, c in zip(selE, cols)] + [("R", int(n), posR[int(n)], c) for n, c in zip(selR, colsR)]


def get(kind, k):
    if kind == "E":
        return vE[k], fE[k], mE[k], int(sE[k]) + 1          # 서브스텝 1 부터
    return vR[k], fR[k], mR[k], int(sR[k]) + 1


def lab(kind, n, sd, m):
    w = "끝까지 넘음" if kind == "E" else "넘었다 돌아옴"
    return f"노드 {n} ({w}, 발산 {sd}, 질량 {m / mmed:.2f} 배)"


# ---- 1 원래 축 ----
fig, ax = plt.subplots(figsize=(13, 6.5))
for kind, n, k, c in rows:
    v, f, m, sd = get(kind, k)
    ax.plot(sub, v, color=c, lw=1.1, ls="-" if kind == "E" else "--", label=lab(kind, n, sd, m))
    ax.axvline(sd, color=c, ls=":", lw=0.9)
    ax.plot([sd], [v[sd - 1]], "o", color=c, ms=6, mec="k", mew=0.6)
ax.axhline(vthr, color="k", lw=1.0, ls="-.", label=f"발산 판정 속도 {vthr:.1f} m/s (0.25 EXT / h)")
ax.plot(sub, Z["vQm"], color="0.5", lw=1.6, label=f"한 번도 안 넘은 점유 노드 {int(Z['nQ']):,} 개 평균")
ax.set_xlim(1, N); ax.set_ylim(0, None)
ax.set_xlabel(f"서브스텝 (프레임 4 안, 1 에서 {N})"); ax.set_ylabel("노드 속력 |v| (m/s)")
ax.set_title(f"수박 발산 프레임 4: 대표 노드 서브스텝별 속력 (끝까지 넘은 {len(E)} 개 중 {len(selE)} 개 + 넘었다 돌아온 {len(R):,} 개 중 {len(selR)} 개)\n"
             "점 · 세로 점선 = 그 노드가 처음 발산 판정 속도를 넘은 서브스텝")
ax.legend(fontsize=8, loc="upper left", ncol=2); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(a.out + "_raw.png", dpi=130); plt.close(fig)

# ---- 2 정규화 ----
fig, ax = plt.subplots(figsize=(11, 6.5))
for kind, n, k, c in rows:
    v, f, m, sd = get(kind, k)
    ax.plot(sub / (2 * sd), v / (2 * vthr), color=c, lw=1.1, ls="-" if kind == "E" else "--", label=lab(kind, n, sd, m))
ax.plot([0.5], [0.5], "k*", ms=12, zorder=5, label="발산 지점 (0.5, 0.5)")
ax.axvline(0.5, color="k", ls=":", lw=0.8); ax.axhline(0.5, color="k", ls=":", lw=0.8)
ax.set_xlim(0, 1); ax.set_ylim(0, None)
ax.set_xlabel("서브스텝 / (2 x 그 노드의 발산 서브스텝)"); ax.set_ylabel(f"|v| / (2 x {vthr:.1f} m/s)")
ax.set_title("정규화 축: 노드마다 발산 서브스텝을 0.5, 발산 판정 속도를 0.5 로 (1 을 넘는 뒤쪽은 잘림)")
ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(a.out + "_norm.png", dpi=130); plt.close(fig)

# ---- 3 평균 ± 표준편차 ----
xg = (np.arange(a.bins) + 0.5) / a.bins
Yn = []
for k in range(len(E)):
    sd = int(sE[k]) + 1
    x = sub / (2 * sd)
    y = np.interp(xg, x, vE[k] / (2 * vthr), left=np.nan, right=np.nan)   # 노드마다 가로 0..1 을 같은 칸으로 보간 (늦게 넘은 노드는 x > N/(2 s_div) 칸이 없다)
    Yn.append(y)
Yn = np.array(Yn)
cnt = np.isfinite(Yn).sum(0)
mY, sY = np.nanmean(Yn, 0), np.nanstd(Yn, 0)
fig, axs = plt.subplots(1, 2, figsize=(16, 6.3))
ax = axs[0]
ax.fill_between(xg, mY - sY, mY + sY, color="C0", alpha=0.25, label="평균 ± 표준편차")
ax.plot(xg, mY, color="C0", lw=2, label=f"끝까지 넘은 {len(E)} 개 평균")
ax.plot([0.5], [0.5], "k*", ms=12, zorder=5, label="발산 지점 (0.5, 0.5)")
ax.axvline(0.5, color="k", ls=":", lw=0.8); ax.axhline(0.5, color="k", ls=":", lw=0.8)
ax2 = ax.twinx(); ax2.plot(xg, cnt, color="0.6", lw=0.8, ls="--"); ax2.set_ylabel("칸마다 값이 있는 노드 수 (점선)", color="0.4"); ax2.set_ylim(0, len(E) * 1.05)
ax.set_xlim(0, 1); ax.set_ylim(0, None)
ax.set_xlabel("서브스텝 / (2 x 발산 서브스텝)"); ax.set_ylabel(f"|v| / (2 x {vthr:.1f} m/s)")
ax.set_title(f"(가) 정규화 축, 끝까지 넘은 {len(E)} 개 (가로 {a.bins} 칸 보간)")
ax.legend(fontsize=9, loc="upper left"); ax.grid(alpha=0.3)
ax = axs[1]
mE_, sE_ = vE.mean(0), vE.std(0)
ax.fill_between(sub, mE_ - sE_, mE_ + sE_, color="C3", alpha=0.25)
ax.plot(sub, mE_, color="C3", lw=2, label=f"끝까지 넘은 {len(E)} 개 평균 ± 표준편차")
mR_, sR_ = vR.mean(0), vR.std(0)
ax.fill_between(sub, mR_ - sR_, mR_ + sR_, color="C1", alpha=0.18)
ax.plot(sub, mR_, color="C1", lw=1.6, label=f"넘었다 돌아온 {len(R):,} 개 평균 ± 표준편차")
ax.fill_between(sub, Z["vQm"] - Z["vQs"], Z["vQm"] + Z["vQs"], color="0.5", alpha=0.3)
ax.plot(sub, Z["vQm"], color="0.3", lw=1.6, label=f"한 번도 안 넘은 점유 노드 {int(Z['nQ']):,} 개 평균 ± 표준편차")
ax.axhline(vthr, color="k", lw=1.0, ls="-.", label=f"발산 판정 속도 {vthr:.1f} m/s")
ax.hist(sE + 1, bins=np.arange(0, N + 9, 8), weights=np.full(len(sE), 0.25), color="C3", alpha=0.5,
        label="끝까지 넘은 노드의 발산 서브스텝 분포 (8 서브스텝 칸, 노드 수 x 0.25)")
ax.set_xlim(1, N); ax.set_ylim(0, None)
ax.set_xlabel(f"서브스텝 (1 에서 {N})"); ax.set_ylabel("|v| (m/s)")
ax.set_title("(나) 원래 축: 발산 노드 vs 발산 안 한 점유 노드")
ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(a.out + "_meanstd.png", dpi=130); plt.close(fig)

# ---- 4 원인: 힘·질량 ----
fig, axs = plt.subplots(1, 3, figsize=(19, 6.0), gridspec_kw=dict(width_ratios=[1.25, 1.25, 1]))
for j, (key, ttl, yl) in enumerate([("f", "걸음 직전 내부력 |f|", "|f| (N)"), ("a", "|f| / 노드 질량", "|f| / m (m/s²)")]):
    ax = axs[j]
    for kind, n, k, c in rows:
        v, f, m, sd = get(kind, k)
        y = f if key == "f" else f / m
        ax.plot(sub, y, color=c, lw=0.9, ls="-" if kind == "E" else "--", label=lab(kind, n, sd, m))
        ax.plot([sd], [y[sd - 1]], "o", color=c, ms=5, mec="k", mew=0.5)
    qm, qs = (Z["fQm"], Z["fQs"]) if key == "f" else (Z["aQm"], Z["aQs"])
    ax.fill_between(sub, np.maximum(qm - qs, qm * 1e-2), qm + qs, color="0.5", alpha=0.3)
    ax.plot(sub, qm, color="0.3", lw=1.6, label="한 번도 안 넘은 점유 노드 평균 ± 표준편차 (로그 축이라 아래 끝은 평균의 1 % 에서 자름)")
    if key == "a":
        ax.axhline(vthr / float(Z["hs"]) / 1.0, color="k", ls="-.", lw=0.8, label="한 서브스텝에 발산 판정 속도만큼 바꾸는 가속도")
    ax.set_yscale("log"); ax.set_xlim(1, N)
    ax.set_xlabel("서브스텝"); ax.set_ylabel(yl); ax.set_title(ttl); ax.grid(alpha=0.3, which="both")
    if j == 0:
        ax.legend(fontsize=7, loc="lower right")
ax = axs[2]
cnts, edges = Z["mQ_hist"]
cnts = np.asarray(cnts, float); edges = np.asarray(edges, float)
w = np.diff(edges)
ax.bar(edges[:-1], cnts / cnts.sum() / w, width=w, align="edge", color="0.6", alpha=0.7, label=f"한 번도 안 넘은 점유 노드 {int(Z['nQ']):,} 개")
for X, c, nm in [(mE, "C3", f"끝까지 넘은 {len(E)} 개"), (mR, "C1", f"넘었다 돌아온 {len(R):,} 개")]:
    h_, _ = np.histogram(np.log10(X / mmed), bins=edges)
    ax.step(edges[:-1], h_ / max(h_.sum(), 1) / w, where="post", color=c, lw=1.6, label=nm)
for kind, n, k, c in rows:
    m = get(kind, k)[2]
    ax.axvline(np.log10(m / mmed), color=c, lw=0.8, ls=":")
ax.set_xlabel("log10 (노드 질량 / 전체 점유 노드 중앙값)"); ax.set_ylabel("밀도"); ax.set_title("노드 질량 분포 (점선 = 대표 노드)")
ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.suptitle("수박 발산 프레임 4: 같은 대표 노드의 내부력과 질량 (점 = 발산 서브스텝)")
fig.tight_layout(); fig.savefig(a.out + "_force_mass.png", dpi=130); plt.close(fig)
print("저장", a.out + "_{raw,norm,meanstd,force_mass}.png")

# ======== 10-31 추가: 셀 응력·힘 성분을 속도와 같은 세 그림으로 ========
EXTRA = [str(k) for k in Z["extra"]] if "extra" in Z.files else []


def qget(k):
    """(E 배열 [nE, N], R 배열, Q 평균, Q 표준편차)"""
    if k == "f_int":
        return fE, fR, Z["fQm"], Z["fQs"]
    return Z[k + "_E"], Z[k + "_R"], Z[k + "_Qm"], Z[k + "_Qs"]


def qtriple(prefix, specs, ttl):
    nr = len(specs)
    # 1 원래 축
    fig, axs = plt.subplots(nr, 1, figsize=(13, 4.6 * nr), squeeze=False)
    for r, (k, nm, un, lg) in enumerate(specs):
        ax = axs[r, 0]; XE, XR, qm, qs = qget(k)
        for kind, n, kk, c in rows:
            y = (XE if kind == "E" else XR)[kk]; sd = get(kind, kk)[3]
            ax.plot(sub, y, color=c, lw=1.0, ls="-" if kind == "E" else "--", label=lab(kind, n, sd, get(kind, kk)[2]))
            ax.axvline(sd, color=c, ls=":", lw=0.8)
            ax.plot([sd], [y[sd - 1]], "o", color=c, ms=5, mec="k", mew=0.5)
        ax.plot(sub, qm, color="0.4", lw=1.6, label=f"한 번도 안 넘은 점유 노드 {int(Z['nQ']):,} 개 평균")
        if lg: ax.set_yscale("log")
        ax.set_xlim(1, N); ax.set_xlabel(f"서브스텝 (프레임 4 안, 1 에서 {N})"); ax.set_ylabel(f"{nm} ({un})")
        ax.set_title(f"{nm}: 대표 노드 (점 · 세로 점선 = 그 노드의 발산 서브스텝)"); ax.grid(alpha=0.3, which="both")
        if r == 0: ax.legend(fontsize=7, loc="upper left", ncol=2)
    fig.suptitle(f"수박 발산 프레임 4: {ttl} (원래 축)"); fig.tight_layout(); fig.savefig(prefix + "_raw.png", dpi=120); plt.close(fig)
    # 2 정규화: 가로 s / (2 s_div), 세로 y / (2 y(s_div))
    fig, axs = plt.subplots(nr, 1, figsize=(11, 4.6 * nr), squeeze=False)
    for r, (k, nm, un, lg) in enumerate(specs):
        ax = axs[r, 0]; XE, XR, qm, qs = qget(k); skip = []
        for kind, n, kk, c in rows:
            y = (XE if kind == "E" else XR)[kk]; sd = get(kind, kk)[3]; y0 = y[sd - 1]
            if not (np.isfinite(y0) and y0 > 0):
                skip.append(str(n)); continue
            ax.plot(sub / (2 * sd), y / (2 * y0), color=c, lw=1.0, ls="-" if kind == "E" else "--", label=lab(kind, n, sd, get(kind, kk)[2]))
        ax.plot([0.5], [0.5], "k*", ms=11, zorder=5)
        ax.axvline(0.5, color="k", ls=":", lw=0.8); ax.axhline(0.5, color="k", ls=":", lw=0.8)
        if lg: ax.set_yscale("log")
        ax.set_xlim(0, 1)
        ax.set_xlabel("서브스텝 / (2 x 그 노드의 발산 서브스텝)"); ax.set_ylabel(f"{nm} / (2 x 발산 서브스텝 값)")
        ax.set_title(f"{nm}: 정규화 (발산 지점 (0.5, 0.5))" + (f" -- 발산 서브스텝 값이 0 이라 뺀 노드: {', '.join(skip)}" if skip else ""), fontsize=10)
        ax.grid(alpha=0.3, which="both")
        if r == 0: ax.legend(fontsize=7, loc="upper left")
    fig.suptitle(f"수박 발산 프레임 4: {ttl} (정규화 축)"); fig.tight_layout(); fig.savefig(prefix + "_norm.png", dpi=120); plt.close(fig)
    # 3 평균 ± 표준편차
    fig, axs = plt.subplots(nr, 2, figsize=(16, 4.8 * nr), squeeze=False)
    for r, (k, nm, un, lg) in enumerate(specs):
        XE, XR, qm, qs = qget(k)
        Yn, nz = [], 0
        for q in range(len(E)):
            sd = int(sE[q]) + 1; y0 = XE[q][sd - 1]
            if not (np.isfinite(y0) and y0 > 0):
                nz += 1; continue
            Yn.append(np.interp(xg, sub / (2 * sd), XE[q] / (2 * y0), left=np.nan, right=np.nan))
        ax = axs[r, 0]
        if Yn:
            Yn = np.array(Yn); mY, sY = np.nanmean(Yn, 0), np.nanstd(Yn, 0)
            ax.fill_between(xg, np.maximum(mY - sY, mY * 1e-2) if lg else mY - sY, mY + sY, color="C0", alpha=0.25, label="평균 ± 표준편차")
            ax.plot(xg, mY, color="C0", lw=2, label=f"끝까지 넘은 {len(Yn)} 개 평균" + (f" (발산 값 0 인 {nz} 개 뺌)" if nz else ""))
        ax.plot([0.5], [0.5], "k*", ms=11, zorder=5, label="발산 지점 (0.5, 0.5)")
        ax.axvline(0.5, color="k", ls=":", lw=0.8); ax.axhline(0.5, color="k", ls=":", lw=0.8)
        if lg: ax.set_yscale("log")
        ax.set_xlim(0, 1); ax.set_xlabel("서브스텝 / (2 x 발산 서브스텝)"); ax.set_ylabel(f"{nm} / (2 x 발산 서브스텝 값)")
        ax.set_title(f"(가) {nm}: 정규화 축 (가로 {a.bins} 칸 보간)", fontsize=10); ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.3, which="both")
        ax = axs[r, 1]
        for X_, c, nm2, al in [(XE, "C3", f"끝까지 넘은 {len(E)} 개", 0.25), (XR, "C1", f"넘었다 돌아온 {len(R):,} 개", 0.15)]:
            m_, s_ = np.nanmean(X_, 0), np.nanstd(X_, 0)
            ax.fill_between(sub, np.maximum(m_ - s_, m_ * 1e-2) if lg else m_ - s_, m_ + s_, color=c, alpha=al)
            ax.plot(sub, m_, color=c, lw=1.8, label=nm2 + " 평균 ± 표준편차")
        ax.fill_between(sub, np.maximum(qm - qs, qm * 1e-2) if lg else qm - qs, qm + qs, color="0.5", alpha=0.3)
        ax.plot(sub, qm, color="0.3", lw=1.6, label=f"한 번도 안 넘은 점유 노드 {int(Z['nQ']):,} 개 평균 ± 표준편차")
        if lg: ax.set_yscale("log")
        ax.set_xlim(1, N); ax.set_xlabel(f"서브스텝 (1 에서 {N})"); ax.set_ylabel(f"{nm} ({un})")
        ax.set_title(f"(나) {nm}: 원래 축, 발산 노드 vs 발산 안 한 점유 노드", fontsize=10); ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.3, which="both")
    fig.suptitle(f"수박 발산 프레임 4: {ttl} (평균 ± 표준편차)"); fig.tight_layout(); fig.savefig(prefix + "_meanstd.png", dpi=120); plt.close(fig)
    print("저장", prefix + "_{raw,norm,meanstd}.png")


if "tau_max" in EXTRA:
    qtriple(a.out + "_stress", [("tau_max", "셀 응력 ‖τ‖_F 최대", "Pa", True), ("tau_mean", "셀 응력 ‖τ‖_F 평균", "Pa", True)],
            "노드가 속한 점유 셀의 Kirchhoff 응력 크기 (셀들 최대·평균)")
if "f_total" in EXTRA:
    qtriple(a.out + "_force", [("f_int", "내부력 |f_int|", "N", True), ("f_floor", "바닥 힘 |f_바닥|", "N", True), ("f_total", "합력 |f_int + m g + f_바닥|", "N", True)],
            "노드에 주어지는 힘")

# ======== 10-31 추가: 노드별 프레임 안 최고 속력 분포 ========
if "vmax_Q" in Z.files:
    grp = [(Z["vmax_Q"], "0.55", f"한 번도 안 넘은 점유 노드 {len(Z['vmax_Q']):,} 개"),
           (Z["vmax_R"], "C1", f"넘었다 돌아온 노드 {len(Z['vmax_R']):,} 개"),
           (Z["vmax_E"], "C3", f"프레임 끝까지 넘은 노드 {len(Z['vmax_E'])} 개")]
    allv = np.concatenate([g[0] for g in grp])
    bins = np.linspace(0, float(allv.max()) * 1.02, 121)
    fig, ax = plt.subplots(figsize=(12, 6.5))
    for (v_, c, nm), al in zip(grp, (0.6, 0.6, 0.85)):                       # 겹쳐 그림 (쌓지 않음) -- 무리마다 같은 칸
        ax.hist(v_, bins=bins, color=c, alpha=al, label=nm, histtype="stepfilled", edgecolor=c, lw=1.0)
    ax.axvline(vthr, color="k", lw=1.4, ls="--")
    ax.text(vthr, 0.97, f" 발산 판정 속도 {vthr:.2f} m/s (0.25 EXT / h)", transform=ax.get_xaxis_transform(), va="top", fontsize=10)
    ax.set_yscale("log"); ax.set_xlim(0, bins[-1])
    ax.set_xlabel("노드별 최고 속력 (프레임 4 의 서브스텝 480 개 중 |v| 최대, m/s)"); ax.set_ylabel("노드 수 (로그, 무리마다 겹쳐 그림)")
    ax.set_title(f"수박 발산 프레임 4: 점유 노드 {len(allv):,} 개의 프레임 안 최고 속력 분포")
    ax.legend(fontsize=9, loc="upper right"); ax.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(os.path.join(os.path.dirname(a.out), "wm_divnode_vmax_hist.png"), dpi=130); plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 6.5))
    xs = np.sort(allv); ax.plot(xs, np.arange(1, len(xs) + 1) / len(xs), color="k", lw=2, label=f"점유 노드 전체 {len(xs):,} 개")
    for v_, c, nm in grp:
        xs_ = np.sort(v_); ax.plot(xs_, np.arange(1, len(xs_) + 1) / len(xs_), color=c, lw=1.5, label=nm)
    fr = float((allv > vthr).mean())
    ax.axvline(vthr, color="k", lw=1.4, ls="--")
    ax.text(vthr, 0.05, f" 발산 판정 속도 {vthr:.2f} m/s\n 넘은 노드 비율 {fr * 100:.3f} %", transform=ax.get_xaxis_transform(), fontsize=10)
    ax.set_xlim(0, bins[-1]); ax.set_ylim(0, 1.01)
    ax.set_xlabel("노드별 최고 속력 (m/s)"); ax.set_ylabel("누적 비율 (무리마다 따로)")
    ax.set_title("수박 발산 프레임 4: 노드별 최고 속력 누적 분포")
    ax.legend(fontsize=9, loc="center right"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(os.path.dirname(a.out), "wm_divnode_vmax_cdf.png"), dpi=130); plt.close(fig)
    print("저장 wm_divnode_vmax_{hist,cdf}.png")

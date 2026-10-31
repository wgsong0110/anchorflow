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
    ax.fill_between(sub, np.maximum(qm - qs, 1e-30), qm + qs, color="0.5", alpha=0.3)
    ax.plot(sub, qm, color="0.3", lw=1.6, label="한 번도 안 넘은 점유 노드 평균 ± 표준편차")
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

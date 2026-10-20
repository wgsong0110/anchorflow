"""psi_color.py 의 psi_hist.npz -> 프레임별 분포 패널 PNG (렌더와 프레임 동기화) + 공통 컬러바 PNG.

왼쪽: 그 프레임의 log10(psi+eps) 히스토그램, 정답·출력만 최적화 겹침 (같은 구간, 가중 없음 = 입자 수 비율).
      아래에 렌더와 같은 컬러맵 띠와 고정 범위 [lo, hi] 표시.
오른쪽: 평균 psi·중앙값 psi 시계열 (로그 축), 현재 프레임 세로선.

  python exe/psi_dist_vid.py --hist psi_hist.npz --range range.json --w 1600 --h 520 --out_dir frames --cbar cbar.png
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
for _p in (os.path.expanduser("~/.fonts/NotoSansCJKkr-Regular.otf"),):
    try:
        fm.fontManager.addfont(_p)
    except Exception:
        pass
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams["font.family"] = "Noto Sans CJK KR"
plt.rcParams["axes.unicode_minus"] = False
ap = argparse.ArgumentParser()
ap.add_argument("--hist", required=True)
ap.add_argument("--range", required=True)
ap.add_argument("--out_dir", required=True)
ap.add_argument("--cbar", required=True)
ap.add_argument("--w", type=int, default=1600)
ap.add_argument("--h", type=int, default=520)
a = ap.parse_args()
os.makedirs(a.out_dir, exist_ok=True)
Z = np.load(a.hist)
R = json.load(open(a.range))
lo, hi, eps, cmap = R["lo"], R["hi"], R["eps"], matplotlib.colormaps[R["cmap"]]
E = Z["edges"]; cen = 0.5 * (E[:-1] + E[1:]); bw = E[1] - E[0]
Hg, Ho = Z["h_gt"], Z["h_ov"]
T = Hg.shape[0]
C_GT, C_OV = "#222222", "#d9480f"
ymax = 1.1 * max(Hg[1:].max(), Ho[1:].max())
tt = np.arange(T)
ser = [Z["mean_gt"], Z["mean_ov"], Z["med_gt"], Z["med_ov"]]
pos = np.concatenate([s[1:] for s in ser]); pos = pos[pos > 0]
s_lo, s_hi = pos.min() / 2, pos.max() * 2
dpi = 100
for t in range(T):
    fig = plt.figure(figsize=(a.w / dpi, a.h / dpi), dpi=dpi)
    ax = fig.add_axes([0.06, 0.25, 0.42, 0.62])
    ax.step(cen, Hg[t], where="mid", color=C_GT, lw=1.8, label="정답 (PG)")
    ax.step(cen, Ho[t], where="mid", color=C_OV, lw=1.8, label="출력만 최적화")
    ax.fill_between(cen, Hg[t], step="mid", color=C_GT, alpha=0.12)
    ax.fill_between(cen, Ho[t], step="mid", color=C_OV, alpha=0.12)
    ax.axvline(lo, color="#888", lw=0.8, ls="--"); ax.axvline(hi, color="#888", lw=0.8, ls="--")
    ax.set_xlim(E[0], E[-1]); ax.set_ylim(0, ymax)
    ax.set_ylabel("입자 비율 (가중 없음)")
    ax.set_title(f"log10(ψ + {eps:g}) 분포  ·  프레임 {t}", fontsize=12)
    ax.legend(loc="upper right", frameon=False)
    if t == 0:
        ax.text(0.5, 0.5, "0 프레임: F = I, ψ = 0 (모두 왼쪽 끝 칸)", transform=ax.transAxes, ha="center", color="#555")
    ax.tick_params(labelbottom=False)
    cb = fig.add_axes([0.06, 0.14, 0.42, 0.07])
    xs = np.linspace(E[0], E[-1], 600)
    cb.imshow(cmap(np.clip((xs - lo) / (hi - lo), 0, 1))[None], aspect="auto", extent=(E[0], E[-1], 0, 1))
    cb.set_yticks([]); cb.set_xlim(E[0], E[-1])
    cb.set_xlabel(f"log10(ψ + {eps:g})  [ψ: Pa = J/m³]   색 범위 {lo:.2f} ~ {hi:.2f} (고정)", fontsize=10)
    ax2 = fig.add_axes([0.57, 0.14, 0.40, 0.73])
    for s, c, ls, nm in ((Z["mean_gt"], C_GT, "-", "정답 평균"), (Z["mean_ov"], C_OV, "-", "출력만 최적화 평균"),
                         (Z["med_gt"], C_GT, ":", "정답 중앙값"), (Z["med_ov"], C_OV, ":", "출력만 최적화 중앙값")):
        ax2.plot(tt[1:], s[1:], color=c, ls=ls, lw=1.6, label=nm)
        ax2.plot([t], [max(s[t], s_lo)], "o", color=c, ms=4)
    ax2.axvline(t, color="#888", lw=0.8)
    ax2.set_yscale("log"); ax2.set_ylim(s_lo, s_hi); ax2.set_xlim(0, T - 1)
    ax2.set_xlabel("프레임"); ax2.set_ylabel("ψ [Pa]")
    ax2.set_title("입자 ψ 평균·중앙값 (가중 없음)", fontsize=12)
    ax2.legend(loc="lower right", frameon=False, fontsize=9, ncol=2)
    for s_ in (ax, ax2):
        s_.spines[["top", "right"]].set_visible(False)
    fig.savefig(f"{a.out_dir}/{t:04d}.png", dpi=dpi)
    plt.close(fig)
# 위 줄 렌더 아래에 붙일 공통 컬러바 (렌더 색과 같은 매핑)
fig = plt.figure(figsize=(a.w / dpi, 90 / dpi), dpi=dpi)
cb = fig.add_axes([0.05, 0.42, 0.90, 0.30])
xs = np.linspace(lo, hi, 800)
cb.imshow(cmap((xs - lo) / (hi - lo))[None], aspect="auto", extent=(lo, hi, 0, 1))
cb.set_yticks([])
tk = np.linspace(lo, hi, 7)
cb.set_xticks(tk); cb.set_xticklabels([f"{v:.2f}" for v in tk], fontsize=9)
cb.set_title(f"가우시안 색 = log10(ψ + {eps:g}), ψ = FCR 탄성 에너지 밀도 [Pa];  고정 범위 {lo:.2f} ~ {hi:.2f} "
             f"(두 궤적 합친 {R['q'][0]:g}~{R['q'][1]:g}% 분위수, 밖은 끝 색) — 같은 ψ 면 같은 색", fontsize=10)
fig.savefig(a.cbar, dpi=dpi)
plt.close(fig)
print(f"[분포 패널] {a.out_dir} {T} 장, 컬러바 {a.cbar}", flush=True)

"""exe/fem_dv_stats.py 결과 (dv_stats.npz/json) 그림 (로컬, torch 없이).
사용: python exe/fem_dv_plot.py <출력 접두> <라벨>=<실행 디렉토리> ...
  <접두>_dist.png : 실행마다 한 행 -- 지표별 상자그림 (로그축), 시점 (시작 전 / 시작 / 직전 2 / 직전 1) x 집단 (정상 파랑 / 폭주 주황).
  <접두>_ts.png   : 실행마다 대표 폭주 노드 (직전 |v| 최대) 의 서브스텝별 m_n·|f_n|·|f|/m·|v|·이웃 셀 ‖∇λ‖ 비 최대·det A_T 최소·‖P‖ 최대,
                    정상 노드 표본 중앙~99 % 띠, x = 발산 시점 기준 서브스텝."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                                       # noqa: E402
from matplotlib import font_manager                                                   # noqa: E402
import numpy as np                                                                    # noqa: E402

for f in ("/home/wgsong/.fonts/NotoSansCJKkr-Regular.otf", "/home/wgsong/.fonts/NotoSansCJKkr-Bold.otf"):
    if os.path.exists(f):
        font_manager.fontManager.addfont(f)
plt.rcParams["font.family"] = "Noto Sans CJK KR"
plt.rcParams["axes.unicode_minus"] = False
C_N, C_R = "#2a78d6", "#eb6834"
INK2 = "#52514e"
pref = sys.argv[1]
RUNS = [q.split("=", 1) for q in sys.argv[2:]]
TPN = [("pre", "시작 전"), ("onset", "폭주 시작"), ("last2", "직전 2"), ("last1", "직전 1")]
MET = [("n", "m", "노드 질량 m_n (kg)", False), ("n", "f", "노드 내부력 |f_n| (N)", False), ("n", "acc", "가속도 |f_n|/m_n (m/s²)", False),
       ("c", "mT", "셀 질량 m_T (kg)", False), ("c", "occ", "셀 점유율 V_T/V_사면체", False), ("c", "P", "셀 응력 ‖P‖ (Pa)", False),
       ("c", "grat", "‖∇λ‖ 최대 / 기준", False), ("c", "detA", "det A_T (프레임 안 셀 사상)", True), ("c", "fv", "셀→노드 힘 V_T‖P F̄^T ∇λ‖ (N)", False)]

fig, axs = plt.subplots(len(RUNS), len(MET), figsize=(3.0 * len(MET), 3.1 * len(RUNS)), squeeze=False)
for ri, (lab, run) in enumerate(RUNS):
    Z = np.load(os.path.join(run, "dv_stats.npz"))
    info = json.load(open(os.path.join(run, "dv_stats.json")))
    for mi, (kind, q, title, lin) in enumerate(MET):
        ax = axs[ri, mi]
        data, pos, cols = [], [], []
        for ti, (tk, _) in enumerate(TPN):
            for gi, (grp, c) in enumerate((("norm", C_N), ("run", C_R))):
                key = f"{tk}_{kind}_{grp}_{q}"
                v = Z[key] if key in Z else np.array([])
                v = v[np.isfinite(v)]
                if not lin:
                    v = v[v > 0]
                data.append(v if v.size else np.array([np.nan]))
                pos.append(ti * 3 + gi)
                cols.append(c)
        bp = ax.boxplot(data, positions=pos, widths=0.8, whis=(0, 100), patch_artist=True, showfliers=False,
                        medianprops=dict(color="#0b0b0b", lw=1.2))
        for b_, c in zip(bp["boxes"], cols):
            b_.set_facecolor(c)
            b_.set_alpha(0.75)
            b_.set_edgecolor(c)
        for w_, c in zip(bp["whiskers"], np.repeat(cols, 2)):
            w_.set_color(c)
        for w_, c in zip(bp["caps"], np.repeat(cols, 2)):
            w_.set_color(c)
        if not lin:
            ax.set_yscale("log")
        else:
            ax.axhline(0, color=INK2, lw=0.6, ls=":")
            ax.set_ylim(-2.0, 3.0)                                                   # 바깥 값은 표 (dv_stats.json) 에
        ax.set_xticks([0.5, 3.5, 6.5, 9.5])
        tp = info["timepoints"]
        ax.set_xticklabels([f"{n}\n({tp[k]['rel']:+d})" for k, n in TPN], fontsize=7)
        ax.grid(axis="y", color="#e6e5e0", lw=0.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        if ri == 0:
            ax.set_title(title, fontsize=9)
        if mi == 0:
            ax.set_ylabel(lab, fontsize=9)
fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=C_N, alpha=0.75), plt.Rectangle((0, 0), 1, 1, color=C_R, alpha=0.75)],
           labels=["정상 노드 / 정상 셀 (무작위 표본)", "폭주 노드 / 그 이웃 셀"], loc="upper right", ncol=2, fontsize=10, frameon=False, bbox_to_anchor=(1.0, 1.0))
fig.suptitle("lego lag 판 발산: 폭주 vs 정상 (상자 = 25~75 %, 수염 = 최소~최대, 괄호 = 발산 시점 기준 서브스텝, det A_T 축은 −2~3 으로 자름)", x=0.01, ha="left", y=1.0, fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 1 - 0.08 / len(RUNS)))
fig.savefig(pref + "_dist.png", dpi=130, bbox_inches="tight")
plt.close(fig)

TSM = [("mn", "nq_mn", "m_n (kg)", True), ("fn", "nq_fn", "|f_n| (N)", True), ("acc", "nq_acc", "|f_n|/m_n (m/s²)", True), ("sp", "nq_sp", "|v_n| (m/s)", True),
       ("nG", "nq_nG", "이웃 ‖∇λ‖ 최대/기준", True), ("nD", "nq_nD", "이웃 det A_T 최소", False), ("nP", "nq_nP", "이웃 ‖P‖ 최대 (Pa)", True)]
fig, axs = plt.subplots(len(TSM), len(RUNS), figsize=(4.6 * len(RUNS), 1.9 * len(TSM)), squeeze=False, sharex="col")
for ri, (lab, run) in enumerate(RUNS):
    Z = np.load(os.path.join(run, "dv_stats.npz"))
    S, td = int(Z["S"]), int(Z["t_div"])
    sp_ = Z["ts_sp"]
    if sp_.shape[0] == 0:
        continue
    gl = int(Z["g_last"])
    i0 = int(np.nanargmax(np.where(np.isfinite(sp_[:, gl]), sp_[:, gl], -1)))
    g = np.arange(sp_.shape[1])
    xr = g - td
    lo = max(0, int(Z["g_on"]) - 3 * S // 4)
    sel = (g >= lo) & (g <= gl)
    for mi, (k, nk, title, lg) in enumerate(TSM):
        ax = axs[mi, ri]
        y = Z[f"ts_{k}"][i0]
        nq = Z[nk]
        ax.fill_between(xr[sel], nq[1][sel], nq[3][sel], color=C_N, alpha=0.18, lw=0, label="정상 표본 중앙~99 %" if k != "nD" else "정상 표본 1 %~중앙")
        if k == "nD":
            ax.fill_between(xr[sel], nq[0][sel], nq[1][sel], color=C_N, alpha=0.18, lw=0)
            ax.axhline(0, color=INK2, lw=0.6, ls=":")
        ax.plot(xr[sel], nq[1][sel], color=C_N, lw=1)
        ax.plot(xr[sel], y[sel], color=C_R, lw=1.6, label="대표 폭주 노드")
        for fb in range(1, len(Z["frames"])):
            if lo <= fb * S <= gl:
                ax.axvline(fb * S - td, color=INK2, lw=0.6, ls="--")
        ax.axvline(int(Z["g_on"]) - td, color=C_R, lw=0.8, ls=":")
        if lg:
            ax.set_yscale("log")
        ax.grid(color="#e6e5e0", lw=0.5)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        if ri == 0:
            ax.set_ylabel(title, fontsize=8)
        if mi == 0:
            ax.set_title(lab, fontsize=9)
            if ri == 0:
                ax.legend(fontsize=7, frameon=False, loc="upper left")
    axs[-1, ri].set_xlabel("발산 시점 기준 서브스텝 (점선 = 프레임 경계, 주황 점선 = 폭주 시작)", fontsize=8)
fig.tight_layout()
fig.savefig(pref + "_ts.png", dpi=130, bbox_inches="tight")
plt.close(fig)
print(pref + "_dist.png", pref + "_ts.png")

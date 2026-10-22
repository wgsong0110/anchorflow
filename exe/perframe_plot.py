"""프레임별 맞춤 곡선 (exe/perframe.sh 결과) -> PNG.

  python exe/perframe_plot.py <내려받은 lbours/perframe_fit 디렉토리> <출력.png>
열 = 궤적 (L000, 0004), 행:
  1) 출력만 최적화 상태 t 에서 E - E* 의 상대 초과 (E/E* - 1, %) -- 갱신 전(직전 프레임 망)·갱신 뒤
  2) 출력 오차 (망 한 스텝 vs 출력만 최적화 다음 상태, EXT 대비 %) -- 갱신 전·뒤
  3) 위치 오차 (정답 대비, %) -- 프레임별 맞춤 모델 롤아웃 · 출력만 최적화 · 다섯 궤적 모델
  4) 프레임별 맞춤 모델 롤아웃 상태와 출력만 최적화 상태의 거리 (%)
반복 수별(1000, 3000 있으면) 선을 같이 그린다. 모든 칸 공통 축 범위(행별로 두 궤적 같은 범위).
"""
import json
import os
import sys

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
U, OUT = sys.argv[1], sys.argv[2]
TR = ("L000", "0004")
C_PF = {1000: "#2a78d6", 3000: "#1c5cab"}
C_OV, C_MM = "#eb6834", "#1baf7a"
INK, MUTED = "#2b2b2b", "#8a8a85"


def J(p, default=None):
    return json.load(open(p)) if os.path.exists(p) else default


fig, ax = plt.subplots(4, 2, figsize=(14, 15), sharex=True, sharey="row")
for c, T in enumerate(TR):
    E = {r["state"]: r["E_min"] for r in J(os.path.join(U, f"estar_{T}.json"), [])}
    me = J(os.path.join(U, f"multi_err_{T}.json"), [])
    ov_drawn = False
    for N in (1000, 3000):
        R = J(os.path.join(U, f"n{N}", T, "records.json"), [])
        if not R:
            continue
        t = np.array([r["t"] for r in R])
        col = C_PF[N]
        ls_b = dict(color=col, lw=1.2, ls=":", alpha=0.9)
        ls_a = dict(color=col, lw=2.0)
        es = np.array([E.get(r["t"], np.nan) for r in R])
        exb = 100 * (np.array([r["Eu_before"] for r in R]) / es - 1)
        exa = 100 * (np.array([r["Eu_after"] for r in R]) / es - 1)
        ax[0, c].plot(t, np.clip(exb, 1e-3, None), **ls_b, label=f"갱신 전 (직전 프레임 망) · {N} 반복")
        ax[0, c].plot(t, np.clip(exa, 1e-3, None), **ls_a, label=f"갱신 뒤 · {N} 반복")
        ax[1, c].plot(t, 100 * np.array([r["out_err_before"] for r in R]), **ls_b, label=f"갱신 전 · {N} 반복")
        ax[1, c].plot(t, 100 * np.array([r["out_err_after"] for r in R]), **ls_a, label=f"갱신 뒤 · {N} 반복")
        pe = 100 * np.array([r["pos_err_model"] for r in R])
        ax[2, c].plot(t + 1, pe, **ls_a, label=f"프레임별 맞춤 모델 · {N} 반복 (평균 {pe.mean():.2f}%)")
        if not ov_drawn:
            oe = 100 * np.array([r["ov_err"] for r in R])
            ax[2, c].plot(t + 1, oe, color=C_OV, lw=2.0, label=f"출력만 최적화 (평균 {oe.mean():.2f}%)")
            if me:
                ax[2, c].plot(np.arange(1, len(me) + 1), 100 * np.array(me), color=C_MM, lw=2.0,
                              label=f"다섯 궤적 학습 모델 (평균 {100 * np.mean(me):.2f}%)")
            ov_drawn = True
        ax[3, c].plot(t + 1, 100 * np.array([r["dist_model_ov"] for r in R]), **ls_a, label=f"{N} 반복")
    ax[0, c].set_title(f"{T}" + (" (다섯 궤적 학습에 안 쓴 궤적)" if T == "0004" else " (다섯 궤적 학습에 쓴 궤적)"), color=INK)
for r_, (yl, lg) in enumerate([("E/E* − 1 (%)  [출력만 최적화 상태 t]", True), ("출력 오차 (%)  [망 한 스텝 vs 출력만 최적화 다음 상태]", False),
                              ("위치 오차 (%)  [정답 대비]", False), ("모델 롤아웃 − 출력만 최적화 상태 거리 (%)", False)]):
    ax[r_, 0].set_ylabel(yl, color=INK)
    for c in range(2):
        a_ = ax[r_, c]
        if lg:
            a_.set_yscale("log")
        a_.grid(True, color="#e6e6e3", lw=0.6)
        for s in ("top", "right"):
            a_.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            a_.spines[s].set_color(MUTED)
        a_.tick_params(colors=INK)
        a_.legend(fontsize=9, frameon=False)
for c in range(2):
    ax[3, c].set_xlabel("프레임", color=INK)
fig.suptitle("프레임별 맞춤: 프레임 t 의 출력만 최적화 상태 하나로 망 갱신 → 모델 자신의 상태에서 한 스텝", color=INK, fontsize=14)
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(OUT, dpi=110)
print("저장", OUT)

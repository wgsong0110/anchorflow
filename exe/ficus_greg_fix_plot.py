"""ficus 로컬 변형 수정판 비교 그림 (로컬, 클러스터 ficus_greg_fix.sh 산출물 읽기).

  정답(PG) · 선형 출력만 최적화 · 이전 Gregory · 수정 Gregory 의
  (1) 프레임별 탄성 에너지 (2) 운동 에너지 (3) 무게중심 x (4) 줄기 끝 x
  (5) 노드 반경 ρ_raw 분포 (이전·수정) (6) 스텝별 ΔE_el 의 로컬/글로벌 몫 (수정)
  + 진동 요약: 극값 사이 반주기, 멈추는 시점 (끝값 ±5% 띠 = 정답 최대 진폭의 5% 안에 머물기 시작하는 프레임)
  python exe/ficus_greg_fix_plot.py --ana .../ficus_v0_greg_fix/ana --out_png ficus_greg_fix.png --out_md ficus_greg_fix.md
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

for _p in (os.path.expanduser("~/.fonts/NotoSansCJKkr-Regular.otf"),):
    try:
        fm.fontManager.addfont(_p)
    except Exception:
        pass
plt.rcParams["font.family"] = "Noto Sans CJK KR"
plt.rcParams["axes.unicode_minus"] = False

ap = argparse.ArgumentParser()
ap.add_argument("--ana", required=True)
ap.add_argument("--out_png", required=True)
ap.add_argument("--out_md", required=True)
ap.add_argument("--dt", type=float, default=0.04)
a = ap.parse_args()
osc = json.load(open(f"{a.ana}/osc.json"))
en = json.load(open(f"{a.ana}/energy/energy.json"))
ns = json.load(open(f"{a.ana}/nodes_stats.json"))
sp = json.load(open(f"{a.ana}/split_el/greg_split_el.json"))
L = [("정답 (PG)", "gt", "#222222", 2.4), ("선형", "선형", "#d9480f", 1.5),
     ("이전 Gregory", "이전 Gregory", "#868e96", 1.5), ("수정 Gregory", "수정 Gregory", "#1c7ed6", 1.8)]


def extrema(y, thr):
    """부호가 바뀌는 기울기 점 중 진폭(직전 극값과의 차)이 thr 넘는 것만."""
    ex = [0]
    for t in range(1, len(y) - 1):
        if (y[t] - y[t - 1]) * (y[t + 1] - y[t]) < 0 and abs(y[t] - y[ex[-1]]) > thr:
            ex.append(t)
    return ex[1:]


def stop_frame(y, band):
    yend = y[-1]
    far = np.nonzero(np.abs(y - yend) > band)[0]
    return int(far[-1] + 1) if len(far) else 0


gt_tip = np.array(osc["gt"]["tip"])[:, 0]
band = 0.05 * float(np.abs(gt_tip).max())
fig, axs = plt.subplots(3, 2, figsize=(15, 12))
summ = []
for lab, key, col, lw in L:
    if key not in osc:
        continue
    tip = np.array(osc[key]["tip"])[:, 0]
    com = np.array(osc[key]["com"])[:, 0]
    tt = np.arange(len(tip))
    axs[1, 0].plot(tt, com * 100, color=col, lw=lw, label=lab)
    axs[1, 1].plot(tt, tip * 100, color=col, lw=lw, label=lab)
    ex = extrema(tip, band)
    half = np.diff(ex) if len(ex) > 1 else np.array([])
    E = en.get(lab, {}).get("E")
    K = en.get(lab, {}).get("K")
    if E is not None:
        axs[0, 0].semilogy(np.arange(1, len(E) + 1), np.maximum(E, 1e-8), color=col, lw=lw, label=lab)
        axs[0, 1].semilogy(np.arange(1, len(K) + 1), np.maximum(K, 1e-8), color=col, lw=lw, label=lab)
    summ.append(dict(lab=lab, ex=ex, half=half.tolist(), stop=stop_frame(tip, band),
                     amp=[float(abs(tip[ex[i + 1]] - tip[ex[i]])) for i in range(len(ex) - 1)],
                     E_mean=(float(np.mean(E)) if E else None), E_last20=(float(np.mean(E[-20:])) if E else None),
                     K_last20=(float(np.mean(K[-20:])) if K else None)))
for ax, tt_ in zip(axs[0], ("탄성 에너지 Σ V ψ(F)", "운동 에너지 ½ Σ m |v|²")):
    ax.set_title(tt_); ax.set_xlabel("프레임"); ax.legend(frameon=False, fontsize=9)
for ax, tt_ in zip(axs[1], ("무게중심 x 변위 (화분 밖 입자)", "줄기 끝 x 변위 (0 프레임 높이 상위 1%)")):
    ax.axhline(0, color="#bbb", lw=0.8); ax.set_title(tt_); ax.set_xlabel("프레임"); ax.set_ylabel("[0.01 시뮬 단위]")
    ax.legend(frameon=False, fontsize=9)
axs[1, 1].axhspan((gt_tip[-1] - band) * 100, (gt_tip[-1] + band) * 100, color="#222", alpha=0.06)
# 반경 분포
for key, col in (("이전 Gregory", "#868e96"), ("수정 Gregory", "#1c7ed6")):
    if key not in ns:
        continue
    R = ns[key]
    t = [r["t"] for r in R]
    axs[2, 0].fill_between(t, [r["rho_raw_p1"] for r in R], [r["rho_raw_p99"] for r in R], color=col, alpha=0.25)
    axs[2, 0].plot(t, [r["rho_raw_med"] for r in R], color=col, lw=1.5, label=f"{key} ρ_raw 중앙 (띠 1~99%)")
axs[2, 0].set_title("노드 반경 출력 ρ_raw (0 = 시작 반경 0.525 h)"); axs[2, 0].set_xlabel("프레임"); axs[2, 0].legend(frameon=False, fontsize=9)
# 로컬/글로벌
t = [r["t"] for r in sp]
loc = np.array([r["b_rho_only"]["dE_el"] for r in sp])
tot = np.array([r["c_both"]["dE_el"] for r in sp])
axs[2, 1].plot(t, tot, color="#222", lw=1.5, label="ΔE_el 실제 스텝 (U + ρ)")
axs[2, 1].plot(t, loc, color="#1c7ed6", lw=1.3, label="로컬만 (U = 0, ρ)")
axs[2, 1].plot(t, [r["a_U_only"]["dE_el"] for r in sp], color="#d9480f", lw=1.0, label="글로벌만 (U, ρ = ρ_ref)")
axs[2, 1].axhline(0, color="#bbb", lw=0.8)
axs[2, 1].set_yscale("symlog", linthresh=1e-2)
axs[2, 1].set_title("수정 Gregory: 스텝별 탄성 에너지 변화 분해"); axs[2, 1].set_xlabel("스텝"); axs[2, 1].legend(frameon=False, fontsize=9)
for ax in axs.flat:
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("ficus 균일 초기 속도 — 정답 · 선형 · 이전 Gregory(로컬 변형 버그) · 수정 Gregory", fontsize=14)
fig.tight_layout()
fig.savefig(a.out_png, dpi=120)
md = ["| 궤적 | 줄기 끝 극값 프레임 | 반주기 (프레임) | 진폭 (×0.01) | 멈추는 프레임 | 탄성 평균 | 탄성 마지막 20 | 운동 마지막 20 |", "|---|---|---|---|---|---|---|---|"]
for s in summ:
    md.append(f"| {s['lab']} | {s['ex'][:12]} | {s['half'][:10]} | {[round(x * 100, 3) for x in s['amp'][:8]]} | {s['stop']} | "
              f"{s['E_mean']:.4e} | {s['E_last20']:.4e} | {s['K_last20']:.4e} |")
md.append(f"\n멈추는 프레임: 줄기 끝 x 가 끝값 ± (정답 최대 진폭의 5% = {band * 100:.3f}×0.01) 띠 밖에 마지막으로 있던 프레임 + 1")
pos = loc[tot < 0]
md.append(f"\n수정 Gregory 로컬/글로벌: 로컬만 ΔE 평균 {loc.mean():+.4e}, 실제 스텝 ΔE 평균 {tot.mean():+.4e}; 탄성 감소 스텝 {int((tot < 0).sum())} 개에서 로컬 ΔE 평균 "
          f"{(pos.mean() if len(pos) else float('nan')):+.4e}")
open(a.out_md, "w").write("\n".join(md) + "\n")
print("\n".join(md))

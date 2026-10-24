"""ficus 셀 안 버블 회전(--local_rot bubble) 비교 그림·표 (로컬, 클러스터 ficus_localrot.sh 산출물 읽기).

  정답(PG) · 선형 출력만 최적화 · 수정 Gregory · 회전판 의
  (1) 탄성 에너지 (2) 운동 에너지 (3) ψ 평균 (빠른 분석) (4) 주 진동 방향 평균 변위 (5) 프레임당 이동 (6) 줄기 끝 x
  (7) 셀별 잠긴 에너지 비 L/E (8) 회전판 노드 회전 벡터 크기 분포 (9) 회전판 det J_loc 최소·셀 밖 입자 수
  + 표: 봉우리 프레임, 멈추는 프레임, ψ·에너지·잠긴 에너지 요약
  python exe/ficus_localrot_plot.py --ana .../ficus_v0_localrot/ana --old_quick .../ficus_v0_greg_fix/osc_quick.json --out_png .. --out_md ..
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
ap.add_argument("--old_quick", default="", help="ficus_v0_greg_fix/osc_quick.json (정답·선형 값 검산)")
ap.add_argument("--out_png", required=True)
ap.add_argument("--out_md", required=True)
a = ap.parse_args()
osc = json.load(open(f"{a.ana}/osc.json"))
oq = json.load(open(f"{a.ana}/osc_quick.json"))
en = json.load(open(f"{a.ana}/energy/energy.json"))
th = json.load(open(f"{a.ana}/theta_stats.json"))
lk = json.load(open(f"{a.ana}/locked/summary.json"))
# 라벨, osc 키, osc_quick 키, 잠긴 에너지 키, 색, 굵기
L = [("정답 (PG)", "gt", "gt", "gt", "#222222", 2.4), ("선형", "선형", "lin", "lin", "#d9480f", 1.5),
     ("수정 Gregory", "수정 Gregory", "gfix", "gfix", "#1c7ed6", 1.5), ("회전판", "회전판", "rot", "rot", "#2b8a3e", 1.9)]
chk = []
if a.old_quick and os.path.exists(a.old_quick):
    o = json.load(open(a.old_quick))
    for k_new, k_old in (("gt", "gt"), ("lin", "lin"), ("gfix", "fix")):
        if k_old in o:
            dif = max(float(np.nanmax(np.abs(np.array(oq[k_new][q]) - np.array(o[k_old][q])))) for q in ("sig", "speed", "psi"))
            chk.append(f"{oq[k_new]['name']}: 10-23 빠른 분석과 최대차 {dif:.2e}")


def extrema(y, thr):
    ex = [0]
    for t in range(1, len(y) - 1):
        if (y[t] - y[t - 1]) * (y[t + 1] - y[t]) < 0 and abs(y[t] - y[ex[-1]]) > thr:
            ex.append(t)
    return ex[1:]


def stop_band(y, band):
    far = np.nonzero(np.abs(y - y[-1]) > band)[0]
    return int(far[-1] + 1) if len(far) else 0


def stop_speed(sp, thr):
    far = np.nonzero(sp > thr)[0]
    return int(far[-1] + 1) if len(far) else 0


gsig = np.array(oq["gt"]["sig"]); band = 0.05 * float(np.abs(gsig).max())
gsp = np.array(oq["gt"]["speed"]); sthr = 0.05 * float(gsp.max())
gtip = np.array(osc["gt"]["tip"])[:, 0]; tband = 0.05 * float(np.abs(gtip).max())
fig, axs = plt.subplots(3, 3, figsize=(20, 15))
summ = []
for lab, ko, kq, kl, col, lw in L:
    q = oq[kq]
    sig, sp, ps = np.array(q["sig"]), np.array(q["speed"]), np.array(q["psi"], dtype=float)
    tt = np.arange(len(sig))
    tip = np.array(osc[ko]["tip"])[:, 0]
    E = en.get(lab, {}).get("E"); K = en.get(lab, {}).get("K")
    axs[0, 0].semilogy(np.arange(1, len(E) + 1), np.maximum(E, 1e-8), color=col, lw=lw, label=lab)
    axs[0, 1].semilogy(np.arange(1, len(K) + 1), np.maximum(K, 1e-8), color=col, lw=lw, label=lab)
    axs[0, 2].semilogy(tt[1:], np.maximum(ps[1:], 1e-8), color=col, lw=lw, label=lab)
    axs[1, 0].plot(tt, sig * 100, color=col, lw=lw, label=lab)
    axs[1, 1].plot(tt, sp * 100, color=col, lw=lw, label=lab)
    axs[1, 2].plot(tt, tip * 100, color=col, lw=lw, label=lab)
    st = lk[kl]["st"]
    le = np.array([s["L"] / max(s["E"], 1e-300) for s in st])
    axs[2, 0].plot(np.arange(len(le))[1:], le[1:], color=col, lw=lw, label=lab)
    ex = extrema(sig, band)
    exT = extrema(tip, tband)
    summ.append(dict(lab=lab, ex=ex, half=np.diff(ex).tolist() if len(ex) > 1 else [], exT=exT,
                     amp=[float(abs(sig[ex[i + 1]] - sig[ex[i]])) for i in range(len(ex) - 1)],
                     stop=stop_band(sig, band), stop_sp=stop_speed(sp, sthr), stopT=stop_band(tip, tband),
                     peak=int(np.argmax(np.abs(sig))), peak_v=float(sig[np.argmax(np.abs(sig))]),
                     psi_mean=float(np.nanmean(ps[1:])), psi_last20=float(np.nanmean(ps[-20:])),
                     sp_mean=float(sp[1:].mean()), sp_last20=float(sp[-20:].mean()),
                     E_mean=float(np.mean(E)), E_last20=float(np.mean(E[-20:])), K_last20=float(np.mean(K[-20:])),
                     LE={f: float(le[f]) for f in (1, 10, 30, 60, 124) if f < len(le)},
                     L={f: float(st[f]["L"]) for f in (1, 10, 30, 60, 124) if f < len(st)}))
for ax, t_ in zip(axs[0], ("탄성 에너지 Σ V ψ(F)", "운동 에너지 ½ Σ m |v|²", "ψ 입자 평균 (빠른 분석 식)")):
    ax.set_title(t_); ax.set_xlabel("프레임"); ax.legend(frameon=False, fontsize=9)
for ax, t_ in zip(axs[1], ("주 진동 방향 평균 변위 (궤적별 주성분 축, 대각 대비) [%]", "프레임당 평균 이동 (대각 대비) [%]",
                           "줄기 끝 x 변위 [0.01 시뮬 단위]")):
    ax.axhline(0, color="#bbb", lw=0.8); ax.set_title(t_); ax.set_xlabel("프레임"); ax.legend(frameon=False, fontsize=9)
axs[2, 0].set_title("셀별 잠긴 에너지 비 L/E (셀마다 아핀 하나로 못 푸는 몫)"); axs[2, 0].set_xlabel("프레임")
axs[2, 0].legend(frameon=False, fontsize=9)
tm = th["theta_max"]
R = th["nodes"]
t = [r["t"] for r in R]
axs[2, 1].fill_between(t, [r["eff_med"] for r in R], [r["eff_p99"] for r in R], color="#2b8a3e", alpha=0.25, label="상한 뒤 |θ̃| 중앙~99%")
axs[2, 1].plot(t, [r["med"] for r in R], color="#2b8a3e", lw=1.2, ls="--", label="출력 |θ| 중앙")
axs[2, 1].plot(t, [r["p99"] for r in R], color="#2b8a3e", lw=1.2, ls=":", label="출력 |θ| 99%")
axs[2, 1].axhline(tm, color="#c92a2a", lw=1.0, label=f"상한 θmax = {tm:.4f} rad")
axs[2, 1].set_yscale("log"); axs[2, 1].set_title("회전판 노드 회전 벡터 크기 [rad]"); axs[2, 1].set_xlabel("프레임")
axs[2, 1].legend(frameon=False, fontsize=9)
inv = th["inv"]
fr = [r["frame"] for r in inv]
axs[2, 2].plot(fr, [r["local_rot"]["detJloc_min"] for r in inv], color="#2b8a3e", lw=1.5, label="det J_loc 최소 (입자)")
axs[2, 2].axhline(0.1 ** 3, color="#c92a2a", lw=0.8, ls="--", label="보장 하한 (1−κ)³ = 0.001")
axs[2, 2].set_title("회전판 셀 안 사상: det J_loc 최소 · 셀 밖 입자 수"); axs[2, 2].set_xlabel("프레임")
ax2 = axs[2, 2].twinx()
ax2.plot(fr, [r["local_rot"]["out_of_cell"] for r in inv], color="#862e9c", lw=1.0, label="셀 밖 입자 수")
ax2.set_ylabel("셀 밖 입자 수"); ax2.set_ylim(-0.5, max(5, max(r["local_rot"]["out_of_cell"] for r in inv) + 1))
h1, l1 = axs[2, 2].get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
axs[2, 2].legend(h1 + h2, l1 + l2, frameon=False, fontsize=9, loc="lower right")
for ax in axs.flat:
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("ficus 균일 초기 속도 · 출력만 최적화 — 정답 · 선형 · 수정 Gregory · 셀 안 버블 회전판", fontsize=15)
fig.tight_layout()
fig.savefig(a.out_png, dpi=110)
md = ["| 궤적 | 주 방향 봉우리 (프레임, 값 %) | 주 방향 극값 프레임 | 반주기 | 진폭 (%) | 멈춤: 주 방향 띠 | 멈춤: 이동 < 5% | 멈춤: 줄기 끝 띠 | "
      "이동 평균 (%) | 이동 마지막 20 | ψ 평균 | ψ 마지막 20 | 탄성 평균 | 운동 마지막 20 |",
      "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for s in summ:
    md.append(f"| {s['lab']} | {s['peak']}, {s['peak_v'] * 100:+.3f} | {s['ex'][:10]} | {s['half'][:8]} | {[round(x * 100, 3) for x in s['amp'][:6]]} | "
              f"{s['stop']} | {s['stop_sp']} | {s['stopT']} | {s['sp_mean'] * 100:.4f} | {s['sp_last20'] * 100:.4f} | {s['psi_mean']:.4e} | "
              f"{s['psi_last20']:.4e} | {s['E_mean']:.4e} | {s['K_last20']:.4e} |")
md.append(f"\n멈춤 정의: 주 방향 띠 = sig 가 끝값 ± (정답 최대 |sig| 의 5% = {band * 100:.4f}%) 밖에 마지막으로 있던 프레임 + 1; "
          f"이동 < 5% = 프레임당 이동이 정답 최대 이동의 5% ({sthr * 100:.4f}%) 를 마지막으로 넘은 프레임 + 1; 줄기 끝 띠 = ficus_greg_fix 표와 같은 정의.")
md.append("\n| 궤적 | L/E 프레임 1 | 10 | 30 | 60 | 124 | L [J] 30 | L [J] 124 |\n|---|---|---|---|---|---|---|---|")
for s in summ:
    le = s["LE"]
    md.append(f"| {s['lab']} | " + " | ".join(f"{le.get(f, float('nan')):.4f}" for f in (1, 10, 30, 60, 124)) +
              f" | {s['L'].get(30, float('nan')):.4e} | {s['L'].get(124, float('nan')):.4e} |")
md.append("\n주의: 회전판은 셀 안 사상이 아핀이 아니라서 \"셀마다 아핀 하나로 못 푸는 에너지\" L 에는 회전판이 이미 셀 안에서 만든 F 퍼짐도 들어간다.")
lr = [r["local_rot"] for r in inv]
md.append(f"\n회전판 셀 안 사상: θmax {tm:.4f} rad (모든 셀 같음, κ 0.9), det J_loc 최소 (전 프레임) {min(q['detJloc_min'] for q in lr):.5f}, "
          f"셀 밖 입자 합 {int(sum(q['out_of_cell'] for q in lr))}, |φ| 최대 {max(q['phi_max'] for q in lr):.3e} rad, "
          f"‖K‖_F 최대 {max(q['K_fro_max'] for q in lr):.3e}; 노드 |θ| 99% (프레임 중앙) {np.median([r['p99'] for r in R]):.3e} rad, "
          f"포화(|θ|>2θmax) 비율 최대 {max(r['frac_sat'] for r in R):.3f}")
if chk:
    md.append("\n검산 (같은 식으로 다시 계산한 정답·선형·수정 Gregory): " + "; ".join(chk))
open(a.out_md, "w").write("\n".join(md) + "\n")
print("\n".join(md))

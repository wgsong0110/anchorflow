"""ficus 텐서곱 B-스플라인 FFD (--interp bspline2 / bspline3) 비교 그림·표 (로컬, 클러스터 ficus_bspline.sh 산출물 읽기).

  정답(PG) · 선형 출력만 최적화 · 수정 Gregory · 회전판 · A(2 차 + det 장벽) · B(3 차 + 변위 차 상한) 의
  에너지·ψ·주 진동 방향 변위·프레임당 이동·줄기 끝·잠긴 에너지 비·det F·위치 오차 + 표 (ficus_localrot_plot.py 와 같은 정의)
  python exe/ficus_bspline_plot.py --ana .../ficus_v0_bspline/ana --old_quick .../ficus_v0_localrot/ana/osc_quick.json --out_png .. --out_md ..
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
ap.add_argument("--old_quick", default="")
ap.add_argument("--out_png", required=True)
ap.add_argument("--out_md", required=True)
a = ap.parse_args()
osc = json.load(open(f"{a.ana}/osc.json"))
oq = json.load(open(f"{a.ana}/osc_quick.json"))
en = json.load(open(f"{a.ana}/energy/energy.json"))
ds = json.load(open(f"{a.ana}/detstats.json"))
lk = json.load(open(f"{a.ana}/locked/summary.json"))
# 표시 라벨, osc/energy/detstats 키, osc_quick 키, 잠긴 에너지 키, 색, 굵기
L = [("정답 (PG)", "gt", "gt", "gt", "#222222", 2.4), ("선형", "선형", "lin", "lin", "#d9480f", 1.4),
     ("수정 Gregory", "수정 Gregory", "gfix", "gfix", "#1c7ed6", 1.2), ("회전판", "회전판", "rot", "rot", "#2b8a3e", 1.2),
     ("A 2차+det 장벽", "A", "bsp2", "bsp2", "#e8590c", 2.0), ("B 3차+차 상한", "B", "bsp3", "bsp3", "#7048e8", 2.0)]
chk = []
if a.old_quick and os.path.exists(a.old_quick):
    o = json.load(open(a.old_quick))
    for k in ("gt", "lin", "gfix", "rot"):
        if k in o:
            dif = max(float(np.nanmax(np.abs(np.array(oq[k][q]) - np.array(o[k][q])))) for q in ("sig", "speed", "psi"))
            chk.append(f"{oq[k]['name']}: 회전판 분석(10-24)과 최대차 {dif:.2e}")


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
fig, axs = plt.subplots(3, 4, figsize=(26, 15))
summ = []
for lab, ko, kq, kl, col, lw in L:
    q = oq[kq]
    sig, sp, ps = np.array(q["sig"]), np.array(q["speed"]), np.array(q["psi"], dtype=float)
    tt = np.arange(len(sig))
    tip = np.array(osc[ko]["tip"])[:, 0]
    ek = "정답 (PG)" if ko == "gt" else ko
    E = en[ek]["E"]; K = en[ek]["K"]
    axs[0, 0].semilogy(np.arange(1, len(E) + 1), np.maximum(E, 1e-8), color=col, lw=lw, label=lab)
    axs[0, 1].semilogy(np.arange(1, len(K) + 1), np.maximum(K, 1e-8), color=col, lw=lw, label=lab)
    axs[0, 2].semilogy(tt[1:], np.maximum(ps[1:], 1e-8), color=col, lw=lw, label=lab)
    axs[1, 0].plot(tt, sig * 100, color=col, lw=lw, label=lab)
    axs[1, 1].plot(tt, sp * 100, color=col, lw=lw, label=lab)
    axs[1, 2].plot(tt, tip * 100, color=col, lw=lw, label=lab)
    st = lk[kl]["st"]
    le = np.array([s["L"] / max(s["E"], 1e-300) for s in st])
    axs[2, 0].plot(np.arange(len(le))[1:], le[1:], color=col, lw=lw, label=lab)
    d = ds.get(ko)
    if d is not None:
        fr = np.arange(1, len(d["err_h"]) + 1)
        axs[0, 3].plot(fr, d["err_h"], color=col, lw=lw, label=lab)
        axs[1, 3].plot(fr, d["detF_min"], color=col, lw=lw, label=lab)
        axs[2, 1].plot(np.arange(1, len(d["step_det_min"]) + 1), d["step_det_min"], color=col, lw=lw, label=f"{lab} 입자")
        sm = [np.nan if x is None else x for x in d["smp_det_min"]]
        if not all(np.isnan(sm)):
            axs[2, 1].plot(np.arange(1, len(sm) + 1), sm, color=col, lw=lw, ls="--", label=f"{lab} 조각 표본")
    ex = extrema(sig, band)
    exT = extrema(tip, tband)
    summ.append(dict(lab=lab, ko=ko, ex=ex, half=np.diff(ex).tolist() if len(ex) > 1 else [], exT=exT,
                     amp=[float(abs(sig[ex[i + 1]] - sig[ex[i]])) for i in range(len(ex) - 1)],
                     stop=stop_band(sig, band), stop_sp=stop_speed(sp, sthr), stopT=stop_band(tip, tband),
                     peak=int(np.argmax(np.abs(sig))), peak_v=float(sig[np.argmax(np.abs(sig))]),
                     psi_mean=float(np.nanmean(ps[1:])), psi_last20=float(np.nanmean(ps[-20:])),
                     sp_mean=float(sp[1:].mean()), sp_last20=float(sp[-20:].mean()),
                     E_mean=float(np.mean(E)), K_last20=float(np.mean(K[-20:])),
                     LE={f: float(le[f]) for f in (1, 10, 30, 60, 124) if f < len(le)},
                     L={f: float(st[f]["L"]) for f in (1, 10, 30, 60, 124) if f < len(st)}))
for ax, t_ in zip(axs[0], ("탄성 에너지 Σ V ψ(F)", "운동 에너지 ½ Σ m |v|²", "ψ 입자 평균", "평균 위치 오차 |x − x_PG| [node_h]")):
    ax.set_title(t_); ax.set_xlabel("프레임"); ax.legend(frameon=False, fontsize=8)
for ax, t_ in zip(axs[1], ("주 진동 방향 평균 변위 (궤적별 주성분 축, 대각 대비) [%]", "프레임당 평균 이동 (대각 대비) [%]",
                           "줄기 끝 x 변위 [0.01 시뮬 단위]", "누적 det F 최소 (입자)")):
    ax.axhline(0, color="#bbb", lw=0.8); ax.set_title(t_); ax.set_xlabel("프레임"); ax.legend(frameon=False, fontsize=8)
axs[2, 0].set_title("셀(Kuhn 사면체)별 잠긴 에너지 비 L/E"); axs[2, 0].set_xlabel("프레임"); axs[2, 0].legend(frameon=False, fontsize=8)
axs[2, 1].axhline(0.3, color="#c92a2a", lw=0.8, ls=":", label="장벽 Ĵ = 0.3")
axs[2, 1].set_title("이번 스텝 det F_step 최소 (실선 입자, 점선 조각 표본점)"); axs[2, 1].set_xlabel("프레임")
axs[2, 1].legend(frameon=False, fontsize=7, ncol=2)
cap = [c for c in ds["B"]["cap"]]
fr = np.arange(1, len(cap) + 1)
axs[2, 2].plot(fr, [c["dV_max"] for c in cap], color="#7048e8", lw=1.2, ls="--", label="사상 전 이웃 차 최대 |ΔV|/Δ")
axs[2, 2].plot(fr, [c["dU_max"] for c in cap], color="#7048e8", lw=1.8, label="사상 뒤 이웃 차 최대 |ΔU|/Δ")
axs[2, 2].axhline(0.48, color="#c92a2a", lw=0.8, label="상한 0.48Δ")
axs[2, 2].set_title("B: 축 이웃 노드 변위 차 (마지막 평가)"); axs[2, 2].set_xlabel("프레임")
ax2 = axs[2, 2].twinx(); ax2.plot(fr, [c["s"] for c in cap], color="#495057", lw=1.0, label="축소 계수 s"); ax2.set_ylabel("s")
h1, l1 = axs[2, 2].get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
axs[2, 2].legend(h1 + h2, l1 + l2, frameon=False, fontsize=8)
for lab, ko, kq, kl, col, lw in L:
    d = ds.get(ko)
    if d is not None:
        axs[2, 3].plot(np.arange(1, len(d["detF_neg"]) + 1), d["detF_neg"], color=col, lw=lw, label=lab)
axs[2, 3].set_title("누적 det F ≤ 0 입자 수"); axs[2, 3].set_xlabel("프레임"); axs[2, 3].legend(frameon=False, fontsize=8)
for ax in axs.flat:
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("ficus 균일 초기 속도 · 출력만 최적화 — 정답 · 선형 · 수정 Gregory · 회전판 · B-스플라인 FFD (A 2차+det 장벽, B 3차+변위 차 상한)", fontsize=15)
fig.tight_layout()
fig.savefig(a.out_png, dpi=100)
md = ["| 궤적 | 주 방향 봉우리 (프레임, 값 %) | 주 방향 극값 프레임 | 반주기 | 진폭 (%) | 멈춤: 주 방향 띠 | 멈춤: 이동 < 5% | 멈춤: 줄기 끝 띠 | "
      "이동 평균 (%) | 이동 마지막 20 | ψ 평균 | ψ 마지막 20 | 탄성 평균 | 운동 마지막 20 |",
      "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for s in summ:
    md.append(f"| {s['lab']} | {s['peak']}, {s['peak_v'] * 100:+.3f} | {s['ex'][:10]} | {s['half'][:8]} | {[round(x * 100, 3) for x in s['amp'][:6]]} | "
              f"{s['stop']} | {s['stop_sp']} | {s['stopT']} | {s['sp_mean'] * 100:.4f} | {s['sp_last20'] * 100:.4f} | {s['psi_mean']:.4e} | "
              f"{s['psi_last20']:.4e} | {s['E_mean']:.4e} | {s['K_last20']:.4e} |")
md.append(f"\n멈춤 정의: 주 방향 띠 = sig 가 끝값 ± (정답 최대 |sig| 의 5% = {band * 100:.4f}%) 밖에 마지막으로 있던 프레임 + 1; "
          f"이동 < 5% = 프레임당 이동이 정답 최대 이동의 5% ({sthr * 100:.4f}%) 를 마지막으로 넘은 프레임 + 1; 줄기 끝 띠 = 같은 정의 (줄기 끝 x).")
md.append("\n| 궤적 | L/E 프레임 1 | 10 | 30 | 60 | 124 | L [J] 30 | L [J] 124 |\n|---|---|---|---|---|---|---|---|")
for s in summ:
    le = s["LE"]
    md.append(f"| {s['lab']} | " + " | ".join(f"{le.get(f, float('nan')):.4f}" for f in (1, 10, 30, 60, 124)) +
              f" | {s['L'].get(30, float('nan')):.4e} | {s['L'].get(124, float('nan')):.4e} |")
md.append("\n잠긴 에너지 셀 = 기존 Kuhn 사면체 (모든 궤적 같은 셀 정의). B-스플라인 판은 사면체 안에서도 F 가 변하므로 L 에 그 셀 안 F 퍼짐도 들어간다.")
md.append("\n| 궤적 | 평균 위치 오차 (err_ov) | 평균 |x−x_PG| [node_h] | 누적 det F 최소 | 누적 뒤집힘 (입자-프레임) | 스텝 det 최소 (입자) | 스텝 뒤집힘 합 | 조각 표본 det 최소 | 표본 det ≤ 0 합 |\n|---|---|---|---|---|---|---|---|---|")
for s in summ:
    d = ds.get(s["ko"])
    if d is None:
        continue
    sm = [x for x in d["smp_det_min"] if x is not None]
    sn = [x for x in d["smp_det_neg"] if x is not None]
    md.append(f"| {s['lab']} | {d['err_ov']} | {np.mean(d['err_h']):.4f} | {min(d['detF_min']):.4f} | {sum(d['detF_neg'])} | "
              f"{min(d['step_det_min']):.4f} | {sum(d['step_inv'])} | {(f'{min(sm):.4f}' if sm else '—')} | {(sum(sn) if sn else '—')} |")
cB = ds["B"]["cap"]
md.append(f"\nB 상한 사상 (마지막 평가, 프레임별): 사상 전 이웃 차 최대 {max(c['dV_max'] for c in cB):.3f}Δ (중앙 {np.median([c['dV_max'] for c in cB]):.3f}Δ), "
          f"사상 뒤 최대 {max(c['dU_max'] for c in cB):.4f}Δ (< 0.48Δ), 축소 계수 s 최소 {min(c['s'] for c in cB):.4f} (중앙 {np.median([c['s'] for c in cB]):.5f})")
if chk:
    md.append("\n검산 (같은 식으로 다시 계산한 기존 궤적): " + "; ".join(chk))
open(a.out_md, "w").write("\n".join(md) + "\n")
print("\n".join(md))

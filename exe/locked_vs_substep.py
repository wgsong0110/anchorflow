"""ficus 서브스텝 수에 따른 셀별 잠긴 에너지 -- cell_locked_energy.py 결과(summary.json)를 모아 그림·표.

  python exe/locked_vs_substep.py --run "gt:정답 (PG):#222222:0:DIR/summary.json:gt" \
     --run "n1:N1:#d9480f:1:DIR/summary.json:lin" --run "n8:N8:...:8:DIR8/summary.json:n8" ... \
     --fig fig.png --table table.md

--run 키:이름:색:서브스텝(0=정답):summary.json:그 안의 키. 프레임은 모든 실행의 공통 구간 1..T-1.
"""
import argparse
import json
import os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--run", action="append", required=True)
ap.add_argument("--fig", required=True)
ap.add_argument("--table", required=True)
ap.add_argument("--table_frames", type=int, nargs="+", default=[1, 10, 30, 60])
a = ap.parse_args()

R = []
for s in a.run:
    k, lab, col, n, p, sk = s.split(":", 5)
    st = json.load(open(p))[sk]["st"]
    R.append(dict(key=k, lab=lab, col=col, n=int(n), st=st, src=p))
NF = {r["key"]: len(r["st"]) for r in R}
T = min(NF.values())
last = T - 1
print(f"[프레임 수] {NF} -> 공통 0..{last}")
tt = np.arange(1, T)
for r in R:
    r["E"] = np.array([r["st"][f]["E"] for f in tt]); r["L"] = np.array([r["st"][f]["L"] for f in tt])

frames = [f for f in a.table_frames if f < T] + [last]
with open(a.table, "w") as fo:
    fo.write("# ficus_v0 서브스텝 수에 따른 셀별 잠긴 탄성 에너지 (선형 출력만 최적화, 균일 초기속도)\n\n")
    fo.write("E = Σ V ψ(F), L = Σ_c min_J Σ_{p∈c} V ψ(J F) (셀 = Kuhn 사면체 node_h 0.056569, 각자 현재 위치, 점유 셀 전부), "
             "ψ = jelly FCR (E 2e6, ν 0.4). 정의·셀 분할·ψ 는 lbours/ficus_locked 와 같다 (exe/cell_locked_energy.py).\n\n")
    nfs = ", ".join("%s %d" % (r["lab"], NF[r["key"]]) for r in R)
    fo.write(f"덤프 프레임 수 (0 프레임 포함): {nfs} -> 공통 프레임 1..{last}\n\n")
    fo.write("| 프레임 | 궤적 | E [J] | L [J] | L/E | 셀 수 (1 입자) |\n|---|---|---|---|---|---|\n")
    for f in frames:
        for r in R:
            s = r["st"][f]
            fo.write(f"| {f} | {r['lab']} | {s['E']:.4e} | {s['L']:.4e} | {s['L'] / max(s['E'], 1e-300):.4f} | {s['C']} ({s['C1']}) |\n")
    fo.write("\n수렴 (프레임 1..마지막 공통): 궤적 | 반복 최대 | ‖g‖/척도 최대 | 미수렴 셀 합 | 두 시작 |ΔL|/E_c 최대\n\n")
    for r in R:
        S = r["st"][1:T]
        fo.write(f"- {r['lab']}: {max(s['it_max'] for s in S)} | {max(s['gn_rel_max'] for s in S):.1e} | "
                 f"{sum(s['nonconv'] for s in S)} | {max(s['start_diff_rel_max'] for s in S):.1e}\n")
print(open(a.table).read())

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
_FONTS = [os.path.expanduser("~/.fonts/NotoSansCJKkr-Regular.otf")] + [p for p in fm.findSystemFonts() if "NotoSansCJK" in os.path.basename(p)]
for _p in _FONTS:
    if os.path.exists(_p):
        fm.fontManager.addfont(_p); plt.rcParams["font.family"] = fm.FontProperties(fname=_p).get_name(); break
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(1, 3, figsize=(22, 6.8), dpi=110, gridspec_kw=dict(width_ratios=[1.25, 1.25, 0.9]))
for r in R:
    ax[0].plot(tt, r["E"], color=r["col"], lw=1.8, label=f"{r['lab']} E")
    ax[0].plot(tt, r["L"], color=r["col"], lw=1.5, ls="--", label=f"{r['lab']} L")
    ax[1].plot(tt, r["L"] / r["E"], color=r["col"], lw=1.8, label=r["lab"])
ax[0].set_yscale("log"); ax[0].set_xlabel("프레임"); ax[0].set_ylabel("에너지 [J] (세로축 로그)")
ax[0].set_title("1. 탄성 에너지 E (실선) · 잠긴 에너지 L (점선)", fontsize=13); ax[0].grid(alpha=0.3, which="both")
ax[0].legend(fontsize=8.5, ncol=2)
ax[1].set_xlabel("프레임"); ax[1].set_ylabel("L / E"); ax[1].set_ylim(0, 1); ax[1].grid(alpha=0.3)
ax[1].set_title("2. 잠긴 비율 L/E (셀 아핀 하나로 못 푸는 몫)", fontsize=13); ax[1].legend(fontsize=10)
sub = [r for r in R if r["n"] > 0]
ns = np.array([r["n"] for r in sub])
Ll = np.array([r["st"][last]["L"] for r in sub]); El = np.array([r["st"][last]["E"] for r in sub])
ax[2].plot(ns, El, "o-", color="#555555", lw=1.6, label="출력만 최적화 E")
ax[2].plot(ns, Ll, "s--", color="#d9480f", lw=1.8, ms=8, label="출력만 최적화 L")
for r in R:
    if r["n"] == 0:
        ax[2].axhline(r["st"][last]["E"], color=r["col"], lw=1.4, label=f"{r['lab']} E")
        ax[2].axhline(r["st"][last]["L"], color=r["col"], lw=1.4, ls="--", label=f"{r['lab']} L")
for n_, l_, e_ in zip(ns, Ll, El):
    ax[2].annotate(f"L {l_:.2e}\nL/E {l_ / e_:.3f}", (n_, l_), textcoords="offset points", xytext=(6, -26), fontsize=8.5)
ax[2].set_xscale("log", base=2); ax[2].set_yscale("log"); ax[2].set_xticks(ns); ax[2].set_xticklabels([str(n) for n in ns])
ax[2].set_xlabel("프레임당 서브스텝 수 N (가로축 로그)"); ax[2].set_ylabel("에너지 [J] (세로축 로그)")
ax[2].set_title(f"3. 마지막 공통 프레임 {last} 의 E · L 대 서브스텝 수", fontsize=13); ax[2].grid(alpha=0.3, which="both")
ax[2].legend(fontsize=9)
fig.suptitle(f"ficus_v0 선형 출력만 최적화 — 서브스텝 수에 따른 셀별 잠긴 탄성 에너지 (셀 = Kuhn 사면체 node_h 0.056569, 공통 프레임 1..{last})",
             fontsize=14)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(a.fig); plt.close(fig)
print(f"[그림] {a.fig}")

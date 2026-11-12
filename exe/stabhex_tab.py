"""서브스텝 안정성 '우리(육면체)' 표·그림 (11-12 지시, 묶음 stabhex): exe/stabhex_an.py 의 JSON (+ 단독 시간 summary) -> md·tex·png.

  python exe/stabhex_tab.py --an stabhex_an.json [--speed speed 폴더] --out 폴더
출력: stabhex_table.md, stabhex_tables.tex (N_min 열 추가판·물리 정합성 육면체 행·오차 표·시간 표), stabhex_err.png (N-오차 곡선)
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
from matplotlib import font_manager                               # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--an", required=True)
ap.add_argument("--speed", default="")
ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
font_manager.fontManager.addfont("/home/wgsong/.fonts/NotoSansCJKkr-Regular.otf")
plt.rcParams["font.family"] = "Noto Sans CJK KR"
plt.rcParams["axes.unicode_minus"] = False
A = json.load(open(a.an))
SH = ["lego", "mic", "ficus"]
SHN = {"lego": "레고", "mic": "mic", "ficus": "ficus"}
METH = [("pg", "PG", "#2a78d6", "o"), ("gf", "GF", "#eb6834", "s"), ("ipg", "i-PG", "#1baf7a", "^"),
        ("ovs", "우리(사면체)", "#eda100", "D"), ("hex", "우리(육면체)", "#e87ba4", "P")]
NMIN_OLD = {"lego": dict(pg="43 (43)", gf="16 (16)", ipg="8 (8)", ovs="1"),          # 기존 표 (stab_results.tex) 값 그대로
            "mic": dict(pg="45 (44)", gf="16 (16)", ipg="12 (8)", ovs="1"),
            "ficus": dict(pg="41 (40)", gf="16 (16)", ipg="10 (8)", ovs="1")}
md, tex = [], []


def stop_s(t):
    t = t or ""
    if "Tried to allocate" in t:
        return "용량 넘침 다시 할당 메모리 부족 (발산)"
    if "CUDA error: out of memory" in t:
        return "그래프 캡처 메모리 부족"
    return "" if t == "끝까지" else t.splitlines()[0][:80] if t else ""


def pc(v, d=2):
    return "--" if v is None else f"{100 * v:.{d}f}"


# ---- 1. 육면체 판별 결과
md.append("# 서브스텝 안정성: 우리(육면체) (stabhex)\n")
cap = None
for S in SH:
    H = A["shapes"][S]["methods"]["hex"]
    md.append(f"## {SHN[S]} (입자 {A['shapes'][S]['n_particles']}, L {A['shapes'][S]['L']:.3f} m)\n")
    md.append("| 판 | N | 상태 | 표 판정 안정 | stabstat 첫 불안정 프레임 | 에너지 변화 최대 | 관통/L % | det≤0 최대 % | 뉴턴/프레임 | PCG/프레임 | 뉴턴 상한 걸린 프레임 | PCG 상한 걸린 프레임 | 기준 대비 RMSE/L % |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for nm, r in sorted(H["runs"].items(), key=lambda kv: (kv[1]["N"], kv[0])):
        mt, it = r.get("metrics") or {}, r.get("it") or {}
        cap = cap or (r.get("newton"), r.get("cg"), r.get("tolN"), r.get("tolP"))
        e = H["err"].get(str(r["N"])) if r.get("rep") == "a" else None
        md.append(f"| {nm} | {r['N']} | {r['status']} ({r.get('frames')} 프레임{', ' + stop_s(r.get('stop')) if stop_s(r.get('stop')) else ''}) | {'예' if r['stable'] else '아니오'} | "
                  f"{r['unstable_frame']} | {mt.get('e_drift_max', float('nan')):.3g} | {pc(mt.get('pen_depth_max'))} | {pc(mt.get('detneg_max'))} | "
                  f"{it.get('nN_mean', float('nan')):.2f} | {it.get('nP_mean', float('nan')):.1f} | {it.get('capN_frames', '--')} | {it.get('capP_frames', '--')} | {pc(e['mean'], 3) if e else ''} |")
    md.append(f"\n기준 궤적 N{H['ref']} ({'수렴' if H['ref_converged'] else '미수렴 -- 가장 큰 N'}), 두 배 짝 차이 (RMSE/L %): "
              + ", ".join(f"{p[0]}-{p[1]} {100 * p[2]:.3f}" for p in H["ref_pairs"]))
    md.append("재현성 바닥 (같은 설정 두 판, 평균 RMSE/L %): " + ", ".join(f"N{n} {100 * f['mean']:.4f}" for n, f in sorted(H["floor"].items(), key=lambda kv: int(kv[0]))))
    md.append(f"N_min = {H['N_min']} (그 N 판 안정 {H['N_min_runs']}), N 별 안정 판/판 {H['stable_by_N']}, N_min 위 불안정 {H['unstable_above_Nmin']}\n")
md.insert(1, f"반복 상한: 뉴턴 {cap[0]} x PCG {cap[1]} (허용치 조기 종료 tolN {cap[2]} · tolP {cap[3]}), 시간 예산 끔, 적응 쪼개기 (--ksplit) 끔 -- 모든 판 같음.\n" if cap else "")

# ---- 2. 방법별 기준·오차 (모든 방법 같은 규칙)
md.append("## 방법별 자기 기준 궤적 대비 RMSE/L (%; 60 프레임 평균, 부분표본 32768 같은 번호)\n")
NS = [2 ** i for i in range(12)]
for S in SH:
    md.append(f"### {SHN[S]}\n")
    md.append("| 방법 | 기준 N | " + " | ".join(f"N{n}" for n in NS) + " |")
    md.append("|---|---|" + "---|" * len(NS))
    for m, name, _, _ in METH:
        M = A["shapes"][S]["methods"].get(m)
        if not M:
            continue
        cells = []
        for n in NS:
            e = M["err"].get(str(n))
            runs = [r for r in M["runs"].values() if r["N"] == n]
            if e and str(n) == str(M["ref"]):
                cells.append("기준")
            elif e:
                cells.append(pc(e["mean"], 3) if all(r["stable"] for r in runs[:1]) else "불안정†")
            elif runs:
                cells.append("발산" if runs[0]["status"] == "발산" else runs[0]["status"])
            else:
                cells.append("")
        md.append(f"| {name} | {M['ref']}{'' if M['ref_converged'] else '*'} | " + " | ".join(cells) + " |")
    md.append("\n† 60 프레임은 끝났으나 표 판정 (에너지 변화 ≤ 1·관통 ≤ 0.5 L) 불안정. * 두 배 짝이 0.5 % 안으로 안 들어와 (미수렴) 가장 작은 끝난 N 에서 두 배씩 이어지는 끝난 N 의 끝을 기준으로 씀.\n")

# ---- 3. 그림: N-오차 곡선
fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.2))
for ax, S in zip(axs, SH):
    for m, name, col, mk in METH:
        M = A["shapes"][S]["methods"].get(m)
        if not M:
            continue
        st0 = {}
        for r in M["runs"].values():
            st0.setdefault(r["N"], r["stable"])
        pts = sorted((int(n), e["mean"]) for n, e in M["err"].items() if int(n) != int(M["ref"]) and st0.get(int(n), True))
        if not pts:
            continue
        st = {r["N"]: r["stable"] for r in M["runs"].values() if r.get("rep", "a") in ("a",) or m != "hex"}
        ax.plot([p[0] for p in pts], [100 * p[1] for p in pts], color=col, lw=2, label=name if S == SH[0] else None, zorder=2)
        for n, e in pts:
            ok = st.get(n, True)
            ax.plot(n, 100 * e, marker=mk, ms=8, color=col, mfc=col if ok else "white", mec=col, mew=1.6, ls="", zorder=3)
    Hf = A["shapes"][S]["methods"]["hex"]["floor"]
    if Hf:
        fl = max(f["mean"] for f in Hf.values())
        ax.axhline(100 * fl, color="#e87ba4", lw=1, ls="--", zorder=1)
        ax.text(1.05, 100 * fl * 1.08, "우리(육면체) 재현성 바닥 (같은 설정 두 판, N 중 최대)", fontsize=8, color="#52514e", va="bottom")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_title(SHN[S] + " (기준 N: " + " · ".join(f"{nm} {A['shapes'][S]['methods'][m]['ref']}" for m, nm, _, _ in METH if m in A["shapes"][S]["methods"]) + ")", fontsize=8.5); ax.set_xlabel("프레임당 서브스텝 N")
    ax.grid(True, which="major", color="#e5e4e0", lw=0.6); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
axs[0].set_ylabel("자기 기준 궤적 대비 RMSE / L (%)")
fig.legend(loc="lower center", ncol=len(METH), frameon=False, fontsize=9)
fig.suptitle("N-오차 곡선 (방법마다 자기 기준 궤적 대비; 60 프레임을 끝내고 에너지·관통 판정도 안정한 판만, 발산 N 은 점 없음)", fontsize=10)
fig.tight_layout(rect=(0, 0.07, 1, 0.95))
fig.savefig(os.path.join(a.out, "stabhex_err.png"), dpi=170)

# ---- 4. tex: N_min 표 (기존 열 그대로 + 육면체 열)
tex += [r"\begin{table}[h]\centering\small",
        r"\caption{발산하지 않는 최소 서브스텝 수 $N$ -- 반복 확정값 (같은 판정: NaN·오류 없음, 운동+위치 에너지 변화 최대 $\le 1$, 바닥 관통 $\le 0.5L$). "
        r"괄호: 정수 스캔 1 회 때의 값. 우리(사면체) = 기존 열 (2026-10-14, 기본 실행 조합 ovs) 그대로, 우리(육면체) = 이차 B-스플라인 암시적 H9 (2026-11-12, "
        + (f"뉴턴 {cap[0]}$\\times$PCG {cap[1]} 상한, 시간 예산·적응 쪼개기 끔" if cap else "") + r"; 괄호 = 그 $N$ 에서 안정한 판/판).}",
        r"\begin{tabular}{lccccc}", r"\toprule", r"형상 & PG & GF & i-PG & 우리(사면체) & 우리(육면체) \\", r"\midrule"]
for S in SH:
    H = A["shapes"][S]["methods"]["hex"]
    o = NMIN_OLD[S]
    hv = "--" if H["N_min"] is None else f"{H['N_min']} ({sum(H['N_min_runs'])}/{len(H['N_min_runs'])})"
    tex.append(f"{S} & {o['pg']} & {o['gf']} & {o['ipg']} & {o['ovs']} & {hv} \\\\")
tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
# 물리 정합성: 육면체 행 (N_min 판 평균)
KEYS = ("detneg", "pen", "jit", "ff", "cd")
tex += ["% 물리 정합성 표에 넣을 우리(육면체) 행 (형상별, N_min 판 평균; 통과 = 모든 판 문턱 안)"]
for S in SH:
    H = A["shapes"][S]["methods"]["hex"]
    rs = [r for r in H["runs"].values() if r["N"] == H["N_min"] and r.get("phys")]
    if not rs:
        tex.append(f"% {S}: 육면체 N_min 판 없음"); continue
    cells = []
    for k in KEYS:
        vs = [r["phys"][k][0] for r in rs if r["phys"][k][0] is not None]
        oks = [r["phys"][k][1] for r in rs if r["phys"][k][1] is not None]
        v = (sum(vs) / len(vs)) if vs else None
        cells.append(pc(v) + ("" if not oks else (r"\,\checkmark" if all(oks) else r"\,$\times$")))
    tex.append(f"%{S}: & 우리(육면체) & {H['N_min']} & {len(rs)} & " + " & ".join(cells) + r" \\")
# 오차 표 (tex)
tex += ["", r"\begin{table}[h]\centering\scriptsize",
        r"\caption{방법마다 자기 기준 궤적 (N 을 두 배씩 올려 $N$ 과 $2N$ 의 차가 0.5\,\% 미만이 되는 $2N$; * = 미수렴이라 끝난 $N$ 의 두 배 사슬 끝) 대비 부분표본 RMSE$/L$ (\%, 60 프레임 평균, "
        r"같은 번호 32768 점). 빈 칸 = 결과 없음, 발산 = 60 프레임을 못 끝냄, $\dagger$ = 끝났으나 에너지·관통 판정 불안정. 기존 방법 값은 기존 실행 결과를 같은 식으로 다시 잰 것.}",
        r"\begin{tabular}{ll" + "c" * len(NS) + "}", r"\toprule", r"형상 & 방법 (기준 $N$) & " + " & ".join(str(n) for n in NS) + r" \\", r"\midrule"]
for S in SH:
    first = True
    for m, name, _, _ in METH:
        M = A["shapes"][S]["methods"].get(m)
        if not M:
            continue
        cells = []
        for n in NS:
            e = M["err"].get(str(n)); runs = [r for r in M["runs"].values() if r["N"] == n]
            if e and str(n) == str(M["ref"]):
                cells.append("기준")
            elif e:
                cells.append(f"{100 * e['mean']:.2f}" if all(r["stable"] for r in runs[:1]) else r"불안정$^\dagger$")
            elif runs:
                cells.append("발산" if runs[0]["status"] == "발산" else "")
            else:
                cells.append("")
        tex.append(f"{S if first else ''} & {name} ({M['ref']}{'' if M['ref_converged'] else '*'}) & " + " & ".join(cells) + r" \\")
        first = False
    tex.append(r"\midrule")
tex[-1] = r"\bottomrule"
tex += [r"\end{tabular}", r"\end{table}", ""]

# ---- 5. 단독 시간 (있으면)
if a.speed and os.path.isdir(a.speed):
    T = {}
    for f in glob.glob(f"{a.speed}/t_*_N*/summary.json"):
        n = os.path.basename(os.path.dirname(f)); _, s, k = n.split("_"); s_ = json.load(open(f))
        ap_ = open(os.path.join(os.path.dirname(f), "apps_at_start.txt")).read().strip() if os.path.exists(os.path.join(os.path.dirname(f), "apps_at_start.txt")) else "?"
        T.setdefault(s, {})[int(k[1:])] = (s_.get("ms_mean"), s_.get("ms_p90"), s_.get("diverged"), ap_ == "")
    if T:
        KS = sorted({k for v in T.values() for k in v})
        md.append("## 우리(육면체) 단독 시간 (ms/프레임, 재생 GPU 시간, 렌더 없음; 카드 하나에 다른 계산 프로세스 0 일 때 차례로)\n")
        md.append("| 형상 | " + " | ".join(f"N{k}" for k in KS) + " |"); md.append("|---|" + "---|" * len(KS))
        tex += [r"\begin{table}[h]\centering\small", r"\caption{우리(육면체) 프레임당 시간 (ms, A6000 카드 단독, 시뮬 재생 GPU 시간 평균 (90 백분위), 렌더 없음, 시간 예산 끔).}",
                r"\begin{tabular}{l" + "c" * len(KS) + "}", r"\toprule", "형상 & " + " & ".join(f"$N{{=}}{k}$" for k in KS) + r" \\", r"\midrule"]
        for s in SH:
            if s not in T:
                continue
            row = [T[s].get(k) for k in KS]
            md.append(f"| {SHN[s]} | " + " | ".join("" if r is None else f"{r[0]:.1f} ({r[1]:.1f}){' 발산' if r[2] else ''}{'' if r[3] else ' 공유?'}" for r in row) + " |")
            tex.append(f"{s} & " + " & ".join("" if r is None else f"{r[0]:.1f} ({r[1]:.1f})" for r in row) + r" \\")
        tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
open(os.path.join(a.out, "stabhex_table.md"), "w").write("\n".join(md) + "\n")
open(os.path.join(a.out, "stabhex_tables.tex"), "w").write("\n".join(tex) + "\n")
print("[stabhex 표]", a.out)

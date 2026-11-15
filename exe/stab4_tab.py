"""[stab4] 서브스텝 안정성 4 장면 (레고·vasedeck·plane·pillow2sofa) 표·그림 -> 논문 폴더 stab_results_1115.tex + fig/stab4_err.png.
입력: <수집 폴더>/<장면>/{an.json (exe/stab1115_an.py --hextag _s4), vis/vis.json, speed/t_N<n>_s4/log.txt, hex/<장면>_N<n>_a_s4/log.txt}
레고의 PG·i-PG·GF 는 11-12 기존 결과 재사용 (N_min·물리 = writing/anchorflow/stab_results.tex, 오차 = R2 lbours/stabhex/stabhex_an.json); 시각 지표는 이번에 다시 렌더.
  python exe/stab4_tab.py <수집 폴더> <stabhex_an.json> <논문 폴더>
"""
import glob, json, os, re, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
for f in glob.glob(os.path.expanduser("~/.fonts/NotoSansCJKkr-*.otf")):
    font_manager.fontManager.addfont(f)
plt.rcParams["font.family"] = "Noto Sans CJK KR"; plt.rcParams["axes.unicode_minus"] = False
root, hexan, paper = sys.argv[1], sys.argv[2], sys.argv[3]
SC = ["lego", "vasedeck", "plane", "pillow2sofa"]
SN = dict(lego="레고", vasedeck="vasedeck", plane="plane", pillow2sofa="pillow2sofa")
MS = [("pg", "PG"), ("ipg", "i-PG"), ("gf", "GF"), ("omni", "OmniPhysGS"), ("ours", "우리")]
TAG = "_s4"
AN = {s: json.load(open(f"{root}/{s}/an.json")) for s in SC if os.path.exists(f"{root}/{s}/an.json")}
VIS = {s: (json.load(open(f"{root}/{s}/vis/vis.json")) if os.path.exists(f"{root}/{s}/vis/vis.json") else {}) for s in SC}
OLD = json.load(open(hexan))["shapes"]["lego"]["methods"]
# 레고 기존 (11-12 표 stab_results.tex 그대로): N_min, 물리 (뒤집힘, 관통, 떨림, 자유낙하, Chamfer, 통과 표시)
LEGO_OLD = dict(pg=(43, ["0.00", "0.61", "0.44", "0.00", "0.73"], [1, 1, 1, 1, 1]),
                gf=(16, ["0.00", "0.79", "0.27", "0.00", "2.38"], [1, 1, 1, 1, 1]),
                ipg=(8, ["0.00", "0.63", "0.39", "0.00", "0.52"], [0, 1, 1, 1, 1]))
LEGO_REF = ["0.00", "0.54", "0.28", "0.00"]


def M(s, m):
    return AN.get(s, {}).get("methods", {}).get(m)


def nmin_val(s, m):
    if s == "lego" and m in LEGO_OLD:
        return LEGO_OLD[m][0]
    x = M(s, m)
    return None if x is None else x.get("N_min")


def nmin(s, m):
    if s == "lego" and m in LEGO_OLD:
        return f"{LEGO_OLD[m][0]}$^\\dagger$"
    x = M(s, m)
    if x is None: return "--"
    n = x.get("N_min")
    if n is None:
        mx = max([int(k) for k in x.get("stable_by_N", {})], default=None)
        return f"$>{mx}$" if mx else "--"
    return str(n)


def f2(v, d=2):
    return "--" if v is None else f"{v:.{d}f}"


def resolves(s, n, rep="a"):
    """우리 판 그 N 의 다시 푼 프레임 수 (ssfix --ssf_split; 로그 끝 '프레임 60 다시 풂 X 깊이별 {...}')"""
    p = f"{root}/{s}/hex/{s}_N{n}_{rep}{TAG}/log.txt"
    if not os.path.exists(p): return None
    mm = re.findall(r"프레임 \d+ 다시 풂 (\d+) 깊이별 (\{[^}]*\})", open(p, errors="ignore").read())
    return mm[-1] if mm else None


out = [r"""% lualatex stab_results_1115.tex
\documentclass[10pt]{article}
\usepackage[margin=0.7in]{geometry}
\usepackage{fontspec}
\setmainfont{Noto Serif CJK KR}
\usepackage{booktabs,amsmath,amssymb,graphicx}
\graphicspath{{./}{fig/}}
\title{서브스텝 안정성: 장면 4 개 (레고 $\cdot$ vasedeck $\cdot$ plane $\cdot$ pillow2sofa)}
\date{2026-11-15}
\begin{document}\maketitle"""]
out.append(open(f"{root}/settings.tex").read() if os.path.exists(f"{root}/settings.tex") else "")
# 표 1: N_min
t = [r"\begin{table}[h]\centering\small\caption{발산하지 않는 최소 서브스텝 수 $N$ (판정: NaN 없이 60 프레임, 운동+위치 에너지 변화 최대 $\le 1$, 바닥 관통 $\le 0.5L$; 2 의 거듭제곱으로 잰 뒤 발산한 $N$ 과 안정한 $N$ 사이 정수를 아래부터 올라가 처음 안정한 $N$, 같은 $N$ 세 번 모두 안정이면 확정). $^\dagger$ = 2026-11-12 기존 결과 재사용. $>X$ = 시험한 최대 $N$ 까지 안정 없음, -- = 측정 못 함.}",
     r"\begin{tabular}{l" + "c" * len(MS) + r"}\toprule", "장면 & " + " & ".join(n for _, n in MS) + r" \\\midrule"]
for s in SC:
    t.append(SN[s] + " & " + " & ".join(nmin(s, m) for m, _ in MS) + r" \\")
t.append(r"\bottomrule\end{tabular}\end{table}"); out.append("\n".join(t))
# 표 1b: 우리 판 다시 푼 프레임
t = [r"\begin{table}[h]\centering\small\caption{우리 판의 다시 푼 프레임 수 (60 프레임 중; 프레임 안 걸음이 무너지면 입자 상태를 되돌려 걸음 수를 두 배로 다시 풂, 최대 8 배 -- 공통 규칙 \texttt{ssfix\_wrap --ssf\_split 3}; 괄호 = 두 배 횟수별 프레임 수). 첫 판 (a) 기준.}",
     r"\begin{tabular}{l" + "c" * 5 + r"}\toprule 장면 & $N{=}1$ & $N{=}2$ & $N{=}4$ & $N{=}8$ & $N{=}16$ \\\midrule"]
for s in SC:
    cells = []
    for n in (1, 2, 4, 8, 16):
        r = resolves(s, n)
        if r is None: cells.append("")
        else:
            d = {int(k): v for k, v in re.findall(r"(\d+): (\d+)", r[1])}
            cells.append(r[0] + (" (" + ", ".join(f"{2 ** k}배 {d[k]}" for k in sorted(d) if k > 0) + ")" if int(r[0]) else ""))
    t.append(SN[s] + " & " + " & ".join(cells) + r" \\")
t.append(r"\bottomrule\end{tabular}\end{table}"); out.append("\n".join(t))
# 표 2: 물리 정합성
t = [r"\begin{table}[h]\centering\scriptsize\caption{확정 최소 $N$ 에서 물리 정합성 (그 $N$ 의 세 실행 평균; $\checkmark$ = 세 실행 모두 문턱 안). 기준 = PG $N{=}1024$. 문턱: 뒤집힘 0, 관통 $\le\max(2\times$기준, $0.01L)$, 떨림 $\le 2\times$기준, 자유낙하 운동량 오차 $\le 5\%$, Chamfer $\le 5\%L$. 값은 \% (관통·떨림·Chamfer 는 $L$ 대비). $^\dagger$ = 2026-11-12 기존 값 (기준 = 그때의 PG $N{=}1024$). -- = 해당 없음.}",
     r"\begin{tabular}{llcccccc}\toprule", r"장면 & 방법 & $N$ & 뒤집힘 & 바닥 관통 & 떨림 & 자유낙하 & Chamfer \\\midrule"]
CK = lambda ok: r"\,\checkmark" if ok else r"\,$\times$"
for s in SC:
    first = True
    pr = AN.get(s, {}).get("phys_ref")
    if pr:
        t.append(f"{SN[s]} & 기준 PG & 1024 & {f2(100 * (pr.get('detneg_max') or 0))} & {f2(100 * (pr.get('pen_depth_max') or 0))} & {f2(100 * (pr.get('jitter_d2') or 0))} & {f2(None if pr.get('freefall_mom_err') is None else 100 * pr['freefall_mom_err'])} & 0 \\\\"); first = False
    for m, nm in MS:
        if s == "lego" and m in LEGO_OLD:
            n, v, ok = LEGO_OLD[m]
            t.append(f"{'' if not first else SN[s]} & {nm}$^\\dagger$ & {n} & " + " & ".join(a + CK(b) for a, b in zip(v, ok)) + r" \\"); first = False
            continue
        x = M(s, m)
        if not x or not x.get("N_min"): continue
        n = x["N_min"]
        rs = [r for k, r in x["runs"].items() if r["N"] == n and r.get("phys")]
        if not rs: continue
        def col(key, sc=100.0):
            vs = [r["phys"][key][0] for r in rs if r["phys"][key][0] is not None]
            oks = [r["phys"][key][1] for r in rs if r["phys"][key][1] is not None]
            if not vs: return "--"
            return f"{sc * np.mean(vs):.2f}" + CK(oks and all(oks))
        t.append(f"{'' if not first else SN[s]} & {nm} & {n} & {col('detneg')} & {col('pen')} & {col('jit')} & {col('ff')} & {col('cd')} \\\\"); first = False
    t.append(r"\midrule")
t[-1] = r"\bottomrule"; t.append(r"\end{tabular}\end{table}"); out.append("\n".join(t))
# 표 3: 시각 지표
t = [r"\begin{table}[h]\centering\small\caption{확정 최소 $N$ 에서 시각 지표 (같은 렌더러 rep\_ip, F 반영 3DGS; 기준 = PG $N{=}1024$ 렌더, 프레임 1--60 평균; 모든 행 이번에 다시 렌더). 괄호: 마지막 프레임 PSNR.}",
     r"\begin{tabular}{llcccc}\toprule", r"장면 & 방법 & $N$ & PSNR $\uparrow$ & SSIM $\uparrow$ & LPIPS $\downarrow$ \\\midrule"]
for s in SC:
    first = True
    for m, nm in MS:
        n = nmin_val(s, m)
        if not n: continue
        v = VIS[s].get(f"{m}{TAG if m == 'ours' else ''}_N{n}")
        if not v: continue
        t.append(f"{'' if not first else SN[s]} & {nm} & {n} & {v['psnr']:.2f} ({v['psnr_last']:.1f}) & {v['ssim']:.3f} & {v['lpips']:.3f} \\\\"); first = False
    t.append(r"\midrule")
t[-1] = r"\bottomrule"; t.append(r"\end{tabular}\end{table}"); out.append("\n".join(t))
# 표 4: N-오차
NS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]


def err_rows(s, m):
    """-> (기준 글자, {N: 글자}, [(N, %)] 곡선 점)"""
    if s == "lego" and m in OLD and m in LEGO_OLD:
        x = OLD[m]; cells = {}; pts = []
        for k, r in x["runs"].items():
            n = r["N"]
            if n not in NS or not k.endswith(f"_N{n}"): continue
            e = x["err"].get(str(n))
            if r["stable"] and e is not None:
                cells[n] = "기준" if x["ref"] == n else f"{100 * e['mean']:.2f}"
                if x["ref"] != n: pts.append((n, 100 * e["mean"]))
            else:
                cells[n] = "발산"
        return (str(x["ref"]) + ("" if x["ref_converged"] else "*") + "$^\\dagger$"), cells, sorted(pts)
    x = M(s, m)
    if not x: return None
    cells = {}; pts = []
    for n in NS:
        e = x["err"].get(str(n))
        rr = [r for k, r in x["runs"].items() if r["N"] == n and r["rep"] == "a"]
        if not rr: continue
        if e is not None and rr[0]["stable"]:
            cells[n] = "기준" if x["ref"] == n else f"{100 * e['mean']:.2f}"
            if x["ref"] != n: pts.append((n, 100 * e["mean"]))
        else:
            cells[n] = "발산"
    ref = x["ref"]
    return ((str(ref) + ("" if x["ref_converged"] else "*")) if ref else "--"), cells, pts


t = [r"\begin{table}[h]\centering\tiny\caption{방법마다 자기 기준 궤적 ($N$ 을 두 배씩 올려 $N$ 과 $2N$ 의 차가 0.5\,\% 미만이 되는 $2N$; * = 미수렴이라 끝난 $N$ 의 두 배 사슬 끝) 대비 부분표본 RMSE$/L$ (\%, 60 프레임 평균, 같은 번호 32768 점). 빈 칸 = 안 돌림, 발산 = 안정 판정 실패. $^\dagger$ = 2026-11-12 기존 결과.}"]
t.append(r"\begin{tabular}{llc" + "c" * len(NS) + r"}\toprule 장면 & 방법 & 기준 " + "".join(f"& {n} " for n in NS) + r"\\\midrule")
CURVE = {}
for s in SC:
    first = True
    for m, nm in MS:
        r = err_rows(s, m)
        if r is None: continue
        rtxt, cells, pts = r; CURVE[(s, m)] = pts
        t.append(f"{'' if not first else SN[s]} & {nm} & {rtxt} & " + " & ".join(cells.get(n, "") for n in NS) + r" \\"); first = False
    t.append(r"\midrule")
t[-1] = r"\bottomrule"; t.append(r"\end{tabular}\end{table}"); out.append("\n".join(t))
# 그림: 오차 곡선
fig, axs = plt.subplots(1, 4, figsize=(16, 3.6), sharey=True)
cols = dict(pg="#1f77b4", ipg="#ff7f0e", gf="#2ca02c", omni="#9467bd", ours="#d62728")
for ax, s in zip(axs, SC):
    for m, nm in MS:
        pts = CURVE.get((s, m))
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", color=cols[m], label=nm, ms=4)
    ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_title(SN[s]); ax.set_xlabel("프레임당 서브스텝 $N$"); ax.grid(alpha=.3)
axs[0].set_ylabel("자기 기준 궤적 대비 RMSE/L (%)"); axs[0].legend(fontsize=7)
plt.tight_layout(); os.makedirs(f"{paper}/fig", exist_ok=True); plt.savefig(f"{paper}/fig/stab4_err.png", dpi=150)
out.append(r"\begin{figure}[h]\centering\includegraphics[width=\linewidth]{fig/stab4_err.png}\caption{$N$-오차 곡선 (방법마다 자기 기준 궤적 대비, 60 프레임 평균; 안정 판정에 통과한 $N$ 만, 점이 없는 $N$ 은 발산$\cdot$미측정$\cdot$기준). 레고의 PG$\cdot$i-PG$\cdot$GF 는 2026-11-12 기존 결과.}\end{figure}")
# 표 5: 프레임 시간
t = [r"\begin{table}[h]\centering\small\caption{우리 판 프레임당 시간 (ms, RTX A6000 단독: 같은 카드에 다른 계산 프로세스 없음, 시뮬 재생 GPU 시간 평균, 다시 푼 프레임은 버린 시도까지 합산, 루프 안 렌더 제외; 빈 칸 = 발산 또는 미측정).}",
     r"\begin{tabular}{lccccc}\toprule 장면 & $N{=}1$ & $N{=}2$ & $N{=}4$ & $N{=}8$ & $N{=}16$ \\\midrule"]
for s in SC:
    cells = []
    for n in (1, 2, 4, 8, 16):
        p = f"{root}/{s}/speed/t_N{n}{TAG}/log.txt"
        v = ""
        if os.path.exists(p):
            mm = re.findall(r"프레임당 ([0-9.]+) ms", open(p, errors="ignore").read())
            if mm: v = f"{float(mm[-1]):.1f}"
        cells.append(v)
    t.append(SN[s] + " & " + " & ".join(cells) + r" \\")
t.append(r"\bottomrule\end{tabular}\end{table}"); out.append("\n".join(t))
out.append(open(f"{root}/notes.tex").read() if os.path.exists(f"{root}/notes.tex") else "")
out.append(r"\end{document}")
open(f"{paper}/stab_results_1115.tex", "w").write("\n\n".join(out))
print("ok")

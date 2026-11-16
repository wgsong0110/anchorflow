#!/usr/bin/env python3
"""절 CM192 (사용자 형식 11-16): 파레토 = 방법마다 대표 설정 하나 (우리 K=1 만), 표 둘 (오차·속도·정답 없는 물리 / 시각, 행 같음, 끝에 네 장면 평균 행),
그림 = 네 장면 평균 하나 (가로 평균 ms 로그, 세로 평균 오차), 정답 없는 지표마다 평균 그림 하나 (fig/pareto_rf_<지표>.png), 고르기 표 (K=1 만).
입력: <캐시>/fpa_summary.json (exe/fpa_table.py 가 만든 파레토 행), 지표 모음 폴더 (exe/ffx_phys_merge.py).
  python3 exe/ffx_pareto_final.py <캐시> <지표 모음> <논문 폴더>"""
import json, math, os, statistics as S, sys

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

C, PH, OUT = sys.argv[1:4]
_FP = "/home/wgsong/.fonts/NotoSansCJKkr-Regular.otf"
try:
    font_manager.fontManager.addfont(_FP); plt.rcParams["font.family"] = font_manager.FontProperties(fname=_FP).get_name()
except Exception:
    pass
plt.rcParams["axes.unicode_minus"] = False
SC = [("L000", "Lego"), ("pillow2sofa", "pillow2sofa"), ("bread", "bread"), ("wolf", "wolf")]
METH = [("ours", "우리"), ("mpm", "정답 MPM (PG)"), ("ipg", "i-PG"), ("omni", "OmniPhysGS"), ("phystwin", "PhysTwin"), ("vrgs", "VR-GS"), ("springgaus", "Spring-Gaus")]
MN = dict(METH)
COL = dict(ours="#2a78d6", mpm="#eb6834", ipg="#1baf7a", omni="#eda100", phystwin="#e87ba4", vrgs="#008300", springgaus="#6250d6")
MK = dict(ours="o", mpm="*", ipg="s", omni="D", phystwin="^", vrgs="v", springgaus="P")
PHYS = [("detneg_max", "뒤집힘 (\\%)", 100, 3, True), ("pen_depth_max", "관통 (\\%$L$)", 100, 2, True), ("e_drift_max", "에너지 증가", 1, 3, True),
        ("freefall_mom_err", "운동량 (\\%)", 100, 2, True), ("jitter_d2", "떨림 ($10^{-3}L$)", 1e3, 2, True), ("dV_absmax", "부피 변화 (\\%)", 100, 2, True)]
VIS = [("niqe", "NIQE$\\downarrow$", 1, 2, True), ("brisque", "BRISQUE$\\downarrow$", 1, 1, True), ("musiq", "MUSIQ$\\uparrow$", 1, 1, False), ("clipiqa", "CLIP-IQA$\\uparrow$", 1, 3, False),
       ("warp_err", "깜빡임 ($10^{-3}$)$\\downarrow$", 1e3, 2, True), ("psnr", "PSNR$\\uparrow$", 1, 2, False), ("ssim", "SSIM$\\uparrow$", 1, 3, False), ("lpips", "LPIPS$\\downarrow$", 1, 3, True)]
RFN = dict(detneg_max="뒤집힘 (det F≤0 비율 최대)", pen_depth_max="바닥 관통 깊이 / L", e_drift_max="에너지 증가 (최대, E0 대비)", freefall_mom_err="자유낙하 운동량 오차",
           jitter_d2="떨림 (2 차 차분 / L)", jitter_hf="떨림 (고주파 에너지 비)", dV_absmax="부피 변화 |dV| 최대", niqe="NIQE", brisque="BRISQUE", clipiqa="CLIP-IQA", musiq="MUSIQ", warp_err="깜빡임 (warping error)")
RFLOW = dict(detneg_max=True, pen_depth_max=True, e_drift_max=True, freefall_mom_err=True, jitter_d2=True, jitter_hf=True, dV_absmax=True, niqe=True, brisque=True, clipiqa=False, musiq=False, warp_err=True)

P = json.load(open(f"{C}/fpa_summary.json"))["pareto"]
P = [p for p in P if p.get("err") is not None and p.get("ms") and p["method"] in MN and (p["method"] != "ours" or int(p["N"]) == 1)]


def mdir(m, s, N, rep=1):
    if m == "ours":
        c = [f"{PH}/ours_{s}_def" + ("" if rep == 1 else f"_r{rep}")]
    elif m == "mpm":
        c = [f"{PH}/mpm_{s}"]
    else:
        import glob
        c = [f"{PH}/{m}_{s}_n{N}", f"{PH}/{m}_{s}_nsg"] + sorted(glob.glob(f"{PH}/{m}_{s}*"))
    for d in c:
        if os.path.exists(f"{d}/phys.json") or os.path.exists(f"{d}/vis.json"):
            return d
    return None


def load(d, n):
    f = f"{d}/{n}.json"
    if not (d and os.path.exists(f)):
        return {}
    j = json.load(open(f))
    return j if isinstance(j, dict) else {}


def num(v):
    return v if isinstance(v, (int, float)) and math.isfinite(v) else None


ROWS = {}
for p in P:
    d = mdir(p["method"], p["scene"], p["N"])
    ph, vi = load(d, "phys"), load(d, "vis")
    if p["scene"] in ("wolf",):                                                         # 장면 조건상 자유낙하 아님 (층별 순차 풀림 구속) -- ffx_phys_tab 과 같은 규칙
        ph = dict(ph, freefall_mom_err=None)
    if ph.get("freefall_frames") == 0:
        ph = dict(ph, freefall_mom_err=None)
    if ph.get("E0") is not None and abs(ph["E0"]) < 1e-12:
        ph = dict(ph, e_drift_max=None)
    r = dict(err=p["err"], ms=p["ms"], ext=p.get("ext") == "material", N=p["N"], dir=d)
    for k, *_ in PHYS + [("jitter_hf",)]:
        r[k] = num(ph.get(k))
    for k, *_ in VIS:
        r[k] = num(vi.get(k))
    ROWS[(p["method"], p["scene"])] = r


def f(v, sc, d):
    return "--" if v is None else f"{v * sc:.{d}f}"


def table(cols, first):
    L = [r"\begin{center}\scriptsize\setlength{\tabcolsep}{3pt}\begin{tabular}{ll" + "r" * (len(cols) + (2 if first else 0)) + "}\\toprule",
         "장면 & 방법 & " + (r"오차 (\%) & ms & " if first else "") + " & ".join(c[1] for c in cols) + r"\\\midrule"]
    for s, sn in SC:
        q = [(m, ROWS[(m, s)]) for m, _ in METH if (m, s) in ROWS]
        for i, (m, r) in enumerate(q):
            lab = MN[m] + ("$^\\dagger$" if r["ext"] else "")
            cells = ([f(r["err"], 1, 2), f(r["ms"], 1, 1)] if first else []) + [f(r[c[0]], c[2], c[3]) for c in cols]
            L.append(f"{sn if i == 0 else ''} & {lab} & " + " & ".join(cells) + r"\\")
        L.append(r"\midrule")
    for i, (m, mn) in enumerate(METH):                                                 # 네 장면 평균 (있는 장면만; 괄호 = 장면 수)
        rs = [ROWS[(m, s)] for s, _ in SC if (m, s) in ROWS]
        if not rs:
            continue
        def avg(k):
            v = [r[k] for r in rs if r[k] is not None]
            return (S.mean(v), len(v)) if v else (None, 0)
        def cell(k, sc, d):
            a, n = avg(k)
            return "--" if a is None else (f"{a * sc:.{d}f}" + ("" if n == 4 else f"$^{{({n})}}$"))
        cells = ([cell("err", 1, 2), cell("ms", 1, 1)] if first else []) + [cell(c[0], c[2], c[3]) for c in cols]
        L.append(f"{'평균' if i == 0 else ''} & {mn} & " + " & ".join(cells) + r"\\")
    L.append(r"\bottomrule\end{tabular}\end{center}")
    return L


T = ["% 자동 생성 (exe/ffx_pareto_final.py) -- 방법마다 대표 설정 하나 (우리 = K=1)", r"\subsection*{오차$\cdot$속도$\cdot$정답 없는 물리 지표}"] + table(PHYS, True)
T += [r"\subsection*{시각 지표 (행 구성은 위 표와 같음)}"] + table(VIS, False)
T += [r"{\footnotesize $^\dagger$ 재질 확장 구현 (원 논문은 탄성 전용; wolf$\cdot$bread 에 PhysGaussian 소성 식을 넣음, 절 CM193 가지 final\_pa\_bl). "
      r"평균 행 = 네 장면 산술 평균, 위첨자 = 값이 있는 장면 수 (4 가 아니면). -- = 해당 없음 (바닥 없는 장면의 관통$\cdot$운동량, $F$ 없는 스프링--질량의 뒤집힘$\cdot$부피, bread 에너지 비 $E_0{=}0$, Lego$\cdot$wolf 자유낙하 구간 없음) 또는 미측정. "
      r"오차 = 롤아웃 평균 $|x-x_{gt}|/\mathrm{EXT}$, ms = RTX A6000 단독 3 회 중앙값. 정의 R2 \texttt{lbours/final\_fix/metrics\_def.md}.}"]
open(f"{OUT}/pareto_final_1115_table.tex", "w").write("\n".join(T) + "\n")


# 그림: 네 장면 평균 (방법마다 점 하나)
def avg_pts(key):
    out = []
    for m, mn in METH:
        rs = [(s, ROWS[(m, s)]) for s, _ in SC if (m, s) in ROWS and ROWS[(m, s)].get(key) is not None]
        if not rs:
            continue
        out.append(dict(m=m, n=len(rs), sc=[s for s, _ in rs], ms=S.mean(r["ms"] for _, r in rs), v=S.mean(r[key] for _, r in rs)))
    return out


def plot(key, ylab, fn, low=True, logy=False):
    A = avg_pts(key)
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for a in A:
        lab = MN[a["m"]] + ("" if a["n"] == 4 else f" ({a['n']} 장면: {', '.join(dict(SC)[s] for s in a['sc'])})")
        ax.scatter([a["ms"]], [a["v"]], s=220 if a["m"] == "mpm" else 80, marker=MK[a["m"]], c=COL[a["m"]], edgecolors="#fcfcfb", linewidths=1.5, label=lab, zorder=3)
    ax.set_xscale("log"); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    if logy:
        ax.set_yscale("log")
    ax.axvline(1000 / 60, color="#c3c2b7", lw=1, ls=":", zorder=1)
    ax.set_xlabel("네 장면 평균 프레임 시간 (ms, RTX A6000 단독, 로그; 점선 60 fps)", fontsize=9, color="#52514e")
    ax.set_ylabel(ylab + (" (낮을수록 좋음)" if low else " (높을수록 좋음)"), fontsize=9, color="#0b0b0b")
    ax.grid(True, color="#ecebe8", lw=0.6, zorder=0); [sp.set_visible(False) for sp in (ax.spines["top"], ax.spines["right"])]
    ax.tick_params(labelsize=8, colors="#52514e")
    ax.legend(fontsize=7.5, frameon=False, loc="best")
    fig.tight_layout(); fig.savefig(fn, dpi=170); plt.close(fig)
    return A


os.makedirs(f"{OUT}/fig", exist_ok=True)
PA = plot("err", "네 장면 평균 롤아웃 오차 (%)", f"{OUT}/fig/pareto_final_1115.png")
for k in RFN:
    plot(k, "네 장면 평균 " + RFN[k], f"{OUT}/fig/pareto_rf_{k}.png", low=RFLOW[k])


# 고르기 표 (K=1 만, 장면별 값으로 셈) -- exe/ffx_rf_choice.py 와 같은 정의
def spearman(a, b):
    def rk(x):
        o = sorted(range(len(x)), key=lambda i: x[i]); r = [0.0] * len(x); i = 0
        while i < len(o):
            j = i
            while j + 1 < len(o) and x[o[j + 1]] == x[o[i]]:
                j += 1
            for q in range(i, j + 1):
                r[o[q]] = (i + j) / 2
            i = j + 1
        return r
    if len(a) < 3:
        return None
    ra, rb = rk(a), rk(b); ma, mb = S.mean(ra), S.mean(rb)
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return sum((x - ma) * (y - mb) for x, y in zip(ra, rb)) / den if den > 0 else None


CH = []
for k in RFN:
    low = RFLOW[k]; sep, mr, cor, noi, cov = [], [], [], [], [0, 0]
    for s, _ in SC:
        V = [(m, ROWS[(m, s)]) for m, _ in METH if (m, s) in ROWS]
        cov[1] += len(V); G = [(m, r[k], r["err"]) for m, r in V if r.get(k) is not None]; cov[0] += len(G)
        if len(G) < 2:
            continue
        vs = [g[1] for g in G]; md = abs(S.median(vs))
        if md > 0:
            sep.append((max(vs) - min(vs)) / md)
        mv = [g[1] for g in G if g[0] == "mpm"]
        if mv:
            srt = sorted(vs, reverse=not low); mr.append(srt.index(mv[0]) / max(len(vs) - 1, 1))
        nm = [(g[2], g[1]) for g in G if g[0] != "mpm"]
        c = spearman([a for a, _ in nm], [b for _, b in nm])
        if c is not None:
            cor.append(c if low else -c)
        rep = [num(load(mdir("ours", s, 1, r), "phys" if k in [x[0] for x in PHYS] + ["jitter_hf"] else "vis").get(k)) for r in (1, 2, 3)]
        rep = [x for x in rep if x is not None]
        if len(rep) >= 2 and S.pstdev(vs) > 0:
            noi.append(S.pstdev(rep) / S.pstdev(vs))
    CH.append(dict(key=k, name=RFN[k], cov=f"{cov[0]}/{cov[1]}", spread=S.median(sep) if sep else None, mpm=S.mean(mr) if mr else None, nm=len(mr),
                   corr=S.mean(cor) if cor else None, nc=len(cor), noise=S.mean(noi) if noi else None, nn=len(noi)))
g = lambda x, fm="%.2f": "--" if x is None else fm % x
L = ["% 자동 생성 (exe/ffx_pareto_final.py): 정답 없는 지표 고르기 (우리 K=1 만)", r"\begin{center}\scriptsize\begin{tabular}{lrrrrr}\toprule",
     r"지표 & 값 있는 점 & 방법 사이 폭 & 정답 MPM 순위 (0 = 가장 좋음) & 오차와 순위 상관 & 노이즈 (3 회 / 방법 사이)\\\midrule"]
for r in CH:
    L.append(f"{r['name']} & {r['cov']} & {g(r['spread'])} & {g(r['mpm'])} ({r['nm']}) & {g(r['corr'])} ({r['nc']}) & {g(r['noise'], '%.3f')} ({r['nn']})\\\\")
L.append(r"\bottomrule\end{tabular}\end{center}")
open(f"{OUT}/pareto_final_1115_rf_choice_table.tex", "w").write("\n".join(L) + "\n")
json.dump(dict(rows={f"{m}:{s}": {k: v for k, v in r.items() if k != "dir"} for (m, s), r in ROWS.items()}, avg=PA, choice=CH), open(f"{C}/pareto_final.json", "w"), ensure_ascii=False, indent=1)
for r in CH:
    print(r["key"].ljust(17), r["cov"], g(r["spread"]), g(r["mpm"]), g(r["corr"]), g(r["noise"], "%.3f"))
for a in PA:
    print("평균", a["m"], a["n"], round(a["ms"], 1), round(a["v"], 2))

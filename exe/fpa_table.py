#!/usr/bin/env python3
"""절 CM190 (final_pa) 표·그림 생성 (로컬 CPU). 입력: <캐시 디렉토리> 아래 runs/<장면>_<판>_r<반복>/metrics.json (R2 lbours/final_pa/runs 에서 내려받음),
all.json (R2 lbours/paper_remeasure2/final2/all.json: 정답 MPM wolf·bread·pillow2sofa), pareto_bl.json (R2 lbours/final_pa/bl/pareto_bl.json: 정답 MPM 레고·VR-GS·PhysTwin·Spring-Gaus).
출력 (<논문 폴더>): pareto_final_1115_table.tex, ablation_final_1115_tables.tex, pareto_final_1115.png, pareto_final_1115_lin.png, 그리고 <캐시>/fpa_summary.json.
  python exe/fpa_table.py <캐시> <논문 폴더>"""
import json, os, statistics as S, sys

C, OUT = sys.argv[1:3]
SC = [("L000", "레고"), ("pillow2sofa", "pillow2sofa"), ("bread", "bread"), ("wolf", "wolf")]
MAT = dict(L000="탄성 (jelly), 바닥 충돌", pillow2sofa="탄성 (jelly), 물체 간 충돌·고정 구속", bread="소성 (plasticine, 손상 연화) 찢어짐, 양쪽 당김", wolf="모래 (Drucker--Prager), 순차 방출")
FR = dict(L000=80, pillow2sofa=300, bread=300, wolf=50)


EXCL = {} if os.environ.get("FPA_ND") == "1" else {"bread_k4_r3": "같은 카드에서 영상 렌더가 겹쳐 돌아 시간 무효 -> r4 로 다시 잼"}      # 절 CM191 runs_nd 는 제외 없음


def runs(s, v):
    out = []
    for r in (1, 2, 3, 4):
        if f"{s}_{v}_r{r}" in EXCL:
            continue
        f = f"{C}/runs/{s}_{v}_r{r}/metrics.json"
        if os.path.exists(f):
            m = json.load(open(f)); m["rep"] = r; out.append(m)
    return out


def done(m):
    return m.get("frames_done") is not None and m.get("frames") and m["frames_done"] >= m["frames"] and not m.get("diverged") and m.get("err") is not None


def agg(s, v):
    rs = runs(s, v)
    ok = [m for m in rs if done(m)]
    a = dict(n=len(rs), ok=len(ok), err=S.median([100 * m["err"] for m in ok]) if ok else None, err_all=[100 * m["err"] for m in ok],
             ms=S.median([m["ms"] for m in ok]) if ok else None, ms_all=[m["ms"] for m in ok],
             div=[(m.get("diverged") or (m.get("frames_done"), m.get("frames"))) for m in rs if not done(m)],
             flip=max([m.get("pF_det_le0_max") or 0 for m in rs], default=None) if rs else None,
             nJ=max([m.get("nJneg_max") or 0 for m in rs], default=None) if rs else None,
             rise=max([((m.get("energy") or {}).get("rise_rel") or 0) for m in rs], default=None) if rs else None,
             eig=next(((m.get("eig") or {}).get("n_zero") for m in rs if isinstance(m.get("eig"), dict)), None))
    return a


def cell(a, k="err", fmt="%.2f"):
    if a is None or a["n"] == 0:
        return "--"
    if a[k] is None and k == "ms":
        return "--"
    if a[k] is None:
        d = a["div"][0] if a["div"] else None
        if isinstance(d, (int, float)):
            return "발산 f%d" % d
        if isinstance(d, (list, tuple)) and d[0] is not None:
            return "중단 %s/%s" % (d[0], d[1])
        return "실패"
    s = fmt % a[k]
    if a["ok"] < a["n"]:
        s += "$^{%d/%d}$" % (a["ok"], a["n"])
    return s


# ---------------- 파레토 ----------------
P = []
for s, _ in SC:
    for v, K in (("def", 1), ("k2", 2), ("k4", 4)):
        a = agg(s, v)
        if a["n"]:
            P.append(dict(method="ours", scene=s, N=K, setting=f"K={K}", reps=a["n"], ok=a["ok"], err=a["err"], err_all=a["err_all"], ms=a["ms"], ms_all=a["ms_all"], div=a["div"]))
AJ = json.load(open(f"{C}/all.json"))
for r in AJ:
    if r["method"] == "mpm" and (r["scene"], r["N"]) in (("wolf", 1999), ("bread", 100), ("pillow2sofa", 200)):
        P.append(dict(r, setting=f"서브스텝 {r['N']} (공식)", src="paper_remeasure2/final2/all.json"))
NA = {}
BLF = f"{C}/pareto_bl.json" if os.path.exists(f"{C}/pareto_bl.json") else f"{C}/pareto_bl_partial.json"
BLX = json.load(open(f"{C}/bl_extra.json")) if os.path.exists(f"{C}/bl_extra.json") else []      # 절 CM191: 다시 잰 베이스라인 칸 (Spring-Gaus·VR-GS pillow2sofa) 이 같은 (방법, 장면) 을 대체
if os.path.exists(BLF):
    BLR = [r for r in json.load(open(BLF)) if (r["method"], r["scene"]) not in {(x["method"], x["scene"]) for x in BLX}] + BLX
    for r in BLR:
        if r["scene"] in ("wolf", "bread") and r.get("err") is None:             # 해당 없음 (탄성 전용) -- 표에서 따로 적음
            continue
        SN = {("vrgs", "pillow2sofa"): "실패: 3 회 모두 f128 발산 (직접 구현에 물체 충돌$\\cdot$상자 경계 없음)",
              ("springgaus", "pillow2sofa"): "미완: 공식 동역학 학습 14/300 반복 (반복당 114 s, 예상 13.7 h) 에서 예산으로 중단"}
        if r.get("err") is None and (r["method"], r["scene"]) in SN:
            r = dict(r, note=SN[(r["method"], r["scene"])], N=r.get("N") or 0)
        if r.get("err") is None or not r.get("ms"):
            NA[(r["method"], r["scene"])] = r.get("note") or "실패"
        P.append(dict(r, src="final_pa/bl/pareto_bl.json"))
json.dump(dict(pareto=P, na={f"{k[0]}:{k[1]}": v for k, v in NA.items()}), open(f"{C}/fpa_summary.json", "w"), ensure_ascii=False, indent=1)
SHORT = dict(ours="K={N}", mpm="서브스텝 {N} (공식)", vrgs="서브스텝 {N} $\\times$ 반복 11", phystwin="서브스텝 $N_0$={N}", springgaus="학습 n\\_step {N}")
for p in P:
    if p.get("N"):
        p["setting"] = SHORT[p["method"]].format(N=p["N"])
        if p["method"] == "vrgs" and p.get("iters"):                                # 절 CM191: 논문 6.1 반복 1 판
            p["setting"] = p["setting"].replace("반복 11", "반복 %d" % p["iters"])
    elif p.get("N") == 0:
        p["setting"] = "--"
LAB = dict(ours="우리", mpm="정답 MPM (PhysGaussian)", vrgs="VR-GS", phystwin="PhysTwin", springgaus="Spring-Gaus")
L = ["% 자동 생성 (exe/fpa_table.py)", r"\begin{center}\scriptsize\begin{tabular}{llllrrr}\toprule",
     r"장면 & 방법 & 설정 & 완주/회 & 오차 \% (3 회) & ms (3 회 중앙값) & ms 3 회 \\\midrule"]
for s, nm in SC:
    for m in ("ours", "mpm", "vrgs", "phystwin", "springgaus"):
        q = [p for p in P if p["scene"] == s and p["method"] == m]
        if not q:
            if m in ("vrgs", "phystwin", "springgaus") and s in ("bread", "wolf"):
                L.append(f"{nm} & {LAB[m]} & \\multicolumn{{5}}{{l}}{{해당 없음 (탄성 전용, 아래 근거)}}\\\\")
            continue
        for p in sorted(q, key=lambda p: p["N"]):
            if p.get("err") is None:
                L.append(f"{nm} & {LAB[m]} & {p.get('setting', p['N'])} & {p.get('ok', 0)}/{p.get('reps', 0)} & \\multicolumn{{3}}{{l}}{{{p.get('note') or '발산'}}}\\\\")
                continue
            ea = ", ".join("%.2f" % e for e in p["err_all"]) if p.get("err_all") else "%.2f" % p["err"]
            ma = ", ".join("%.1f" % e for e in p["ms_all"]) if p.get("ms_all") else "%.1f" % p["ms"]
            L.append(f"{nm} & {LAB[m]} & {p.get('setting', p['N'])} & {p.get('ok')}/{p.get('reps')} & {p['err']:.2f} ({ea}) & {p['ms']:.1f} & {ma}\\\\")
    L.append(r"\midrule")
L[-1] = r"\bottomrule\end{tabular}\end{center}"
open(f"{OUT}/pareto_final_1115_table.tex", "w").write("\n".join(L) + "\n")

# 그림 (점만, 로그판·선형판)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                                       # noqa: E402
from matplotlib import font_manager as fm                                            # noqa: E402
f = "/home/wgsong/.fonts/NotoSansCJKkr-Regular.otf"
try:
    fm.fontManager.addfont(f); plt.rcParams["font.family"] = fm.FontProperties(fname=f).get_name()
except Exception:
    pass
ST = dict(ours=("우리", "#d62728", "o"), mpm=("정답 MPM (PhysGaussian, 공식 서브스텝)", "#1f77b4", "s"), vrgs=("VR-GS", "#2ca02c", "^"),
          phystwin=("PhysTwin", "#ff7f0e", "D"), springgaus=("Spring-Gaus", "#17becf", "X"))
TT = dict(L000="레고 (탄성)", pillow2sofa="pillow2sofa (탄성, 충돌)", bread="bread (소성, 찢어짐)", wolf="wolf (모래)")
for lin in (False, True):
    fig, ax = plt.subplots(2, 2, figsize=(11, 8.4), squeeze=False)
    for k, (s, _) in enumerate(SC):
        a = ax[k // 2][k % 2]
        for m, (lab, c, mk) in ST.items():
            q = [p for p in P if p["scene"] == s and p["method"] == m and p.get("err") is not None and p.get("ms")]
            if not q:
                continue
            YF = 1e-3
            xs = [p["ms"] for p in q]; ys = [p["err"] if lin else max(p["err"], YF) for p in q]
            a.scatter(xs, ys, c=c, marker=mk, s=46, label=lab, alpha=0.9, zorder=3)
            if not lin:
                lo = [(x_, y_) for x_, y_, p in zip(xs, ys, q) if p["err"] < YF]
                if lo:
                    a.scatter([x_ for x_, _ in lo], [y_ for _, y_ in lo], facecolors="white", edgecolors=c, marker=mk, s=46, zorder=4)
            if m == "ours":
                for x_, y_, p in zip(xs, ys, q):
                    a.annotate(f"K={p['N']}", (x_, y_), textcoords="offset points", xytext=(5, 4), fontsize=7, color=c)
        a.axvline(1000 / 60, c="gray", ls="--", lw=0.8)
        if not lin:
            a.set_xscale("log"); a.set_yscale("log")
        else:
            a.set_xlim(left=0); a.set_ylim(bottom=0)
        a.margins(0.1)
        a.set_xlabel("프레임 시간 (ms, RTX A6000 단독, 3 회 중앙값)"); a.set_ylabel("롤아웃 평균 오차 (%)")
        a.set_title(f"{TT[s]} (점선 = 60 fps)"); a.grid(alpha=0.3, which="both")
    h = {}
    for r_ in ax:
        for a in r_:
            for hh, ll in zip(*a.get_legend_handles_labels()):
                h[ll] = hh
    ax[0][0].legend(h.values(), h.keys(), fontsize=7)
    fig.tight_layout()
    o = f"{OUT}/fig/pareto_final_1115{'_lin' if lin else ''}.png"
    os.makedirs(os.path.dirname(o), exist_ok=True)
    fig.savefig(o, dpi=170); print("저장", o)

# ---------------- ablation ----------------
VN = dict(def_=os.environ.get("FPA_DEF_LABEL", "기본 (2$^3$ 가우스점, HHT $\\rho_\\infty{=}0.8$, F 가우스점 보간)"), q1="가우스점 1$^3$", be="후퇴 오일러", mid="중점 (HHT $\\rho_\\infty{=}1$)",
          nofpi=os.environ.get("FPA_NOFPI_LABEL", "F 전달: G2P 누적 (가우스점 보간 없음)"), q27="가우스점 3$^3$", q64="가우스점 4$^3$", c075="칸 $\\times$0.75", c15="칸 $\\times$1.5", c20="칸 $\\times$2")


def vname(v):
    return VN["def_" if v == "def" else v]


def table(title, vs, extra=True):
    T = [f"\\subsection*{{{title}}}", r"\begin{center}\scriptsize\begin{tabular}{l" + "rr" * len(SC) + "}\\toprule",
         "판 & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{nm}}}" for _, nm in SC) + r"\\", " & " + " & ".join("오차 \\% & ms" for _ in SC) + r"\\\midrule"]
    for v in vs:
        T.append(vname(v) + " & " + " & ".join(cell(agg(s, v)) + " & " + cell(agg(s, v), "ms", "%.1f") for s, _ in SC) + r"\\")
    T.append(r"\bottomrule\end{tabular}\end{center}")
    if extra:
        T += [r"\begin{center}\scriptsize\begin{tabular}{llrrrr}\toprule", r"장면 & 판 & 완주/회 & 입자 $\det F\le0$ 최대 & 에너지 최대 증가 (규모 대비) & 영 에너지 모드 수\\\midrule"]
        for s, nm in SC:
            for v in vs:
                a = agg(s, v)
                if not a["n"]:
                    continue
                T.append(f"{nm} & {vname(v)} & {a['ok']}/{a['n']} & {a['flip'] if a['flip'] is not None else '--'} & "
                         f"{'%.3g' % a['rise'] if a['rise'] is not None else '--'} & {a['eig'] if (a['eig'] is not None and v in ('def', 'q1', 'q27')) else '--'}\\\\")
        T.append(r"\bottomrule\end{tabular}\end{center}")
    return T


A = ["% 자동 생성 (exe/fpa_table.py)"]
A += table("A. 설계 선택 (가우스점 수$\\cdot$적분기$\\cdot$격자 전달)", ["def", "q1", "be", "mid", "nofpi"])
A += table("B. 가우스점 수 (축당 1--4)", ["q1", "def", "q27", "q64"])
A += table("C. 칸 크기 (기본 칸의 0.75--2 배)", ["c075", "def", "c15", "c20"])
A += ["\\subsection*{D. 재질 일반성 (기본 판 하나로 네 장면)}", r"\begin{center}\small\begin{tabular}{llrrr}\toprule", r"장면 & 재질$\cdot$경계 & 완주/회 & 오차 \% & ms\\\midrule"]
for s, nm in SC:
    a = agg(s, "def")
    A.append(f"{nm} & {MAT[s]} & {a['ok']}/{a['n']} & {cell(a)} & {cell(a, 'ms', '%.1f')}\\\\")
A.append(r"\bottomrule\end{tabular}\end{center}")
A += ["\\subsection*{E. 서브스텝 수 $N$ 강인성 (프레임당 걸음 $N$, 50 프레임, 1 회; 오차 \\% / ms)}", r"\begin{center}\small\begin{tabular}{r" + "rr" * len(SC) + "}\\toprule",
      "$N$ & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{nm}}}" for _, nm in SC) + r"\\", " & " + " & ".join("오차 \\% & ms" for _ in SC) + r"\\\midrule"]
for n in (1, 2, 4, 8, 16, 32, 64, 128):
    A.append(f"{n} & " + " & ".join(cell(agg(s, f'ss{n}')) + " & " + cell(agg(s, f'ss{n}'), "ms", "%.0f") for s, _ in SC) + r"\\")
A.append(r"\bottomrule\end{tabular}\end{center}")
open(f"{OUT}/ablation_final_1115_tables.tex", "w").write("\n".join(A) + "\n")
print("표 저장", OUT)

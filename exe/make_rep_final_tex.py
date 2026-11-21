"""rep_final (11-15): 표현력 실험 (정답 궤적을 L2 로 따라가기) 최종 표 -- L2 오차 (RMSE/L, %) 하나.
  python exe/make_rep_final_tex.py --old repflow_results.tex --out rep_final_1115.tex
  [repro] python exe/make_rep_final_tex.py --raw $RAW --pt best --out <논문>/rep_final_1115.tex --table_out <논문>/results_all/rep_table.tex
장면 = wolf + 레고 L000 세팅 PhysGaussian MPM 장면 셋 (R2 lbours/rep_final1115/<장면>/<판>/<판>_<장면>.log 의 [요약] RMSE).
  wolf 열: --raw 가 있으면 R2 원 로그 (repflow_hex·archive_20261009/r10·repflow_vrgs·repflow1115/sg, 아래 WOLF_LOG) 에서 읽고, --old 가 같이 있으면 그 표 값과 대조;
           --raw 가 없으면 기존 표 --old 의 RMSE 열 그대로.
방법 = 우리 (육면체), PhysTwin (공식 강성 k 1·10·100·1000), VR-GS, Spring-Gaus. 굵게 = 장면별 최저.
  --pt all (기본) = PhysTwin k 넷 다; --pt best = 네 장면 평균이 가장 낮은 k 한 행만 'PhysTwin' 으로 (11-16 손 편집을 대체); --pt 1 등 = 그 k 한 행만.
  --table_out = 논문 repo results_all/rep_table.tex (표만, 짧은 캡션; paper_tables.py 입력)."""
import argparse, os, re, subprocess

ap = argparse.ArgumentParser(); ap.add_argument("--old", default=""); ap.add_argument("--out", required=True)
ap.add_argument("--r2", default="r2:storage/result/anchorflow/lbours/rep_final1115")
ap.add_argument("--raw", default="", help="repro/fetch_raw.sh 받은 폴더 (r2:storage/result/anchorflow 아래와 같은 경로); 주면 --r2 대신 로컬 파일")
ap.add_argument("--pt", default="all"); ap.add_argument("--table_out", default="")
a = ap.parse_args()
LAB = [("hex", "Ours (hexahedral)"), ("pt1", "PhysTwin ($k{=}1$)"), ("pt10", "PhysTwin ($k{=}10$)"), ("pt100", "PhysTwin ($k{=}100$)"),
       ("pt1000", "PhysTwin ($k{=}10^3$)"), ("vrgsx", "VR-GS"), ("sg", "Spring-Gaus")]
OLDLAB = {"hex": "Ours (hex)"}
SC = [("wolf", "wolf"), ("bread_foam_drop_lego", "bread (foam)"), ("drums_honey_lego", "drums (foam)"), ("chair_paste_lego", "chair (plasticine)")]


def old_rmse(scene):
    s = open(a.old).read()
    m = re.search(r"\\caption\{A: L2 tracking, %s[ .(].*?\\end\{tabular\}" % scene, s, re.S)
    out = {}
    for l in m.group(0).split("\n"):
        if l.count("&") >= 3 and not l.startswith("Method"):
            c = [re.sub(r"\\textbf\{(.*)\}", r"\1", q.strip()) for q in l.rstrip().rstrip("\\").rstrip().split("&")]
            out[c[0]] = float(c[2])
    return out


def _rmse(txt):
    m = re.findall(r"\[요약\].*?RMSE ([0-9.]+)%", txt)
    return float(m[-1]) if m else float("nan")


def new_rmse(scene, key):
    if a.raw:
        p = f"{a.raw}/lbours/rep_final1115/{scene}/{key}/{key}_{scene}.log"
        return _rmse(open(p, errors="ignore").read()) if os.path.exists(p) else float("nan")
    log = subprocess.run(["rclone", "cat", f"{a.r2}/{scene}/{key}/{key}_{scene}.log"], capture_output=True, text=True, timeout=300).stdout
    return _rmse(log)


# wolf 열 원 로그 (네 칸이 서로 다른 실행: README 4.6) -- r2:storage/result/anchorflow 아래 경로
WOLF_LOG = dict(hex="lbours/repflow_hex/main/wolf_hex.log", pt1="archive_20261009/r10/wolf_phystwin_k1.txt", pt10="archive_20261009/r10/wolf_phystwin_k10.txt",
                pt100="archive_20261009/r10/wolf_phystwin_k100.txt", pt1000="archive_20261009/r10/wolf_phystwin_k1000.txt",
                vrgsx="lbours/repflow_vrgs/main/wolf_vrgsx.log", sg="lbours/repflow1115/sg/main/wolf/sg_wolf.log")


V = {}
for sc, _ in SC:
    if sc == "wolf":
        if a.raw:
            V[sc] = {k: _rmse(open(f"{a.raw}/{WOLF_LOG[k]}", errors="ignore").read()) for k, _ in LAB}
            if a.old:                                                                    # 기존 표 (repflow_results.tex) 와 대조: 표에 찍힌 두 자리까지 같아야 함
                o = old_rmse("wolf")
                for k, lab in LAB:
                    assert f"{V[sc][k]:.2f}" == f"{o[OLDLAB.get(k, lab)]:.2f}", ("wolf", k, V[sc][k], o[OLDLAB.get(k, lab)])
        else:
            o = old_rmse("wolf"); V[sc] = {k: o[OLDLAB.get(k, lab)] for k, lab in LAB}
    else:
        V[sc] = {k: new_rmse(sc, k) for k, _ in LAB}
PTK = [k for k, _ in LAB if k.startswith("pt")]
if a.pt != "all":                                                                       # PhysTwin 한 행만 (대표 k)
    keep = min(PTK, key=lambda k: sum(V[sc][k] for sc, _ in SC) / len(SC)) if a.pt == "best" else f"pt{a.pt}"
    assert keep in PTK, a.pt
    PT_KEEP = keep
    LAB = [(k, ("PhysTwin" if k == keep else lab)) for k, lab in LAB if not k.startswith("pt") or k == keep]
    print("[PhysTwin 대표]", keep, {k: round(sum(V[sc][k] for sc, _ in SC) / len(SC), 3) for k in PTK})
best = {sc: min([v for v in V[sc].values() if v == v] or [float("nan")]) for sc, _ in SC}
rows = []
for k, lab in LAB:
    cells = []
    for sc, _ in SC:
        x = V[sc][k]; t = "--" if x != x else f"{x:.2f}"
        if x == x and abs(x - best[sc]) < 1e-12: t = "\\textbf{%s}" % t
        cells.append(t)
    rows.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    if k == "hex": rows.append("\\midrule")
doc = r"""\documentclass[10pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{booktabs,amsmath}
\begin{document}
\begin{table}[t]\centering\small
\caption{Representation expressiveness: L2 tracking of a reference trajectory. Each representation follows the reference point
trajectory by minimizing the per-point L2 distance on its own geometry ($\Delta=-(G+\varepsilon\lambda_{\max}I)^{-1}\nabla L$, Armijo line search;
$\varepsilon,\eta$ of every method chosen once on wolf and used unchanged). Reported: frame-mean RMSE over all points divided by the rest
bounding-box diagonal $L$ (\%, lower is better; bold = lowest per scene). wolf: all opacity-filtered Gaussians, reference = Gaussian velocity
field, 120 frames. bread, drums, chair: PhysGaussian MPM reference (all simulated particles, no subsampling), one setting for all three
(initial rotation, $\Delta t_\text{sub}=2.0\times10^{-5}$, 60 fps, 80 frames, zero initial velocity, $n_\text{grid}=100$, bounding box and sticky floor);
materials foam ($E=2{\times}10^4$, $\nu=0.4$), foam ($E=2{\times}10^4$, $\nu=0.4$), plasticine ($E=2{\times}10^5$, $\nu=0.3$).
""" + (r"PhysTwin is shown at its official stiffness settings $k$" if a.pt == "all" else
       r"PhysTwin uses a single representative setting (spring stiffness weight $k{=}%s$, the most accurate of $k\in\{1,10,10^2,10^3\}$)" % {"1": "1", "10": "10", "100": "10^2", "1000": "10^3"}[PT_KEEP[2:]]) + r"""; Spring-Gaus uses its official initial (pre-training) representation.}\label{tab:rep_l2}
\begin{tabular}{l""" + "r" * len(SC) + r"""}\toprule
Method & """ + " & ".join(h for _, h in SC) + r""" \\\midrule
""" + "\n".join(rows) + r"""
\bottomrule\end{tabular}
\end{table}
\end{document}
"""
open(a.out, "w").write(doc)
if a.table_out:                                                                         # 논문 repo results_all/rep_table.tex (짧은 캡션, 표만)
    open(a.table_out, "w").write(r"""\begin{table}[t]\centering\small
\caption{L2 추종 오차 (RMSE/$L$, \%).}\label{tab:rep_l2}
\begin{tabular}{l""" + "r" * len(SC) + r"""}\toprule
Method & """ + " & ".join(h for _, h in SC) + r""" \\\midrule
""" + "\n".join(rows) + r"""
\bottomrule\end{tabular}
\end{table}
""")
print("[표]", V)

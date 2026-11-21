"""rep_final (11-15): 표현력 실험 (정답 궤적을 L2 로 따라가기) 최종 표 -- L2 오차 (RMSE/L, %) 하나.
  python exe/make_rep_final_tex.py --old repflow_results.tex --out rep_final_1115.tex
장면 = wolf (기존 표 --old 의 RMSE 열 그대로) + 레고 L000 세팅 PhysGaussian MPM 장면 셋 (R2 lbours/rep_final1115/<장면>/<판>/<판>_<장면>.log 의 [요약] RMSE).
방법 = 우리 (육면체), PhysTwin (공식 강성 k 1·10·100·1000), VR-GS, Spring-Gaus. 굵게 = 장면별 최저."""
import argparse, re, subprocess

ap = argparse.ArgumentParser(); ap.add_argument("--old", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--r2", default="r2:storage/result/anchorflow/lbours/rep_final1115")
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


def new_rmse(scene, key):
    log = subprocess.run(["rclone", "cat", f"{a.r2}/{scene}/{key}/{key}_{scene}.log"], capture_output=True, text=True, timeout=300).stdout
    m = re.findall(r"\[요약\].*?RMSE ([0-9.]+)%", log)
    return float(m[-1]) if m else float("nan")


V = {}
for sc, _ in SC:
    if sc == "wolf":
        o = old_rmse("wolf"); V[sc] = {k: o[OLDLAB.get(k, lab)] for k, lab in LAB}
    else:
        V[sc] = {k: new_rmse(sc, k) for k, _ in LAB}
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
PhysTwin is shown at its official stiffness settings $k$; Spring-Gaus uses its official initial (pre-training) representation.}\label{tab:rep_l2}
\begin{tabular}{l""" + "r" * len(SC) + r"""}\toprule
Method & """ + " & ".join(h for _, h in SC) + r""" \\\midrule
""" + "\n".join(rows) + r"""
\bottomrule\end{tabular}
\end{table}
\end{document}
"""
open(a.out, "w").write(doc)
print("[표]", V)

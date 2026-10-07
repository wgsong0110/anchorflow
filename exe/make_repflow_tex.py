"""표현력 비교(repflow) 결과를 LaTeX 표로 모은다 (로컬에서 돈다).

입력은 클러스터에서 가져온 요약 폴더 (--src):
  A/<shape>_<method>.json   rep_metrics 출력 + 요약 (summary 키: rmse, cd, det_min, inv_max, dof)
  B/<sim>_<shape>_<method>.json   같은 형식 (+ IP_mean)
출력: writing/anchorflow/repflow_results.tex (단독 컴파일되는 문서)

  python exe/make_repflow_tex.py --src /tmp/repflow_sum --out /home/wgsong/workspace/writing/anchorflow/repflow_results.tex
"""
import argparse
import glob
import json
import os

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--commit_a", default="")
ap.add_argument("--commit_b", default="")
a = ap.parse_args()

NAME = {"ours": "Ours", "phystwin_k1": "PhysTwin ($k{=}1$)", "phystwin_k10": "PhysTwin ($k{=}10$)",
        "phystwin_k100": "PhysTwin ($k{=}100$)", "phystwin_k1000": "PhysTwin ($k{=}10^3$)",
        "phystwin": "PhysTwin", "vrgs": "GS-Verse (VR-GS)", "gaussim": "GausSim", "simplicits": "Simplicits"}
ORDER = ["ours", "phystwin", "phystwin_k1", "phystwin_k10", "phystwin_k100", "phystwin_k1000", "vrgs",
         "gaussim", "simplicits"]


def f(v, d=2, pct=False):
    if v is None or v != v:
        return "--"
    if pct:
        return f"{v:.{d}f}"
    if abs(v) >= 1e4 or (abs(v) < 1e-2 and v != 0):
        return f"{v:.1e}".replace("e+0", "e").replace("e-0", "e-")
    return f"{v:.{d}f}"


def load(d):
    R = {}
    for p in sorted(glob.glob(os.path.join(a.src, d, "*.json"))):
        R[os.path.splitext(os.path.basename(p))[0]] = json.load(open(p))
    return R


def bold_best(rows, col, lower=True):
    vals = [r[col] for r in rows if r[col] is not None and r[col] == r[col]]
    if not vals:
        return None
    return min(vals) if lower else max(vals)


def table(title, label, R, keys):
    """keys: [(표시 이름, json 키)] 순서대로 행."""
    cols = [("DoF", "dof", None), ("RMSE (\\%)", "rmse", True), ("CD (\\%)", "cd", True),
            ("EMD (\\%)", "EMD_pct", True), ("$\\min\\det F$", "det_min", False),
            ("inv.\\ (\\%)", "inv_max", True), ("$\\Delta$KE (\\%)", "KE", True),
            ("$\\Delta P$ (\\%)", "P", True), ("$\\Delta L$ (\\%)", "Lang", True),
            ("$\\Delta V$ (\\%)", "vol", True), ("PSNR", "PSNR", False), ("SSIM", "SSIM", False),
            ("LPIPS", "LPIPS", True), ("FID", "FID", True)]
    has_ip = any("IP_mean" in R[k] for _, k in keys if k in R)
    if has_ip:
        cols.insert(6, ("IP", "IP", True))
    rows = []
    for nm, k in keys:
        if k not in R:
            continue
        j = R[k]; s = j.get("summary", {}); ph = j.get("phys", {}); vi = j.get("visual", {})
        rows.append(dict(name=nm, dof=s.get("dof"), rmse=s.get("rmse"), cd=s.get("cd"),
                         EMD_pct=j.get("EMD_pct"), det_min=s.get("det_min"), inv_max=s.get("inv_max"),
                         IP=j.get("IP_mean"), KE=ph.get("KE_err_pct"), P=ph.get("P_err_pct"),
                         Lang=ph.get("Lang_err_pct"), vol=ph.get("vol_err_pct"), PSNR=vi.get("PSNR"),
                         SSIM=vi.get("SSIM"), LPIPS=vi.get("LPIPS"), FID=vi.get("FID")))
    if not rows:
        return ""
    best = {c[1]: bold_best(rows, c[1], c[2]) for c in cols if c[2] is not None}
    out = [f"\\begin{{table*}}[t]\\centering\\scriptsize\\setlength{{\\tabcolsep}}{{3pt}}",
           f"\\caption{{{title}}}\\label{{{label}}}",
           "\\begin{tabular}{l" + "r" * len(cols) + "}\\toprule",
           "Method & " + " & ".join(c[0] for c in cols) + " \\\\\\midrule"]
    for r in rows:
        cells = []
        for _, key, low in cols:
            v = r[key]
            if key == "dof":
                cells.append("--" if v is None else f"{int(v):,}".replace(",", "{,}"))
                continue
            sv = f(v, 3 if key in ("SSIM", "LPIPS") else 2)
            if low is not None and v is not None and v == v and best.get(key) is not None and v == best[key]:
                sv = "\\textbf{" + sv + "}"
            cells.append(sv)
        out.append(r["name"] + " & " + " & ".join(cells) + " \\\\")
    out += ["\\bottomrule\\end{tabular}\\end{table*}", ""]
    return "\n".join(out)


A, B = load("A"), load("B")
doc = [r"""\documentclass[10pt]{article}
\usepackage[margin=0.6in,landscape]{geometry}
\usepackage{booktabs,amsmath}
\title{Representation comparison (repflow): results}
\date{}
\begin{document}\maketitle
\section*{Protocol}
\textbf{A (L2 tracking).} Shapes wolf, bread, ship; all official-opacity-filtered 3DGS Gaussian centers
(wolf 139{,}018, bread 53{,}606, ship 200{,}000), no interior fill, no subsampling. Target: a single Gaussian
velocity field swapped every 10 frames ($\Delta t = 0.5\sigma^2/A$), 120 frames. Each representation follows the target
by minimizing per-particle L2 on its own geometry: $\Delta=-(G+\varepsilon\lambda_{\max}I)^{-1}\nabla L$ with $G$
recomputed every step and an Armijo backtracking line search. $G$: Ours -- cell-determinant log barrier;
PhysTwin -- spring elastic energy (stiffness multiplier $k$); GS-Verse -- triangle-determinant log barrier;
Simplicits -- particle-determinant log barrier; GausSim -- unconstrained Adam (lr $10^{-3}$).
$\det F$ is the analytic Jacobian of each map (cumulative from rest).
""" + (f"Code commit {a.commit_a}." if a.commit_a else "") + r"""

\textbf{B (incremental potential).} Shapes lego, mic, ficus with PhysGaussian interior fill. Each representation
replaces L2 by the simulator's incremental potential per frame (inertia + elastic/plastic energy of the
simulator's constitutive model through the representation Jacobian + gravity + floor/contact penalties),
60 frames, compared against the simulator's own trajectory. Simulators: i-PhysGaussian (viscoplastic drop),
GaussianFluent (throw onto floor, watermelon CD-MPM with its hidden defaults), Fracture-GS (two-object fast
collision; no public code, reimplemented as Collision-MPM with per-object grids and momentum-conserving
interface forces, NACC with the paper's Teapot parameters).
""" + (f"Code commit {a.commit_b}." if a.commit_b else "") + r"""

\textbf{Metrics.} RMSE and Chamfer distance (CD) over all Gaussians, divided by the fixed rest bounding-box
diagonal. EMD: exact Hungarian on the same 8{,}192 indices for every method and frame (exact EMD on all points is
intractable; the tested approximations -- Sinkhorn, multiscale Sinkhorn (KeOps), sparse $k$NN matching, GPU auction
-- deviated by 3--15\% or did not terminate). Physics: kinetic energy, linear and angular momentum and volume ratio
($\sum V\det F/\sum V$) against the reference, normalized by the reference's frame-mean magnitude.
Visual: PSNR / SSIM / LPIPS (AlexNet) per frame and FID over frames, against the reference rendered with the same
official 3DGS rasterizer and camera.
"""]
for S in ("wolf", "bread", "ship"):
    keys = [(NAME[m], f"{S}_{m}") for m in ORDER]
    doc.append(table(f"A: L2 tracking, {S}.", f"tab:A_{S}", A, keys))
SIM = {"col": "Fracture-GS (collision)", "ipg": "i-PhysGaussian (viscoplastic)", "gf": "GaussianFluent (throw)"}
for sc in ("ipg", "gf", "col"):
    for S in ("lego", "mic", "ficus"):
        keys = [(NAME[m], f"{sc}_{S}_{m}") for m in ORDER]
        doc.append(table(f"B: incremental potential, {SIM[sc]}, {S}.", f"tab:B_{sc}_{S}", B, keys))
doc.append(r"\end{document}")
os.makedirs(os.path.dirname(a.out), exist_ok=True)
open(a.out, "w").write("\n".join(doc))
print(f"[표] {a.out}  A {len(A)} 개, B {len(B)} 개")

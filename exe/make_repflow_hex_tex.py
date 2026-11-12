"""repflow A 표 (wolf·bread·ship) 를 '우리(육면체)' 행을 넣어 다시 만든다 (로컬에서 돈다, 사용자 11-12).

비교 방법·기존 우리(사면체) 값 = r10 측정 m10c 그대로 (--old: archive_20261009/r10/metrics/<형상>_<방법>.json),
우리(육면체) = repflow_hex main 측정 (--new: <형상>_hex.json). 열·서식·굵게(최선) 규칙은 exe/make_repflow_tex.py 와 같다.
기존 사면체 행은 지우지 않고 'Ours (tet, old)' 별도 행으로 남긴다. 굵게는 사면체 행을 뺀 행들 중 최선.

  python exe/make_repflow_hex_tex.py --old ~/workspace/result/anchorflow/archive_20261009/r10/metrics \
      --new /path/to/repflow_hex/main/metrics --tune tune.json --commit <커밋> --out .../repflow_results.tex
"""
import argparse
import json
import os

ap = argparse.ArgumentParser()
ap.add_argument("--old", required=True)
ap.add_argument("--new", required=True)
ap.add_argument("--tune", default="")
ap.add_argument("--commit", default="")
ap.add_argument("--edge", default="", help="칸 변 설명 (본문용)")
ap.add_argument("--out", required=True)
ap.add_argument("--md", default="", help="같은 표의 마크다운 (보고용)")
a = ap.parse_args()

ROWS = [("Ours (hex)", "new", "hex"), ("Ours (tet, old)", "old", "ours"),
        ("PhysTwin ($k{=}1$)", "old", "phystwin_k1"), ("PhysTwin ($k{=}10$)", "old", "phystwin_k10"),
        ("PhysTwin ($k{=}100$)", "old", "phystwin_k100"), ("PhysTwin ($k{=}10^3$)", "old", "phystwin_k1000"),
        ("GS-Verse (VR-GS)", "old", "vrgs"), ("GausSim", "old", "gaussim"), ("Simplicits", "old", "simplicits")]
COLS = [("DoF", "dof", None), ("RMSE (\\%)", "rmse", True), ("CD (\\%)", "cd", True),
        ("EMD (\\%)", "EMD_pct", True), ("$\\min\\det F$", "det_min", False),
        ("inv.\\ (\\%)", "inv_max", True), ("$\\Delta$KE (\\%)", "KE", True),
        ("$\\Delta P$ (\\%)", "P", True), ("$\\Delta L$ (\\%)", "Lang", True),
        ("$\\Delta V$ (\\%)", "vol", True), ("PSNR", "PSNR", False), ("SSIM", "SSIM", False),
        ("LPIPS", "LPIPS", True), ("FID", "FID", True)]


def f(v, d=2):
    if v is None or v != v:
        return "--"
    if abs(v) >= 1e4 or (abs(v) < 1e-2 and v != 0):
        return f"{v:.1e}".replace("e+0", "e").replace("e-0", "e-")
    return f"{v:.{d}f}"


def row_of(p):
    j = json.load(open(p)); s = j.get("summary", {}); ph = j.get("phys", {}); vi = j.get("visual", {})
    return dict(dof=s.get("dof"), rmse=s.get("rmse"), cd=s.get("cd"), EMD_pct=j.get("EMD_pct"),
                det_min=s.get("det_min"), inv_max=s.get("inv_max"), KE=ph.get("KE_err_pct"),
                P=ph.get("P_err_pct"), Lang=ph.get("Lang_err_pct"), vol=ph.get("vol_err_pct"),
                PSNR=vi.get("PSNR"), SSIM=vi.get("SSIM"), LPIPS=vi.get("LPIPS"), FID=vi.get("FID"))


def table(S):
    rows = []
    for nm, src, m in ROWS:
        p = os.path.join(a.new if src == "new" else a.old, f"{S}_{m}.json")
        if os.path.exists(p):
            rows.append(dict(name=nm, **row_of(p)))
    best = {}
    for _, k, low in COLS:
        if low is None:
            continue
        vals = [r[k] for r in rows if r["name"] != "Ours (tet, old)" and r[k] is not None and r[k] == r[k]]
        best[k] = (min(vals) if low else max(vals)) if vals else None
    out = ["\\begin{table*}[t]\\centering\\scriptsize\\setlength{\\tabcolsep}{3pt}",
           f"\\caption{{A: L2 tracking, {S}. Ours (hex) = quadratic B-spline hexahedral grid, metric = Gauss-point "
           f"$\\det$ log barrier$^{{\\dagger}}$; Ours (tet, old) = previous tetrahedral lattice with Gregory/radial "
           f"weights and cell-$\\det$ log barrier (kept for reference, not bolded).}}\\label{{tab:A_{S}}}",
           "\\begin{tabular}{l" + "r" * len(COLS) + "}\\toprule",
           "Method & " + " & ".join(c[0] for c in COLS) + " \\\\\\midrule"]
    md = ["| 방법 | " + " | ".join(c[1] for c in COLS) + " |", "|---" * (len(COLS) + 1) + "|"]
    for r in rows:
        cells, mcells = [], []
        for _, k, low in COLS:
            v = r[k]
            if k == "dof":
                sv = "--" if v is None else f"{int(v):,}".replace(",", "{,}")
                cells.append(sv); mcells.append(sv.replace("{,}", ","))
                continue
            sv = f(v, 3 if k in ("SSIM", "LPIPS") else 2)
            mcells.append(sv)
            if low is not None and r["name"] != "Ours (tet, old)" and v is not None and v == v and v == best.get(k):
                sv = "\\textbf{" + sv + "}"
            cells.append(sv)
        out.append(r["name"] + " & " + " & ".join(cells) + " \\\\")
        if r["name"] == "Ours (hex)":
            out.append("\\midrule")
        md.append(f"| {r['name']} | " + " | ".join(mcells) + " |")
    out += ["\\bottomrule\\end{tabular}",
            "\\par\\smallskip\\footnotesize$^{\\dagger}$ Ours (hex): $\\det F$ is barrier-controlled only at the "
            "$2{\\times}2{\\times}2$ Gauss points of each cell; it is not guaranteed positive at every point inside a cell "
            "(the $\\min\\det F$ / inv.\\ columns are measured at the Gaussians through the analytic Jacobian).",
            "\\end{table*}", ""]
    return "\n".join(out), "\n".join(md)


tune = json.load(open(a.tune)) if a.tune else None
tl = (f" Ours (hex) $\\varepsilon={tune['best']['eps']}$, $\\eta={tune['best']['eta']}$ (chosen by the same rule as the "
      f"other methods: lowest mean RMSE over the first 3 wolf frames).") if tune else ""
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
recomputed every step and an Armijo backtracking line search. $G$: Ours (hex) -- Gauss-point determinant log barrier;
Ours (tet, old) -- cell-determinant log barrier;
PhysTwin -- spring elastic energy (stiffness multiplier $k$); GS-Verse -- triangle-determinant log barrier;
Simplicits -- particle-determinant log barrier; GausSim -- unconstrained Adam (lr $10^{-3}$).
$\det F$ is the analytic Jacobian of each map (cumulative from rest).
Code commit 224443c (all methods except Ours (hex)).

\textbf{Ours (hex).} Quadratic B-spline hexahedral grid, the same grid and cell-size rule as the real-time solver
(cube cell edge $=$ cell $\times$ scene MPM $dx$, lattice anchored at the simulation origin""" + (", " + a.edge if a.edge else "") + r"""),
re-laid on the previous positions every frame; the unknowns are the node positions and particles follow the
B-spline-weighted interpolation $y=\sum_i w_i(X)\,x_i$, $F=I+\sum_i u_i\otimes\nabla w_i$. The metric is the
log barrier of $\det F$ at the $2{\times}2{\times}2$ Gauss points of every occupied cell (in place of the cell-determinant
log barrier); everything else (shapes, frames, field swaps, optimizer, metrics) is unchanged.""" + tl +
       (f" Code commit {a.commit}." if a.commit else "") + r"""

\textbf{Metrics.} RMSE and Chamfer distance (CD) over all Gaussians, divided by the fixed rest bounding-box
diagonal. EMD: exact Hungarian on the same 8{,}192 indices for every method and frame (exact EMD on all points is
intractable; the tested approximations -- Sinkhorn, multiscale Sinkhorn (KeOps), sparse $k$NN matching, GPU auction
-- deviated by 3--15\% or did not terminate). Physics: kinetic energy, linear and angular momentum and volume ratio
($\sum V\det F/\sum V$) against the reference, normalized by the reference's frame-mean magnitude.
Visual: PSNR / SSIM / LPIPS (AlexNet) per frame and FID over frames, against the reference rendered with the same
official 3DGS rasterizer and camera.
"""]
MD = []
for S in ("wolf", "bread", "ship"):
    t_, m_ = table(S)
    doc.append(t_); MD.append(f"### {S}\n\n" + m_ + "\n")
doc.append(r"\end{document}")
open(a.out, "w").write("\n".join(doc))
if a.md:
    open(a.md, "w").write("\n".join(MD))
print(f"[표] {a.out}")

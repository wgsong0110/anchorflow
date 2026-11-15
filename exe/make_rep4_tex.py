"""rep4 (11-15): 표현력 실험 (A, 정답 궤적을 L2 로 따라가기) 표 -- L2 오차 (RMSE/L, %) 만 (사용자 11-15).
  python exe/make_rep4_tex.py --old repflow_results.tex --ns chair_paste,<새 형상> --out repflow_results.tex --preamble P.tex
wolf·bread: 기존 표 (--old) 의 RMSE 열 (같은 실행·같은 값; 우리 판 = 기존 hex 그대로).
새 형상 장면: R2 lbours/rep4/<장면>/<판>/<판>_<장면>.log 의 [요약] RMSE (프레임 평균 RMSE / L, rep_track_sg).
방법 = 우리 (hex), PhysTwin (k 1·10·100·1000), VR-GS, Spring-Gaus. (GS-Verse·옛 사면체 판·Simplicits·GausSim·GaussianFluent 장면은 넣지 않는다.)
굵게 = 장면별 최소."""
import argparse, re, subprocess

ap = argparse.ArgumentParser(); ap.add_argument("--old", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--ns", default=""); ap.add_argument("--preamble", required=True)
a = ap.parse_args()
LAB = [("hex", "Ours (hex)"), ("pt1", "PhysTwin ($k{=}1$)"), ("pt10", "PhysTwin ($k{=}10$)"), ("pt100", "PhysTwin ($k{=}100$)"),
       ("pt1000", "PhysTwin ($k{=}10^3$)"), ("vrgsx", "VR-GS"), ("sg", "Spring-Gaus")]
HEAD = {"wolf": "wolf", "bread": "bread", "chair_paste": "chair (paste)", "chair_plast": "chair (plasticine)",
        "drums_honey": "drums (honey)", "materials_sand": "materials (sand)"}
R4 = "r2:storage/result/anchorflow/lbours/rep4"


def old_rmse(scene):
    s = open(a.old).read()
    m = re.search(r"\\caption\{A: L2 tracking, %s[ .(].*?\\end\{tabular\}" % scene, s, re.S)
    out = {}
    for l in m.group(0).split("\n"):
        if l.count("&") >= 3 and not l.startswith("Method"):
            c = [re.sub(r"\\textbf\{(.*)\}", r"\1", q.strip()) for q in l.rstrip().rstrip("\\").rstrip().split("&")]
            out[c[0]] = float(c[2])
    return out


def ns_rmse(scene, key):
    try:
        log = subprocess.run(["rclone", "cat", f"{R4}/{scene}/{key}/{key}_{scene}.log"], capture_output=True, text=True, timeout=120).stdout
    except Exception:
        return float("nan")
    m = re.findall(r"\[요약\].*?RMSE ([0-9.]+)%", log)
    return float(m[-1]) if m else float("nan")


scenes = ["wolf", "bread"] + [q for q in a.ns.split(",") if q]
V = {}
for sc in scenes:
    if sc in ("wolf", "bread"):
        o = old_rmse(sc); V[sc] = {k: o[lab] for k, lab in LAB}
    else:
        V[sc] = {k: ns_rmse(sc, k) for k, _ in LAB}
best = {sc: min([v for v in V[sc].values() if v == v] or [float("nan")]) for sc in scenes}
rows = []
for k, lab in LAB:
    cells = []
    for sc in scenes:
        x = V[sc][k]; t = "--" if x != x else f"{x:.2f}"
        if x == x and abs(x - best[sc]) < 1e-12: t = "\\textbf{%s}" % t
        cells.append(t)
    rows.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    if k == "hex": rows.append("\\midrule")
tab = ["", "\\begin{table}[t]\\centering\\small",
       "\\caption{A: L2 tracking of the reference trajectory. Frame-mean RMSE over all points divided by the rest bounding-box diagonal $L$ (\\%, lower is better).}\\label{tab:A_l2}",
       "\\begin{tabular}{l" + "r" * len(scenes) + "}\\toprule",
       "Method & " + " & ".join(HEAD.get(sc, sc.replace("_", " ")) for sc in scenes) + " \\\\\\midrule"] + rows + ["\\bottomrule\\end{tabular}", "\\end{table}"]
s = open(a.old).read()
head = s[:s.index("\\section*{Protocol}")]
open(a.out, "w").write(head + open(a.preamble).read() + "\n".join(tab) + "\n\n\\end{document}\n")
print("[표]", {sc: V[sc] for sc in scenes})

"""rep4 (11-15): 표현력 실험 (A, L2 추적) 표를 장면 4 개 (wolf·bread·kiwi·lollipop) 로 다시 쓴다.
  python exe/make_rep4_tex.py --old repflow_results.tex --res DIR --out repflow_results.tex
DIR/<장면>/ 아래 json:
  wolf·bread  hex_<장면>.json (rep_metrics, 새 hex) -- 나머지 방법 행은 --old 표에서 그대로 읽는다 (같은 실행·같은 값)
  kiwi·lollipop <방법>_<장면>.json (rep_metrics_gf), <방법>_<장면>_visual.json (gfs_post vid), <방법>_<장면>_topo.json (rep_topo_gf)
방법 = 우리 (hex), PhysTwin (k 1·10·100·1000), GS-Verse, VR-GS, Spring-Gaus. (옛 사면체 판·Simplicits·GausSim 은 넣지 않는다.)
굵게 = 열별 최고 (RMSE·CD·EMD·ΔKE·ΔP·ΔL·ΔV·LPIPS·이탈률 최소, min det·PSNR·SSIM 최대, 뒤집힘 최소, 조각 수는 정답에 가장 가까운 것)."""
import argparse, json, os, re

ap = argparse.ArgumentParser(); ap.add_argument("--old", required=True); ap.add_argument("--res", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--preamble", required=True, help="표 앞에 넣을 설명 (LaTeX 조각 파일)")
a = ap.parse_args()
LAB = [("hex", "Ours (hex)"), ("pt1", "PhysTwin ($k{=}1$)"), ("pt10", "PhysTwin ($k{=}10$)"), ("pt100", "PhysTwin ($k{=}100$)"),
       ("pt1000", "PhysTwin ($k{=}10^3$)"), ("gsverse", "GS-Verse"), ("vrgsx", "VR-GS"), ("sg", "Spring-Gaus")]
NAMES = {"wolf": "wolf (PhysGaussian, Gaussian velocity field)", "bread": "bread (PhysGaussian, Gaussian velocity field)",
         "kiwi": "kiwi (GaussianFluent CD-MPM trajectory)", "lollipop": "lollipop (GaussianFluent CD-MPM trajectory)"}
NBASE = 12          # RMSE CD EMD detmin inv KE P L V PSNR SSIM LPIPS


def fnum(x, p=2):
    if x != x: return "--"
    if abs(x) >= 1e4 or (0 < abs(x) < 1e-2): return f"{x:.1e}".replace("e-0", "e-").replace("e+0", "e+")
    return f"{x:.{p}f}"


def J(p):
    return json.load(open(p)) if os.path.exists(p) else None


def vals_pg(d):
    s, p, v = d["summary"], d["phys"], d.get("visual", {})
    return int(round(s["dof"])), [s["rmse"], s["cd"], d["EMD_pct"], s["det_min"], s["inv_max"], p["KE_err_pct"], p["P_err_pct"],
                                  p["Lang_err_pct"], p["vol_err_pct"], v.get("PSNR", float("nan")), v.get("SSIM", float("nan")), v.get("LPIPS", float("nan"))]


def old_rows(scene):
    """기존 표 (--old) 에서 scene 표의 행 -> {표시 이름: (DoF, 12 값)} (FID 열 버림)."""
    s = open(a.old).read()
    m = re.search(r"\\caption\{A: L2 tracking, %s\..*?\\end\{tabular\}" % scene, s, re.S)
    out = {}
    for l in m.group(0).split("\n"):
        if l.count("&") == 14 and not l.startswith("Method"):
            c = [re.sub(r"\\textbf\{(.*)\}", r"\1", q.strip()) for q in l.rstrip().rstrip("\\").rstrip().split("&")]
            out[c[0]] = (int(c[1].replace("{,}", "")), [float(q.replace("--", "nan")) for q in c[2:2 + NBASE]])
    return out


RULE = {0: "min", 1: "min", 2: "min", 3: "max", 4: "min", 5: "min", 6: "min", 7: "min", 8: "min", 9: "max", 10: "max", 11: "min", 12: "min", 13: "near"}


def table(scene, rows, gt):
    gf = gt is not None; gt_topo = gt["pieces_last"] if gf else None
    best = {}
    for c, rl in RULE.items():
        vs = [r[2][c] for r in rows if c < len(r[2]) and r[2][c] == r[2][c]]
        if not vs: continue
        best[c] = (max(vs) if rl == "max" else min(vs, key=lambda x: abs(x - gt_topo)) if rl == "near" else min(vs))
    lines = []
    for lab, dof, v in rows:
        f = []
        for i, x in enumerate(v):
            t = fnum(x, 3) if i in (10, 11) else (str(int(x)) if i == 13 and x == x else fnum(x))
            if i in best and x == x and abs(x - best[i]) <= 1e-12 * max(1, abs(best[i])):
                t = "\\textbf{%s}" % t
            f.append(t)
        lines.append(f"{lab} & {dof:,}".replace(",", "{,}") + " & " + " & ".join(f) + " \\\\")
        if lab.startswith("Ours"):
            lines.append("\\midrule")
    cols = "l" + "r" * (1 + NBASE + (2 if gf else 0))
    hdr = ("Method & DoF & RMSE (\\%) & CD (\\%) & EMD (\\%) & $\\min\\det F$ & inv.\\ (\\%) & $\\Delta$KE (\\%) & $\\Delta P$ (\\%) & "
           "$\\Delta L$ (\\%) & $\\Delta V$ (\\%) & PSNR & SSIM & LPIPS" + (" & esc.\\ (\\%) & pieces" if gf else "") + " \\\\\\midrule")
    cap = f"A: L2 tracking, {NAMES[scene]}."
    if gf:
        cap += (f" esc.\\ = neighbor-escape rate (max over frames; share of the 16 initial nearest neighbors that moved beyond 3$\\times$ the initial distance, "
                f"4{{,}}000 fixed samples), pieces = connected pieces (${{\\ge}}1\\%$ of samples) at the last frame; reference: esc.\\ {100 * gt['esc_max']:.2f}\\%, pieces {gt_topo}. "
                "Bold for pieces = closest to the reference.")
    return ["", "\\begin{table*}[t]\\centering\\scriptsize\\setlength{\\tabcolsep}{3pt}", f"\\caption{{{cap}}}\\label{{tab:A_{scene}}}",
            f"\\begin{{tabular}}{{{cols}}}\\toprule", hdr] + lines + ["\\bottomrule\\end{tabular}", "\\end{table*}"]


out = []
for sc in ("wolf", "bread"):
    old = old_rows(sc); rows = []
    for key, lab in LAB:
        if key == "hex":
            d = J(os.path.join(a.res, sc, f"hex_{sc}.json")); assert d, f"{sc} hex json 없음"
            dof, v = vals_pg(d)
        else:
            dof, v = old[lab]
        rows.append((lab, dof, v))
    out += table(sc, rows, None)
for sc in ("kiwi", "lollipop"):
    rows = []; gt = None
    for key, lab in LAB:
        b = os.path.join(a.res, sc, f"{key}_{sc}")
        d, vi, tp = J(b + ".json"), J(b + "_visual.json"), J(b + "_topo.json")
        assert d and vi and tp, f"{b} json 없음"
        dof, v = vals_pg(d)
        v[9:12] = [vi["psnr"], vi["ssim"], vi["lpips"]]
        v += [100 * tp["ours"]["esc_max"], float(tp["ours"]["pieces_last"])]
        gt = tp["gt"]; rows.append((lab, dof, v))
    out += table(sc, rows, gt)
s = open(a.old).read()
head = s[:s.index("\\section*{Protocol}")]
body = open(a.preamble).read()
open(a.out, "w").write(head + body + "\n".join(out) + "\n\n\\end{document}\n")
print("[표] 장면 4 개 ->", a.out)

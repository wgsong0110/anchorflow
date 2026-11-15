"""repflow A 표 갱신 (사용자 11-15): repflow_results.tex 의 wolf·bread 표에 Spring-Gaus 행 추가, ship 표 삭제, PG 장면 표 추가 (ficus 등).
  python exe/make_repflow1115_tex.py --tex repflow_results.tex --res DIR   (DIR/<장면>/<이름>.json = rep_metrics 출력, wolf·bread 의 Spring-Gaus 도 같은 모양)
굵게 = 옛 사면체 (old) 행을 뺀 열별 최고 (RMSE·CD·EMD·ΔKE·ΔP·ΔL·ΔV·LPIPS·FID 최소, min det·PSNR·SSIM 최대, 뒤집힘 최소)."""
import argparse, glob, json, os, re

ap = argparse.ArgumentParser(); ap.add_argument("--tex", required=True); ap.add_argument("--res", required=True)
ap.add_argument("--add", default="", help="새 표를 만들 장면 (쉼표)"); ap.add_argument("--drop_ship", action="store_true")
a = ap.parse_args()
LAB = [("hex", "Ours (hex)"), ("pt1", "PhysTwin ($k{=}1$)"), ("pt10", "PhysTwin ($k{=}10$)"), ("pt100", "PhysTwin ($k{=}100$)"),
       ("pt1000", "PhysTwin ($k{=}10^3$)"), ("gsverse", "GS-Verse"), ("vrgsx", "VR-GS"), ("sg", "Spring-Gaus")]
def fnum(x, p=2):
    if x != x: return "--"
    if abs(x) >= 1e4 or (0 < abs(x) < 1e-2): return f"{x:.1e}".replace("e-0", "e-").replace("e+0", "e+")
    return f"{x:.{p}f}"
def row(label, d):
    s, p, v = d["summary"], d["phys"], d.get("visual", {})
    dof = f"{int(round(s['dof'])):,}".replace(",", "{,}")
    vals = [s["rmse"], s["cd"], d["EMD_pct"], s["det_min"], s["inv_max"], p["KE_err_pct"], p["P_err_pct"], p["Lang_err_pct"], p["vol_err_pct"],
            v.get("PSNR", float("nan")), v.get("SSIM", float("nan")), v.get("LPIPS", float("nan")), v.get("FID", float("nan"))]
    return label, dof, vals
def fmt_vals(vals):
    out = []
    for i, x in enumerate(vals):
        out.append(fnum(x, 3) if i in (10, 11) else fnum(x))
    return out
RULE = {0: "min", 1: "min", 2: "min", 3: "max", 4: "min", 5: "min", 6: "min", 7: "min", 8: "min", 9: "max", 10: "max", 11: "min", 12: "min"}
def bold(rows):
    """rows: [(label, dof, vals)] -> 굵게 표시를 넣은 문자열 목록. old 행은 제외."""
    elig = [r for r in rows if "old" not in r[0]]
    best = {}
    for c, rl in RULE.items():
        vs = [r[2][c] for r in elig if r[2][c] == r[2][c]]
        best[c] = (max if rl == "max" else min)(vs) if vs else None
    lines = []
    for r in rows:
        f = fmt_vals(r[2])
        for c in RULE:
            if "old" not in r[0] and best[c] is not None and r[2][c] == r[2][c] and abs(r[2][c] - best[c]) <= 1e-12 * max(1, abs(best[c])):
                f[c] = "\\textbf{%s}" % f[c]
        lines.append(f"{r[0]} & {r[1]} & " + " & ".join(f) + " \\\\")
    return lines
def load(scene, name):
    p = os.path.join(a.res, scene, name + ".json")
    return json.load(open(p)) if os.path.exists(p) else None
s = open(a.tex).read().split("\n")
# 기존 표의 행 -> 값 (wolf·bread)
def parse_rows(lines):
    rows = []
    for l in lines:
        if l.count("&") == 14 and not l.startswith("Method"):
            c = [re.sub(r"\\textbf\{(.*)\}", r"\1", q.strip()) for q in l.rstrip().rstrip("\\").rstrip().split("&")]
            def num(x):
                x = x.replace("{,}", "").replace("--", "nan")
                return float(x)
            rows.append((c[0], c[1], [num(q) for q in c[2:]]))
    return rows
out = []; i = 0
while i < len(s):
    l = s[i]
    if l.startswith("\\begin{table"):
        j = i
        while not s[j].startswith("\\end{table"): j += 1
        blk = s[i:j + 1]
        cap = next((q for q in blk if q.startswith("\\caption")), "")
        scene = next((k for k in ("wolf", "bread", "ship") if f"tracking, {k}." in cap), None)
        if scene == "ship" and a.drop_ship:
            i = j + 1; continue
        if scene in ("wolf", "bread") and load(scene, "sg_" + scene):
            rows = parse_rows(blk)
            rows = [r for r in rows if not r[0].startswith("Spring-Gaus")]
            rows.append(row("Spring-Gaus", load(scene, "sg_" + scene)))
            new = bold(rows)
            k0 = next(k for k, q in enumerate(blk) if q.count("&") == 14 and not q.startswith("Method"))
            k1 = max(k for k, q in enumerate(blk) if q.count("&") == 14 and not q.startswith("Method"))
            # old 행 앞의 구분선 (\midrule) 은 그대로 둔다: 행 순서를 (hex, \midrule, old, 나머지) 로 다시 씀
            hexr = [q for q in new if q.startswith("Ours (hex)")]; oldr = [q for q in new if q.startswith("Ours (tet, old)")]
            rest = [q for q in new if not q.startswith("Ours (")]
            blk = blk[:k0] + hexr + ["\\midrule"] + oldr + rest + blk[k1 + 1:]
            blk = [q for q in blk if q != "\\midrule" or True]
        out += blk; i = j + 1; continue
    out.append(l); i += 1
# 새 장면 표 (PG 장면): 표 바로 앞 (\end{document}) 에 붙인다
HDR = ("\\begin{tabular}{lrrrrrrrrrrrrrr}\\toprule\nMethod & DoF & RMSE (\\%) & CD (\\%) & EMD (\\%) & $\\min\\det F$ & inv.\\ (\\%) & $\\Delta$KE (\\%) & $\\Delta P$ (\\%) & "
       "$\\Delta L$ (\\%) & $\\Delta V$ (\\%) & PSNR & SSIM & LPIPS & FID \\\\\\midrule")
NAMES = {"ficus": "ficus (elastic)", "plane": "plane (metal plasticity)", "pineple": "pineapple (NACC)", "milk": "milk (fluid)", "sand_castle": "sand castle (sand collapse)"}
add = [q for q in a.add.split(",") if q]
tabs = []
for sc in add:
    rows = []
    for key, lab in LAB:
        d = load(sc, f"{key}_{sc}" if key != "sg" else f"sg_{sc}")
        if d: rows.append(row(lab, d))
    if not rows: continue
    body = bold(rows)
    tabs += ["", "\\begin{table*}[t]\\centering\\scriptsize\\setlength{\\tabcolsep}{3pt}",
             f"\\caption{{A: L2 tracking, {NAMES.get(sc, sc)}. Ours (hex) = quadratic B-spline hexahedral grid, metric = Gauss-point $\\det$ log barrier$^{{\\dagger}}$; all Gaussians kept (no subsampling).}}\\label{{tab:A_{sc}}}",
             HDR] + body + ["\\bottomrule\\end{tabular}", "\\end{table*}"]
k = max(q for q, l in enumerate(out) if l.startswith("\\end{document}"))
# 같은 장면 표가 이미 있으면 (이전 실행) 그대로 두고 새로 만든 것만 붙인다
out = out[:k] + tabs + [""] + out[k:]
open(a.tex, "w").write("\n".join(out))
print("[표] 장면", add, "행", [len(t) for t in [tabs]])

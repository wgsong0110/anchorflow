"""[paper_remeasure 11-15] 지표 비교표 LaTeX: 베이스라인 = 옛 R2 lbours/stab/stab.json (A6000 단독, 우리 판과 무관), 우리 = 새 stab.json (현재 기본 판).
사용: python exe/pr_table.py <old stab.json> <new stab.json> <출력 tex>"""
import json
import sys

old, new = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
ROWS = [("mpm", "정답 MPM"), ("vrgs", "VR-GS"), ("bnv", "명시적 B-스플라인"), ("phystwin", "PhysTwin"), ("simplicits", "Simplicits"), ("springgaus", "Spring-Gaus"), ("ours", "우리 (현재 기본 판)")]
ELASTIC = ("L000", "ficus")
SC = [("L000", "레고"), ("ficus", "ficus"), ("wolf", "wolf"), ("bread", "bread")]


def f(x, k=2):
    return "" if x is None else f"%.{k}f" % x


T = ["\\begin{center}\\footnotesize\\begin{tabular}{@{}llrlrrrrr@{}}", "\\toprule", "씬 & 방법 & 최소 $N$ & 발산 쪽 경계 & 오차 \\% & ms (최대) & PSNR & SSIM & LPIPS \\\\ \\midrule"]
for j, (t, tn) in enumerate(SC):
    if j:
        T.append("\\midrule")
    for m, nm in ROWS:
        c = [tn, nm]
        q = (new if m == "ours" else old).get(f"{m}_{t}")
        if t not in ELASTIC and m != "mpm" and m != "ours":
            c += ["\\multicolumn{7}{l}{해당 없음 (탄성 전용)}"]
        elif q is None:
            c += ["\\multicolumn{7}{l}{측정 안 함}"]
        elif q.get("nmin") is None:
            c += ["\\multicolumn{7}{l}{" + str(q.get("state", "")).replace("_", "\\_").replace("%", "\\%") + "}"]
        else:
            ns, lo = q["nmin"], q.get("lo")
            b = q["probes"][str(ns)] if str(ns) in q["probes"] else q["probes"][ns]
            pl = q["probes"].get(str(lo)) if lo is not None else None
            bd = "--" if lo is None else (f"{lo} ($\\times${pl['div']})" if pl and pl["stable"] is False else f"{lo} (중단)")
            tm, v = q.get("time") or {}, q.get("vis") or {}
            ms = "" if tm.get("ms") is None else f"{tm['ms']:.2f} ({tm['ms_max']:.2f})" + ("$^\\ddagger$" if tm.get("part") else "")
            dag = "$^\\dagger$" if q.get("mono") and any(s_ is False for _, s_ in q["mono"]) else ""
            c += [str(ns) + dag, bd, f(b["err"]), ms, f(v.get("psnr")), f(v.get("ssim"), 3), f(v.get("lpips"), 3)]
        T.append(" & ".join(c) + " \\\\")
T += ["\\bottomrule", "\\end{tabular}\\end{center}"]
open(sys.argv[3], "w").write("\n".join(T) + "\n")
print("저장", sys.argv[3])

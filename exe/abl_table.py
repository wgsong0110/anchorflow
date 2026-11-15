#!/usr/bin/env python3
"""절 CM183 표·그림 생성: R2 lbours/ablation/runs/*/metrics.json (판당 3 회) -> 중앙값 표 (ablation_1115_tables.tex) + 그림 (fig/abl_*.png).
사용: python exe/abl_table.py <논문 폴더 /home/wgsong/workspace/writing/anchorflow>"""
import json, os, statistics as S, subprocess, sys, glob
OUT = sys.argv[1]; CACHE = "/tmp/ablruns"
subprocess.run(["rclone", "copy", "r2:storage/result/anchorflow/lbours/ablation/runs", CACHE, "--include", "*/metrics.json", "-q"], check=False)
R = {}
for f in glob.glob(f"{CACHE}/*/metrics.json"):
    m = json.load(open(f))
    if m.get("scene") is None or m.get("frames_done") is None:
        continue
    R.setdefault((m["scene"], m["var"]), []).append(m)
def med(x):
    x = [v for v in x if v is not None]
    return S.median(x) if x else None
def cell(sc, v, bc=False):
    rs = R.get((sc, "defb" if (bc and v == "def" and (sc, "defb") in R) else v), [])
    if not rs:
        return None
    ok = [r for r in rs if r.get("diverged") is None and r["frames_done"] == r["frames"]]
    d = dict(n=len(rs), ok=len(ok), err=med([r["err"] for r in ok]), ms=med([r["ms"] for r in ok]),
             lo=min([r["err"] for r in ok], default=None), hi=max([r["err"] for r in ok], default=None),
             div=[r["diverged"] for r in rs if r.get("diverged") is not None], flip=max([r.get("nJneg_max") or 0 for r in rs]),
             fdet=max([r.get("pF_det_le0_max") or 0 for r in rs]),
             erise=max([(r.get("energy") or {}).get("rise_rel") or 0 for r in rs]) if any(r.get("energy") for r in rs) else None,
             efin=med([ (r["energy"]["Efinal"] / r["energy"]["E0"]) if r.get("energy") and r["energy"].get("E0") else None for r in rs]),
             eig=next((r["eig"]["n_zero"] for r in rs if isinstance(r.get("eig"), dict)), None))
    return d
def fe(d):
    if d is None: return "--"
    if d["ok"] == 0: return "발산 %s" % (",".join("f%s" % x for x in d["div"][:1]) or "실패")
    s = "%.2f" % (100 * d["err"])
    if d["ok"] < d["n"]: s += "$^{%d/%d}$" % (d["ok"], d["n"])
    return s
def fm(d): return "--" if d is None or d["ms"] is None else "%.1f" % d["ms"]
def tex_table(title, scenes, vars_, names, extra_eig=True, bc=False):
    cols = "l" + "rr" * len(scenes)
    t = ["\\begin{center}\\small\\begin{tabular}{%s}\\toprule" % cols,
         "판 & " + " & ".join("\\multicolumn{2}{c}{%s}" % s.replace("_", "\\_") for s in scenes) + "\\\\",
         " & " + " & ".join("오차 \\% & ms" for _ in scenes) + "\\\\\\midrule"]
    for v in vars_:
        t.append(names.get(v, v) + " & " + " & ".join("%s & %s" % (fe(cell(s, v, bc)), fm(cell(s, v, bc))) for s in scenes) + "\\\\")
    t.append("\\bottomrule\\end{tabular}\\end{center}")
    return "\n".join(t)
def tex_detail(scenes, vars_, names):
    t = ["\\begin{center}\\scriptsize\\begin{tabular}{llrrrrr}\\toprule", "장면 & 판 & 완주/회 & 뒤집힘 (입자 $\\det F\\le0$ 최대) & 에너지 최대 증가 (규모 대비) & 마지막/처음 에너지 & 영 모드 수 \\\\\\midrule"]
    for s in scenes:
        for v in vars_:
            d = cell(s, v)
            if d is None: continue
            t.append("%s & %s & %d/%d & %d (%d) & %s & %s & %s\\\\" % (s.replace("_", "\\_"), names.get(v, v), d["ok"], d["n"], d["flip"], d["fdet"],
                     "--" if d["erise"] is None else "%.3g" % d["erise"], "--" if d["efin"] is None else "%.3g" % d["efin"], "--" if d["eig"] is None else d["eig"]))
    t.append("\\bottomrule\\end{tabular}\\end{center}")
    return "\n".join(t)
N = {"def": "기본 (2$^3$ 점, HHT $\\rho_\\infty{=}0.8$)", "q1": "가우스 1$^3$ 점", "q27": "가우스 3$^3$ 점 (27)", "q64": "가우스 4$^3$ 점 (64)",
     "rho0": "$\\rho_\\infty{=}0$ (일반화-$\\alpha$)", "rho1": "$\\rho_\\infty{=}1$ (중점)", "c075": "칸 $\\times$0.75", "c15": "칸 $\\times$1.5", "c20": "칸 $\\times$2"}
SA = ["ficus", "pillow2sofa", "bread", "toast", "sand_castle"]; SB = ["ficus", "bread", "wolf", "pillow2sofa"]; SD = ["ficus", "vasedeck", "plane", "wolf", "bread", "pillow2sofa"]
out = []
out.append("% 자동 생성 (exe/abl_table.py)")
out.append("\\subsection*{A. 설계 선택 ablation}\n" + tex_table("A", SA, ["def", "q1", "rho0", "rho1"], N) + "\n" + tex_detail(SA, ["def", "q1", "rho0", "rho1"], N))
out.append("\\subsection*{B. 가우스점 수}\n" + tex_table("B", SB, ["q1", "def", "q27", "q64"], N, bc=True) + "\n" + tex_detail(SB, ["q1", "def", "q27", "q64"], N))
out.append("\\subsection*{C. 칸 크기}\n" + tex_table("C", SB, ["c075", "def", "c15", "c20"], N, bc=True) + "\n" + tex_detail(SB, ["c075", "def", "c15", "c20"], N))
out.append("\\subsection*{D. 재질 일반성}\n" + tex_table("D", SD, ["def"], N))
open(f"{OUT}/ablation_1115_tables.tex", "w").write("\n\n".join(out))
# 그림: B (가우스점 수 대 오차) · C (칸 배수 대 오차·시간)
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib import font_manager as _fm; _fm.fontManager.addfont(os.path.expanduser("~/.fonts/NotoSansCJKkr-Regular.otf")); plt.rcParams["font.family"] = "Noto Sans CJK KR"
    os.makedirs(f"{OUT}/fig", exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    for s in SB:
        xs, ys = [], []
        for k, v in ((1, "q1"), (8, "def"), (27, "q27"), (64, "q64")):
            d = cell(s, v)
            if d and d["err"] is not None: xs.append(k); ys.append(100 * d["err"])
        ax[0].plot(xs, ys, "o-", label=s)
    ax[0].set_xscale("log"); ax[0].set_xlabel("칸당 가우스 점 수"); ax[0].set_ylabel("오차 (%)"); ax[0].legend(fontsize=7)
    for s in SB:
        xs, ys = [], []
        for k, v in ((0.75, "c075"), (1, "def"), (1.5, "c15"), (2, "c20")):
            d = cell(s, v)
            if d and d["err"] is not None: xs.append(k); ys.append(100 * d["err"])
        ax[1].plot(xs, ys, "o-", label=s)
    ax[1].set_xlabel("칸 크기 (기본의 배수)"); ax[1].set_ylabel("오차 (%)"); ax[1].legend(fontsize=7)
    plt.tight_layout(); plt.savefig(f"{OUT}/fig/abl_B_C.png", dpi=160)
except Exception as e:
    print("그림 실패", e)
print("표 쓴 위치", f"{OUT}/ablation_1115_tables.tex", "셀", len(R))

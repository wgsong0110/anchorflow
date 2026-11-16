#!/usr/bin/env python3
"""절 CM192: 정답 없는 지표로 파레토 세로축 고르기 -- 지표마다 '세로축 = 지표, 가로축 = A6000 단독 시간 (로그)' 그림 + 고르기 비교 표.

입력 (로컬 R2 FUSE ~/r2/result/anchorflow/lbours):
  점 (방법, 장면, ms, 정답 대비 오차) = <캐시>/fpa_summary.json 의 pareto 행 (exe/fpa_table.py 가 만든 것; 우리 K=1·2·4, 정답 MPM, 베이스라인)
  지표 = 방법·장면마다 phys.json·vis.json 폴더 (아래 DIRS 규칙; 우리 3 회 판은 노이즈용)
출력: <논문 폴더>/fig/pareto_rf_<지표>.png, <논문 폴더>/pareto_final_1115_rf_choice.tex, <캐시>/rf_choice.json
  python exe/ffx_rf_choice.py <캐시> <논문 폴더>
"""
import glob, json, math, os, statistics as S, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

C, OUT = sys.argv[1:3]
LB = os.path.expanduser("~/r2/result/anchorflow/lbours")
PH = os.environ.get("RF_PHYS", "/tmp/cm192/physm")                                  # exe/ffx_phys_merge.py 가 만든 모음 (VR-GS 레고·pillow2sofa = 반복 1, 우리 레고·pillow2sofa = CM192 판)
SC = [("L000", "레고"), ("pillow2sofa", "pillow2sofa"), ("bread", "bread"), ("wolf", "wolf")]
# 방법 순서 = 색 칸 순서 (고정, 순위로 바꾸지 않음)
METH = [("ours", "우리"), ("mpm", "정답 MPM (PG)"), ("ipg", "i-PG"), ("omni", "OmniPhysGS"), ("phystwin", "PhysTwin"), ("vrgs", "VR-GS"), ("springgaus", "Spring-Gaus")]
COL = dict(ours="#2a78d6", mpm="#eb6834", ipg="#1baf7a", omni="#eda100", phystwin="#e87ba4", vrgs="#008300", springgaus="#6250d6")
MK = dict(ours="o", mpm="*", ipg="s", omni="D", phystwin="^", vrgs="v", springgaus="P")
# 지표: (키, 출처 phys|vis, 이름, 낮을수록 좋음, 변환)
MET = [("detneg_max", "phys", "뒤집힘 (det F≤0 비율 최대)", True), ("pen_depth_max", "phys", "바닥 관통 깊이 / L", True),
       ("e_drift_max", "phys", "에너지 증가 (최대, E0 대비)", True), ("freefall_mom_err", "phys", "자유낙하 운동량 오차", True),
       ("jitter_d2", "phys", "떨림 (2 차 차분 / L)", True), ("jitter_hf", "phys", "떨림 (고주파 에너지 비)", True),
       ("dV_absmax", "phys", "부피 변화 |dV| 최대", True),
       ("niqe", "vis", "NIQE", True), ("brisque", "vis", "BRISQUE", True), ("clipiqa", "vis", "CLIP-IQA", False), ("musiq", "vis", "MUSIQ", False),
       ("warp_err", "vis", "깜빡임 (warping error)", True)]

for f in ("Noto Sans CJK KR", "Noto Serif CJK KR", "NanumGothic"):
    if any(f in x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = f; break
plt.rcParams["axes.unicode_minus"] = False

P = json.load(open(f"{C}/fpa_summary.json"))["pareto"]
for extra in (f"{C}/pareto_extra_rows.json",):                                       # i-PG·OmniPhysGS·재질 확장 등 (표 생성기가 합친 행)
    if os.path.exists(extra):
        P += json.load(open(extra))


def dirs(m, s, N, rep=1):
    """지표 폴더 후보 (있는 첫 것). 우리 = ours_<장면>_<판> (rep 2·3 = RF_OURS_REP 폴더), 베이스라인 = <방법>_<장면>_n<N>[it1]."""
    if m == "ours":
        v = {1: "def", 2: "k2", 4: "k4"}[int(N)]
        c = [f"{PH}/ours_{s}_{v}" + ("" if rep == 1 else f"_r{rep}")]
    elif m == "mpm":
        c = [f"{PH}/mpm_{s}"]
    else:
        c = [f"{PH}/{m}_{s}_n{N}", f"{PH}/{m}_{s}_nsg"] + sorted(glob.glob(f"{PH}/{m}_{s}*"))
    for d in c:
        if os.path.exists(f"{d}/phys.json") or os.path.exists(f"{d}/vis.json"):
            return d
    return None


def load(d, src):
    f = f"{d}/{src}.json"
    return json.load(open(f)) if d and os.path.exists(f) else {}


pts = []
for r in P:
    m, s = r["method"], r["scene"]
    if r.get("err") is None or not r.get("ms") or m not in COL:
        continue
    d = dirs(m, s, r["N"])
    pts.append(dict(method=m, scene=s, N=r["N"], ms=r["ms"], err=r["err"], dir=d, phys=load(d, "phys"), vis=load(d, "vis")))


def val(p, k, src):
    v = p[src].get(k)
    return v if isinstance(v, (int, float)) and math.isfinite(v) else None


def spearman(a, b):
    def rk(x):
        o = sorted(range(len(x)), key=lambda i: x[i]); r = [0.0] * len(x)
        i = 0
        while i < len(o):
            j = i
            while j + 1 < len(o) and x[o[j + 1]] == x[o[i]]:
                j += 1
            for q in range(i, j + 1):
                r[o[q]] = (i + j) / 2
            i = j + 1
        return r
    if len(a) < 3:
        return None
    ra, rb = rk(a), rk(b); ma, mb = S.mean(ra), S.mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb)); den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return num / den if den > 0 else None


os.makedirs(f"{OUT}/fig", exist_ok=True)
rows = []
for k, src, name, low in MET:
    fig, axs = plt.subplots(1, 4, figsize=(16, 3.9))
    sep, mpm_ok, cors, noise, cover = [], [], [], [], [0, 0]
    for ax, (s, sn) in zip(axs, SC):
        Q = [p for p in pts if p["scene"] == s]
        V = [(p, val(p, k, src)) for p in Q]
        cover[0] += sum(v is not None for _, v in V); cover[1] += len(V)
        good = [(p, v) for p, v in V if v is not None]
        for mm, mn in METH:
            G = [(p, v) for p, v in good if p["method"] == mm]
            if not G:
                continue
            ax.scatter([p["ms"] for p, _ in G], [v for _, v in G], s=64 if mm != "mpm" else 120, marker=MK[mm], c=COL[mm], edgecolors="#fcfcfb", linewidths=1.5, label=mn, zorder=3)
            if mm == "ours":
                for p, v in G:
                    ax.annotate(f"K={p['N']}", (p["ms"], v), textcoords="offset points", xytext=(5, 4), fontsize=7, color="#52514e")
        ax.set_xscale("log"); ax.set_title(sn, fontsize=11, color="#0b0b0b"); ax.set_xlabel("A6000 단독 ms/프레임 (로그)", fontsize=8, color="#52514e")
        ax.axvline(16.7, color="#c3c2b7", lw=1, ls=":", zorder=1)
        ax.grid(True, color="#ecebe8", lw=0.6, zorder=0); [sp.set_visible(False) for sp in (ax.spines["top"], ax.spines["right"])]
        ax.tick_params(labelsize=8, colors="#52514e")
        vs = [v for _, v in good]
        if len(vs) >= 2:
            md = abs(S.median(vs)); sep.append((max(vs) - min(vs)) / md if md > 0 else None)   # 방법 사이 폭 = (최대 - 최소) / |중앙값|
            mv = [v for p, v in good if p["method"] == "mpm"]
            if mv:                                                                   # 정답 MPM 이 좋은 쪽 절반 안인지 (순위 / 점 수)
                srt = sorted(vs, reverse=not low); rnk = srt.index(mv[0]) / max(len(vs) - 1, 1); mpm_ok.append(rnk)
            nm = [(p["err"], v) for p, v in good if p["method"] != "mpm"]
            c = spearman([a for a, _ in nm], [b for _, b in nm])
            if c is not None:
                cors.append(c if low else -c)
        on = [v for p, v in good if p["method"] == "ours" and int(p["N"]) == 1]
        reps = on + [val(dict(phys=load(dirs("ours", s, 1, r), "phys"), vis=load(dirs("ours", s, 1, r), "vis")), k, src) for r in (2, 3)]
        reps = [x for x in reps if x is not None]
        if len(reps) >= 2 and len(vs) >= 2 and S.pstdev(vs) > 0:
            noise.append(S.pstdev(reps) / S.pstdev(vs))
    axs[0].set_ylabel(name + (" (낮을수록 좋음)" if low else " (높을수록 좋음)"), fontsize=9, color="#0b0b0b")
    h, l = [], []
    for ax in axs:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in l:
                h.append(hh); l.append(ll)
    fig.legend(h, l, loc="lower center", ncol=len(l), fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(f"{OUT}/fig/pareto_rf_{k}.png", dpi=150); plt.close(fig)
    sp_ = [x for x in sep if x is not None]
    rows.append(dict(key=k, name=name, low=low, coverage=f"{cover[0]}/{cover[1]}", spread=(S.median(sp_) if sp_ else None),
                     mpm_rank=(S.mean(mpm_ok) if mpm_ok else None), n_mpm=len(mpm_ok),
                     corr=(S.mean(cors) if cors else None), n_corr=len(cors),
                     noise=(S.mean(noise) if noise else None), n_noise=len(noise)))
json.dump(dict(rows=rows, points=[{k: v for k, v in p.items() if k not in ("phys", "vis")} for p in pts]), open(f"{C}/rf_choice.json", "w"), ensure_ascii=False, indent=1)


def f(x, fmt="%.2f"):
    return "--" if x is None else fmt % x


L = ["% 자동 생성 (exe/ffx_rf_choice.py)", "\\begin{center}\\scriptsize\\begin{tabular}{llrrrrr}\\toprule",
     "지표 & 출처 & 값 있는 점 & 방법 사이 폭 & 정답 MPM 순위 (0 = 가장 좋음) & 정답 대비 오차와 순위 상관 & 노이즈 (우리 3 회 편차 / 방법 사이 편차)\\\\\\midrule"]
for r in rows:
    L.append(f"{r['name']} & {'물리' if r['key'] in [m[0] for m in MET if m[1]=='phys'] else '시각'} & {r['coverage']} & {f(r['spread'])} & {f(r['mpm_rank'])} ({r['n_mpm']}) & {f(r['corr'])} ({r['n_corr']}) & {f(r['noise'], '%.3f')} ({r['n_noise']})\\\\")
L += ["\\bottomrule\\end{tabular}\\end{center}"]
open(f"{OUT}/pareto_final_1115_rf_choice_table.tex", "w").write("\n".join(L) + "\n")
print(json.dumps(rows, ensure_ascii=False, indent=1))

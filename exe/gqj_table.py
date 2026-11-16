#!/usr/bin/env python3
"""절 CM194: gq_justify 결과 -> 논문 폴더 표·그림.
  python exe/gqj_table.py <결과 캐시 (runs/<판>/gq.json 들, eig/*.json)> <논문 results_all 폴더>
표 gq_justify_table.tex (장면 블록 + 평균 블록, 행 = 판), gq_justify_stress.tex (조건별, 네 장면 합), 그림 fig/gq_stability_map.png.
정의: 오차·모래시계 에너지 비 = 기본 조건 3 회 중앙값 (반복 1 덤프 + 2·3 lean), ms = 시간 모드 (A6000 단독) 3 회 중앙값,
이방성·det F·시각 = 기본 조건 반복 1 덤프, 발산 = 모든 스트레스 조건 (기본 포함 7 조건 x 3 회) 에서 끝까지 못 간 판 수."""
import glob, json, os, sys
import numpy as np

C, OUTD = sys.argv[1], sys.argv[2]
R1 = os.environ.get("GQ_R1", "1") == "1"
MODES = ("dump",) if R1 else ("dump", "met")
SC = ["L000", "pillow2sofa", "bread", "wolf"]
SCN = {"L000": "Lego", "pillow2sofa": "pillow2sofa", "bread": "bread", "wolf": "wolf"}
V = ["q1", "q1g001", "q1g01", "q1g1", "def", "q27"]
VN = {"q1": "$1^3$", "q1g001": "$1^3{+}\\gamma_{0.01}$", "q1g01": "$1^3{+}\\gamma_{0.1}$", "q1g1": "$1^3{+}\\gamma_{1}$", "def": "$2^3$ (기본)", "q27": "$3^3$"}
CONDS = ["base", "e10", "e100", "c15", "c20", "v2", "k2"]
CN = {"base": "기본", "e10": "$E{\\times}10$", "e100": "$E{\\times}100$", "c15": "칸 ${\\times}1.5$", "c20": "칸 ${\\times}2$", "v2": "충격 ${\\times}2$", "k2": "$K{=}2$"}
R = {}
for f in glob.glob(f"{C}/runs/*/gq.json"):
    try:
        o = json.load(open(f))
    except Exception:
        continue
    R[(o["scene"], o["var"], o["cond"], o["mode"], o["rep"])] = o


def runs(s, v, c, modes=("dump", "met")):
    return [o for (s_, v_, c_, m_, r_), o in R.items() if s_ == s and v_ == v and c_ == c and m_ in modes]


def done(o):
    return o.get("frames_done") is not None and o.get("frames_done") == o.get("frames") and o.get("rc", 0) in (0,)


def med(x):
    x = [q for q in x if q is not None and np.isfinite(q)]
    return float(np.median(x)) if x else None


def g(o, *ks):
    for k in ks:
        if o is None:
            return None
        o = o.get(k) if isinstance(o, dict) else None
    return o


def cell(s, v):
    b = runs(s, v, "base")
    d = [o for o in b if o["mode"] == "dump"]
    d = d[0] if d else None
    t = runs(s, v, "base", ("time",))
    allr = [o for c in CONDS for o in runs(s, v, c, MODES)]
    if R1:                                                                           # 사용자 11-16: r1 (덤프) 값 먼저
        b = [o for o in b if o["rep"] == 1]; t = [o for o in t if o["rep"] == 1]
    ndiv = sum(1 for o in allr if not done(o))
    return dict(err=med([o["err"] * 100 for o in b if done(o) and o.get("err") is not None]), nerr=sum(1 for o in b if done(o)),
                ms=med([o["ms"] for o in t if done(o)]), hg=med([g(o, "hgm", "ratio") for o in b if done(o)]),
                an99=g(d, "aniso", "rel_p99"), anmax=g(d, "aniso", "rel_max"), detmin=g(d, "phys", "detmin"),
                lpips=g(d, "vis", "lpips"), warp=(g(d, "vis", "warp_err") * 1e3 if g(d, "vis", "warp_err") is not None else None),
                ndiv=ndiv, nrun=len(allr))


def EZ(ez):
    return "--" if ez is None else (("$\\ge$" if ez[1] else "") + str(ez[0]))


def fmt(x, p):
    return "--" if x is None else f"{x:.{p}f}"


EIG = {}
RIG = {"L000": 6, "bread": 6, "wolf": 6, "pillow2sofa": 30}                        # 강체 모드 (연결 성분마다 6; ablation 표 abl_eig 와 같은 빼기)
for f in glob.glob(f"{C}/eig/*_q*_hg*.json"):
    b = os.path.basename(f)[:-5]
    s, q, h = b.rsplit("_", 2)
    try:
        c0 = json.load(open(f))["cases"][0]
        EIG[(s, int(q[1:]), float(h[2:]))] = (max(c0["n_zero"] - RIG[s], 0), bool(c0.get("all_kb_small")) and c0["n_zero"] > 1000)   # 고유값 창이 모두 영에 가까우면 하한
    except Exception:
        pass
VQ = {"q1": (1, 0.0), "q1g001": (1, 0.01), "q1g01": (1, 0.1), "q1g1": (1, 1.0), "def": (8, 0.0), "q27": (27, 0.0)}
hdr = "장면 & 판 & 오차 \\% & ms & 모래시계 에너지 비 & 이방성 상위 1\\% & 이방성 최대 & det 최소 & 발산 & LPIPS & 깜빡임 $\\times10^3$ & 영 에너지 모드\\\\"
L = ["% 자동 생성 (exe/gqj_table.py, 절 CM194)", "\\begin{center}\\scriptsize\\setlength{\\tabcolsep}{3pt}\\begin{tabular}{llrrrrrrrrrr}\\toprule", hdr + "\\midrule"]
ALL = {v: [] for v in V}
for s in SC:
    for i, v in enumerate(V):
        x = cell(s, v); ALL[v].append(x)
        ez = EIG.get((s, *VQ[v]))
        L.append(f"{SCN[s] if i == 0 else ''} & {VN[v]} & {fmt(x['err'], 2)} & {fmt(x['ms'], 1)} & {fmt(x['hg'], 3)} & {fmt(x['an99'], 1)} & {fmt(x['anmax'], 0)} & {fmt(x['detmin'], 3)} & "
                 f"{x['ndiv']}/{x['nrun']} & {fmt(x['lpips'], 4)} & {fmt(x['warp'], 3)} & {EZ(ez)}\\\\")
    L.append("\\midrule")
for i, v in enumerate(V):
    xs = ALL[v]
    m = lambda k: (float(np.mean([x[k] for x in xs])) if all(x[k] is not None for x in xs) else None)
    dmin = [x["detmin"] for x in xs if x["detmin"] is not None]
    L.append(f"{'평균' if i == 0 else ''} & {VN[v]} & {fmt(m('err'), 2)} & {fmt(m('ms'), 1)} & {fmt(m('hg'), 3)} & {fmt(m('an99'), 1)} & {fmt(m('anmax'), 0)} & "
             f"{fmt(min(dmin) if len(dmin) == len(xs) else None, 3)} & {sum(x['ndiv'] for x in xs)}/{sum(x['nrun'] for x in xs)} & {fmt(m('lpips'), 4)} & {fmt(m('warp'), 3)} & --\\\\")
L.append("\\bottomrule\\end{tabular}\\end{center}")
open(f"{OUTD}/gq_justify_table.tex", "w").write("\n".join(L) + "\n")
# 조건별 (네 장면): 모래시계 에너지 비 중앙값 · 발산 수
L = ["% 자동 생성 (exe/gqj_table.py, 절 CM194) -- 조건별, 네 장면", "\\begin{center}\\scriptsize\\setlength{\\tabcolsep}{3pt}\\begin{tabular}{ll" + "r" * len(V) + "}\\toprule",
     "지표 & 조건 & " + " & ".join(VN[v] for v in V) + "\\\\\\midrule"]
for k, nm in (("hg", "모래시계 에너지 비"), ("div", "발산 (판 수)")):
    for j, c in enumerate(CONDS):
        row = []
        for v in V:
            rr = [o for s in SC for o in runs(s, v, c, MODES)]
            if k == "hg":
                row.append(fmt(med([g(o, "hgm", "ratio") for o in rr if done(o)]), 3))
            else:
                row.append(f"{sum(1 for o in rr if not done(o))}/{len(rr)}")
        L.append(f"{nm if j == 0 else ''} & {CN[c]} & " + " & ".join(row) + "\\\\")
    L.append("\\midrule")
L[-1] = "\\bottomrule\\end{tabular}\\end{center}"
open(f"{OUTD}/gq_justify_stress.tex", "w").write("\n".join(L) + "\n")
json.dump({f"{s}|{v}": cell(s, v) for s in SC for v in V}, open(f"{OUTD}/gq_justify_cells.json", "w"), indent=1, ensure_ascii=False)
# 안정 영역 그림: E x 칸 평면 (레고·bread), 판별 칸 = 완주 몫 (색) + 모래시계 에너지 비 (글자)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
for fp in (os.path.expanduser("~/.fonts/NotoSansCJKkr-Regular.otf"), "/usr/share/fonts/truetype/nanum/NanumGothic.ttf", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"):
    if os.path.exists(fp):
        font_manager.fontManager.addfont(fp); plt.rcParams["font.family"] = font_manager.FontProperties(fname=fp).get_name(); break
plt.rcParams["axes.unicode_minus"] = False
EG, CG = ["", "e10", "e100"], ["", "c15", "c20"]


def cname(e, c):
    return (e + c) if (e or c) else "base"


fig, axs = plt.subplots(2, len(V), figsize=(2.3 * len(V), 5.0), constrained_layout=True)
for a_, s in enumerate(["L000", "bread"]):
    for b_, v in enumerate(V):
        ax = axs[a_, b_]
        M = np.full((3, 3), np.nan); T = [["" for _ in range(3)] for _ in range(3)]
        for i, e in enumerate(EG):
            for j, c in enumerate(CG):
                rr = runs(s, v, cname(e, c), ("dump", "met") if cname(e, c) not in CONDS else MODES)
                if not rr:
                    continue
                M[i, j] = sum(1 for o in rr if done(o)) / len(rr)
                h = med([g(o, "hgm", "ratio") for o in rr if done(o)])
                T[i][j] = f"{sum(1 for o in rr if done(o))}/{len(rr)}" + (f"\n{h:.2f}" if h is not None else "")
        ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=1, origin="lower")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, T[i][j], ha="center", va="center", fontsize=7)
        ax.set_xticks(range(3)); ax.set_xticklabels(["1", "1.5", "2"]); ax.set_yticks(range(3)); ax.set_yticklabels(["1", "10", "100"])
        if a_ == 1:
            ax.set_xlabel("칸 크기 배율")
        if b_ == 0:
            ax.set_ylabel(f"{SCN[s]}\n영률 배율")
        ax.set_title(VN[v].replace("$", "").replace("{+}", "+").replace("\\gamma_", "γ").replace("{", "").replace("}", "").replace("^3", "³"), fontsize=9)
fig.suptitle("안정 영역 (칸: 완주 판 수 / 전체, 아래 숫자 = 모래시계 에너지 비 중앙값; 초록 = 모두 완주)", fontsize=10)
os.makedirs(f"{OUTD}/fig", exist_ok=True)
fig.savefig(f"{OUTD}/fig/gq_stability_map.png", dpi=160)
print("ok", len(R))

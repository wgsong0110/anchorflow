"""절 AH 표: 판마다 lego 홀드아웃 20 궤적 + L000 + ficus 오차 (비측정 스윕, 태그 ah_<판>_<궤적>_r<회>).
사용: python exe/fem_ah_table.py <lbours 루트> <판1,판2,...> [--long] [--r R]
  기본: 판마다 한 줄 (만족 수, 평균, 미달 궤적), --long: 궤적별 오차 표 (행 = 궤적, 열 = 판)."""
import json
import os
import sys

root = sys.argv[1]
VS = sys.argv[2].split(",")
LONG = "--long" in sys.argv
RR = int(sys.argv[sys.argv.index("--r") + 1]) if "--r" in sys.argv else 1
TR = [f"{i:04d}" for i in range(20)] + ["L000"]


def load(sub, tag):
    f = os.path.join(root, sub, tag, "summary.json")
    return json.load(open(f)) if os.path.exists(f) else None


def cell(s):
    if s is None:
        return None, "-"
    if s["diverged"]:
        return None, f"발산 {s['diverged']}"
    return s["err_mean"], f"{100 * s['err_mean']:.2f}"


def runs(v, t):
    if t == "F":
        return [load("fem_ah_ficus", f"ah_{v}_F_r{r}") for r in range(1, RR + 1)]
    return [load("fem_ah_lego", f"ah_{v}_{t}_r{r}") for r in range(1, RR + 1)]


rows = {}
for v in VS:
    rows[v] = {}
    for t in TR + ["F"]:
        S = [s for s in runs(v, t) if s]
        if not S:
            rows[v][t] = (None, "-", 0, 0)
            continue
        fin = [s for s in S if not s["diverged"]]
        e = sum(s["err_mean"] for s in fin) / len(fin) if len(fin) == len(S) else None
        txt = (f"{100 * e:.2f}" if e is not None else "/".join(cell(s)[1] for s in S))
        rows[v][t] = (e, txt, len(fin), len(S))
if LONG:
    print("| 궤적 | " + " | ".join(VS) + " |")
    print("|---|" + "---|" * len(VS))
    for t in TR + ["F"]:
        print(f"| {t if t != 'F' else 'ficus'} | " + " | ".join(rows[v][t][1] + ("" if rows[v][t][0] is None or rows[v][t][0] < 0.10 else " ✗") for v in VS) + " |")
print()
print("| 판 | lego 20 만족 | L000 | ficus | 22 개 만족 | lego 20 평균 | 미달 |")
print("|---|---|---|---|---|---|---|")
for v in VS:
    R_ = rows[v]
    ok20 = [t for t in TR[:20] if R_[t][0] is not None and R_[t][0] < 0.10]
    n20 = sum(1 for t in TR[:20] if R_[t][3])
    es = [R_[t][0] for t in TR[:20] if R_[t][0] is not None]
    okL = R_["L000"][0] is not None and R_["L000"][0] < 0.10
    okF = R_["F"][0] is not None and R_["F"][0] < 0.10
    bad = [f"{t} {R_[t][1]}" for t in TR[:20] if R_[t][3] and t not in ok20]
    print(f"| {v} | {len(ok20)}/{n20} | {R_['L000'][1]} | {R_['F'][1]} | {len(ok20) + okL + okF}/{n20 + (1 if R_['L000'][3] else 0) + (1 if R_['F'][3] else 0)} | "
          f"{100 * sum(es) / len(es) if es else float('nan'):.2f} ({len(es)}) | {', '.join(bad)} |")

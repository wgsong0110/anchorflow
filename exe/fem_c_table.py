"""절 C 표: 실행 폴더들을 판 (태그에서 _r<회> 를 뺀 이름) 별로 묶어 완주 수·오차·ms (단독 측정만)·적응 N 평균, 그리고 결과 불변 대조 --
같은 판 안 두 실행 (반복 바닥) / 판 사이 (기준 판 대비) 위치 궤적 차 = 프레임 평균 |Δx|/EXT 의 평균 (pred.pt 가 있는 완주 실행끼리).
사용: python exe/fem_c_table.py [--ref <판 이름>] <폴더...>"""
import argparse
import itertools
import json
import os
import re
from collections import OrderedDict

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--ref", default="")
ap.add_argument("dirs", nargs="+")
a = ap.parse_args()
G = OrderedDict()
for d in a.dirs:
    if not os.path.exists(f"{d}/summary.json"):
        continue
    k = re.sub(r"_r\d+$", "", os.path.basename(d.rstrip("/")))
    G.setdefault(k, []).append(d)
P = {}


def pred(d):
    if d not in P:
        f = f"{d}/pred.pt"
        P[d] = torch.load(f, map_location="cpu", weights_only=False) if os.path.exists(f) else None
    return P[d]


def tdiff(d1, d2):
    A, B = pred(d1), pred(d2)
    if A is None or B is None:
        return None
    return float(((A["pred"].float() - B["pred"].float()).norm(dim=-1).mean(1) / float(A["EXT"])).mean())


print("| 판 | 완주 | 오차 (완주 평균; 회별) | ms 평균 (단독) | 발산 프레임 | N 평균 (적응) | 같은 판 반복 궤적 차 |" + (f" vs {a.ref} 궤적 차 |" if a.ref else ""))
print("|---|---|---|---|---|---|---|" + ("---|" if a.ref else ""))
for k, ds in G.items():
    S = [json.load(open(f"{d}/summary.json")) for d in ds]
    ok = [(d, s) for d, s in zip(ds, S) if s["diverged"] is None]
    errs = [s["err_mean"] * 100 for _, s in ok]
    tim = [s["ms_mean"] for d, s in zip(ds, S) if os.path.exists(f"{d}/TIMED") and s["diverged"] is None]
    dv = [str(s["diverged"]) for s in S if s["diverged"] is not None]
    nm = [s["adapt"]["N_mean"] for s in S if s.get("adapt")]
    rep = [tdiff(x[0], y[0]) for x, y in itertools.combinations(ok, 2)]
    rep = [r for r in rep if r is not None]
    row = (f"| {k} | {len(ok)}/{len(S)} | " + (f"{sum(errs) / len(errs):.2f} % (" + ", ".join(f"{e:.2f}" for e in errs) + ")" if errs else "-")
           + f" | {(sum(tim) / len(tim)) if tim else float('nan'):.2f} | {'·'.join(dv) or '-'} | {(sum(nm) / len(nm)) if nm else float('nan'):.1f} | "
           + (f"{sum(rep) / len(rep):.2e} ({len(rep)} 쌍)" if rep else "-") + " |")
    if a.ref and a.ref in G and k != a.ref:
        rk = [(d, json.load(open(f"{d}/summary.json"))) for d in G[a.ref]]
        rk = [d for d, s in rk if s["diverged"] is None]
        cr = [tdiff(x[0], y) for x in ok for y in rk]
        cr = [r for r in cr if r is not None]
        row += (f" {sum(cr) / len(cr):.2e} ({len(cr)} 쌍) |" if cr else " - |")
    elif a.ref:
        row += " - |"
    print(row)

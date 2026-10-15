"""확정 최소 N 에서 물리 정합성 표 (stab_scan_table.py 와 같은 지표·문턱, N 만 확정값).

확정 N 의 실행은 여러 번이다: 정수 스캔 때의 한 번 (res 또는 scan) + 반복 확정 (robust/r1..r3).
값 = 그 실행들의 평균, 통과 = 모든 실행이 문턱 안. 발산한 실행이 있으면 그 수를 적는다.

  python exe/stab_phys_confirmed.py --root /home/dkta/work/stab --out phys_confirmed.json --tex phys_confirmed.tex
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import stab_metrics as sm                                   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="/home/dkta/work/stab")
ap.add_argument("--out", required=True)
ap.add_argument("--tex", required=True)
a = ap.parse_args()

# stab_scan_table.py 와 같은 문턱
THR = dict(detneg=0.0, pen_mult=2.0, pen_floor=0.01, jit_mult=2.0, ff=0.05, cd=0.05)
METH = [("pg", "PG"), ("gf", "GF"), ("ipg", "i-PG"), ("ovs", "우리")]

NCONF = {"lego": dict(pg=43, gf=16, ipg=8, ovs=1),
         "mic": dict(pg=45, gf=16, ipg=12, ovs=1),
         "ficus": dict(pg=41, gf=16, ipg=10, ovs=1)}
KEYS = ("detneg", "pen", "jit", "ff", "cd")

R = {}
for S in ("lego", "mic", "ficus"):
    import h5py
    cfg = json.load(open(f"{a.root}/{S}/config.json"))
    zf = float([b for b in cfg["boundary_conditions"] if b["type"] == "surface_collider"][0]["point"][2])
    g = float(cfg["g"][2]); dt = float(cfg["frame_dt"])
    with h5py.File(f"{a.root}/{S}/init/sim_0000000000.h5") as h:
        x0 = np.array(h["x"]); x0 = x0.T if x0.shape[0] == 3 else x0
    dxg = float(cfg["grid_lim"]) / int(cfg["n_grid"])
    M = float(cfg["density"]) * dxg ** 3 * len(np.unique(np.floor(x0 / dxg).astype(np.int64), axis=0))
    refr = sm.load(f"{a.root}/{S}/res", "pg", 1024)
    ref = refr["X"]
    L = float(np.linalg.norm(ref[0].max(0) - ref[0].min(0)))
    refm = sm.metrics(refr, ref, L, zf, g, dt, M, dxg)
    R[S] = {"ref": refm, "L": L}
    for m, name in METH:
        n = NCONF[S][m]
        dirs = [f"{a.root}/{S}/{s}" for s in ("res", "scan")] + \
               [f"{a.root}/{S}/robust/r{k}" for k in (1, 2, 3)]
        runs, srcs, bad = [], [], 0
        seen_main = False
        for d in dirs:
            if not os.path.isdir(d):
                continue
            if d.endswith(("res", "scan")) and seen_main:
                continue                                    # 스캔 때 실행은 한 번만 센다
            r = sm.load(d, m, n)
            if r is None:
                continue
            if d.endswith(("res", "scan")):
                seen_main = True
            if r["status"] != "끝":
                bad += 1; srcs.append(os.path.relpath(d, a.root) + "(발산)"); continue
            mt = sm.metrics(r, ref, L, zf, g, dt, M, dxg)
            ch = {
                "detneg": (mt.get("detneg_max"), (mt.get("detneg_max") or 0.0) <= THR["detneg"]),
                "pen": (mt.get("pen_depth_max"),
                        (mt.get("pen_depth_max") or 0.0) <= max(THR["pen_mult"] * (refm.get("pen_depth_max") or 0.0), THR["pen_floor"])),
                "jit": (mt.get("jitter_d2"), (mt.get("jitter_d2") or 0.0) <= THR["jit_mult"] * refm["jitter_d2"]),
                "ff": (mt.get("freefall_mom_err"), None if mt.get("freefall_mom_err") is None else mt["freefall_mom_err"] <= THR["ff"]),
                "cd": (mt.get("cd_mpm"), (mt.get("cd_mpm") or 1.0) <= THR["cd"]),
            }
            runs.append(ch); srcs.append(os.path.relpath(d, a.root))
        rec = dict(n=n, runs=len(runs), diverged=bad, src=srcs)
        if runs:
            agg = {}
            for k in KEYS:
                vs = [c[k][0] for c in runs if c[k][0] is not None]
                oks = [c[k][1] for c in runs if c[k][1] is not None]
                agg[k] = dict(value=float(np.mean(vs)) if vs else None,
                              ok=(all(oks) if oks else None), per_run=[c[k][0] for c in runs])
            rec["check"] = agg
        R[S][m] = rec
        print(S, name, n, rec.get("runs"), rec.get("diverged"), srcs, flush=True)
json.dump(R, open(a.out, "w"), indent=1, default=float)


def f(v):
    return "--" if v is None else f"{100 * v:.2f}"


def mark(o):
    return "" if o is None else (r"\,\checkmark" if o else r"\,$\times$")


T = [r"\begin{table}[h]\centering\small",
     r"\caption{확정 최소 $N$ 에서 물리 정합성. 값 = 그 $N$ 의 실행 전부(정수 스캔 1 회 + 반복 확정) 평균, "
     r"통과($\checkmark$) = 모든 실행이 문턱 안. 기준 = PG $N{=}1024$. 문턱: 뒤집힘 비율 0, "
     r"관통 깊이 $\le \max(2\times\text{기준}, 0.01L)$, 떨림 $\le 2\times$기준, 자유낙하 운동량 오차 $\le 5\%$, "
     r"PG $N{=}1024$ 대비 Chamfer $\le 5\%L$. 값은 \% (관통·떨림·Chamfer 는 $L$ 대비). 실행: 평균에 든 실행 수.}",
     r"\begin{tabular}{llccccccc}", r"\toprule",
     r"형상 & 방법 & $N$ & 실행 & 뒤집힘 & 바닥 관통 & 떨림 & 자유낙하 & Chamfer \\", r"\midrule"]
for S in ("lego", "mic", "ficus"):
    rf = R[S]["ref"]
    T.append(f"{S} & 기준 PG & 1024 & 1 & {f(rf.get('detneg_max'))} & {f(rf.get('pen_depth_max'))} & "
             f"{f(rf.get('jitter_d2'))} & {f(rf.get('freefall_mom_err'))} & 0 \\\\")
    for m, name in METH:
        rr = R[S][m]
        runs = f"{rr['runs']}" + (f" (발산 {rr['diverged']})" if rr["diverged"] else "")
        if not rr.get("check"):
            T.append(f" & {name} & {rr['n']} & {runs} & & & & & \\\\"); continue
        c = rr["check"]
        T.append(f" & {name} & {rr['n']} & {runs} & " + " & ".join(f(c[k]['value']) + mark(c[k]['ok']) for k in KEYS) + r" \\")
    T.append(r"\midrule")
T[-1] = r"\bottomrule"
T += [r"\end{tabular}", r"\end{table}", ""]
open(a.tex, "w").write("\n".join(T))
print("[표]", a.tex)

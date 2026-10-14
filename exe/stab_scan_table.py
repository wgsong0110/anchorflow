"""최소 안정 서브스텝 수 (정수 단위) + 그 N 에서의 물리 정합성 검사 -> json + LaTeX 표 조각.

입력: stab/<형상>/res (2 의 거듭제곱 N) + stab/<형상>/scan (사이 정수 N, exe 실행 스크립트 stab_scan.sh).
안정 = 60 프레임을 끝까지 돌았고 (NaN·오류 없음) 값이 터지지 않음 -- stab_metrics.py 와 같은 판정
(총에너지 변화 최대 > 1 또는 바닥 관통 깊이 > 0.5 L 이면 발산).
최소 N = 안정한 가장 작은 N. 그보다 큰 N 중 불안정한 것이 있으면 같이 적는다 (단조가 아닐 수 있다).

물리 정합성 (그 방법의 최소 N 에서, 기준 = PG N1024 = 가장 정밀한 실행):
  뒤집힘     det F ≤ 0 비율 최대 = 0
  바닥 관통  최대 깊이 / L ≤ max(2 × 기준, 0.01)
  떨림       위치 2 차 차분 평균 / L ≤ 2 × 기준
  자유낙하   운동량 상대 오차 최대 ≤ 0.05 (공중 구간이 없으면 -)
  궤적       PG N1024 대비 Chamfer / L ≤ 0.05
문턱은 이 파일 THR 에 모아 둔다.

  python exe/stab_scan_table.py --root /home/dkta/work/stab --out scan.json --tex scan_tables.tex
"""
import argparse
import glob
import json
import os
import re
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

THR = dict(detneg=0.0, pen_mult=2.0, pen_floor=0.01, jit_mult=2.0, ff=0.05, cd=0.05)
METH = [("pg", "PG"), ("gf", "GF"), ("ipg", "i-PG"), ("ovs", "우리")]


def ns_of(d, m):
    out = set()
    for p in glob.glob(f"{d}/{m}_N*.*"):
        g = re.match(rf"{m}_N(\d+)\.(npz|log)$", os.path.basename(p))
        if g:
            out.add(int(g.group(1)))
    return out


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
        st = {}
        for d in (f"{a.root}/{S}/res", f"{a.root}/{S}/scan"):
            for n in sorted(ns_of(d, m)):
                if n > 128 or n in st:
                    continue
                r = sm.load(d, m, n)
                if r is None or r["status"] == "진행":
                    continue
                ok = r["status"] == "끝"
                mt = None
                if ok:
                    mt = sm.metrics(r, None, L, zf, g, dt, M, dxg)
                    if not np.isfinite(mt.get("e_drift_max", 0.0)) or mt.get("e_drift_max", 0.0) > 1.0 \
                            or (mt.get("pen_depth_max") or 0.0) > 0.5:
                        ok = False
                st[n] = ok
        stable = sorted(n for n, ok in st.items() if ok)
        nmin = stable[0] if stable else None
        gaps = sorted(set(range(1, (nmin or 0))) - set(st)) if nmin else []
        bad_above = sorted(n for n, ok in st.items() if nmin and n > nmin and not ok)
        rec = dict(nmin=nmin, tested=sorted(st), unstable_above=bad_above, untested_below=gaps)
        if nmin:
            d = f"{a.root}/{S}/res" if nmin in ns_of(f"{a.root}/{S}/res", m) else f"{a.root}/{S}/scan"
            r = sm.load(d, m, nmin)
            mt = sm.metrics(r, ref, L, zf, g, dt, M, dxg)
            chk = {
                "detneg": (mt.get("detneg_max"), (mt.get("detneg_max") or 0.0) <= THR["detneg"]),
                "pen": (mt.get("pen_depth_max"),
                        (mt.get("pen_depth_max") or 0.0) <= max(THR["pen_mult"] * (refm.get("pen_depth_max") or 0.0), THR["pen_floor"])),
                "jit": (mt.get("jitter_d2"), (mt.get("jitter_d2") or 0.0) <= THR["jit_mult"] * refm["jitter_d2"]),
                "ff": (mt.get("freefall_mom_err"), None if mt.get("freefall_mom_err") is None else mt["freefall_mom_err"] <= THR["ff"]),
                "cd": (mt.get("cd_mpm"), (mt.get("cd_mpm") or 1.0) <= THR["cd"]),
            }
            rec["check"] = {k: dict(value=v, ok=o) for k, (v, o) in chk.items()}
        R[S][m] = rec
        print(S, name, rec, flush=True)
json.dump(R, open(a.out, "w"), indent=1, default=float)


def f(v, pct=False):
    if v is None:
        return "--"
    return f"{100 * v:.2f}" if pct else (f"{v:.1e}" if v and abs(v) < 1e-2 else f"{v:.3f}")


def mark(o):
    return "" if o is None else (r"\,\checkmark" if o else r"\,$\times$")


T = [r"\begin{table}[h]\centering\small",
     r"\caption{발산하지 않는 최소 서브스텝 수 $N$ (정수 단위로 다시 잼, 60 프레임). 괄호: 최소 $N$ 보다 큰데 발산한 $N$.}",
     r"\begin{tabular}{l" + "c" * len(METH) + "}", r"\toprule", "형상 & " + " & ".join(n for _, n in METH) + r" \\", r"\midrule"]
for S in ("lego", "mic", "ficus"):
    cells = []
    for m, _ in METH:
        rr = R[S][m]
        c = str(rr["nmin"]) if rr["nmin"] else "--"
        if rr["unstable_above"]:
            c += " (" + ",".join(map(str, rr["unstable_above"])) + ")"
        if rr["untested_below"]:
            c += r"$^\ast$"
        cells.append(c)
    T.append(S + " & " + " & ".join(cells) + r" \\")
T += [r"\bottomrule", r"\end{tabular}", r"\\[2pt]{\footnotesize $^\ast$ 그보다 작은 N 중 아직 안 돌린 것이 있음}", r"\end{table}", ""]
T += [r"\begin{table}[h]\centering\small",
      r"\caption{각 방법의 최소 $N$ 에서 물리 정합성. 기준 = PG $N{=}1024$. 통과($\checkmark$) 문턱: 뒤집힘 비율 0, "
      r"관통 깊이 $\le \max(2\times\text{기준}, 0.01L)$, 떨림 $\le 2\times$기준, 자유낙하 운동량 오차 $\le 5\%$, "
      r"PG $N{=}1024$ 대비 Chamfer $\le 5\%L$. 값은 \% (관통·떨림·Chamfer 는 $L$ 대비).}",
      r"\begin{tabular}{llcccccc}", r"\toprule",
      r"형상 & 방법 & 최소 $N$ & 뒤집힘 & 바닥 관통 & 떨림 & 자유낙하 & Chamfer \\", r"\midrule"]
for S in ("lego", "mic", "ficus"):
    rf = R[S]["ref"]
    T.append(f"{S} & 기준 PG & 1024 & {f(rf.get('detneg_max'), True)} & {f(rf.get('pen_depth_max'), True)} & "
             f"{f(rf.get('jitter_d2'), True)} & {f(rf.get('freefall_mom_err'), True)} & 0 \\\\")
    for m, name in METH:
        rr = R[S][m]
        if not rr.get("check"):
            T.append(f" & {name} & -- & & & & & \\\\"); continue
        c = rr["check"]
        T.append(f" & {name} & {rr['nmin']} & " + " & ".join(f(c[k]['value'], True) + mark(c[k]['ok'])
                                                           for k in ("detneg", "pen", "jit", "ff", "cd")) + r" \\")
    T.append(r"\midrule")
T[-1] = r"\bottomrule"
T += [r"\end{tabular}", r"\end{table}", ""]
open(a.tex, "w").write("\n".join(T))
print("[표]", a.tex)

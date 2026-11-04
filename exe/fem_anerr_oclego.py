"""절 AN.err4 lego 표 (팔분 가중): 9 궤적 x 판 (칸 "a / b" = 1·2 회차). 사용: python exe/fem_anerr_galego.py <lbours 경로>"""
import json
import os
import sys

import numpy as np

B = sys.argv[1]
T = ["0001", "0003", "0015", "0016", "0017", "0018", "0005", "0013", "L000"]
V = [("명시적 bnv", ["fem_ak_bselc/{t}_bnv"]), ("실시간 판 중점 K1", ["fem_an/{t}_rl1n2c20"]),
     ("후퇴 K2 칸 가우스 (원본)", ["fem_anerr/r2/g_be_K2_{t}", "fem_anerr/ga3/be_K2_{t}", "fem_anerr/ga6b/be_K2_{t}"]),
     ("후퇴 K2 팔분 가중 κ0.1", ["fem_anerr/oc1/be_K2_{t}", "fem_anerr/oc1b/be_K2_{t}"]),
     ("HHT ρ0.7 K2 칸 가우스 (원본)", ["fem_anerr/ga6/hht7_K2_{t}", "fem_anerr/ga6b/hht7_K2_{t}"]),
     ("HHT ρ0.7 K2 팔분 가중 κ0.1", ["fem_anerr/oc1/hht7_K2_{t}", "fem_anerr/oc1b/hht7_K2_{t}"]),
     ("HHT ρ0.7 K2 팔분 가중 κ0.25", ["fem_anerr/oc2/hk25_K2_{t}"]),
     ("HHT ρ0.7 K2 팔분 가중 κ0.5", ["fem_anerr/oc2/hk5_K2_{t}"]),
     ("HHT ρ0.7 K2 팔분 가중 κ0 (1 차 시험)", ["fem_anerr/oc0/hk0_{t}"])]


def one(p):
    f = os.path.join(B, p, "summary.json")
    lg = os.path.join(B, p, "log.txt")
    if os.path.exists(f):
        s = json.load(open(f))
        if s.get("diverged"):
            return f"x{s['diverged']}", None
        if s.get("frames_done", 0) >= s.get("frames", 1):
            return f"{100 * s['err_mean']:.2f}", 100 * s["err_mean"]
    if os.path.exists(lg) and "OutOfMemoryError" in open(lg, errors="ignore").read():
        return "폭주", None
    return "", None


print("| 판 | " + " | ".join(T) + " | 평균 | <10% |")
print("|---|" + "---|" * (len(T) + 2))
for nm, pats in V:
    c, es, k, n = [], [], 0, 0
    for t in T:
        vals = [one(p.format(t=t)) for p in pats]
        vals = [v for v in vals if v[0]]
        c.append(" / ".join(v[0] for v in vals))
        if vals:
            n += 1
            e = vals[0][1]
            if e is not None:
                es.append(e)
                k += int(e < 10)
    print(f"| {nm} | " + " | ".join(c) + f" | {np.mean(es) if len(es) == n and n else float('nan'):.2f} | {k}/{n} |")

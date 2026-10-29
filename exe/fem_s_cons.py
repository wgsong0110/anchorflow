"""절 S: 질량·운동량 보존 (fem.jsonl 의 mass_p·mass_n·P_p0·P_n0·P_n1·P_p1) 요약.
사용: python exe/fem_s_cons.py <실행 디렉토리>...   -- 프레임마다 |Σm_n/Σm_p - 1|, P2G |P_n0 - P_p0|/|P_p0|, G2P |P_p1 - P_n1|/|P_n1|, 서브스텝 |P_n1 - P_n0 - M g h|/|P_p0| 의 최대 (유한 프레임)"""
import json
import math
import os
import sys

import numpy as np

for d in sys.argv[1:]:
    f = os.path.join(d, "fem.jsonl")
    if not os.path.exists(f):
        continue
    R = [json.loads(l) for l in open(f)]
    cmd = open(os.path.join(d, "cmd.txt")).read() if os.path.exists(os.path.join(d, "cmd.txt")) else ""
    rows = []
    for r in R:
        if "P_p0" not in r:
            continue
        P = {k: np.array(r[k]) for k in ("P_p0", "P_n0", "P_n1", "P_p1")}
        if not all(np.isfinite(v).all() for v in P.values()):
            continue
        n0 = np.linalg.norm(P["P_p0"]) + 1e-30
        cm = abs(r["mass_c"] / r["mass_p"] - 1) if "mass_c" in r else float("nan")                       # 절 W: 입자 -> 셀 질량
        cp = np.linalg.norm(np.array(r["P_c"]) - P["P_p0"]) / n0 if "P_c" in r else float("nan")           # 절 W: 입자 -> 셀 운동량
        rows.append((r["t"] + 1, abs(r["mass_n"] / r["mass_p"] - 1), np.linalg.norm(P["P_n0"] - P["P_p0"]) / n0,
                     np.linalg.norm(P["P_p1"] - P["P_n1"]) / (np.linalg.norm(P["P_n1"]) + 1e-30), n0, np.linalg.norm(P["P_n1"]), cm, cp))
    if not rows:
        print(f"{os.path.basename(d)}: 유한 프레임 없음")
        continue
    A = np.array(rows)
    print(f"{os.path.basename(d)}: 유한 프레임 {len(rows)} (1..{int(A[-1, 0])}), 질량 상대오차 최대 {A[:, 1].max():.2e}, "
          f"P2G 운동량 상대오차 최대 {A[:, 2].max():.2e}, G2P 운동량 상대오차 최대 {A[:, 3].max():.2e} (중앙 {np.median(A[:, 3]):.2e}), "
          f"|P| 프레임 1 {A[0, 4]:.4g} -> 마지막 {A[-1, 5]:.4g}"
          + (f", [셀] 질량 {np.nanmax(A[:, 6]):.2e} 운동량 {np.nanmax(A[:, 7]):.2e}" if np.isfinite(A[:, 6]).any() else ""))

"""절 AN.err lego 표: 궤적 x 판 평균 오차 (%) -- 명시적 bnv·gqw (lbours/fem_ak_bselc), 실시간 판 (lbours/fem_an/<t>_rl1n2c20), 이 절 판 (lbours/fem_anerr/*). 발산은 x<프레임>.
사용: python exe/fem_anerr_lego.py <lbours 경로>"""
import glob
import json
import os
import sys

import numpy as np

B = sys.argv[1]
T = ["0001", "0003", "0016", "0017", "0018", "0005", "0013", "L000"]
COLS = [("명시적 bnv", "fem_ak_bselc/{t}_bnv"), ("명시적 gqw", "fem_ak_bselc/{t}_gqw"), ("실시간 판 (중점 K1 야코비 2x20)", "fem_an/{t}_rl1n2c20"),
        ("중점 K1", "fem_anerr/*/g_mid_K1_{t}"), ("중점+하한 K1", "fem_anerr/*/gm_mid_K1_{t}"), ("BDF2 K1", "fem_anerr/*/g_bdf2_K1_{t}"),
        ("후퇴 K1", "fem_anerr/*/g_be_K1_{t}"), ("후퇴 K1 입자 적분점", "fem_anerr/*/pb_be_K1_{t}"), ("후퇴 K1 명시적 바닥", "fem_anerr/*/b_flx_K1_{t}"),
        ("후퇴 K1 고착 뉴턴마다", "fem_anerr/*/bi_be_K1_{t}"), ("중점 K1 고착 뉴턴마다", "fem_anerr/*/g_flit_K1_{t}"),
        ("후퇴 K267", "fem_anerr/*/g_be_K267_{t}"), ("중점+하한 K267", "fem_anerr/*/gm_mid_K267_{t}")]


def val(pat):
    for p in glob.glob(os.path.join(B, pat)):
        f = os.path.join(p, "summary.json")
        lg = os.path.join(p, "log.txt")
        if not os.path.exists(f) and os.path.exists(lg) and "OutOfMemoryError" in open(lg, errors="ignore").read():
            n = sum(1 for _ in open(os.path.join(p, "fem.jsonl"))) if os.path.exists(os.path.join(p, "fem.jsonl")) else 0
            return f"폭주 OOM@{n + 1}", None                                            # 국소 폭주로 노드 상자가 커져 메모리 초과 (발산과 같게 셈)
        if os.path.exists(f):
            s = json.load(open(f))
            if s.get("frames_done", 0) < 80 and not s.get("diverged"):
                continue
            if s.get("diverged"):
                return f"x{s['diverged']}", None
            e = 100 * (s.get("err_mean") if s.get("err_mean") is not None else s["err_mean_done"])
            return f"{e:.2f}", e
    return "", None


print("| 판 | " + " | ".join(T) + " | <10% |")
print("|---|" + "---|" * (len(T) + 1))
for nm, pat in COLS:
    c, n, k = [], 0, 0
    for t in T:
        s, e = val(pat.format(t=t))
        c.append(s)
        if e is not None or s.startswith("x") or s.startswith("폭주"):
            n += 1
            k += int(e is not None and e < 10)
    print(f"| {nm} | " + " | ".join(c) + f" | {k}/{n} |")

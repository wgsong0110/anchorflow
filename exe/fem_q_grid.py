"""절 Q: 후보별 N x K 격자 요약 (summary.json·period.json) -- 사용: python exe/fem_q_grid.py <fem_q_lego 또는 fem_q_ficus 디렉토리>
칸: 완주면 '오차% ms [주기]', 발산이면 'D<프레임> (발산 전 오차%)'.  1 차·2 차 모두 비측정 병렬 (ms 는 참고)."""
import glob
import json
import os
import re
import sys

R = {}
for d in sorted(glob.glob(f"{sys.argv[1]}/q_*")):
    f = d + "/summary.json"
    m = re.match(r"q_[lf]_(\w+?)_(?:(vel|full)_)?n(\d+)_k(\d+)$", os.path.basename(d))
    if not (m and os.path.exists(f)):
        continue
    s = json.load(open(f))
    v, md, N, K = m.group(1), m.group(2) or "-", int(m.group(3)), int(m.group(4))
    pj = d + "/period.json"
    per = json.load(open(pj))["T_zc"] if os.path.exists(pj) else None
    R.setdefault((v, md), {})[(N, K)] = (s["diverged"], 100 * (s["err_mean"] if s["err_mean"] is not None else s["err_mean_done"]), s["ms_mean"], per)
for (v, md), g in sorted(R.items()):
    Ns = sorted({k[0] for k in g})
    Ks = sorted({k[1] for k in g})
    print(f"\n-- {v} {md}  (열 K = {Ks})")
    for N in Ns:
        row = []
        for K in Ks:
            if (N, K) not in g:
                row.append(f"{'':>17}")
                continue
            dv, e, ms, per = g[(N, K)]
            row.append(f"{'D%d (%.1f)' % (dv, e):>17}" if dv else f"{'%.2f%% %.1f' % (e, ms) + (' T%.2f' % per if per else ''):>17}")
        print(f"N{N:4d} |" + "|".join(row))

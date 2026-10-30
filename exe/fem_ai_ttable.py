"""절 AI 단독 측정 표: <root>/*_t[g]N/{summary.json, gpu_at_start.txt} -> 씬 x 판: 완주, 오차 % (회별), ms 평균 (회별), fps, 측정 시작 때 카드 1~6 최대 MiB.
사용: python exe/fem_ai_ttable.py <root> [<root> ...]"""
import glob
import json
import os
import re
import sys
from collections import defaultdict

rows = defaultdict(list)
for R in sys.argv[1:]:
    for d in glob.glob(os.path.join(R, "*_t*[0-9]")):
        tg = os.path.basename(d)
        m = re.match(r"(\w+?)_(a\d+b\d_\w+?)_(tg|t)(\d)$", tg)
        if not m:
            continue
        try:
            s = json.load(open(os.path.join(d, "summary.json")))
        except Exception:
            continue
        busy = 0
        try:
            for ln in open(os.path.join(d, "gpu_at_start.txt")):
                i, mem = ln.split(",")[:2]
                if int(i) >= 1:
                    busy = max(busy, int(mem.strip().split()[0]))
        except Exception:
            pass
        rows[(m.group(1), m.group(2) + ("·새바닥" if m.group(3) == "tg" else ""), os.path.basename(R))].append((s, busy))
print("| 씬 | 판 (코드) | 완주 | 오차 % (회별) | ms/프레임 평균 (회별) | fps | 기준 (≤ 16.7 ms · < 10 %) | 측정 시작 카드 1~6 최대 MiB |")
print("|---|---|---|---|---|---|---|---|")
for k in sorted(rows):
    rr = rows[k]
    fin = [s for s, _ in rr if s.get("diverged") is None]
    e = [100 * s["err_mean"] for s in fin]
    ms = [s["ms_mean"] for s, _ in rr]
    ok = fin and len(fin) == len(rr) and sum(ms) / len(ms) <= 16.7 and sum(e) / len(e) < 10
    print(f"| {k[0]} | {k[1]} ({k[2]}) | {len(fin)}/{len(rr)} | {sum(e) / len(e):.2f} ({', '.join('%.2f' % x for x in e)}) | {sum(ms) / len(ms):.2f} ({', '.join('%.2f' % x for x in ms)}) | "
          f"{1000 / (sum(ms) / len(ms)):.1f} | {'만족' if ok else '미달'} | {max(b for _, b in rr)} |" if e else f"| {k[0]} | {k[1]} | 0/{len(rr)} | | | | 미달 | |")

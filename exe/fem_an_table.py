"""절 AN 표: 판 이름 목록 x 궤적 목록의 summary.json -> 오차 % (발산이면 x프레임), 10 % 미만 개수, 평균. 사용: python exe/fem_an_table.py <루트> <판,...> <궤적,...>"""
import json
import os
import sys

R, VS, TS = sys.argv[1], sys.argv[2].split(","), sys.argv[3].split(",")
print("| 궤적 | " + " | ".join(VS) + " |")
print("|---|" + "---|" * len(VS))
acc = {v: [] for v in VS}
for t in TS:
    row = []
    for v in VS:
        p = os.path.join(R, f"{t}_{v}", "summary.json")
        if not os.path.exists(p):
            row.append("-")
            continue
        s = json.load(open(p))
        if s.get("diverged"):
            row.append(f"x{s['diverged']}")
            acc[v].append(None)
        else:
            e = 100 * s["err_mean"]
            row.append(f"{e:.2f}")
            acc[v].append(e)
    print(f"| {t} | " + " | ".join(row) + " |")
out = []
for v in VS:
    ok = [e for e in acc[v] if e is not None]
    nd = sum(1 for e in acc[v] if e is None)
    out.append(f"{sum(1 for e in ok if e < 10)}/{len(acc[v])} (발산 {nd}, 평균 {sum(ok) / max(len(ok), 1):.2f})")
print("| <10 % | " + " | ".join(out) + " |")

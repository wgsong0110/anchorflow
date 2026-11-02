"""절 AN.err3 단위 시험 표: <묶음 디렉토리> 의 <판>_{stretch,shear,ficus,drop1} 에서 블록 주기·진폭 감쇠율, ficus 주기 비·진폭 비, 낙하 반발·튀어 오름, 완주·최대 |g|/|g0|.
진폭 감쇠율 = ln(극값 진폭 끝/처음) / 기록 길이 (1/s) -- 기록 길이가 정답 (80 프레임) 과 달라도 비교되게. 사용: python exe/fem_anerr_unit_table.py <묶음 디렉토리> ..."""
import glob
import json
import math
import os
import re
import sys

GT = dict(stretch=(74.4, 0.447, 80), shear=(87.2, 0.074, 80))
rows = {}
for D in sys.argv[1:]:
    for p in sorted(glob.glob(os.path.join(D, "*_*"))):
        if not os.path.isdir(p):
            continue
        v, t = os.path.basename(p).rsplit("_", 1)
        o = rows.setdefault(v, {})
        sm = os.path.join(p, "summary.json")
        if not os.path.exists(sm):
            o[t] = "중단" if os.path.exists(os.path.join(p, "log.txt")) else ""
            continue
        s = json.load(open(sm))
        if s.get("diverged"):
            o[t] = f"발산 {s['diverged']}"
            continue
        n = s["frames_done"]
        if t in ("stretch", "shear"):
            f = os.path.join(p, "block_ana.txt")
            m = re.search(r"주기 영점 ([\d.na]+) ms .*끝/처음 ([\d.]+)", open(f).read()) if os.path.exists(f) else None
            if m:
                per, amp = float(m.group(1)), float(m.group(2))
                rate = math.log(max(amp, 1e-6)) / (n / 60.0)
                o[t] = f"{per:.1f} ms ({per / GT[t][0]:.2f}) / {rate:+.2f}/s"
        elif t == "ficus":
            f = os.path.join(p, "period.txt")
            m = re.search(r"정답 대비 ([\d.]+)\), 최대 진폭 [\d.]+ m \(비 ([\d.]+)\)", open(f).read()) if os.path.exists(f) else None
            o[t] = f"{m.group(1)} / {m.group(2)} / {100 * s['err_mean']:.2f}%" if m else f"{100 * s['err_mean']:.2f}%"
        elif t == "drop1":
            f = os.path.join(p, "drop_ana.txt")
            l = open(f).read().strip().split("\n")[-1] if os.path.exists(f) else ""
            m = re.search(r"반발 계수 ([\d.]+), 튀어 오름 ([-\d.]+) mm", l)
            o[t] = f"{m.group(1)} / {m.group(2)} mm / {100 * s['err_mean']:.2f}%" if m else f"{100 * s['err_mean']:.2f}%"
for t, (per, amp, n) in GT.items():
    pass
print("| 판 | 블록 신축 주기 (비) / 진폭 감쇠율 | 블록 전단 주기 (비) / 진폭 감쇠율 | ficus 주기 비 / 진폭 비 / 오차 | 낙하 반발 / 튀어 오름 / 오차 |")
print("|---|---|---|---|---|")
print(f"| 정답 MPM | 74.4 ms (1.00) / {math.log(0.447) / (80 / 60):+.2f}/s | 87.2 ms (1.00) / {math.log(0.074) / (80 / 60):+.2f}/s | 1.00 / 1.00 | 0.750 / 83.4 mm |")
print("| 명시적 bnv (절 AK.33) | 75.3 ms (1.01) | 89.8 ms (1.03) | 1.03 / 0.27% | 0.571 / 69.9 mm |")
print("| 명시적 gqw (칸 가우스) | 74.1 ms (1.00) | 85.3 ms (0.98) | 0.84 / 1.04% | 0.571 / 69.8 mm |")
for v in sorted(rows):
    o = rows[v]
    print(f"| {v} | {o.get('stretch', '')} | {o.get('shear', '')} | {o.get('ficus', '')} | {o.get('drop1', '')} |")

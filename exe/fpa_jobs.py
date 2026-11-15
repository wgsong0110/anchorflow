#!/usr/bin/env python3
"""절 CM190 대기열: fpa_jobs.py <장면,...> [A|E]  -> 한 줄에 장면:판:반복. A = 파레토 K + ablation (반복이 바깥 고리, 같은 반복 안에서 장면 번갈아), E = 서브스텝 N 1~128 (1 회)."""
import sys
sc = sys.argv[1].split(","); part = sys.argv[2] if len(sys.argv) > 2 else "A"
V = "def be mid q1 nofpi k2 k4 q27 q64 c075 c15 c20".split()
if part == "A":
    print("\n".join(f"{s}:{v}:{r}" for r in (1, 2, 3) for v in V for s in sc))
else:
    print("\n".join(f"{s}:ss{n}:1" for n in (1, 2, 4, 8, 16, 32, 64, 128) for s in sc))

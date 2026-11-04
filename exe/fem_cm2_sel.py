"""절 CM2 예산 규칙 (정의 변경 첫째): 씬마다 PCG 상한 c (K 2 x 뉴턴 2 고정) = 허용치 0 (펼친 반복 전부 실행) 단독 전체 궤적 최대 프레임 <= 16.7 ms 인 가장 큰 c.
python exe/fem_cm2_sel.py next <시간 디렉토리> <씬...>   -> 다음에 잴 "c 씬" 줄 (선형 맞춤 최대 = a + b c 의 예측 c* 와 c*±1 중 안 잰 것)
python exe/fem_cm2_sel.py pick <시간 디렉토리> <씬...>   -> "씬 c" 줄 (잰 c 중 최대 <= 예산인 가장 큰 c; 없으면 0) + 표
시간 판 이름 = z<c>_<씬> (fem.jsonl ms, 첫 2 프레임 뺌)."""
import json
import os
import sys

import numpy as np

BUD = 16.7


def meas(R, t):
    out = {}
    for d in os.listdir(R):
        if not d.startswith("z") or not d.endswith("_" + t):
            continue
        c = d[1:-len(t) - 1]
        if not c.isdigit():
            continue
        p = os.path.join(R, d, "fem.jsonl")
        if not os.path.exists(p):
            continue
        ms = [json.loads(l).get("ms") for l in open(p)][2:]
        ms = [q for q in ms if q is not None and np.isfinite(q)]
        sm = os.path.join(R, d, "summary.json")
        ok = os.path.exists(sm) and not json.load(open(sm)).get("diverged")
        if ms:
            out[int(c)] = (float(np.max(ms)), float(np.mean(ms)), ok, len(ms))
    return out


def fit(m):
    cs = sorted(m)
    if len(cs) < 2:
        return None
    x = np.array(cs, float); y = np.array([m[c][0] for c in cs])
    b, a = np.polyfit(x, y, 1)
    return a, b


def main():
    mode, R, scs = sys.argv[1], sys.argv[2], sys.argv[3:]
    for t in scs:
        m = meas(R, t)
        if mode == "next":
            f = fit(m)
            if f is None or f[1] <= 0:
                continue
            cs = int(np.floor((BUD - f[0]) / f[1]))
            for c in (cs - 1, cs, cs + 1):
                if c >= 1 and c not in m:
                    print(c, t)
        else:
            good = [c for c in m if m[c][0] <= BUD]
            c = max(good) if good else 0
            f = fit(m)
            print(t, c, "#", "맞춤 최대 = %.3f + %.4f c ms" % f if f else "", "| " + " ".join("c%d %.2f(%.2f)%s" % (q, m[q][0], m[q][1], "" if m[q][2] else "x") for q in sorted(m)))


main()

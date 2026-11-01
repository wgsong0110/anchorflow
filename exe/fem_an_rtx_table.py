"""절 AN 실시간 판 단독 표: <루트>/<판>_r1, _r2 의 fem.jsonl -> 프레임당 ms (첫 2 프레임 뺀 재생 GPU 시간, 1·2 회), 회차 차이 %, 10 프레임 평균 오차, 커널별 ms (프로파일 프레임 평균, 상위 6), 카드 사용률."""
import glob
import json
import os
import sys

import numpy as np

R = sys.argv[1]
runs = sorted({os.path.basename(p)[:-3] for p in glob.glob(os.path.join(R, "*_r[12]"))})
print("| 판 | 프레임 ms 1 회 / 2 회 (차이 %) | 16.7 ms 안 | 10 프레임 평균 오차 % | 커널 ms/프레임 (상위 6, 1 회) | 카드 사용률 % |")
print("|---|---|---|---|---|---|")
for v in runs:
    ms, er, kk, ut = [], None, None, None
    for r in (1, 2):
        p = os.path.join(R, f"{v}_r{r}", "fem.jsonl")
        if not os.path.exists(p):
            ms.append(None); continue
        rs = [json.loads(l) for l in open(p)]
        ms.append(float(np.mean([x["ms"] for x in rs[2:]])) if len(rs) > 2 else None)
        if r == 1:
            er = 100 * float(np.mean([x["err"] for x in rs]))
            ks = [x["kernel_ms"] for x in rs if "kernel_ms" in x]
            if ks:
                tot = {}
                for k in ks:
                    for n, (t, c) in k.items():
                        tot[n] = tot.get(n, 0.0) + t / len(ks)
                kk = ", ".join(f"{n.replace('rt_', '')} {t:.2f}" for n, t in sorted(tot.items(), key=lambda z: -z[1])[:6])
            sp = os.path.join(R, f"{v}_r1", "smi.csv")
            if os.path.exists(sp):
                a = [float(l.split(",")[0]) for l in open(sp) if l.strip() and l.split(",")[0].strip().replace(".", "").isdigit()]
                ut = float(np.mean(a)) if a else None
    d = (100 * (ms[1] - ms[0]) / ms[0]) if (ms[0] and ms[1]) else None
    f = lambda q: f"{q:.2f}" if q is not None else "-"
    ok = "예" if (ms[0] and ms[0] <= 16.7 and (ms[1] is None or ms[1] <= 16.7)) else "아니오"
    print(f"| {v} | {f(ms[0])} / {f(ms[1])} ({f(d)}) | {ok} | {f(er)} | {kk or '-'} | {f(ut)} |")

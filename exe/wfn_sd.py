#!/usr/bin/env python3
"""절 CM193 진단 집계: ssdiag.jsonl (exe/ssdiag_wrap.py, 끝 칸 = 반복 시작 |T f|²) 에서 프레임마다 반복별 ‖g‖²/F² 중앙·최대 (F² = 반복 0 의 |T f|²), 반복 수, 보폭 최소.
사용: wfn_sd.py <ssdiag.jsonl> ..."""
import json, sys
import statistics as st
for fn in sys.argv[1:]:
    print("==", fn)
    for l in open(fn):
        d = json.loads(l); K = d["K"]; rows = d["rows"]
        per = {}; nits = []; smin = []
        for r in rows:
            its = r["its"]
            if not its:
                continue
            F2 = its[0][7] if len(its[0]) > 7 else None
            nits.append(len(its)); smin.append(min(q[1] for q in its))
            for i, q in enumerate(its):
                if F2:
                    per.setdefault(i, []).append(q[0] / F2)
        s = " ".join(f"it{i} {st.median(v):.3g}/{max(v):.3g}" for i, v in sorted(per.items()))
        print(f"f{d['f']:02d} K {K} 반복 {st.mean(nits) if nits else 0:.2f} 보폭최소 {min(smin) if smin else 1:.3g} | ‖g‖²/F² 중앙/최대 {s}")

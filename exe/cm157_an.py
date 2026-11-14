#!/usr/bin/env python3
"""절 CM157 집계: 판 디렉토리들 (summary.json·fem.jsonl·w8info.json) -> 표. 사용: cm157_an.py <루트> <판 이름...>"""
import json, os, sys, statistics as st
root = sys.argv[1]
print("| 판 | 프레임 | 오차 % | 반복 뉴턴 평균 · PCG 평균 | 프레임 ms 평균 (첫 2 프레임 뺌) · 최대 | nlow (프레임 중앙 · 최대) | 확장 커널 | 끝 해시 |")
print("|---|---|---|---|---|---|---|---|")
for n in sys.argv[2:]:
    d = os.path.join(root, n)
    try: s = json.load(open(d + "/summary.json"))
    except Exception: print(f"| {n} | summary 없음 |"); continue
    L = [json.loads(l) for l in open(d + "/fem.jsonl")]
    ms = [x["ms"] for x in L[2:]] or [0]
    nl = [x["nlow"] for x in L if "nlow" in x]
    w8 = ""
    try:
        w = json.load(open(d + "/w8info.json")); w8 = json.dumps(w)[:60]
    except Exception: w8 = "w8info 없음"
    dv = s.get("diverged")
    print(f"| {n} | {s['frames_done']}/{s['frames']}" + (f" (발산 {dv})" if dv else "") +
          f" | {100*s['err_mean']:.3f} | {s['it_nN']['mean']:.2f} · {s['it_nP']['mean']:.2f} | {st.mean(ms):.2f} · {max(ms):.1f} | " +
          (f"{int(st.median(nl))} · {max(nl)}" if nl else "기록 없음") + f" | {w8} | {s.get('x_sha1')} |")

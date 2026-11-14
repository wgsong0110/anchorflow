"""절 CM148 (CPU): 판 nl.jsonl 의 프레임별 꼭짓점 (apex) · 꼬리 (tail) 가우스 점 몫 (q_apex/nq, q_tail/nq) 의 f10·f20·f30 값. 사용: python exe/cm148_an.py <O> <판 ...>"""
import json, os, sys
O = sys.argv[1]
for n in sys.argv[2:]:
    p = os.path.join(O, n, "nl.jsonl")
    if not os.path.exists(p): print(n, "nl 없음"); continue
    d = {}
    for l in open(p):
        o = json.loads(l); d[o["f"]] = (o.get("q_apex", 0) / max(o.get("nq", 1), 1), o.get("q_tail", 0) / max(o.get("nq", 1), 1))
    print(f"{n:16s} 꼭짓점 몫 " + " ".join(f"f{f} {d[f][0]:.3f}" for f in (10, 15, 20, 25, 30, 40) if f in d) + " | 꼬리 몫 " + " ".join(f"f{f} {d[f][1]:.3f}" for f in (10, 20, 30) if f in d), flush=True)

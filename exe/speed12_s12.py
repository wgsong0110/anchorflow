"""절 CM85 묶음 speed12 진단 표 (s12.jsonl; json 만 읽음): python exe/speed12_s12.py <결과 디렉토리> -> markdown (표 (9)).
판마다: 'r' 경계 걸음 수, 판정 수 (X2 켬), [z 0 판] X3g 판정 대 '‖g0‖² + S > 1e6 F²' 일치, 바닥 노드 (띠 질량 > 0) 고착 몫 분포 (√k = 0 · 0 < √k ≤ 1e-6 · 1e-6 < √k < 1),
판정 걸음 대 안 된 걸음의 0 < √k ≤ 1e-6 노드 수, 그 노드의 띠 질량 몫 r · 질량 / m̄ 중앙, 가벼운 (m < m̄) 몫.
"""
import json
import os
import statistics as stt
import sys

O = sys.argv[1]
T = 1e6


def jl(p):
    L = []
    if not os.path.exists(p):
        return L
    for l in open(p):
        try:
            L.append(json.loads(l))
        except Exception:
            pass
    return L


def md(v):
    v = [x for x in v if x is not None]
    return stt.median(v) if v else None


def mn(v):
    v = [x for x in v if x is not None]
    return stt.mean(v) if v else None


def f(v, d=2):
    return "-" if v is None else (f"{v:.{d}f}" if isinstance(v, float) else str(v))


print("## (9) 할 일 1 진단 (s12.jsonl, 'r' 경계 걸음마다; z 0 판 = X3g 반복 1 경계, z 1 판 = 반복 0 경계)\n")
print("| 판 | 걸음 | 판정 (X2 켬) | 판정 일치 (z 0: X3g 대 ‖g0‖²+S) | 바닥 노드 평균 | √k = 0 · 0<√k≤1e-6 · 1e-6<√k<1 노드 평균 | 0<√k≤1e-6 노드: 판정 걸음 · 안 된 걸음 평균 | 그 노드 r 중앙 · m/m̄ 중앙 · 가벼운 몫 | r = 1 (포화) 노드 평균 | S/F² 중앙 (판정 걸음) | KEX |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for n in sorted(os.listdir(O)):
    L = jl(os.path.join(O, n, "s12.jsonl"))
    if not L:
        continue
    z = (json.load(open(os.path.join(O, n, "s12info.json"))) if os.path.exists(os.path.join(O, n, "s12info.json")) else {}).get("z")
    tr = [q for q in L if q["trig"]]
    nt = [q for q in L if not q["trig"]]
    agree = "-"
    if z == 0:
        pr = [((q["g0F"] or 0) + (q["SF"] or 0) > T) or q["g0F"] != q["g0F"] for q in L]
        agree = f"{sum(1 for q, p in zip(L, pr) if bool(q['trig']) == p)}/{len(L)}"
    rw = md([q["r_win"][1] for q in L if q.get("r_win")])
    mw = md([q["m_win"][1] for q in L if q.get("m_win")])
    lw = sum(q["nlight_win"] for q in L) / max(1, sum(q["nwin"] for q in L))
    print(f"| {n} | {len(L)} | {len(tr)} | {agree} | {f(mn([q['nfl'] for q in L]), 0)} | {f(mn([q['n0'] for q in L]), 1)} · {f(mn([q['nwin'] for q in L]), 2)} · {f(mn([q['nmid'] for q in L]), 1)} | "
          f"{f(mn([q['nwin'] for q in tr]), 2)} · {f(mn([q['nwin'] for q in nt]), 2)} | {f(rw, 4)} · {f(mw, 1)} · {f(lw, 2)} | {f(mn([q['nsat'] for q in L]), 1)} | {f(md([q['SF'] for q in tr]), 1) if tr else '-'} | {L[0].get('kex')} |")
print()

"""절 CM88 묶음 speed14 진단 표 (s14.jsonl; json 만 읽음): python exe/speed14_s14.py <결과 디렉토리> -> markdown 표 (13).
'r' 경계 걸음마다 (speed13 진단과 같은 걸음): 필터를 자른 점 (k > 0) 의 J (Ut) · J (Ut − c_a a_L') · J (Ut − c_a a_K') (가속도 기억에서 가벼움 몫 · 벌점 강성 전체 몫을 뺀 관성 예측),
27 노드 c_a|a_n| · c_a|a_L'| · c_a|a_K'| 최대 (mm), cos(a_n, a_L'), 가장 가까운 벌점 면 칸까지 체비셰프 거리 (칸), 몸 전체 무거운 노드 (m ≥ 100 m̄) · 나머지의 |a_L'|/|a_n'| · |a_K'|/|a_n'| 중앙.
a_L'·a_K' = 지난 걸음 끝 벌점 가속도 (−∇E/m) 를 입자로 G2P 해 지금 노드로 입자 질량 가중 P2G 한 값 (a_n 이 실제로 건너오는 길의 근사; p2g 상대 차 = 같은 길로 옮긴 pa 대 실제 gaA).
"""
import json
import os
import statistics as stt
import sys

O = sys.argv[1]


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


def f(v, d=2):
    return "-" if v is None else (f"{v:.{d}f}" if isinstance(v, float) else str(v))


def pc(a, b):
    return "-" if not b else f"{100.0 * a / b:.0f}"


R = []
for n in sorted(os.listdir(O)):
    L = jl(os.path.join(O, n, "s14.jsonl"))
    if L:
        R.append((n, L))
print("## (13) 할 일 1 진단 -- 가속도 기억의 벌점 몫 (s14.jsonl; β < 1 걸음 = speed13 표 (9) 와 같은 걸음, 자른 점 = 필터 k > 0)\n")
print("| 판 | β < 1 걸음 · 판정 | 자른 점 (걸음당) | 자른 점 중 J ≤ 0 %: Ut · Ut − c_a a_L' · Ut − c_a a_K' · 고침 M 기억 Ut + c_a (a_M' − a_P') | 자른 점 J 중앙 (고침 M 기억) | 입자 기억 중 가벼운 노드 (m < 100 m̄) 몫 중앙 · |p_M|/|p_a| 중앙 | 자른 점 J 중앙: Ut · −a_L' · −a_K' | 27 노드 최대 (mm, 중앙): c_a|a_n| · c_a|a_L'| · c_a|a_K'| | cos(a_n, a_L') 중앙 | 벌점 면 칸까지 거리 칸 중앙 [최소]: 가벼움 · 강성 전체 | 몸 전체 |a_L'|/|a_n'| 중앙: 무거운 · 나머지 | |a_K'|/|a_n'|: 무거운 · 나머지 | p2g 상대 차 중앙 |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for n, L in R:
    B = [q for q in L if (q.get("beta") is not None and q["beta"] < 1)]
    nc = sum(q["ncut"] for q in B)
    a = [sum(q.get(k, 0) for q in B) for k in ("cut_Ju_le0", "cut_JuL_le0", "cut_JuK_le0", "cut_JuM_le0")]
    g = lambda key, i, S=B, sc=1.0: md([sc * q[key][i] for q in S if q.get(key)])
    print(f"| {n} | {len(B)} · {sum(1 for q in B if q['trig'])} | {f(nc / max(1, len(B)), 1)} | {pc(a[0], nc)} · {pc(a[1], nc)} · {pc(a[2], nc)} · {pc(a[3], nc)} | {f(g('JuM', 1), 3)} | {f(md([q.get('p_light_share') for q in L]), 3)} · {f(md([q.get('p_M_over_pa') for q in L]), 3)} | "
          f"{f(g('Ju', 1), 3)} · {f(g('JuL', 1), 3)} · {f(g('JuK', 1), 3)} | {f(g('am', 1, B, 1000.0))} · {f(g('aLm', 1, B, 1000.0))} · {f(g('aKm', 1, B, 1000.0))} | {f(g('cosL', 1), 2)} | "
          f"{f(g('dL', 1), 1)} [{f(md([q['dL'][0] for q in B if q.get('dL')]), 0)}] · {f(g('dK', 1), 1)} [{f(md([q['dK'][0] for q in B if q.get('dK')]), 0)}] | "
          f"{f(md([q['heavy_rL'] for q in L]), 3)} · {f(md([q['oth_rL'] for q in L]), 3)} | {f(md([q['heavy_rK'] for q in L]), 3)} · {f(md([q['oth_rK'] for q in L]), 3)} | {f(md([q['p2g_rel'] for q in L]), 3)} |")
print()

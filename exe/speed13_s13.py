"""절 CM86 묶음 speed13 진단 표 (s13.jsonl; json 만 읽음): python exe/speed13_s13.py <결과 디렉토리> -> markdown (표 (9)·(10)).
(9) 판마다: 'r' 경계 걸음, β < 1 걸음 · 그중 판정, 필터를 자른 점 (k > 0) 걸음당, 그 점의 위치별 뒤집힘 (J ≤ 0) 몫: 선분 끝 y_t · ŷ (고착 섞음) · Ut (관성 예측) · Yn (가속도 몫 뺀 예측) · 걸음 끝 외삽,
    벌점 면 노드 낌 · 바닥 띠 노드 낌 · 0 < √k ≤ 1e-6 노드 낌 · 가벼운 (최소 노드 질량 < m̄) 몫; 몸 전체 Ut ≤ 0 · Yn ≤ 0 점 걸음당 (그중 벌점 면 몫).
(10) 노드 가속도 몫 |Ut − Yn| · 속도 몫 |(1 − α_f) dt v| (mm): 벌점 면 노드 대 나머지 (걸음 중앙의 중앙), β < 1 걸음 · 나머지 걸음.
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
    L = jl(os.path.join(O, n, "s13.jsonl"))
    if L:
        R.append((n, L))
print("## (9) 할 일 1 진단 -- 시작점 뒤집힘 필터를 자른 점 (s13.jsonl, 'r' 경계 걸음마다; k > 0 = 필터 선분 [y_z, y_z + 2^-k dy] 에서 J ≤ 0 이 되는 점)\n")
print("| 판 | 걸음 | β < 1 걸음 · 그중 판정 | k > 0 점 (β < 1 걸음당) · 최대 k | k > 0 점 중 J ≤ 0 %: y_t · ŷ · Ut · Yn (가속도 뺌) · 끝 외삽 | k > 0 점 중 벌점 면 노드 · 바닥 띠 · √k≈0 · 가벼운 % | 몸 전체 점 중 벌점 면 노드 낌 % | 몸 전체 Ut ≤ 0 점 · Yn ≤ 0 점 (걸음당; 벌점 면 몫 %) | k > 0 점 J 중앙: y_t · Ut · Yn | k > 0 점 높이 칸 중앙 · 최소 노드 질량/m̄ 중앙 |")
print("|---|---|---|---|---|---|---|---|---|---|")
for n, L in R:
    B = [q for q in L if q["beta"] < 1]
    c = [sum(q["cnt"][i] for q in B) for i in range(32)]
    ca = [sum(q["cnt"][i] for q in L) for i in range(32)]
    k = c[2]
    print(f"| {n} | {len(L)} | {len(B)} · {sum(1 for q in B if q['trig'])} | {f(k / max(1, len(B)), 1)} · {max([q['kmax'] for q in L] or [0])} | "
          f"{pc(c[21], k)} · {pc(c[20], k)} · {pc(c[15], k)} · {pc(c[16], k)} · {pc(c[22], k)} | {pc(c[3], k)} · {pc(c[17], k)} · {pc(c[19], k)} · {pc(c[18], k)} | {pc(ca[1], ca[0])} | "
          f"{f(ca[4] / max(1, len(L)), 1)} · {f(ca[6] / max(1, len(L)), 1)} ({pc(ca[5], ca[4])} · {pc(ca[7], ca[6])}) | "
          f"{f(md([q['kJt'][1] for q in B if q.get('kJt')]), 3)} · {f(md([q['kJu'][1] for q in B if q.get('kJu')]), 3)} · {f(md([q['kJn'][1] for q in B if q.get('kJn')]), 3)} | "
          f"{f(md([q['kz'][1] for q in B if q.get('kz')]), 0)} · {f(md([q['kmm'][1] for q in B if q.get('kmm')]), 2)} |")
print()
print("## (10) 노드 가속도 몫 |Ut − Yn| · 속도 몫 |(1 − α_f) dt v| (mm; 걸음마다 중앙 -> 걸음 중앙) -- 벌점 면 노드 대 나머지 활성 노드\n")
print("| 판 | 벌점 면 노드 / 활성 노드 (걸음 평균) | 가속도 몫: 벌점 면 · 나머지 (모든 걸음) | 가속도 몫: 벌점 면 · 나머지 (β < 1 걸음) | 가속도 몫 최대 벌점 면 · 나머지 (β < 1 걸음 중앙) | 속도 몫: 벌점 면 · 나머지 | k > 0 점 27 노드 가속도 몫 최대 · 속도 몫 최대 중앙 |")
print("|---|---|---|---|---|---|---|")
for n, L in R:
    B = [q for q in L if q["beta"] < 1]
    g = lambda S, key, i: md([1000 * q[key][i] for q in S if q.get(key)])
    print(f"| {n} | {f(sum(q['nnf'] for q in L) / max(1, len(L)), 0)} / {f(sum(q['nnode'] for q in L) / max(1, len(L)), 0)} | {f(g(L, 'acc_nf', 0))} · {f(g(L, 'acc_oth', 0))} | {f(g(B, 'acc_nf', 0))} · {f(g(B, 'acc_oth', 0))} | "
          f"{f(g(B, 'acc_nf', 2))} · {f(g(B, 'acc_oth', 2))} | {f(g(L, 'vel_nf', 0))} · {f(g(L, 'vel_oth', 0))} | {f(md([1000 * q['kam'][1] for q in B if q.get('kam')]))} · {f(md([1000 * q['kvm'][1] for q in B if q.get('kvm')]))} |")
print()

"""절 CM92 finterp 진단 표 (읽기만, json 만): 판 디렉토리의 fi.jsonl (lib/anchorflow/fem_finterp.py) -> 프레임별 det F 질량 평균의 단계별 변화.
사용: python exe/finterp_fi.py <판디렉토리> ... > 표.md
단계 (한 프레임): 입자 J_p0 (프레임 시작) -[P2G]-> 가우스 점 J_q0 -[풀이]-> J_qX (커밋 앞) -[RM]-> J_qE (보간 원천) -[보간]-> 입자 J_pA.
  가우스 점 값은 m_q 가중, 입자 값은 m_p 가중. 보간 규칙만의 몫 = J_ar (같은 입자·같은 가중의 Σ w J_q) − J_pA.
  탄성 판 (레고·ficus) 은 커밋이 없어 J_qX 칸 없음 (P2G·풀이·RM 을 J_q0 -> J_qE 로 묶음).
"""
import json
import os
import sys


def jl(p):
    L = []
    if os.path.exists(p):
        for l in open(p):
            try:
                L.append(json.loads(l))
            except Exception:
                pass
    return L


def g(o, k):
    v = o.get(k)
    return float("nan") if v is None else v


def f3(v):
    return "-" if v != v else f"{v:.3f}"


print("| 판 | 프레임 수 | 입자 J f1 → 끝 (중앙 끝) | 합 ΔP2G (J_q0 − J_p0) | 합 Δ풀이 (J_qX − J_q0) | 합 ΔRM (J_qE − J_qX) | 합 Δ보간 (J_pA − J_qE) | 합 보간 규칙 몫 (J_pA − J_ar) | 합 기하/산술 (J_base − J_ar) | 유클리드 대체 질량 % 평균 | 뒤집힌 점 낀 입자 질량 % 평균 |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for d in sys.argv[1:]:
    L = jl(os.path.join(d, "fi.jsonl"))
    if not L:
        print(f"| {os.path.basename(d.rstrip('/'))} | 0 | - | - | - | - | - | - | - | - | - |")
        continue
    hasx = any((o.get("JqX") or 0) != 0 for o in L)
    sP = sum(g(o, "Jq0") - g(o, "Jp0") for o in L)
    sS = sum(g(o, "JqX") - g(o, "Jq0") for o in L) if hasx else sum(g(o, "JqE") - g(o, "Jq0") for o in L)
    sR = sum(g(o, "JqE") - g(o, "JqX") for o in L) if hasx else float("nan")
    sI = sum(g(o, "JpA") - g(o, "JqE") for o in L)
    sA = sum(g(o, "JpA") - g(o, "Jar") for o in L)
    sB = sum(g(o, "Jbase") - g(o, "Jar") for o in L)
    eu = 100 * sum(g(o, "euc") for o in L) / len(L)
    iv = 100 * sum(g(o, "invq") for o in L) / len(L)
    print(f"| {os.path.basename(d.rstrip('/'))} | {len(L)} | {f3(g(L[0], 'Jp0'))} → {f3(g(L[-1], 'JpA'))} ({f3(g(L[-1], 'medA'))}) | {f3(sP)} | {f3(sS)} | {f3(sR)} | {f3(sI)} | {f3(sA)} | {f3(sB)} | {eu:.1f} | {iv:.1f} |")
print()
print("프레임별 (판마다 f 1·5·10·20·30·40·50·끝): J_p0 · J_q0 · J_qX · J_qE · J_ar · J_pA · 중앙 J_pA · 유클리드 %")
for d in sys.argv[1:]:
    L = jl(os.path.join(d, "fi.jsonl"))
    if not L:
        continue
    print(f"\n{os.path.basename(d.rstrip('/'))}:")
    for i in sorted(set([0, 4, 9, 19, 29, 39, 49, len(L) - 1])):
        if i < len(L):
            o = L[i]
            print(f"  f{o['f']}: " + " · ".join(f3(g(o, k)) for k in ("Jp0", "Jq0", "JqX", "JqE", "Jar", "JpA", "medA")) + f" · {100 * g(o, 'euc'):.1f}")

"""절 CM96 묶음 convwhy 표 (로컬, numpy): R2 lbours/convwhy 를 내려받은 디렉토리의 판별 cw.jsonl (다시 풀기 기록) + 저장 판 fem.jsonl (대조) -> 마크다운.

사용: python exe/convwhy_tab.py <내려받은 디렉토리> > convwhy_table.md
r_k = 뉴턴 반복 k 시작 ‖g‖/F (F² = 걸음 첫 반복 |T f|², 실행기 허용치 판정과 같은 척도), εN = 1e-2.
수렴 차수 p = log(r_{k+1}/r_k) / log(r_k/r_{k−1}) (2 = 2 차, 1 = 1 차), 감소율 ρ = r_{k+1}/r_k.
보폭 = 장벽 실현 가능 보폭 2^-bk (재질 판은 에너지 선탐색 없음). κ = PCG 랑초시 삼중대각 극값 비 (전처리 연산자; J 판은 야코비 = 대각 척도 헤시안).
"""
import json
import math
import os
import sys

import numpy as np

EPS = 1e-2
D = sys.argv[1]
FR = [10, 12, 15, 20, 25, 30]
VAR = ["R", "B", "a", "b", "c", "d", "ab", "ac", "ad", "bc", "bd", "cd", "abcd", "a4", "a4b", "a4c", "a4d", "J", "Jd"]
VNAME = dict(R="H 그대로 (대조)", B="진단 바탕 (PCG 5)", a="(a) PCG 정확", b="(b) 장벽 강성 ×0.01", c="(c) 소성 끔", d="(d) 빈 노드 고정",
             ab="a+b", ac="a+c", ad="a+d", bc="b+c", bd="b+d", cd="c+d", abcd="a+b+c+d", a4="(a) PCG 400 (뉴턴 12)", a4b="a4+b", a4c="a4+c", a4d="a4+d", J="야코비 PCG (조건수)", Jd="야코비 PCG + d")


def load(n):
    p = os.path.join(D, n, "cw.jsonl")
    if not os.path.exists(p):
        return None
    L = [json.loads(l) for l in open(p) if l.strip()]
    return {q["f"]: q for q in L if q.get("mode") == "load"}


def med(v):
    v = [q for q in v if q is not None and math.isfinite(q)]
    return float(np.median(v)) if v else None


def f(x, d=2):
    if x is None:
        return "-"
    if isinstance(x, str):
        return x
    if abs(x) >= 1e4 or (0 < abs(x) < 1e-2):
        return f"{x:.1e}"
    return f"{x:.{d}f}"


def step_stats(q):
    s = q["steps"][0]
    its = [o for o in s["its"] if not o.get("stop") or (o.get("r") is not None and 0 < o["r"] <= EPS)]   # 멈춘 줄은 허용치 도달 (새로 잰 r) 만 -- xit 문으로 멈춘 줄 (R 판) 은 r 이 안 새로 잼
    rs = [o["r"] for o in its if o.get("r") is not None]
    nconv = next((i for i, r in enumerate(rs) if r <= EPS), None)
    ran = [o for o in its if not o.get("stop") and (o.get("nP") or 0) >= 0]
    rho = [rs[i + 1] / rs[i] for i in range(len(rs) - 1) if rs[i] > 0]
    pord = [math.log(rs[i + 1] / rs[i]) / math.log(rs[i] / rs[i - 1]) for i in range(1, len(rs) - 1)
            if rs[i] > 0 and rs[i - 1] > 0 and rs[i + 1] > 0 and abs(math.log(rs[i] / rs[i - 1])) > 1e-3]
    bk = [o.get("bk") for o in ran if o.get("bk") is not None]
    kap = [o.get("kap") for o in ran if o.get("kap")]
    spk = sum(1 for i in range(len(rs) - 1) if rs[i] > 0 and rs[i + 1] > 5 * rs[i])                  # 튐: 한 반복에 r 5 배 넘게 커짐
    return dict(spk=spk, nconv=nconv, nit=len(ran), r0=rs[0] if rs else None, rlast=rs[-1] if rs else None, rmin=min(rs) if rs else None,
                rho=med(rho), rho_tail=med(rho[-5:]) if rho else None, p=med(pord), full=(sum(1 for b in bk if b == 0) / len(bk)) if bk else None,
                stepmed=med([2.0 ** -b for b in bk]), kap0=(ran[0].get("kap") if ran else None), kapmax=max(kap) if kap else None,
                nP=med([o.get("nP") for o in ran]), Prel=med([o.get("Prel") for o in ran]), g2end=s.get("g2end"), rs=rs, bks=bk,
                apex=(q.get("counts") or {}).get("q_apex"), rm=(q.get("counts") or {}).get("p_rm"))


def nc_str(st, cap):
    return str(st["nconv"]) if st["nconv"] is not None else f">{cap}"


out = []
out.append("# 절 CM96 convwhy -- wolf 접촉 걸음 수렴 원인 가르기 (진단, 결과 불변)\n")
out.append("r = 반복 시작 ‖g‖/F, εN = 1e-2 (H8 허용치), 뉴턴 상한 25 (R = H 그대로 6 · xit, a4 12, J 3), 프레임 = 그 프레임을 푸는 걸음 (시작 상태 = 저장 판 그 프레임 앞). "
           "ρ = r_{k+1}/r_k, p = 경험 수렴 차수 (2 = 2 차), 보폭 = 장벽 실현 가능 보폭 2^-bk, κ = PCG 랑초시 (MG 전처리 연산자; 야코비 판은 대각 척도 헤시안) -- PCG 5 판의 κ 는 5 걸음 추정이라 하한.\n")
for sc, H in (("8", "H8"), ("5", "H5n")):
    out.append(f"\n## {H} -- 판별 요약 (프레임 {', '.join('f%d' % q for q in FR)})\n")
    out.append("| 판 | εN 도달 반복 (프레임별) | 튐 수 (6 프레임 합) | 끝 r 중앙 | 최소 r 중앙 | ρ 중앙 · 끝 5 | 차수 p 중앙 | 온 보폭 몫 · 보폭 중앙 | PCG /반복 · 상대 잔차 | κ 첫 반복 중앙 · 최대 |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    rows = {}
    for v in VAR:
        L = load(f"r{sc}_{v}")
        if not L:
            out.append(f"| {VNAME[v]} | (없음) | | | | | | | | |")
            continue
        S = {fr: step_stats(L[fr]) for fr in FR if fr in L}
        rows[v] = S
        cap = 6 if v == "R" else (3 if v.startswith("J") else (12 if v.startswith("a4") else 25))
        nc = " · ".join(nc_str(S[fr], cap) for fr in FR if fr in S)
        g = lambda k: med([S[fr][k] for fr in S])
        out.append(f"| {VNAME[v]} | {nc} | {sum(S[fr]['spk'] for fr in S)} | {f(g('rlast'))} | {f(g('rmin'))} | {f(g('rho'))} · {f(g('rho_tail'))} | {f(g('p'))} | {f(g('full'))} · {f(g('stepmed'))} | "
                   f"{f(g('nP'), 0)} · {f(g('Prel'))} | {f(g('kap0'), 0)} · {f(max([S[fr]['kapmax'] or 0 for fr in S]) or None, 0)} |")
    for v in ("B", "a", "c", "ac", "abcd"):
        if v not in rows:
            continue
        out.append(f"\n### {H} {VNAME[v]} -- 프레임별 r 이력 (반복 0, 1, 2, ..; 보폭 2^-bk)\n")
        out.append("| 프레임 | r_k | bk | 꼭짓점 점 · 반환 입자 |")
        out.append("|---|---|---|---|")
        for fr in FR:
            if fr not in rows[v]:
                continue
            s = rows[v][fr]
            out.append(f"| f{fr} | {' '.join(f(q) for q in s['rs'][:26])} | {' '.join(str(q) for q in s['bks'][:25])} | {s['apex']} · {s['rm']} |")
    sv = os.path.join(D, f"sv{sc}", "fem.jsonl")
    if os.path.exists(sv) and "R" in rows:
        E = {json.loads(l)["t"] + 1: json.loads(l) for l in open(sv) if l.strip()}
        out.append(f"\n### {H} 대조 -- 저장 판 그 프레임 (실행 그대로) 대 R (같은 설정 다시 풀기)\n")
        out.append("| 프레임 | 저장 판 뉴턴 · PCG · 걸음 끝 g2 · K | R 뉴턴 반복 · 걸음 끝 g2 |")
        out.append("|---|---|---|")
        for fr in FR:
            e = E.get(fr)
            r = rows["R"].get(fr)
            if not e or not r:
                continue
            it = e.get("it") or {}
            out.append(f"| f{fr} | {it.get('nN')} · {it.get('nP')} · {f((e.get('uf') or {}).get('g2', [None])[0])} · {e.get('K')} | {r['nit']} · {f(r['g2end'])} |")
print("\n".join(out))

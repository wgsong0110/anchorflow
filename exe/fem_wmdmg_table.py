"""절 CM19 표 (수박 손상 모델, 묶음 wmdmg): python exe/fem_wmdmg_table.py <wmdmg 디렉토리> [정답 wmd_stats.json]
판 디렉토리 <판>_<셀>_<a|b> (판 s = SNH, ct·cu = CDM ψ_c 정수압 인장·단축 인장, pt·pu = 위상장 같은 두 근거; 셀 c15·c2·c25·c3),
단독 시간 z_<판>_<셀>. 정답 손상 몫 = 다시 돌린 GF 정답 logJp > 0 (절 CM13.4 wmd_stats.json)."""
import json
import math
import os
import sys

B = sys.argv[1]
GT = json.load(open(sys.argv[2])) if len(sys.argv) > 2 and os.path.exists(sys.argv[2]) else None
VAR = [("s", "SNH"), ("ct", "CDM ψ_c 정수압 인장"), ("cu", "CDM ψ_c 단축 인장"), ("pt", "위상장 G_c 정수압 인장"), ("pu", "위상장 G_c 단축 인장")]
CEL = [("c15", 1.5), ("c2", 2.0), ("c25", 2.5), ("c3", 3.0)]
FR = (4, 10, 30, 50, 70, 100)


def rows(d):
    p = os.path.join(d, "fem.jsonl")
    out = []
    if os.path.exists(p):
        for ln in open(p):
            try:
                out.append(json.loads(ln))
            except Exception:
                pass
    return out


def js(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def f(x, k=2):
    return "-" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{k}f}"


def pc(x, k=2):
    return "-" if x is None else f"{100 * x:.{k}f}"


def med(v):
    v = sorted(q for q in v if q is not None)
    return v[len(v) // 2] if v else None


print("정답 (GF 다시 돌림) logJp > 0 몫 %: " + (" · ".join(f"f{t} {pc(GT['gt'][t]['gt0'])}" for t in (5, 10, 40, 100)) if GT else "-"))
print()
print("| 셀 | 판 | 실행 | 발산 | 평균 오차 % (끝난 프레임) | 오차 f4·f10·f30·f50·f70·f100 % | 마지막 프레임 logJp>0 % | d>0.5 % | 합집합 % | d>0.1 % | d 평균 | 가우스 점 (중앙) | 노드 (중앙) | 위상장 잔차 최대 |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for cn, cv in CEL:
    for vn, vl in VAR:
        for r in ("a", "b"):
            d = os.path.join(B, f"{vn}_{cn}_{r}")
            if not os.path.isdir(d):
                continue
            sm = js(os.path.join(d, "summary.json")); R = rows(d); st = js(os.path.join(d, "dmg_stats.json"))
            if sm is None and not R:
                print(f"| {cv} | {vl} | {r} | (실행 안 끝남) | | | | | | | | | | |"); continue
            dv = sm.get("diverged") if sm else None
            em = sm.get("err_mean_done") if sm else None
            ec = "·".join(f(100 * R[t - 1]["err"], 1) if len(R) >= t and R[t - 1].get("err") is not None and (dv is None or t < dv) else "x" for t in FR)
            ls = st["frames"][-1] if st and st.get("frames") else {}
            res = [max(q["dmg"]["pfres"]) for q in R if "dmg" in q and q["dmg"].get("pfres")]
            print(f"| {cv} | {vl} | {r} | {'완주' if (sm and dv is None) else ('x%s' % dv if dv else '?')} | {pc(em)} | {ec} | {pc(ls.get('lj0'))} | {pc(ls.get('d05'))} | {pc(ls.get('uni'))} | {pc(ls.get('d01'))} | {f(ls.get('dmean'), 3)} | "
                  f"{med([q.get('nq') for q in R])} | {med([q.get('nn') for q in R])} | {('%.1e' % max(res)) if res else '-'} |")
print()
print("단독 시간 (카드 1, 판 하나씩 차례로, 진단·덤프 없이 `--gq_mass --elx snh [--dmg ...]`; 재생 GPU 시간, 첫 2 프레임 뺌; 실행 중 그 카드의 다른 계산 프로세스 최대 수)")
print()
print("| 셀 | " + " | ".join(vl for _, vl in VAR) + " |")
print("|---|" + "---|" * len(VAR))
for cn, cv in CEL:
    cells = []
    for vn, _ in VAR:
        d = os.path.join(B, f"z_{vn}_{cn}"); sm = js(os.path.join(d, "summary.json")); R = rows(d)
        if sm is None:
            cells.append("-"); continue
        ms = [q["ms"] for q in R[2:] if q.get("ms") is not None and math.isfinite(q["ms"])]
        ap_ = open(os.path.join(d, "apps_during.txt")).read().strip() if os.path.exists(os.path.join(d, "apps_during.txt")) else "?"
        dv = sm.get("diverged")
        cells.append(f"{f(sm.get('ms_mean'), 1)} 평균 · {f(max(ms) if ms else None, 1)} 최대 ms ({'완주' if dv is None else 'x%d' % dv}, 오차 {pc(sm.get('err_mean_done'))} %, 다른 프로세스 {ap_})")
    print(f"| {cv} | " + " | ".join(cells) + " |")
print()
for cn, cv in CEL:
    sm = js(os.path.join(B, f"ct_{cn}_a", "summary.json")) or js(os.path.join(B, f"pt_{cn}_a", "summary.json"))
    if sm and sm.get("dmg"):
        g = sm["dmg"]
        print(f"- 셀 {cv}: 칸 {g['cell']:.5f} m, ψ⁺ 정수압 인장 {g['psi_tens']:.3f} · 단축 인장 {g['psi_uni']:.3f} · 순수 전단 {g['psi_shear']:.3f}, l0 = 칸 {g['l0']:.5f} m, "
              f"G_c (정수압) {512 / 27 * g['psi_tens'] * g['l0']:.4g} · (단축) {512 / 27 * g['psi_uni'] * g['l0']:.4g}")

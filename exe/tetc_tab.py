"""절 CM106 표 (로컬, 읽기만): R2 lbours/tetc 에서 내려받은 판 디렉토리 -> 씬별 표 (한국어 markdown).

사용: python exe/tetc_tab.py <판 디렉토리 뿌리>  > 표.md
- 오차·발산 = summary.json, wolf = heap_rows.json (exe/wolfh10_heap.py 판마다), 수박 = wm_rows.json (exe/tet1_wm.py 판마다), ficus = period.json.
- 보존 = tetc.jsonl em (|Σm_n − Σm_p| / Σm_p) · eP (max_a |ΣP_n − ΣP_p| / Σ m_p|v_p|) 의 판 최대.
- 질량 대 강성 = tetc.jsonl rat (k_n = (λ+2μ) Σ_q V_q |F_qᵀ∇w_n|², ω_n h = sqrt(k_n/m_n) h).
"""
import glob
import json
import os
import sys

import numpy as np

ROOT = sys.argv[1]


def jl(p):
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def js(p):
    return json.load(open(p)) if os.path.exists(p) else None


def rng(v, p=2):
    v = [q for q in v if q is not None and np.isfinite(q)]
    if not v:
        return "-"
    return f"{np.mean(v):.{p}f}" + (f" [{min(v):.{p}f}~{max(v):.{p}f}]" if len(v) > 1 else "")


def runs(pre):
    return sorted(d for d in glob.glob(os.path.join(ROOT, pre)) if os.path.isdir(d))


def base(d):
    s = js(os.path.join(d, "summary.json")) or {}
    e = s.get("err_mean") if s.get("err_mean") is not None else s.get("err_mean_done")
    L = jl(os.path.join(d, "tetc.jsonl"))
    return dict(n=os.path.basename(d), err=e, div=s.get("diverged"), done=s.get("frames_done"), L=L)


def errline(R):
    nd = [r for r in R if r["div"]]
    ok = [100 * r["err"] for r in R if not r["div"] and r["err"] is not None]
    dv = ", ".join(f"f{r['div']}" for r in nd)
    return f"{len(nd)}/{len(R)}" + (f" ({dv})" if dv else ""), rng(ok)


def cons(R):
    em = [max((q["em"] for q in r["L"] if q.get("em") is not None), default=None) for r in R]
    eP = [max((q["eP"] for q in r["L"] if q.get("eP") is not None), default=None) for r in R]
    em = [q for q in em if q is not None]; eP = [q for q in eP if q is not None]
    return (f"{max(em):.1e}" if em else "-"), (f"{max(eP):.1e}" if eP else "-")


print("## 결과 (TC = 사면체 단위 질량·운동량, 노드 ¼; T9S = 대조 기록 판)\n")
print("| 씬 | 판 | 발산 | 오차 % | 보존: 질량 · 운동량 상대 차 (판 최대) |")
print("|---|---|---|---|---|")
SC = [("wolf", "cW_*", "cSW"), ("수박", "cM_*", "cSM"), ("레고", "cL_*", "cSL"), ("ficus", "cF_*", "cSF"), ("bread", "cB_*", "cSB")]
ALL = {}
for nm, pc, ps in SC:
    for lab, pat in (("TC", pc), ("T9S", ps)):
        R = [base(d) for d in runs(pat)]
        ALL[(nm, lab)] = (R, runs(pat))
        if not R:
            continue
        a, b = errline(R)
        c1, c2 = cons(R)
        print(f"| {nm} | {lab} ({len(R)}) | {a} | {b} | {c1} · {c2} |")

print("\n## wolf (heap_rows.json; 높이 = 바닥 위 mm)\n")
print("| 판 | 바닥 위 질량중심 f30 · f50 | 꼭대기 f50 99 % · 최대 | 안식각 f50 ° | 채움 f50 | 입자 det F 중앙 f30 · f50 (det≤0 수) | 이탈 판 · 최대 개수 |")
print("|---|---|---|---|---|---|---|")
GT = None
for lab in ("TC", "T9S"):
    R, D = ALL.get(("wolf", lab), ([], []))
    rr = []
    for d in D:
        h = js(os.path.join(d, "heap_rows.json")) or {}
        GT = GT or h.get("정답")
        for g, v in h.items():
            if g != "정답":
                rr += v["runs"]
    if not rr:
        continue
    ok = [r for r in rr if not r["div"]]
    f30 = [r["hp"].get("30") for r in ok]; f50 = [r["hp"].get("50") for r in ok]
    m50 = [r["m"].get("50", {}) for r in ok]; m30 = [r["m"].get("30", {}) for r in ok]
    esc = [r["bb"] for r in rr if r["bb"]]
    det = " / ".join(f"{a.get('detF', float('nan')):.2f} · {b.get('detF', float('nan')):.2f} ({b.get('ninv', '-')})" for a, b in zip(m30, m50) if b)
    print(f"| {lab} ({len(rr)}) | {rng([q - 480 for q in f30 if q], 0)} · {rng([q - 480 for q in f50 if q], 0)} | {rng([m.get('z99') for m in m50], 0)} · {rng([m.get('zmax') for m in m50], 0)} | "
          f"{rng([m.get('rep') for m in m50], 1)} | {rng([m.get('ppv') for m in m50], 2)} | {det or '-'} | {sum(1 for b in esc if b[0] > 20)}/{len(rr)} · {max((b[0] for b in esc), default=0)} |")
if GT:
    g = GT["gt"]["50"]
    print(f"| 정답 | 53 (절 CM102) | {g['z99']:.0f} · {g['zmax']:.0f} | {g['rep']:.1f} | {g['ppv']:.2f} | 1 | - |")

print("\n## 수박 (wm_rows.json, exe/tet1_wm.py)\n")
print("| 판 | 튐 프레임/판 · 최대 속력 m/s | 이탈 판 · 최대 개수 | 손상 % 끝 (f5 · f6 · f8) | 되튐 m | 꿈틀거림 [1,30) · [30,100) m/s |")
print("|---|---|---|---|---|---|")
for lab in ("TC", "T9S"):
    R, D = ALL.get(("수박", lab), ([], []))
    rr, G = [], None
    for d in D:
        h = js(os.path.join(d, "wm_rows.json")) or {}
        G = G or h.get("gt")
        for g, v in (h.get("rows") or {}).items():
            rr += [q for q in v if "rms" in q]
    if not rr:
        continue
    w = lambda r_, a, b: float(np.nanmean(r_["rms"][a:b])) if len(r_.get("rms", [])) > a else float("nan")
    dm = lambda f: rng([100 * r["dmg"].get(str(f), np.nan) for r in rr if r.get("dmg")])
    last = rng([100 * list(r["dmg"].values())[-1] for r in rr if r.get("dmg")])
    nf = [max(r["far"]) for r in rr]
    print(f"| {lab} ({len(rr)}) | {rng([r['jump'] for r in rr], 1)} · {rng([max(r['vmax']) for r in rr], 1)} | {sum(1 for q in nf if q > 20)}/{len(rr)} · {max(nf)} | "
          f"{last} ({dm(5)} · {dm(6)} · {dm(8)}) | {rng([r['bounce'] for r in rr], 3)} | {rng([w(r, 0, 29) for r in rr], 3)} · {rng([w(r, 29, 200) for r in rr], 3)} |")

print("\n## ficus 주기 (period.json; 정답 대비 비)\n")
for lab in ("TC", "T9S"):
    R, D = ALL.get(("ficus", lab), ([], []))
    P = [js(os.path.join(d, "period.json")) for d in D]
    P = [p for p in P if p]
    if P:
        print(f"- {lab}: 주기 비 {rng([p['ratio'] for p in P], 2)}, 진폭 비 {rng([p['amp_ratio'] for p in P], 2)}, 정답 주기 {P[0]['gt']['T_zc']:.3f} s")

print("\n## 노드 질량 대 강성 (tetc.jsonl rat; 판 하나, 프레임 f1 · 가운데 · 마지막 기록; ω h = 국소 진동수 x 프레임 시간)\n")
print("| 씬 | 판 | 프레임 | 강성 노드 수 | 질량 ≤ 0 · < 1e-6 · < 1e-3 x 중앙 (강성 노드 중) | ω h 중앙 · 99 % · 99.9 % · 최대 | ω h > 10 · > 100 노드 수 |")
print("|---|---|---|---|---|---|---|")
for nm, pc, ps in SC:
    for lab in ("T9S", "TC"):
        R, D = ALL.get((nm, lab), ([], []))
        if not R:
            continue
        r0 = R[0]
        Q = [q for q in r0["L"] if "rat" in q and "nK" in q["rat"]]
        if not Q:
            continue
        sel = sorted({Q[min(1, len(Q) - 1)]["f"], Q[len(Q) // 2]["f"], Q[-1]["f"]})
        for q in Q:
            if q["f"] not in sel:
                continue
            a = q["rat"]
            print(f"| {nm} | {lab} {r0['n']} | f{q['f']} | {a['nK']} | {a['n0']} · {a['n6']} · {a['n3']} | {a.get('wh50', float('nan')):.3g} · {a.get('wh99', float('nan')):.3g} · {a.get('wh999', float('nan')):.3g} · {a.get('whmax', float('nan')):.3g} | {a.get('nwh10', '-')} · {a.get('nwh100', '-')} |")

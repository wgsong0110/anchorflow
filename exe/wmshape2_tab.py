"""절 CM41 묶음 wmshape2 표: python exe/wmshape2_tab.py <결과 디렉토리> <gt_jp 디렉토리> -> markdown (stdout) + <결과>/wmshape2_curves.json.
판 = exe/wmshape2_spec.txt, 묶음 = 이름의 끝 _숫자 뺀 것 (비결정 반복). 절 CM40 표 (exe/wmshape_tab.py, 같은 정의) 를 이 판 목록으로 다시 만들고 앞에 이 절 표를 둠.
정답 = <결과>/gt/gt_wmshape.json (절 CM40) + gt_wmshape2.json (exe/wmshape2_gt.py). 손상 = logJp > 0 몫 (정답 끝 6.51 %), 생긴 E = 전 구간 최대 (운동 + 중력 위치 + SNH 탄성) − 처음 7.13 J.
진단 ws2 (lib/anchorflow/fem_wmshape2.py): 커밋 n 등분 RM 호출의 NACC 갈래별 logJp 증분 (0 압축 꼭짓점 · 1 인장 꼭짓점 · 2 타원 전단 인장 쪽 · 3 타원 전단 압축 쪽), 걸음 변형 증분 막대."""
import contextlib
import io
import json
import math
import os
import re
import runpy
import sys

import numpy as np

O, JP = sys.argv[1], sys.argv[2]
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ["WSH_SPEC"] = os.path.join(HERE, "wmshape2_spec.txt")
os.environ["WSH_CUR"] = "wmshape2_curves.json"
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    sys.argv = [os.path.join(HERE, "wmshape_tab.py"), O, JP]
    T = runpy.run_path(os.path.join(HERE, "wmshape_tab.py"), run_name="__main__")
base = buf.getvalue()
W, grp, G, rng, val, dE, shape_err, res = T["W"], T["grp"], T["G"], T["rng"], T["val"], T["dE"], T["shape_err"], T["res"]
spec = [l.split(None, 4) for l in open(os.environ["WSH_SPEC"]) if l.strip() and not l.startswith("#")]
G2 = {}
p = os.path.join(O, "gt", "gt_wmshape2.json")
if os.path.exists(p):
    G2 = {int(k): v for k, v in json.load(open(p)).items()}


def ws2(n):
    return {q["t"] + 1: q["ws2"] for q in W[n]["rows"] if q.get("ws2")}


def summ(n):
    try:
        return json.load(open(os.path.join(O, n, "summary.json")))
    except Exception:
        return None


def zms(g):
    """단독 시간 판 z<묶음>_a·_b: (평균, 최대) ms (첫 2 프레임 뺌), 발산."""
    o = []
    for r_ in ("a", "b"):
        n = f"z{g}_{r_}"
        f = os.path.join(O, n, "fem.jsonl")
        if not os.path.exists(f):
            continue
        rows = [json.loads(l) for l in open(f) if l.strip()]
        m = [x["ms"] for x in rows if x["t"] >= 2 and x.get("ms") is not None and math.isfinite(x["ms"])]
        s = summ(n)
        o.append((r_, float(np.mean(m)) if m else None, float(np.max(m)) if m else None, (s or {}).get("diverged"), s is None))
    return o


def fmt(x, d=3):
    return "-" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{d}f}"


print("# 절 CM41 묶음 wmshape2 -- 수박 깨짐 모양 2 (C2 손상 과다 원인 + 고침 후보)\n")
print(__doc__.split("\n", 1)[1].strip() + "\n")

print("## A. 후보 요약 (비결정 판 + 단독 z a·b; 정답 f4 H 0.505 · 질량중심 0.230, 끝 r95 0.895 · 질량중심 0.166, 손상 6.51 %)\n")
print("| 묶음 | 판 | 메모리·발산 (판정 + 단독) | f4 H · 질량중심 z m | f10 H · r95 m | f100 r95 · 질량중심 z m | 끝 손상 % [범위] | 손상 흩어짐 (판 사이 표준편차 %p) | 모양 오차 높이 · 퍼짐 | 오차 % | 생긴 E J | 단독 ms 평균/최대 a · b |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for g, ns in grp.items():
    last = []
    for n in ns:
        by = W[n]["by"]
        last.append(by[max(by)]["d0"] if by else None)
    bad = sum(1 for n in ns if (summ(n) is None) or summ(n).get("diverged"))
    z = zms(g)
    zbad = sum(1 for q in z if q[3] or q[4])
    s_ = [summ(n) for n in ns]
    er = [100 * s["err_mean_done"] for s in s_ if s]
    se = [shape_err(n) for n in ns]
    lv = [q for q in last if q is not None]
    sd = 100 * float(np.std(lv)) if len(lv) > 1 else None
    zs = " · ".join(f"{fmt(q[1], 1)}/{fmt(q[2], 0)}" + (f" (발산 f{q[3]})" if q[3] else "") for q in z) or "-"
    print(f"| {g} | {len(ns)} | {bad}/{len(ns)} + {zbad}/{len(z)} | {rng([val(n, 4, 'H') for n in ns])} · {rng([val(n, 4, 'zc') for n in ns])} | {rng([val(n, 10, 'H') for n in ns])} · {rng([val(n, 10, 'r95') for n in ns])} | "
          f"{rng([val(n, 100, 'r95') for n in ns])} · {rng([val(n, 100, 'zc') for n in ns])} | {rng(last, 2, True)} | {fmt(sd, 2)} | {rng([q[0] for q in se], 3)} · {rng([q[1] for q in se], 3)} | {rng(er, 2)} | {rng([dE(n) for n in ns], 2)} | {zs} |")

WIN = ((4, 4), (5, 5), (6, 10), (11, 20), (21, 50), (51, 100))
BRN = ("압축 꼭짓점", "인장 꼭짓점", "타원 인장 쪽", "타원 압축 쪽")
print("\n## B. 커밋 RM 갈래별 logJp 증분 (가우스 점, 묶음 판 평균; 받아들인 시도만) -- 구간마다 Σ 양 증분 · Σ 음 증분 (점 수 x logJp) · 0 넘은 호출 수\n")
print("| 묶음 | 구간 | " + " | ".join(f"{b} Σ+ · Σ− · 0 넘음" for b in BRN) + " | 0 넘은 점 수 · 그 점 변형 증분 평균 | 등분 n > 1 점 몫 · 평균 n |")
print("|---|---|" + "---|" * (len(BRN) + 2))
CUR2 = {}
for g, ns in grp.items():
    if not any(ws2(n) for n in ns):
        continue
    CUR2[g] = {}
    for a, b in WIN:
        acc = np.zeros((5, 4)); x0 = np.zeros(2); nsb = np.zeros(2); nde = 0.0; k = 0
        for n in ns:
            w = ws2(n)
            fr = [w[t] for t in range(a, b + 1) if t in w and w[t].get("br")]
            if not fr:
                continue
            k += 1
            for q in fr:
                acc += np.array(q["br"]); x0 += np.array(q["x0"]); nsb += np.array(q["nsub"]); nde += sum(q["hde"])
        if k == 0:
            continue
        acc /= k; x0 /= k; nsb /= k; nde /= k
        CUR2[g][f"{a}-{b}"] = dict(br=acc.tolist(), x0=x0.tolist(), nsub=nsb.tolist(), nde=nde)
        cells = " | ".join(f"{acc[i, 0]:.3g} · {acc[i, 1]:.3g} · {acc[i, 3]:.0f}" for i in range(4))
        print(f"| {g} | f{a}~{b} | {cells} | {x0[0]:.0f} · {(x0[1] / x0[0]) if x0[0] > 0 else float('nan'):.3g} | {(nsb[0] / nde) if nde > 0 else float('nan'):.3f} · {(nsb[1] / nde) if nde > 0 else float('nan'):.2f} |")

print("\n## C. 양 증분 크기 (호출당 Δ logJp, log10 막대 −6~2 0.5 간격) · 걸음 변형 증분 ‖sym ∇ΔU‖ (점당, log10 막대 −5~3) -- 묶음 첫 판, 구간 합에서 분위\n")


def hq(h, lo, qs=(0.5, 0.9, 0.99)):
    h = np.array(h, dtype=np.float64)
    if h.sum() <= 0:
        return ["-"] * len(qs)
    c = np.cumsum(h) / h.sum()
    return [f"{10 ** (lo + (int(np.searchsorted(c, q)) + 0.5) / 2):.2g}" for q in qs]


print("| 묶음 | 구간 | 호출당 Δ logJp 50·90·99 % | 호출 수 | 변형 증분 50·90·99 % | 변형 증분 최대 |")
print("|---|---|---|---|---|---|")
for g, ns in grp.items():
    w = ws2(ns[0])
    if not w or not any(q.get("hdl") for q in w.values()):
        continue
    for a, b in WIN:
        fr = [w[t] for t in range(a, b + 1) if t in w and w[t].get("hdl")]
        if not fr:
            continue
        hl = np.sum([q["hdl"] for q in fr], 0); he = np.sum([q["hde"] for q in fr], 0)
        print(f"| {g} | f{a}~{b} | {' · '.join(hq(hl, -6))} | {int(hl.sum())} | {' · '.join(hq(he, -5))} | {max(q['demax'] for q in fr):.3g} |")

print("\n## D. 손상 입자 상태 (입자; lb = logJp 구간 0~0.01 · 0.01~0.1 · 0.1~0.4 · > 0.4 몫 %, dJt = 손상 입자 전체 부피 비 J·Jp 10·50·90 %, nJt = 새 손상 입자) -- 묶음 첫 판\n")
print("| 묶음 | 프레임 | lb % | dJt | nJt | 탄성 J (손상) | 프레임 Δ logJp 1·50·99 % |")
print("|---|---|---|---|---|---|---|")
for t in (4, 5, 6, 8, 10, 15, 20, 30, 50, 100):
    if t in G2:
        q = G2[t]
        print(f"| 정답 | f{t} | {' · '.join(f'{100 * v:.2f}' for v in q['lb'])} | {q.get('dJt')} | {q.get('nJt')} | {q.get('dJ')} | {q.get('dlq')} |")
    for g, ns in grp.items():
        w = ws2(ns[0])
        if t in w and w[t].get("lb") is not None:
            q = w[t]
            print(f"| {g} | f{t} | {' · '.join(f'{100 * v:.2f}' for v in q['lb'])} | {q.get('dJt')} | {q.get('nJt')} | {q.get('dJ')} | {q.get('dlq')} |")

FC = [(n, q) for g, ns in grp.items() for n in ns for t, w in ws2(n).items() for q in (w.get("fc") or [])]
if FC:
    print("\n## E. 후보 (b) 첫 접촉 프레임 다시 풀기 (E = 운동 + 중력 위치 + SNH 탄성 J; 받음 = E 끝 ≤ E 시작 + 문턱)\n")
    print("| 판 | 프레임 | E 시작 | 보통 E (운동·탄성) | 뉴턴 상한 E (운동·탄성) | 받음 | ms 보통 · 다시 |")
    print("|---|---|---|---|---|---|---|")
    for n, q in FC:
        print(f"| {n} | f{q['t']} | {q['E0'][0]:.3f} | {q['Ea'][0]:.3f} ({q['Ea'][1]:.3f}·{q['Ea'][2]:.3f}) | {q['Eb'][0]:.3f} ({q['Eb'][1]:.3f}·{q['Eb'][2]:.3f}) | {'예' if q['acc'] else '아니오'} | {q['ms'][0]:.0f} · {q['ms'][1]:.0f} |")

print("\n# 절 CM40 과 같은 표 (이 판 목록)\n")
print(base.split("\n", 1)[1])
try:
    cur = json.load(open(os.path.join(O, "wmshape2_curves.json")))
    cur["ws2"] = CUR2
    cur["gt2"] = G2
    json.dump(cur, open(os.path.join(O, "wmshape2_curves.json"), "w"))
except Exception as ex:
    print(f"(곡선 json 덧붙이기 실패 {ex!r})", file=sys.stderr)

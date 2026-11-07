"""절 CM29 묶음 wminv 표 (수박 뒤집힘 필터 국소화). 사용: python exe/wminv_tab.py <출력 디렉토리>
오차 = 끝난 프레임 평균. 손상 = 위치 유한 입자 중 logJp > 0 몫 (%, 마지막 프레임; 정답 6.51 %). 튐 = 질량중심 z 가 그때까지 최저점보다 오른 최대 (m).
E = 운동 + 중력 위치 + SNH 탄성 (입자 F, J). 시간 = b 판 (진단은 재생 밖 --erec 만) 프레임당 재생 GPU ms (첫 2 프레임 뺌; 쪼갠 프레임은 모든 시도 합), 카드 하나에 판 하나.
"""
import glob
import json
import math
import os
import sys

O = sys.argv[1]
NAME = {"base": "기준 판 (--ksplit 3)", "d": "진단 다시 (빈 팔분 높이)", "ksit": "쪼갠 시도 뉴턴 8 x PCG 20", "node": "(a) 노드별 보폭", "jb": "(b) 필터 대신 걸린 점만 J 장벽", "noinv": "(c) 필터 끔 (SNH)"}


def js(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def load(n):
    p = os.path.join(O, n, "fem.jsonl")
    if not os.path.exists(p):
        return None, None
    return [json.loads(l) for l in open(p) if l.strip()], js(os.path.join(O, n, "summary.json"))


def nm(v):
    return " + ".join(NAME.get(x, x) for x in v.split("+"))


def fm(x, f="{:.3g}"):
    return "-" if (x is None or (isinstance(x, float) and not math.isfinite(x))) else f.format(x)


def bounce(zs):
    lo, b = float("inf"), 0.0
    for z in zs:
        if z is None:
            continue
        lo = min(lo, z); b = max(b, z - lo)
    return b


def res(s, rows):
    if s is None:
        return "도는 중" if rows else "없음"
    return ("발산 f%d" % s["diverged"]) if s.get("diverged") else ("완주 %d" % s["frames_done"])


def by_f(rows):
    return {q["t"] + 1: q for q in rows}


def E(q):
    e = (q or {}).get("en") or {}
    return None if e.get("Eel") is None else e["KE"] + e["PE"] + e["Eel"]


def fs(q, k, i=None):
    v = ((q or {}).get("en") or {}).get("fs", {}).get(k)
    return v if (i is None or v is None) else v[i]


def lj_last(rows):
    for q in reversed(rows or []):
        v = (q.get("en") or {}).get("lj0")
        if v is not None and q.get("err") is not None and math.isfinite(q["err"]):
            return v
    return None


def ms(rows):
    v = [q["ms"] for q in rows[2:] if q.get("ms") is not None and math.isfinite(q["ms"])]
    return (sum(v) / len(v), max(v)) if v else (None, None)


GT = js(os.path.join(O, "gt_stats.json"))
G = {r["t"]: r for r in GT["frames"]} if GT else {}


def gE(f):
    r = G.get(f)
    return None if r is None else r["KE"] + r["PE"] + r["Esnh"]


ORD = list(NAME)
vs = sorted({os.path.basename(p)[:-2] for p in glob.glob(os.path.join(O, "*_a"))}, key=lambda v: (ORD.index(v.split("+")[0]) if v.split("+")[0] in ORD else 99, v.count("+"), v))
print("# 절 CM29 묶음 wminv -- 수박 뒤집힘 필터 국소화 (기준 판 = 장벽 바닥 + 보간 교체 K 1 + --ksplit 3)\n")
print(__doc__.split("\n", 1)[1])
print("## 1. 판별 (각 판 비결정 두 번 a · b)\n")
print("| 판 | 결과 a · b | 오차 % a · b | 손상 % a · b | 튐 m a · b | E f3 → f4 → f6 J (a) | f4 SNH 탄성 J (a) | f4 J q1 · 중앙 · J<0.9 (a) | f5 J>1.11 (a) | ms 평균 · 최대 (b) | 쪼갠 프레임 a · b |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for v in vs:
    R = {r: load(f"{v}_{r}") for r in "ab"}
    if not R["a"][0]:
        continue
    a = by_f(R["a"][0])
    c = [" · ".join(res(R[r][1], R[r][0]) for r in "ab"),
         " · ".join(fm(100 * R[r][1]["err_mean_done"], "{:.2f}") if R[r][1] else "-" for r in "ab"),
         " · ".join(fm(100 * lj_last(R[r][0]) if lj_last(R[r][0]) is not None else None, "{:.2f}") for r in "ab"),
         " · ".join(fm(bounce([q.get("en", {}).get("zc") for q in R[r][0]]), "{:.3f}") if R[r][0] else "-" for r in "ab"),
         " → ".join(fm(E(a.get(f)), "{:.2f}") for f in (3, 4, 6)),
         fm((a.get(4, {}).get("en") or {}).get("Eel"), "{:.2f}"),
         f"{fm(fs(a.get(4), 'Jq', 0), '{:.3f}')} · {fm(fs(a.get(4), 'Jq', 3), '{:.3f}')} · {fm(fs(a.get(4), 'J090'), '{:.3f}')}",
         fm(fs(a.get(5), "J111"), "{:.4f}")]
    m_, x_ = ms(R["b"][0]) if R["b"][0] else (None, None)
    c.append(f"{fm(m_, '{:.1f}')} · {fm(x_, '{:.0f}')}")
    c.append(" · ".join(str(sum(1 for q in R[r][0] if (q.get("K") or 1) > 1)) if R[r][0] else "-" for r in "ab"))
    print(f"| {nm(v)} | " + " | ".join(c) + " |")
if G:
    print(f"| 정답 | - | - | 6.51 | 0.092 | " + " → ".join(fm(gE(f), "{:.2f}") for f in (3, 4, 6)) + f" | {fm(G[4]['Esnh'], '{:.2f}')} | {fm(G[4]['Jq'][0], '{:.3f}')} · {fm(G[4]['Jq'][3], '{:.3f}')} · {fm(G[4]['J090'], '{:.3f}')} | {fm(G[5]['J111'], '{:.4f}')} | - | - |")

print("\n## 2. 진단 (1): 원래 필터 (--ls_inv) 가 거는 가우스 점 -- 기준 판 a, 프레임 3~6 시도별 (--wi_rec)\n")
print("높이 = 프레임 시작 팔분 표본점 (팔분 입자 질량 중심) 의 바닥 위 거리 / 칸 크기 (빈 팔분은 그 칸 입자 평균 높이, 프레임 끝). 칸 입자 수 = 그 점의 칸 (8 팔분) 안 입자. 노드 질량 비 = 27 노드 최소 질량 / 노드 질량 중앙값.")
print("J0 = 그 반복 시작 J, J1 = 온 걸음 (보폭 1) J, Jmin = [0, 1] 최소 J. 분위 = 최소 · 10 % · 중앙 · 90 % · 최대.\n")
print("| 판 | 프레임 · K | 걸린 점 (다른 점 수) · 빈 팔분 몫 | 반복별 걸린 점 수 (걸음·반복:수:최대 k) | k | 높이 칸 | 칸 입자 수 (≤1 · ≤2 몫) | 노드 질량 비 최소 (< 1e-2 몫) | J0 | J1 | Jmin | det F_q |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
Q = lambda a, f="{:.3g}": " · ".join(fm(x, f) for x in a) if a else "-"
for v in vs:
    rows, s = load(f"{v}_a")
    if not rows:
        continue
    for q in rows:
        for t in (q.get("wi") or {}).get("rec") or []:
            pi = " ".join(f"{i}:{n}:{k}" for i, n, k in t.get("per_it", [])[:12]) + (" …" if len(t.get("per_it", [])) > 12 else "")
            print(f"| {nm(v)} | f{t['frame']} · K{t['K']} | {t.get('nq', 0)} · {fm(t.get('frac_oct_empty'), '{:.2f}')} | {pi or '-'} | {Q(t.get('k'), '{:.0f}')} | {Q(t.get('h_cells'), '{:.2f}')} | {Q(t.get('nump'), '{:.0f}')} ({fm(t.get('frac_np1'), '{:.2f}')} · {fm(t.get('frac_np2'), '{:.2f}')}) | "
                  f"{Q(t.get('mmin_rel'))} ({fm(t.get('frac_mr1e2'), '{:.2f}')}) | {Q(t.get('J0'))} | {Q(t.get('J1'))} | {Q(t.get('Jmin'))} | {Q(t.get('detFq'))} |")

print("\n## 3. 국소 처리 통계 (a 판, 마지막 시도; st = [국소 자른 반복, 전역 되돌림, 최대 노드 k, 자른 노드 수 합, 프레임 끝 표시 점, 새 표시, 반복 수])\n")
for v in vs:
    rows, s = load(f"{v}_a")
    if not rows or not any("wi" in q for q in rows):
        continue
    if "node" not in v and "jb" not in v:
        continue
    b = by_f(rows)
    seg = []
    for f in range(3, 13):
        q = b.get(f)
        if q and "wi" in q:
            seg.append(f"f{f} K{q.get('K')} {q['wi']['st']}")
    tot = [0] * 7
    for q in rows:
        for i, x in enumerate((q.get("wi") or {}).get("st", [])[:7]):
            tot[i] = tot[i] + x if i != 2 else max(tot[i], x)
    print(f"- {nm(v)}: " + "; ".join(seg) + f" | 전 프레임 합 {tot}")

print("\n## 4. 첫 접촉 프레임 3~8 (a 판·정답): E, 운동, SNH 탄성, J q1 · q50 · q99, J<0.9 · J>1.11, 손상 %, K\n")
print("| 판 | f | E | 운동 | SNH 탄성 | J q1 · q50 · q99 | J<0.9 · J>1.11 | logJp>0 % | K |")
print("|---|---|---|---|---|---|---|---|---|")
for v in vs:
    rows, s = load(f"{v}_a")
    if not rows:
        continue
    b = by_f(rows)
    for f in range(3, 9):
        q = b.get(f)
        if q is None:
            continue
        e = q.get("en") or {}; f_ = e.get("fs") or {}
        print(f"| {nm(v)} | {f} | {fm(E(q), '{:.2f}')} | {fm(e.get('KE'), '{:.2f}')} | {fm(f_.get('Esnh'), '{:.2f}')} | {' · '.join(fm(f_.get('Jq', [None] * 7)[i], '{:.3f}') for i in (0, 3, 6))} | "
              f"{fm(f_.get('J090'), '{:.3f}')} · {fm(f_.get('J111'), '{:.4f}')} | {fm(100 * f_['lj_pos'] if f_.get('lj_pos') is not None else None, '{:.2f}')} | {q.get('K')} |")
for f in range(3, 9):
    r = G.get(f)
    if r is None:
        continue
    print(f"| 정답 | {f} | {fm(gE(f), '{:.2f}')} | {fm(r['KE'], '{:.2f}')} | {fm(r['Esnh'], '{:.2f}')} | {' · '.join(fm(r['Jq'][i], '{:.3f}') for i in (0, 3, 6))} | {fm(r['J090'], '{:.3f}')} · {fm(r['J111'], '{:.4f}')} | "
          f"{fm(100 * r['lj_pos'] if r.get('lj_pos') is not None else None, '{:.2f}')} | - |")

print("\n## 5. f4 뉴턴 반복 (a 판 --nrec: 걸음·반복별 원래 필터 보폭 si · 장벽 k · 쓴 보폭 st, 걸음 끝 ‖g‖/F)\n")
for v in vs:
    rows, s = load(f"{v}_a")
    if not rows:
        continue
    q4 = by_f(rows).get(4) or {}
    nr = q4.get("nr")
    if not nr:
        continue
    it = [r for r in nr["it"] if r.get("ran")]
    seg = " ".join(f"{r['k']}.{r['i']}:{fm(r.get('si'), '{:.3g}')}/{r.get('kb')}/{fm(r.get('st'), '{:.3g}')}" for r in it[:24])
    ge = nr.get("gend") or []
    print(f"- {nm(v)} (K {q4.get('K')}): {seg}{' …' if len(it) > 24 else ''} | 걸음 끝 ‖g‖/F 중앙 {fm(sorted(ge)[len(ge) // 2] if ge else None)} 최대 {fm(max(ge) if ge else None)}")

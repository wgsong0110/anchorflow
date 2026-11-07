"""절 CM28 묶음 wmfrac 표 (수박 깨짐). 사용: python exe/wmfrac_tab.py <출력 디렉토리>
튐 = 질량중심 z 가 그때까지 최저점보다 오른 최대 (m, erec zc). 손상 = 위치 유한 입자 중 logJp > 0 몫 (%, 마지막 프레임; 정답 f100 6.51 %).
E = 운동 + 중력 위치 + SNH 탄성 (입자 F, J). 첫 접촉 통계 en.fs (lib/anchorflow/wmfrac_stats.py) -- 정답은 gt_stats.json (GF h5 f_tensor = 탄성 F_e, 다시 돌린 정답 logJp).
"""
import glob
import json
import math
import os
import sys

O = sys.argv[1]
NAME = {"base": "기준 판 (--ksplit 3)", "seed": "(a) 씨 영역 β 3e9", "ljp0": "(b) logJp 모으기 = 응집력 평균", "ljmax": "(b) logJp 모으기 = 최대",
        "sub16": "(c) 걸음 안 RM 16 등분", "sub64": "(c) 걸음 안 RM 64 등분", "rmsnh": "(d) RM 식 SNH 와 맞춤",
        "ksr1": "(e) 쪼갠 시도 HHT ρ∞ 1", "ksit": "(e) 쪼갠 시도 뉴턴 8 x PCG 20"}
ANA = [("ref", "분석: K 64 고정 · 뉴턴 8 x PCG 20 (잘 푼 기준)"), ("ref_r1", "분석: ref + ρ∞ 1 (수치 감쇠 없음)"), ("ref_norm", "분석: ref + RM 끔"), ("ks3_norm", "분석: 기준 판 + RM 끔")]


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


GT = js(os.path.join(O, "gt_stats.json"))
G = {r["t"]: r for r in GT["frames"]} if GT else {}


def gE(f):
    r = G.get(f)
    return None if r is None else r["KE"] + r["PE"] + r["Esnh"]


vs = sorted({os.path.basename(p)[:-2] for p in glob.glob(os.path.join(O, "*_a"))}, key=lambda v: (v.count("+"), list(NAME).index(v.split("+")[0]) if v.split("+")[0] in NAME else 99, v))
print("# 절 CM28 묶음 wmfrac -- 수박 깨짐 (기준 판 = 장벽 바닥 + 보간 교체 K 1 + --ksplit 3)\n")
print(__doc__.split("\n", 1)[1])
print("## 1. 판별 (각 판 비결정 두 번 a · b)\n")
print("| 판 | 결과 a · b | 평균 오차 % a · b | 튐 m a · b | 손상 % a · b (마지막) | 손상 % f10 a · b | E f3 → f4 → f6 J (a) | f4 SNH 탄성 J (a) | f4 J 중앙 · J<0.9 몫 (a) | f5 J>1.11 몫 (a) | f10 logJp<−0.2 % (a) | 쪼갠 프레임 a · b |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for v in vs:
    R = {}
    for r in "ab":
        R[r] = load(f"{v}_{r}")
    if not R["a"][0]:
        continue
    cells = []
    cells.append(" · ".join(res(R[r][1], R[r][0]) for r in "ab"))
    cells.append(" · ".join(fm(100 * R[r][1]["err_mean_done"], "{:.2f}") if R[r][1] else "-" for r in "ab"))
    cells.append(" · ".join(fm(bounce([q.get("en", {}).get("zc") for q in R[r][0]]), "{:.3f}") if R[r][0] else "-" for r in "ab"))
    cells.append(" · ".join(fm(100 * (R[r][0][-1].get("en", {}).get("lj0") or float("nan")), "{:.2f}") if R[r][0] else "-" for r in "ab"))
    cells.append(" · ".join(fm(100 * (by_f(R[r][0]).get(10, {}).get("en", {}).get("lj0") or float("nan")), "{:.2f}") if R[r][0] else "-" for r in "ab"))
    a = by_f(R["a"][0])
    cells.append(" → ".join(fm(E(a.get(f)), "{:.2f}") for f in (3, 4, 6)))
    cells.append(fm((a.get(4, {}).get("en") or {}).get("Eel"), "{:.2f}"))
    cells.append(f"{fm(fs(a.get(4), 'Jq', 3), '{:.3f}')} · {fm(fs(a.get(4), 'J090'), '{:.3f}')}")
    cells.append(fm(fs(a.get(5), "J111"), "{:.4f}"))
    cells.append(fm(100 * fs(a.get(10), "lj_c02") if fs(a.get(10), "lj_c02") is not None else None, "{:.2f}"))
    cells.append(" · ".join(str(sum(1 for q in R[r][0] if (q.get("K") or 1) > 1)) if R[r][0] else "-" for r in "ab"))
    print(f"| {nm(v)} | " + " | ".join(cells) + " |")
if G:
    print(f"| 정답 | - | - | 0.092 | 6.51 (f100) | {fm(100 * G[10]['lj_pos'] if G.get(10, {}).get('lj_pos') is not None else None, '{:.2f}')} | "
          + " → ".join(fm(gE(f), "{:.2f}") for f in (3, 4, 6)) + f" | {fm(G[4]['Esnh'], '{:.2f}')} | {fm(G[4]['Jq'][3], '{:.3f}')} · {fm(G[4]['J090'], '{:.3f}')} | {fm(G[5]['J111'], '{:.4f}')} | "
          + f"{fm(100 * G[10]['lj_c02'] if G.get(10, {}).get('lj_c02') is not None else None, '{:.2f}')} | - |")
print("\n## 2. 첫 접촉 프레임 3~8 (분석 판·기준 판 a·정답): E = 운동 + 위치 + SNH 탄성 (J), 탄성 SNH · NHB (J), 바닥 띠 (바닥 위 < 0.03) SNH 몫, J 분위 1·50·99 %, J<0.9 · J>1.11 몫, 손상 · 다져짐 %\n")
print("| 판 | 프레임 | E | 운동 | SNH 탄성 · NHB 탄성 | 바닥 띠 몫 | J q1 · q50 · q99 | J<0.9 · J>1.11 | logJp>0 · <−0.2 % | 뒤집힌 입자 | K |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
rows_ana = [("base_a", "기준 판 a")] + ANA
for n, lab in rows_ana:
    rows, s = load(n)
    if not rows:
        continue
    b = by_f(rows)
    for f in range(3, 9):
        q = b.get(f)
        if q is None:
            continue
        e = q.get("en") or {}
        f_ = e.get("fs") or {}
        Eb = f_.get("Ebot"); Es = f_.get("Esnh")
        print(f"| {lab} ({res(s, rows)}) | {f} | {fm(E(q), '{:.2f}')} | {fm(e.get('KE'), '{:.2f}')} | {fm(Es, '{:.2f}')} · {fm(f_.get('Enhb'), '{:.2f}')} | {fm(Eb / Es if (Eb is not None and Es) else None, '{:.2f}')} | "
              f"{' · '.join(fm(f_.get('Jq', [None] * 7)[i], '{:.3f}') for i in (0, 3, 6))} | {fm(f_.get('J090'), '{:.3f}')} · {fm(f_.get('J111'), '{:.4f}')} | "
              f"{fm(100 * f_['lj_pos'] if f_.get('lj_pos') is not None else None, '{:.2f}')} · {fm(100 * f_['lj_c02'] if f_.get('lj_c02') is not None else None, '{:.2f}')} | {f_.get('Jinv', '-')} | {q.get('K')} |")
for f in range(3, 9):
    r = G.get(f)
    if r is None:
        continue
    print(f"| 정답 | {f} | {fm(gE(f), '{:.2f}')} | {fm(r['KE'], '{:.2f}')} | {fm(r['Esnh'], '{:.2f}')} · {fm(r['Enhb'], '{:.2f}')} | {fm(r['Ebot'] / r['Esnh'] if r['Esnh'] else None, '{:.2f}')} | "
          f"{' · '.join(fm(r['Jq'][i], '{:.3f}') for i in (0, 3, 6))} | {fm(r['J090'], '{:.3f}')} · {fm(r['J111'], '{:.4f}')} | "
          f"{fm(100 * r['lj_pos'] if r.get('lj_pos') is not None else None, '{:.2f}')} · {fm(100 * r['lj_c02'] if r.get('lj_c02') is not None else None, '{:.2f}')} | {r['Jinv']} | - |")
print("\n## 3. 분석 판 결과·f4 뉴턴 (nrec)\n")
for n, lab in ANA:
    rows, s = load(n)
    if not rows:
        continue
    b = by_f(rows)
    q4 = b.get(4, {})
    nr = q4.get("nr") or {}
    ge = nr.get("gend")
    print(f"- {lab}: {res(s, rows)}, 끝난 프레임 오차 {fm(100 * s['err_mean_done'] if s else None, '{:.2f}')} %, 튐 {fm(bounce([q.get('en', {}).get('zc') for q in rows]), '{:.3f}')} m, "
          f"f4 걸음별 끝 ‖g‖/F 최대 {fm(max(ge) if ge else None)} · 중앙 {fm(sorted(ge)[len(ge) // 2] if ge else None)}, 덜 풀린 걸음 f4 {sum((q4.get('uf') or {}).get('fail') or [])}/{len((q4.get('uf') or {}).get('fail') or [])}, 프레임당 재생 ms 중앙 {fm(sorted(q['ms'] for q in rows)[len(rows) // 2], '{:.0f}')}")

"""절 CM32 묶음 wmfrac2 표 (수박 첫 접촉: 절 CM30 (a)+(b) 위 뒤집힘 필터 노드별 국소화 (fi) · CM27 (d) · 쪼갠 걸음 8 x 20). 사용: python exe/wmfrac2_tab.py <출력 디렉토리>
오차 = 끝난 프레임 평균. 손상 = 위치 유한 입자 중 logJp > 0 몫 (%, 마지막 프레임; 정답 6.51 %). 튐 = 질량중심 z 가 그때까지 최저점보다 오른 최대 (m).
E = 운동 + 중력 위치 + SNH 탄성 (입자 F, J). 시간 = b 판 (진단은 재생 밖 --erec 만) 프레임당 재생 GPU ms (첫 2 프레임 뺌; 쪼갠 프레임은 모든 시도 합), 카드 하나에 판 하나 (동시 4 판까지).
덜 풀린 걸음 = 받아들인 시도 (마지막 시도) 의 덜 풀린 걸음 합 (판정 커널), K1 덜 풀림 프레임 = 첫 시도 K 1 이 덜 풀린 프레임 수.
"""
import glob
import json
import math
import os
import sys

O = sys.argv[1]
NAME = {"base": "기준 판", "oct": "(a)", "bl": "(b)", "fi": "필터 국소", "d": "CM27 (d)", "kit": "쪼갠 걸음 8 x 20", "nks": "쪼개기 끔"}
CELLS = {"c2": "2", "c25": "2.5", "c3": "3"}


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


def cell(v):
    return next((CELLS[x] for x in v.split("+") if x in CELLS), "1.5")


def nm(v):
    x = [NAME.get(x, x) for x in v.split("+") if x not in CELLS]
    return " + ".join(x) if x else NAME["base"]


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
vs = sorted({os.path.basename(p)[:-2] for p in glob.glob(os.path.join(O, "*_a")) if not os.path.basename(p).startswith("vid_")},
            key=lambda v: (float(cell(v)), sum(1 for x in v.split("+") if x not in CELLS and x != "base"), [ORD.index(x) if x in ORD else 99 for x in v.split("+") if x not in CELLS]))
print("# 절 CM32 묶음 wmfrac2 -- 수박 첫 접촉: (a) --gq_oct 0 · (b) --ws_bar 4 · 필터 국소 --ws_fi · CM27 (d) --y0loc 3 --ue be --uf_res · 쪼갠 걸음 --ks_it 8,20 (기준 판 = 장벽 바닥 + 보간 교체 K 1 + --ksplit 3; 쪼개기 끔 표시 없으면 --ksplit 3)\n")
print(__doc__.split("\n", 1)[1])
print("## 1. 판별 (각 판 비결정 두 번 a · b)\n")
print("| 셀 | 판 | 결과 a · b | 오차 % a · b | 손상 % a · b | 튐 m a · b | E f3 → f4 → f6 J (a) | f4 SNH 탄성 J (a) | f4 J q1 · 중앙 · J<0.9 (a) | f5 J>1.11 (a) | ms 평균 · 최대 (b) | 쪼갠 프레임 a · b | 덜 풀린 걸음 a · b | K1 덜 풀림 프레임 a · b |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
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
    c.append(" · ".join(str(sum(sum((q.get("uf") or {}).get("fail") or []) for q in R[r][0])) if R[r][0] else "-" for r in "ab"))
    c.append(" · ".join(str(sum(1 for q in R[r][0] if ((q.get("ks") or {}).get("tries") or [[1, 0]])[0][1] > 0)) if R[r][0] else "-" for r in "ab"))
    print(f"| {cell(v)} | {nm(v)} | " + " | ".join(c) + " |")
if G:
    print(f"| - | 정답 | - | - | 6.51 | 0.092 | " + " → ".join(fm(gE(f), "{:.2f}") for f in (3, 4, 6)) + f" | {fm(G[4]['Esnh'], '{:.2f}')} | {fm(G[4]['Jq'][0], '{:.3f}')} · {fm(G[4]['Jq'][3], '{:.3f}')} · {fm(G[4]['J090'], '{:.3f}')} | {fm(G[5]['J111'], '{:.4f}')} | - | - | - | - |")

print("\n## 2. 진단 (4) -- 첫 시도 K 1 을 무엇이 자르나 (a 판 --w2rec, 모든 시도; 반복마다 kg = 원래 전역 뒤집힘 필터 k, kb = 원래 전역 장벽 k (온 걸음 후보), km = 쓴 노드 k 최대, 되돌림 fb, 라운드 0 어긴 장벽 표본점 b0 · 뒤집힘 가우스 점 i0, 최종 확인 bF · iF)\n")
for v in vs:
    rows, s_ = load(f"{v}_a")
    if not rows or not any("w2" in q for q in rows):
        continue
    b = by_f(rows)
    print(f"- 셀 {cell(v)} {nm(v)}:")
    for f in (3, 4, 5, 6):
        q = b.get(f)
        if not q or "w2" not in q:
            continue
        for at in q["w2"]:
            it = at["it"]
            seg = " ".join(f"{r['k']}.{r['i']}:{r['kg']}/{r['kb']}/{r['km']}{'*' if r['fb'] else ''}(b{r['bar'][0]},i{r['inv'][0]};b{r['bar'][-1]},i{r['inv'][-1]})" for r in it[:16])
            print(f"  - f{f} K{at['K']} 덜 풀린 걸음 {at['nf']}: {seg}{' …' if len(it) > 16 else ''}")
    # 전 프레임: K 1 시도에서 자른 반복 (km > 0) 의 원인 몫
    n1 = nfb = nbar = ninv = 0
    for q in rows:
        for at in q.get("w2") or []:
            if at["K"] != 1:
                continue
            for r in at["it"]:
                if r["km"] > 0:
                    n1 += 1; nfb += r["fb"]; nbar += int(r["bar"][0] > 0); ninv += int(r["inv"][0] > 0)
    print(f"  - 전 프레임 K 1 시도에서 보폭을 자른 반복 {n1} (되돌림 {nfb}; 첫 라운드 장벽 관통 있음 {nbar}, 뒤집힘 있음 {ninv})")

print("\n## 3. 첫 접촉 프레임 3~8 (a 판·정답): E, 운동, SNH 탄성, J q1 · q50 · q99, J<0.9 · J>1.11, 손상 %, K\n")
print("| 셀 | 판 | f | E | 운동 | SNH 탄성 | J q1 · q50 · q99 | J<0.9 · J>1.11 | logJp>0 % | K |")
print("|---|---|---|---|---|---|---|---|---|---|")
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
        print(f"| {cell(v)} | {nm(v)} | {f} | {fm(E(q), '{:.2f}')} | {fm(e.get('KE'), '{:.2f}')} | {fm(f_.get('Esnh'), '{:.2f}')} | {' · '.join(fm(f_.get('Jq', [None] * 7)[i], '{:.3f}') for i in (0, 3, 6))} | "
              f"{fm(f_.get('J090'), '{:.3f}')} · {fm(f_.get('J111'), '{:.4f}')} | {fm(100 * f_['lj_pos'] if f_.get('lj_pos') is not None else None, '{:.2f}')} | {q.get('K')} |")
for f in range(3, 9):
    r = G.get(f)
    if r is None:
        continue
    print(f"| - | 정답 | {f} | {fm(gE(f), '{:.2f}')} | {fm(r['KE'], '{:.2f}')} | {fm(r['Esnh'], '{:.2f}')} | {' · '.join(fm(r['Jq'][i], '{:.3f}') for i in (0, 3, 6))} | {fm(r['J090'], '{:.3f}')} · {fm(r['J111'], '{:.4f}')} | "
          f"{fm(100 * r['lj_pos'] if r.get('lj_pos') is not None else None, '{:.2f}')} | - |")

print("\n## 4. f4 뉴턴 반복 (a 판 --nrec: 걸음·반복별 원래 필터 보폭 si · 장벽 k · 쓴 보폭 st, 걸음 끝 ‖g‖/F)\n")
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
    print(f"- 셀 {cell(v)} {nm(v)} (K {q4.get('K')}): {seg}{' …' if len(it) > 24 else ''} | 걸음 끝 ‖g‖/F 중앙 {fm(sorted(ge)[len(ge) // 2] if ge else None)} 최대 {fm(max(ge) if ge else None)}")

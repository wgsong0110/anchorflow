"""절 CM88 묶음 speed14 넘는 프레임 표 (json 만 읽음): python exe/speed14_over.py <결과 디렉토리> -> markdown 표 (11)·(12).
단독 판 (rr.jsonl·s6.jsonl 있는 판) 마다 프레임 3~ (t ≥ 2, 절 CM86 (4) 와 같은 범위) 중 벽시계 W > 16.7 ms 인 프레임 하나하나:
  W = 그림 끝 간격 (판정 기준) · Ws = 시뮬 스트림 프레임 끝 간격 · a = 시뮬 GPU (실행기 last_ms) · T = 시간 예산 시계가 잰 시뮬 구간 · b·c = 가우시안 갱신 · 래스터화 (CUDA 이벤트)
  · 나머지 = W − (a + b + c) (호스트 빈틈·스트림 겹침·래스터 끝 흔들림) · host · gap · lag · ng (그래프 재생 수) · 두 시도 (fem.jsonl ks.on 에 1) · 쪼갠 걸음 수.
  몫 = 같은 판의 넘지 않은 프레임 중앙값 대비 늘어난 ms (a · b+c · 나머지), 주된 곳 = 가장 많이 늘어난 칸.
"""
import json
import os
import statistics as stt
import sys

O = sys.argv[1]
LIM = 1000.0 / 60.0


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


def f(v, d=2):
    return "-" if v is None else (f"{v:.{d}f}" if isinstance(v, (float, int)) else str(v))


SCN = {"L": "레고", "W": "wolf", "F": "ficus", "B": "bread", "M": "수박"}
rows, summ = [], []
for n in sorted(os.listdir(O)):
    d = os.path.join(O, n)
    if not (n.startswith("x") and os.path.exists(os.path.join(d, "s6.jsonl")) and os.path.exists(os.path.join(d, "rr.jsonl"))):
        continue
    rr = {q["t"]: q for q in jl(os.path.join(d, "rr.jsonl")) if "t" in q}
    s6 = {q["t"]: q for q in jl(os.path.join(d, "s6.jsonl")) if "t" in q}
    fe = {q["t"]: q for q in jl(os.path.join(d, "fem.jsonl")) if "t" in q}
    F = []
    for t in sorted(set(rr) & set(s6)):
        if t < 2 or s6[t].get("W") is None:
            continue
        q = dict(s6[t]); r = rr[t]
        q["a"] = r.get("a", q.get("a")); q["b"] = r.get("b", 0.0); q["c"] = r.get("c", 0.0)
        q["rest"] = q["W"] - (q["a"] + q["b"] + q["c"])
        ks = (fe.get(t, {}).get("ks") or {})
        q["two"] = int(any(v == 1 for v in (ks.get("on") or [])))
        q["K"] = ks.get("K", fe.get(t, {}).get("K"))
        q["t"] = t
        F.append(q)
    if not F:
        continue
    ok = [q for q in F if q["W"] <= LIM]
    med = {k: (stt.median([q[k] for q in ok]) if ok else 0.0) for k in ("a", "rest")}
    med["bc"] = stt.median([q["b"] + q["c"] for q in ok]) if ok else 0.0
    over = [q for q in F if q["W"] > LIM]
    cnt = {"시뮬": 0, "렌더": 0, "나머지": 0}
    for q in over:
        ex = {"시뮬": q["a"] - med["a"], "렌더": q["b"] + q["c"] - med["bc"], "나머지": q["rest"] - med["rest"]}
        top = max(ex, key=ex.get)
        cnt[top] += 1
        rows.append((n, q, ex, top))
    summ.append((n, len(F), len(over), med, cnt, sum(q["two"] for q in over), stt.mean([q["W"] for q in F])))

print("## (11) 단독 판 넘는 프레임 요약 (프레임 3~, W > 16.7 ms; 몫 = 그 판 넘지 않은 프레임 중앙 대비)\n")
print("| 판 | 프레임 | 넘는 프레임 | W 평균 | 넘지 않은 프레임 중앙: 시뮬 a · 렌더 b+c · 나머지 | 넘는 프레임 주된 곳: 시뮬 · 렌더 · 나머지 | 그중 두 시도 |")
print("|---|---|---|---|---|---|---|")
for n, nf, no, med, cnt, two, wm in summ:
    print(f"| {n} | {nf} | {no} | {f(wm)} | {f(med['a'])} · {f(med['bc'])} · {f(med['rest'])} | {cnt['시뮬']} · {cnt['렌더']} · {cnt['나머지']} | {two} |")
print()
print("## (12) 넘는 프레임 하나하나 (ms; 늘어난 몫 = 그 판 넘지 않은 프레임 중앙 대비)\n")
print("| 판 | 프레임 | W | Ws | 시뮬 a | 예산 시계 T | 갱신 b | 래스터 c | 나머지 W−(a+b+c) | host · gap · lag | 그래프 재생 · 동기화 | 두 시도 · 걸음 K | 늘어난 몫: 시뮬 · 렌더 · 나머지 | 주된 곳 |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for n, q, ex, top in rows:
    print(f"| {n} | {q['t']} | {f(q['W'])} | {f(q.get('Ws'))} | {f(q['a'])} | {f(q.get('T'))} | {f(q['b'])} | {f(q['c'])} | {f(q['rest'])} | {f(q.get('host'))} · {f(q.get('gap'))} · {f(q.get('lag'))} | "
          f"{q.get('ng', '-')} · {q.get('nsync', '-')} | {q['two']} · {q.get('K', '-')} | {f(ex['시뮬'])} · {f(ex['렌더'])} · {f(ex['나머지'])} | {top} |")
print()

"""절 CM62 묶음 speed5 표: python exe/speed5_tab.py <결과 디렉토리> -> markdown (stdout). 판 목록 = exe/speed5_spec.txt.
렌더 포함 합계 (G) = rr.jsonl a (시뮬 프레임 재생 GPU 시간) + b (가우시안 갱신) + c (래스터화), 프레임 3~ (t >= 2), 문턱 16.7 ms.
s5.jsonl: per = 시뮬 시작 -> 시뮬 + 갱신 (+ 겹치기 아니면 래스터화) 끝 (시뮬 스트림 사건 시각, 그래프 사이 호스트 빈틈 포함), int = 앞 프레임 시작 -> 이번 시작 (처리량 간격),
lat = 시뮬 시작 -> 그 프레임 래스터화 끝 (지연), craw = 래스터화 GPU 시간. 겹치기 판의 rr b·c = 시뮬 스트림에 드러난 렌더 몫.
"""
import json
import math
import os
import statistics as stt
import sys

O = sys.argv[1]
HERE = os.path.dirname(os.path.abspath(__file__))
spec = [l.split(None, 4) for l in open(os.path.join(HERE, "speed5_spec.txt")) if l.strip() and not l.startswith("#")]
spec = [s + [""] * (5 - len(s)) for s in spec]
LIM = 1000.0 / 60.0
SCN = {"L": "레고", "W": "wolf", "F": "ficus", "B": "bread", "M3": "수박"}
CATS = ["P2G", "걸음 시작", "W̄", "R2·A2", "벌점", "H4 판정", "요소", "장벽 기울기", "기울기·판정", "PCG", "보폭·선탐색", "걸음 끝", "G2P", "구간 사이", "기타"]


def jl(n, f):
    p = os.path.join(O, n, f)
    try:
        return [json.loads(l) for l in open(p) if l.strip()] if os.path.exists(p) else None
    except Exception:
        return None


def summ(n):
    try:
        return json.load(open(os.path.join(O, n, "summary.json")))
    except Exception:
        return None


def fmt(v, d=2):
    return "-" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{v:.{d}f}"


def res(n):
    s = summ(n)
    if s is None:
        return "없음/실패"
    return ("발산 f%d" % s["diverged"]) if s.get("diverged") else ("완주 %d" % s["frames_done"])


def err(n):
    s = summ(n)
    e = None if s is None else s.get("err_mean_done", s.get("err_mean"))
    return 100 * e if e is not None else None


def lj_last(n):
    for q in reversed(jl(n, "fem.jsonl") or []):
        v = (q.get("en") or {}).get("lj0")
        if v is not None and q.get("err") is not None and math.isfinite(q["err"]):
            return 100 * v
    return None


def stat(v):
    v = [x for x in v if x is not None and math.isfinite(x)]
    if not v:
        return "-"
    return f"{stt.mean(v):.2f} / {max(v):.1f} / {sum(1 for x in v if x > LIM)}"


def frames(n):
    rr = {q["t"]: q for q in (jl(n, "rr.jsonl") or [])}
    s5 = {q["t"]: q for q in (jl(n, "s5.jsonl") or [])}
    out = []
    for t in sorted(rr):
        if t < 2:
            continue
        q = dict(rr[t]); q.update({k: v for k, v in s5.get(t, {}).items() if k not in ("t", "a")})
        out.append(q)
    return out


def cats(q):
    d, nt = {}, 0
    for tr in ((q.get("sp") or {}).get("tries") or []):
        nt += 1
        for k, v in tr["cat"].items():
            d[k] = d.get(k, 0.0) + v
    return d, nt


names = [s[1] for s in spec]
runs = sorted(d for d in os.listdir(O) if os.path.isdir(os.path.join(O, d)))


def rlist(n):
    return [d for d in runs if d == n or d in (n + "_a", n + "_b")]


print("# 절 CM62 speed5 표\n")
# ---------------- (1) 단계 기록
print("## (1) G 단계 기록 판 (p*, 표식 몫만큼 느림; 프레임 3~) -- 넘는 프레임 (a + b + c > 16.7) 과 보통 프레임의 단계별 평균 ms\n")
print("| 판 | 프레임 | 넘는 | 넘는 중 시도 > 1 | 넘는 중 시뮬 혼자 > 16.7 | 넘는 중 시뮬 ≤ 16.7 (렌더가 넘김) | 묶음 | 시도 수 | 시뮬 a | " + " | ".join(CATS) + " | 갱신 b | 래스터 c | 합계 |")
print("|---" * (10 + len(CATS) + 3) + "|")
for s in spec:
    if s[3] != "p":
        continue
    for n in rlist(s[1]):
        F = frames(n)
        if not F:
            continue
        ov = [q for q in F if q["tot"] > LIM]
        nm = [q for q in F if q["tot"] <= LIM]
        n_multi = sum(1 for q in ov if cats(q)[1] > 1)
        n_sim = sum(1 for q in ov if q["a"] > LIM)
        for lab, G_ in (("넘는", ov), ("보통", nm)):
            if not G_:
                continue
            agg = {c: stt.mean([cats(q)[0].get(c, 0.0) for q in G_]) for c in CATS}
            nt = stt.mean([cats(q)[1] for q in G_])
            print(f"| {n} | {len(F)} | {len(ov)} | {n_multi} | {n_sim} | {len(ov) - n_sim} | {lab} {len(G_)} | {nt:.2f} | {stt.mean([q['a'] for q in G_]):.2f} | "
                  + " | ".join(f"{agg[c]:.2f}" for c in CATS) + f" | {stt.mean([q['b'] for q in G_]):.2f} | {stt.mean([q['c'] for q in G_]):.2f} | {stt.mean([q['tot'] for q in G_]):.2f} |")
print()
print("넘는 프레임 목록 (판마다 합계 큰 순 최대 12 개): t · 합계 · 시뮬 · 시도 수 · 시도별 ms · 큰 단계 셋\n")
for s in spec:
    if s[3] != "p":
        continue
    for n in rlist(s[1]):
        F = frames(n)
        ov = sorted([q for q in F if q["tot"] > LIM], key=lambda q: -q["tot"])[:12]
        if not ov:
            continue
        print(f"- {n}: " + "; ".join(
            f"f{q['t'] + 1} {q['tot']:.1f} (시뮬 {q['a']:.1f}, 시도 {cats(q)[1]} [{', '.join(fmt(tr.get('ms'), 1) for tr in (q.get('sp') or {}).get('tries', []))}], "
            + ", ".join(f"{k} {v:.1f}" for k, v in sorted(cats(q)[0].items(), key=lambda kv: -kv[1])[:3]) + ")" for q in ov))
print()
# ---------------- (2) 결정적 해시
print("## (2) 겹치기 결과 불변 확인 (D 바탕 @tb 1000, --det --fhash, 렌더 켬; 끝 해시 = 마지막 프레임 xh)\n")
print("| 씬 | 겹치기 끔 판 · 끝 해시 · 프레임 | 켬 판 · 끝 해시 · 프레임 | 프레임마다 같음 |")
print("|---|---|---|---|")
for sc in ("L", "W", "F", "B"):
    a, b = f"h{sc}0", f"h{sc}1"
    ra, rb = jl(a, "fem.jsonl") or [], jl(b, "fem.jsonl") or []
    xa, xb = [q.get("xh") for q in ra], [q.get("xh") for q in rb]
    same = (len(xa) == len(xb) and len(xa) > 0 and all(p == q and p is not None for p, q in zip(xa, xb)))
    print(f"| {SCN[sc]} | {a} · {xa[-1] if xa else '-'} · {len(xa)} | {b} · {xb[-1] if xb else '-'} · {len(xb)} | {'예' if same else '아니오'} |")
print()
# ---------------- (3) 단독 다섯 씬
print("## (3) 다섯 씬 렌더 포함 단독 (z*, a · b; 평균 / 최악 / > 16.7 프레임 수, 프레임 3~)\n")
print("| 씬 | 판 | 결과 | 오차 % | 수박 끝 손상 % | 시뮬 a | 갱신 b | 래스터 c (드러난) | 래스터 GPU craw | **합계 a+b+c** | 프레임 붙든 시간 per | 처리량 간격 int | 지연 lat |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for s in spec:
    if s[3] != "z":
        continue
    for n in rlist(s[1]):
        F = frames(n)
        if not F:
            print(f"| {SCN.get(s[2], s[2])} | {n} | {res(n)} | | | | | | | | | | |")
            continue
        print(f"| {SCN.get(s[2], s[2])} | {n} | {res(n)} | {fmt(err(n))} | {fmt(lj_last(n)) if s[2] == 'M3' else '-'} | {stat([q['a'] for q in F])} | {stat([q['b'] for q in F])} | "
              f"{stat([q['c'] for q in F])} | {stat([q.get('craw') for q in F])} | **{stat([q['tot'] for q in F])}** | {stat([q.get('per') for q in F])} | {stat([q.get('int') for q in F])} | {stat([q.get('lat') for q in F])} |")
print()
print("판 시작 카드 다른 계산 프로세스 (apps_at_start 끝 줄들):\n")
for d in runs:
    p = os.path.join(O, d, "apps_at_start.txt")
    if d.startswith(("z", "p")) and os.path.exists(p):
        L = [l.strip() for l in open(p) if l.startswith("카드")]
        print(f"- {d}: {L[0] if L else '-'}")
# ---------------- (4) 수박 비결정 판 정확도
def ljk(n, k):
    r = jl(n, "fem.jsonl") or []
    if k >= len(r):
        return None
    v = (r[k].get("en") or {}).get("lj0")
    return None if v is None or r[k].get("err") is None or not math.isfinite(r[k]["err"]) else 100 * v


def e1s(n):
    e = jl(n, "e1.jsonl") or []
    return (max((q.get("nfar", 0) or 0) for q in e), max((q.get("dmax") or 0) for q in e)) if e else (None, None)


print("## (4) 수박 비결정 판 (g*, 진단 D1·D2·D5; 정답 손상 f5 4.41 · f6 6.94 · f8 7.08 · 끝 6.51 %; 이탈 판 = 4 칸 밖 > 20 개)\n")
print("| 후보 | 판 | 발산 | 오차 % 평균 [범위] | 손상 f5 · f6 · f8 평균 % | 끝 손상 평균 [범위] % | 이탈 판 · 최대 수 · 거리 m | X3c 다시 하기 프레임 (판당) |")
print("|---|---|---|---|---|---|---|---|")
for lab, pre in (("G + O", "gO_"), ("G + O + X3e4", "gX_")):
    ns = [d for d in runs if d.startswith(pre)]
    if not ns:
        continue
    dv = sum(1 for n in ns if (summ(n) or {}).get("diverged") or summ(n) is None)
    es = [err(n) for n in ns if err(n) is not None]
    ends = [lj_last(n) for n in ns if lj_last(n) is not None]
    fk = [" · ".join(fmt(stt.mean([v for v in (ljk(n, k) for n in ns) if v is not None]) if any(ljk(n, k) is not None for n in ns) else None) for k in (4, 5, 7))]
    far = [e1s(n) for n in ns]
    far = [f for f in far if f[0] is not None]
    x2 = [sum(1 for q in (jl(n, "s4.jsonl") or []) if (q.get("x1") or [0, 0])[1] > 0 or (q.get("x1") or [0, 0, 0])[2] > 0) for n in ns]
    print(f"| {lab} | {len(ns)} | {dv} | {fmt(stt.mean(es)) if es else '-'} [{fmt(min(es)) if es else '-'}~{fmt(max(es)) if es else '-'}] | {fk[0]} | "
          f"{fmt(stt.mean(ends)) if ends else '-'} [{fmt(min(ends)) if ends else '-'}~{fmt(max(ends)) if ends else '-'}] | "
          f"{sum(1 for f in far if f[0] > 20)}/{len(far)} · {max((f[0] for f in far), default='-')} · {fmt(max((f[1] for f in far), default=None))} | {fmt(stt.mean(x2), 1) if x2 else '-'} |")
print()

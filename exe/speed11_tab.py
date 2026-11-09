"""절 CM82 묶음 speed11 표 (exe/speed10_tab.py 도움 함수 복사): python exe/speed11_tab.py <결과 디렉토리> -> markdown (stdout). 판 목록 = exe/speed9_spec.txt.
커널 프로파일 판 (s7trace.json.gz) 은 exe/speed7_kp.py 로 프레임·단계·커널별 시간.
GPU 합 = rr.jsonl a (시뮬 재생 GPU) + b (갱신) + c (래스터화) (절 CM54·62 정의). 벽시계 W = s6.jsonl 프레임 그림 끝 벽시계 간격 P(t) − P(t−1) (판정 기준, 절 CM65).
프레임 3~ (t >= 2), 문턱 16.7 ms. 프로파일 판 (s6prof.json) 은 GPU 빈 시간을 그때 돌던 파이썬 함수·CUDA API 로.
"""
import json
import math
import os
import statistics as stt
import sys

O = sys.argv[1]
HERE = os.path.dirname(os.path.abspath(__file__))
spec = [l.split(None, 4) for l in open(os.path.join(HERE, "speed11_spec.txt")) if l.strip() and not l.startswith("#")]
spec = [[q.strip() for q in s] + [""] * (5 - len(s)) for s in spec]
LIM = 1000.0 / 60.0
SCN = {"L": "레고", "W": "wolf", "F": "ficus", "B": "bread", "M3": "수박"}


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


def stat(v):
    v = [x for x in v if x is not None and math.isfinite(x)]
    if not v:
        return "-"
    return f"{stt.mean(v):.2f} / {max(v):.1f} / {sum(1 for x in v if x > LIM)}"


def mean(v):
    v = [x for x in v if x is not None and math.isfinite(x)]
    return stt.mean(v) if v else None


def load(n):
    try:
        L = open(os.path.join(O, n, "apps_at_start.txt")).read().strip().split("\n")
        return L[-1].split()[0]
    except Exception:
        return "-"


def frames(n):
    rr = {q["t"]: q for q in (jl(n, "rr.jsonl") or [])}
    s6 = {q["t"]: q for q in (jl(n, "s6.jsonl") or [])}
    out = []
    for t in sorted(set(rr) | set(s6)):
        if t < 2:
            continue
        q = dict(rr.get(t, {}))
        q.update({k: v for k, v in s6.get(t, {}).items() if k not in ("t",)})
        if "a" in rr.get(t, {}):
            q["a"] = rr[t]["a"]
        out.append(q)
    return out


def dmg(n):
    L = jl(n, "fem.jsonl") or []
    d = {q["t"]: (q.get("en") or {}).get("lj0") for q in L}
    last = None
    for q in reversed(L):
        v = (q.get("en") or {}).get("lj0")
        if v is not None and q.get("err") is not None and math.isfinite(q["err"]):
            last = v
            break
    return [d.get(4), d.get(5), d.get(7)], last


def far(n):
    e = jl(n, "e1.jsonl") or []
    return (max((q.get("nfar", 0) or 0) for q in e), max((q.get("dmax") or 0) for q in e)) if e else (None, None)


def x3g(n):
    return sum((q.get("x3g") or 0) for q in (jl(n, "s8w.jsonl") or []))


def x3ms(n):
    v = [q["ms"] for q in (jl(n, "fem.jsonl") or []) if len((q.get("ks") or {}).get("on") or []) > 1 and q.get("ms") is not None]
    return (stt.mean(v), max(v)) if v else (None, None)



def x3c(n):
    c = 0
    for q in jl(n, "fem.jsonl") or []:
        on = (q.get("ks") or {}).get("on") or []
        if any(v == 1 for v in on):
            c += 1
    return c


def grp(n):
    return n.split("_")[0]


runs = []
for st, n, s, md, x in spec:
    if md in ("z", "p"):
        for r in ("a", "b"):
            if os.path.isdir(os.path.join(O, f"{n}_{r}")):
                runs.append((st, f"{n}_{r}", s, md, x))
    elif os.path.isdir(os.path.join(O, n)):
        runs.append((st, n, s, md, x))

print("# 절 CM82 speed11 표\n")
G = {}
for st, n, s, md, x in runs:
    if md in ("z", "p"):
        continue
    G.setdefault((st, grp(n), s, x), []).append(n)


def rng(v, d=2):
    v = [q for q in v if q is not None and math.isfinite(q)]
    return f"{stt.mean(v):.{d}f} [{min(v):.{d}f}~{max(v):.{d}f}]" if v else "-"


print("## (1) 수박 비결정 (CM78.4 형식; 정답 손상 f5 4.41 · f6 6.94 · f8 7.08 · 끝 6.51 %; 이탈 판 = 4 칸 밖 > 20 개)\n")
print("| 단계 | 묶음 | 플래그 | 판 | 발산 | 오차 % 평균 [범위] | 손상 f5 · f6 · f8 % | 끝 손상 평균 [범위] % | 이탈 판 · 최대 수 · 거리 m | X3c 프레임 (판당) · 그중 X3g | 두 시도 프레임 시뮬 ms 평균 / 최대 |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for (st, g, s, x), ns in G.items():
    if s != "M3" or st.endswith("k"):
        continue
    dv = sum(1 for n in ns if (summ(n) or {}).get("diverged") or summ(n) is None)
    D = [dmg(n) for n in ns]
    d3 = [mean([100 * d[0][i] for d in D if d[0][i] is not None]) for i in range(3)]
    fa = [far(n) for n in ns]
    nfar = sum(1 for f in fa if (f[0] or 0) > 20)
    xc = [x3c(n) for n in ns]; xg = [x3g(n) for n in ns]
    ms = [x3ms(n) for n in ns]
    print(f"| {st} | {g} | {x} | {len(ns)} | {dv}/{len(ns)} | {rng([err(n) for n in ns])} | {' · '.join(fmt(v) for v in d3)} | {rng([100 * d[1] if d[1] is not None else None for d in D])} | "
          f"{nfar} · {max((f[0] or 0) for f in fa)} · {fmt(max((f[1] or 0) for f in fa))} | {fmt(mean(xc), 1)} · {fmt(mean(xg), 1)} | {fmt(mean([m[0] for m in ms]), 1)} / {fmt(max([m[1] for m in ms if m[1] is not None], default=None), 1)} |")
print()
print("## (2) 다른 씬 (판정 판)\n")
print("| 단계 | 묶음 | 씬 | 플래그 | 판 | 발산 | 오차 % 평균 [범위] | 이탈 판 · 최대 수 · 거리 m | 두 시도 프레임 (판당) |")
print("|---|---|---|---|---|---|---|---|---|")
for (st, g, s, x), ns in G.items():
    if s == "M3" or st.endswith("k"):
        continue
    dv = sum(1 for n in ns if (summ(n) or {}).get("diverged") or summ(n) is None)
    fa = [far(n) for n in ns]
    print(f"| {st} | {g} | {SCN.get(s, s)} | {x} | {len(ns)} | {dv}/{len(ns)} | {rng([err(n) for n in ns])} | {sum(1 for f in fa if (f[0] or 0) > 20)} · {max((f[0] or 0) for f in fa)} · {fmt(max((f[1] or 0) for f in fa))} | {fmt(mean([x3c(n) for n in ns]), 1)} |")
print()
print("## (3) 렌더 포함 벽시계 단독 (카드 하나에 차례로; GPU 합 a+b+c · 벽시계 W = 평균 / 최악 / > 16.7 프레임 수, 프레임 3~; ms)\n")
print("| 판 | 씬 | 플래그 | GPU 합 a+b+c | **벽시계 W** | W 중앙 | 오차 % | 두 시도 프레임 수 · 시뮬 평균 / 최대 ms | 결과 | 부하 |")
print("|---|---|---|---|---|---|---|---|---|---|")
for st, n, s, md, x in runs:
    if md != "z":
        continue
    F = frames(n)
    tot = [q.get("a", float("nan")) + q.get("b", 0.0) + q.get("c", 0.0) if "b" in q else None for q in F]
    W = [q.get("W") for q in F]
    Wv = [w for w in W if w is not None]
    xm = x3ms(n)
    print(f"| {n} | {SCN.get(s, s)} | {x} | {stat(tot)} | **{stat(W)}** | {fmt(stt.median(Wv) if Wv else None)} | {fmt(err(n))} | {x3c(n)} · {fmt(xm[0], 1)} / {fmt(xm[1], 1)} | {res(n)} | {load(n)} |")
print()


def med(v):
    v = [q for q in v if q is not None and isinstance(q, (int, float)) and math.isfinite(q)]
    return stt.median(v) if v else None


print("## (4) X3g 경계 진단 (s11.jsonl, 첫 시도 반복 0 뒤 = 반복 1 시작; 넘어감 = X3g 판정 ‖g1‖²/F² > 1e6; 값은 걸음 중앙값)\n")
K = [("g0F", "‖g0‖²/F²"), ("g1F", "‖g1‖²/F²"), ("rz1F", "g1ᵀD⁻¹g1/F²"), ("pcap", "PCG 상한 풀이"), ("pcg", "PCG 반복"), ("st", "보폭 SC14"), ("kg", "kg"), ("kmax", "kn 최대"), ("ncut", "kn>kg 노드"), ("gsh_cut", "g1² 몫 kn>kg"),
     ("gsh_m0", "g1² 몫 m<m̄"), ("gsh_m1", "g1² 몫 튐 띠"), ("gsh_m2", "g1² 몫 무거움"), ("gsh_L", "g1² 몫 가벼움 면 노드"), ("fL2F", "‖T f_L‖²/F²"), ("fP2F", "‖T f_P‖²/F²"), ("g_noL", "‖g1 − T f_L‖²/‖g1‖²"),
     ("hL50", "tr B_L / tr 대각 중앙"), ("hL90", "90 %"), ("hLgw", "g1² 가중"), ("hLM50", "tr B_L / 3M/τ² 중앙"), ("ncutL", "kn>kg 가벼움 면 노드")]
print("| 묶음 | 걸음 | " + " | ".join(k[1] for k in K) + " |")
print("|---|---|" + "---|" * len(K))
DG = {}
for st, n, s, md, x in runs:
    L = jl(n, "s11.jsonl")
    if not L:
        continue
    for q in L:
        q = dict(q)
        for i, v in enumerate(q.get("gsh_m") or []):
            q[f"gsh_m{i}"] = v
        for a_, b_ in (("hL_q", "hL"), ("hLM_q", "hLM")):
            v = q.get(a_)
            if v:
                q[b_ + "50"], q[b_ + "90"] = v[0], v[1]
        q["hLgw"] = q.get("hL_gw")
        DG.setdefault((grp(n), q["trig"]), []).append(q)
for (g, tr), L in sorted(DG.items()):
    print(f"| {g} | {'넘어감' if tr else '안 넘어감'} {len(L)} | " + " | ".join(fmt(med([q.get(k[0]) for q in L]), 3) if k[0] not in ("g0F", "g1F", "rz1F", "fL2F", "fP2F") else (lambda m: '-' if m is None else f'{m:.3g}')(med([q.get(k[0]) for q in L])) for k in K) + " |")
print()

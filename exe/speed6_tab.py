"""절 CM65 묶음 speed6 표: python exe/speed6_tab.py <결과 디렉토리> -> markdown (stdout). 판 목록 = exe/speed6_spec.txt.
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
spec = [l.split(None, 4) for l in open(os.path.join(HERE, "speed6_spec.txt")) if l.strip() and not l.startswith("#")]
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


def x3c(n):
    c = 0
    for q in jl(n, "fem.jsonl") or []:
        on = (q.get("ks") or {}).get("on") or []
        if any(v == 1 for v in on):
            c += 1
    return c


runs = []
for st, n, s, md, x in spec:
    if md in ("z", "p"):
        for r in ("a", "b"):
            if os.path.isdir(os.path.join(O, f"{n}_{r}")):
                runs.append((st, f"{n}_{r}", s, md, x))
    elif md == "q":
        if os.path.isdir(os.path.join(O, f"{n}_a")):
            runs.append((st, f"{n}_a", s, md, x))
    elif os.path.isdir(os.path.join(O, n)):
        runs.append((st, n, s, md, x))

print("# 절 CM65 speed6 표\n")
print("## (1) 렌더 포함 프레임 시간: GPU 합 a+b+c 대 벽시계 W (평균 / 최악 / > 16.7 프레임 수, 프레임 3~; ms)\n")
print("| 단계 | 판 | 씬 | 플래그 | 결과 | 오차 % | **GPU 합 a+b+c** | **벽시계 W** | 시뮬 스트림 간격 Ws | 시뮬 a 평균 | 예산 시계 T 평균 | W − T 평균 | per (GPU 시각) 평균 | 호스트 안 host 평균 | 프레임 사이 gap 평균 | 지연 lat 평균 | 래스터 GPU craw | 시작 늦음 lag 평균 | 그래프 재생 ng | 동기화 nsync | 백업 재사용 | 부하 |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for st, n, s, md, x in runs:
    if md == "h":
        continue
    F = frames(n)
    if not F:
        print(f"| {st} | {n} | {SCN.get(s, s)} | {x} | {res(n)} | {fmt(err(n))} |" + " - |" * 16)
        continue
    tot = [q.get("a", float("nan")) + q.get("b", 0.0) + q.get("c", 0.0) if "b" in q else None for q in F]
    wt = [q["W"] - q["T"] for q in F if q.get("W") is not None and q.get("T") is not None]
    print(f"| {st} | {n} | {SCN.get(s, s)} | {x} | {res(n)} | {fmt(err(n))} | {stat(tot)} | {stat([q.get('W') for q in F])} | {stat([q.get('Ws') for q in F])} | {fmt(mean([q.get('a') for q in F]))} | {fmt(mean([q.get('T') for q in F]))} | {fmt(mean(wt))} | "
          f"{fmt(mean([q.get('per') for q in F]))} | {fmt(mean([q.get('host') for q in F]))} | "
          f"{fmt(mean([q.get('gap') for q in F]))} | {fmt(mean([q.get('lat') for q in F]))} | {fmt(mean([q.get('craw') for q in F]))} | {fmt(mean([q.get('lag') for q in F]), 3)} | {fmt(mean([q.get('ng') for q in F]), 1)} | {fmt(mean([q.get('nsync') for q in F]), 1)} | "
          f"{sum(q.get('bkhit', 0) or 0 for q in F)} | {load(n)} |")

H = [r for r in runs if r[3] == "h"]
if H:
    print("\n## (2) 결정적 끝 해시 (--det --fhash; 같은 이름 짝 0 = 원래 실행기·훅, 1 = 가벼운 판)\n")
    print("| 판 | 씬 | 플래그 | 결과 | 끝 해시 x_sha1 | 프레임 해시 끝 |")
    print("|---|---|---|---|---|---|")
    for st, n, s, md, x in H:
        sm = summ(n) or {}
        L = jl(n, "fem.jsonl") or []
        print(f"| {n} | {SCN.get(s, s)} | {x} | {res(n)} | {str(sm.get('x_sha1'))[:10]} | {L[-1].get('xh') if L else '-'} ({len(L)}) |")
    pairs = {}
    for st, n, s, md, x in H:
        pairs.setdefault(n[:-1], {})[n[-1]] = n
    for k, v in pairs.items():
        if "0" in v and "1" in v:
            a = {q["t"]: q.get("xh") for q in (jl(v["0"], "fem.jsonl") or [])}
            b = {q["t"]: q.get("xh") for q in (jl(v["1"], "fem.jsonl") or [])}
            com = sorted(set(a) & set(b))
            diff = [t for t in com if a[t] != b[t]]
            print(f"- {k}: 프레임 해시 같음 {len(com) - len(diff)}/{len(com)}" + (f", 처음 다른 프레임 {diff[0]}" if diff else ""))

G = [r for r in runs if r[3] == "g"]
if G:
    print("\n## (3) 수박 비결정 판 (정답 손상 f5 4.41 · f6 6.94 · f8 7.08 · 끝 6.51 %; 이탈 판 = 4 칸 밖 > 20 개)\n")
    print("| 묶음 | 판 수 | 발산 | 오차 % 평균 [범위] | 손상 f5 · f6 · f8 평균 % | 끝 손상 평균 [범위] % | 이탈 판 · 최대 수 · 거리 m | X3c 다시 하기 프레임 (판당) |")
    print("|---|---|---|---|---|---|---|---|")
    grp = {}
    for st, n, s, md, x in G:
        grp.setdefault(n.rsplit("_", 1)[0], []).append(n)
    for k, ns in grp.items():
        dv = sum(1 for n in ns if (summ(n) or {}).get("diverged") or summ(n) is None)
        es = [err(n) for n in ns if err(n) is not None]
        dd = [dmg(n) for n in ns]
        f568 = [stt.mean([q[0][i] * 100 for q in dd if q[0][i] is not None]) if any(q[0][i] is not None for q in dd) else None for i in range(3)]
        ld = [q[1] * 100 for q in dd if q[1] is not None]
        fr = [far(n) for n in ns]
        nfar = [q[0] for q in fr if q[0] is not None]
        print(f"| {k} | {len(ns)} | {dv}/{len(ns)} | {fmt(mean(es))} [{fmt(min(es)) if es else '-'}~{fmt(max(es)) if es else '-'}] | {' · '.join(fmt(v) for v in f568)} | "
              f"{fmt(mean(ld))} [{fmt(min(ld)) if ld else '-'}~{fmt(max(ld)) if ld else '-'}] | {sum(1 for v in nfar if v > 20)} · {max(nfar) if nfar else '-'} · {fmt(max((q[1] or 0) for q in fr) if fr else None)} | {fmt(mean([x3c(n) for n in ns]), 1)} |")

P = [r for r in runs if os.path.exists(os.path.join(O, r[1], "s6prof.json"))]
if P:
    print("\n## (4) 프로파일 (torch.profiler, 파이썬 호출 추적 켬 -> 호스트가 느려짐; 몫의 순서만 봄): 프레임당 GPU 빈 시간 (ms) 과 그때 호스트에서 돌던 것\n")
    for st, n, s, md, x in P:
        try:
            r = json.load(open(os.path.join(O, n, "s6prof.json")))
        except Exception:
            continue
        F = r.get("frames", [])
        if not F:
            continue
        print(f"### {n} ({SCN.get(s, s)}, {x}) -- 프레임 {len(F)}: 창 {fmt(mean([q['win_ms'] for q in F]))} · 쪼개기 안 {fmt(mean([q['split_ms'] for q in F]))} · GPU 일 {fmt(mean([q['gpu_busy_ms'] for q in F]))} · "
              f"빈 시간 안 {fmt(mean([q['idle_in_ms'] for q in F]))} / 밖 {fmt(mean([q['idle_out_ms'] for q in F]))} · 빈 틈 (> 5 µs, 안) {fmt(mean([q['n_gap_in'] for q in F]), 1)} · 그래프 띄움 {fmt(mean([q['graph_launch'] for q in F]), 1)} · 동기화 API {fmt(mean([q['sync'] for q in F]), 1)} · 복사 API {fmt(mean([q['memcpy'] for q in F]), 1)}\n")
        print("| 자기 몫 (안 = 쪼개기 함수 안, 밖 = 프레임 사이) | ms / 프레임 |")
        print("|---|---|")
        for k, v in r.get("self_per_frame", [])[:20]:
            print(f"| {k.replace('|', '/')[:220]} | {v:.3f} |")
        print("\n| CUDA API (GPU 빈 시간과 겹친 몫) | ms / 프레임 |")
        print("|---|---|")
        for k, v in r.get("api_per_frame", [])[:10]:
            print(f"| {k} | {v:.3f} |")
        print("\n| 포함 몫 상위 (함수 안에서 GPU 가 빈 시간) | ms / 프레임 |")
        print("|---|---|")
        for k, v in r.get("incl_per_frame", [])[:25]:
            print(f"| {k.replace('|', '/')[:200]} | {v:.3f} |")
        print()

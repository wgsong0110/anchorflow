"""절 CM78 묶음 speed10 표 (exe/speed9_tab.py 복사 + (12) 노드 상자 압축 N 확인 · (13) 넘침 다시 할당 프레임): python exe/speed10_tab.py <결과 디렉토리> -> markdown (stdout). 판 목록 = exe/speed9_spec.txt.
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
spec = [l.split(None, 4) for l in open(os.path.join(HERE, "speed10_spec.txt")) if l.strip() and not l.startswith("#")]
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


LAB = {"xH": "H5", "xN": "H6 = H5 + N + K"}


def lab(n, s):
    k = LAB.get(n[:2], n)
    return k


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

print("# 절 CM78 speed10 표\n")
# ---- (0) 최종 요약 (s6: 씬마다 H · H + 가속 같은 카드 나란히 a·b) ----
def _row0(n):
    F = frames(n)
    if not F:
        return None
    tot = [q.get("a", float("nan")) + q.get("b", 0.0) + q.get("c", 0.0) if "b" in q else None for q in F]
    W = [q.get("W") for q in F]
    return dict(tot=stat(tot), W=stat(W), Wmed=stt.median([w for w in W if w is not None]) if any(w is not None for w in W) else None,
                T=mean([q.get("T") for q in F]), WT=mean([q["W"] - q["T"] for q in F if q.get("W") is not None and q.get("T") is not None]), err=err(n), load=load(n), res=res(n))


S6R = [(st, n, s, md, x) for st, n, s, md, x in spec if st == "s2" and md == "z"]
S6R.sort(key=lambda r: (["M3", "L", "W", "F", "B"].index(r[2]) if r[2] in ("M3", "L", "W", "F", "B") else 9))
if S6R:
    print("## (0) 최종 요약 -- 다섯 씬 렌더 포함 벽시계 단독 (s2, 같은 카드에 차례로 a 다 돌고 b; H5 = 공통 후보, H6 = H5 + 노드 상자 압축 N + 첫 할당 상자 배수 K (옛 실행기 = 수박만); 평균 / 최악 / > 16.7 프레임 수, 프레임 3~; ms)\n")
    print("| 씬 | 판 | GPU 합 a+b+c | **벽시계 W** | W 중앙 | 예산 시계 T 평균 | W − T 평균 | 오차 % | 두 시도 프레임 수 · 시뮬 평균 / 최대 ms | X3g | 결과 | 부하 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for st, n, s, md, x in S6R:
        for r in ("a", "b"):
            q = _row0(f"{n}_{r}")
            if q is None:
                print(f"| {SCN.get(s, s)} | {n}_{r} | - | - | - | - | - | - | - | - | {res(n + '_' + r)} | - |")
                continue
            xm = x3ms(f"{n}_{r}")
            print(f"| {SCN.get(s, s)} | {n}_{r} ({lab(n, s)}) | {q['tot']} | **{q['W']}** | {fmt(q['Wmed'])} | {fmt(q['T'])} | {fmt(q['WT'])} | {fmt(q['err'])} | {x3c(n + '_' + r)} · {fmt(xm[0], 1)} / {fmt(xm[1], 1)} | {x3g(n + '_' + r)} | {q['res']} | {q['load']} |")
    print()

print("## (1) 렌더 포함 프레임 시간: GPU 합 a+b+c 대 벽시계 W (평균 / 최악 / > 16.7 프레임 수, 프레임 3~; ms)\n")
print("| 단계 | 판 | 씬 | 플래그 | 결과 | 오차 % | **GPU 합 a+b+c** | **벽시계 W** | 시뮬 스트림 간격 Ws | 시뮬 a 평균 | 예산 시계 T 평균 | W − T 평균 | Rw 몫 평균 | per (GPU 시각) 평균 | 호스트 안 host 평균 | 프레임 사이 gap 평균 | 지연 lat 평균 | 래스터 GPU craw | 시작 늦음 lag 평균 | 그래프 재생 ng | 동기화 nsync | 백업 재사용 | 부하 |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for st, n, s, md, x in runs:
    if md == "h":
        continue
    F = frames(n)
    if not F:
        print(f"| {st} | {n} | {SCN.get(s, s)} | {x} | {res(n)} | {fmt(err(n))} |" + " - |" * 17)
        continue
    tot = [q.get("a", float("nan")) + q.get("b", 0.0) + q.get("c", 0.0) if "b" in q else None for q in F]
    wt = [q["W"] - q["T"] for q in F if q.get("W") is not None and q.get("T") is not None]
    print(f"| {st} | {n} | {SCN.get(s, s)} | {x} | {res(n)} | {fmt(err(n))} | {stat(tot)} | {stat([q.get('W') for q in F])} | {stat([q.get('Ws') for q in F])} | {fmt(mean([q.get('a') for q in F]))} | {fmt(mean([q.get('T') for q in F]))} | {fmt(mean(wt))} | {fmt(mean([(q.get('rw') or [None, None])[1] for q in F]))} | "
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
    print("| 묶음 | 판 수 | 발산 | 오차 % 평균 [범위] | 손상 f5 · f6 · f8 평균 % | 끝 손상 평균 [범위] % | 이탈 판 · 최대 수 · 거리 m | X3c 다시 하기 프레임 (판당) | 그중 X3g 로 일찍 (판당) | 두 시도 프레임 시뮬 ms 평균 / 최대 |")
    print("|---|---|---|---|---|---|---|---|---|---|")
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
              f"{fmt(mean(ld))} [{fmt(min(ld)) if ld else '-'}~{fmt(max(ld)) if ld else '-'}] | {sum(1 for v in nfar if v > 20)} · {max(nfar) if nfar else '-'} · {fmt(max((q[1] or 0) for q in fr) if fr else None)} | {fmt(mean([x3c(n) for n in ns]), 1)} | {fmt(mean([x3g(n) for n in ns]), 1)} | {fmt(mean([x3ms(n)[0] for n in ns]), 1)} / {fmt(max([x3ms(n)[1] or 0 for n in ns]), 1)} |")

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

KPR = [r for r in runs if os.path.exists(os.path.join(O, r[1], "s7trace.json.gz"))]
if KPR:
    sys.path.insert(0, HERE)
    import speed7_kp as KPm
    print("\n## (5) 커널 프로파일 (torch.profiler CUDA 활동만, 그래프 안 커널 하나씩)\n")
    for st, n, s, md, x in KPR:
        try:
            r = KPm.analyze(os.path.join(O, n), top=200)
            if r is None:
                continue
            json.dump(r, open(os.path.join(O, n, "s7kp.json"), "w"), ensure_ascii=False)
            print(f"({SCN.get(s, s)}, {x})\n")
            KPm.md(os.path.join(O, n), r, 45)
        except Exception as e_:
            print(f"(커널 프로파일 {n} 실패: {e_})")

ZR = [r for r in runs if os.path.exists(os.path.join(O, r[1], "s7z.jsonl"))]
if ZR:
    print("\n## (6) 지연 백업 Z 셈 (s7z.jsonl; 확인 판 @s7z 2 = 할 일 없는 되돌리기마다 지금 상태와 프레임 시작 실제 백업 비트 대조)\n")
    print("| 판 | 플래그 | 결과 | 기록 프레임 | 지연 백업 | 실제로 뜸 | 할 일 없는 되돌리기 | 실제 되돌리기 | 비트 대조 | **다름 (되돌리기)** | **다름 (뜰 때)** | 다른 배열 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for st, n, s, md, x in ZR:
        L = jl(n, "s7z.jsonl") or []
        sm = {k: sum((q.get(k) or 0) for q in L) for k in ("lazy", "mat", "rs_noop", "rs_real", "cmp", "mis", "mis_mat")}
        nm_ = sorted({a for q in L for b in (q.get("mis_names") or []) for a in b})
        print(f"| {n} | {x} | {res(n)} | {len(L)} | {sm['lazy']} | {sm['mat']} | {sm['rs_noop']} | {sm['rs_real']} | {sm['cmp']} | {sm['mis']} | {sm['mis_mat']} | {','.join(nm_) or '-'} |")

PR = [r for r in runs if os.path.exists(os.path.join(O, r[1], "s7p.jsonl"))]
if PR:
    print("\n## (7) 장벽 표본점 거르기 P 셈 (s7p.jsonl; 확인 판 @s7p 2 = 거르지 않고 다 센 뒤 '거르라는데 실제 d 가 문턱 밑' 인 점 = 어김)\n")
    print("| 판 | 플래그 | 결과 | 고친 커널 | 확인 프레임 | 활성 점 셈 (커널 부름 합) | 거를 점 | 거를 몫 % | **어김** | Y 판정 수 | 앞 M 안에서 찾음 | **Y 어김** |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for st, n, s, md, x in PR:
        L = jl(n, "s7p.jsonl") or []
        pt = next((q.get("patched") for q in L if q.get("patched")), [])
        V = [q for q in L if "viol" in q]
        a_ = sum(q["act"] for q in V); f_ = sum(q["far"] for q in V); v_ = sum(q["viol"] for q in V)
        Y = [q for q in L if "ychk" in q]
        yc = sum(q["ychk"] for q in Y); yg = sum(q["ygate"] for q in Y); yv = sum(q["yviol"] for q in Y)
        print(f"| {n} | {x} | {res(n)} | {len(pt)} | {len(V)} | {a_} | {f_} | {fmt(100.0 * f_ / a_ if a_ else None)} | {v_} | {yc} | {yg} | {yv} |")

WR = [r for r in runs if any("chk" in q for q in (jl(r[1], "s8w.jsonl") or []))]
if WR:
    print("\n## (8) 워프 모으기 W 확인 (s8w.jsonl; 같은 커널 안에서 원래 방식 (입자마다 원자 합) 을 그림자 버퍼에. 어김 = |W − 원래| > 2 (n − 1) u Σ|항| (u = 2^-24, 두 합 순서의 반올림 한계), 정규화 차 = |Δ| / ((n − 1) u Σ|항|) 최대, 상대 = |Δ| / Σ|항| 최대)\n")
    print("| 판 | 결과 | 확인 프레임 | 버퍼 | 비교 원소 (항 ≥ 2, 합) | 비트 다름 원소 | **어김** | 비유한 다름 | 정규화 차 최대 | 상대 차 최대 |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for st, n, s, md, x in WR:
        L = [q for q in (jl(n, "s8w.jsonl") or []) if "chk" in q]
        for b in ("Aacc", "Cacc", "Sa", "Sc", "gqP", "bx", "e3wq"):
            v = [q["chk"].get(b) for q in L if q["chk"].get(b)]
            if not v:
                continue
            print(f"| {n} | {res(n)} | {len(L)} | {b} | {sum(q['n'] for q in v)} | {sum(q['ndiff'] for q in v)} | **{sum(q['viol'] for q in v)}** | {sum(q['nfin'] for q in v)} | {max(q['rmax'] for q in v):.3g} | {max(q['qmax'] for q in v):.3g} |")
    print("\n| 판 | 커널 | 레인 수 (합) | 대표 = 원자 합 묶음 수 | 줄어든 배수 | 프레임 3 · 50 · 끝 배수 |")
    print("|---|---|---|---|---|---|")
    for st, n, s, md, x in WR:
        L = [q for q in (jl(n, "s8w.jsonl") or []) if "grp" in q]
        for kn in ("rt_p2g", "aq_p2g_s", "gq_p2q", "wb_p2q", "e3_wq"):
            if not all(kn in q["grp"] for q in L):
                continue
            a = sum(q["grp"][kn][0] for q in L); b = sum(q["grp"][kn][1] for q in L)
            fr = [q["grp"][kn][0] / q["grp"][kn][1] if q["grp"][kn][1] else None for q in L]
            pick = [fr[i] for i in (min(3, len(fr) - 1), min(50, len(fr) - 1), len(fr) - 1)] if fr else []
            print(f"| {n} | {kn} | {a} | {b} | {fmt(a / b if b else None)} | {' · '.join(fmt(v) for v in pick)} |")


def s9(n):
    return jl(n, "s9.jsonl") or []


CR = [r for r in runs if any(("chk" in q) or ("ylchk" in q) for q in s9(r[1]))]
if CR:
    print("\n## (9) 압축 C · 조기 끝 Y 확인 (s9.jsonl; C: 원래 커널로 진짜 SC, 압축 판을 그림자에 -- 어김 = |Δ| > 2 (n − 1) u Σ|e| (u = 2^-53), 고른 후보 다름; 기준 = 원래 커널 두 번 (원자 순서 잡음); Y: 원래 고리 대 조기 끝 고리의 i 다름)\n")
    print("| 판 | 결과 | 확인 프레임 | y0e 부름 | 목록 점 (합) | **C 어김** | **C 고른 후보 다름** | 정규화 차 최대 | 기준: 어김 · 고른 후보 다름 · 정규화 차 | Y 점 판정 | **Y 다름** |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for st, n, s, md, x in CR:
        L = s9(n)
        C_ = [q["chk"] for q in L if "chk" in q]
        Y_ = [q["ylchk"] for q in L if "ylchk" in q]
        sm = lambda k: sum(c[k] for c in C_)
        print(f"| {n} | {res(n)} | {len(C_)} | {sm('calls') if C_ else '-'} | {sm('nlist') if C_ else '-'} | **{sm('viol') if C_ else '-'}** | **{sm('pick') if C_ else '-'}** | {max((c['rmax'] for c in C_), default=0):.3g} | {sm('viol_ref') if C_ else '-'} · {sm('pick_ref') if C_ else '-'} · {max((c['rmax_ref'] for c in C_), default=0):.3g} | {sum(q['n'] for q in Y_)} | **{sum(q['mis'] for q in Y_)}** |")

RR = [r for r in runs if any("rc" in q for q in s9(r[1]))]
if RR:
    print("\n## (10) 이어 하기 R 셈 (s9.jsonl rc: 이어 함 · 떠 둔 y 비유한이라 원래 다시 하기)\n")
    print("| 판 | 결과 | 이어 함 | 원래로 | X3c 프레임 | X3g |")
    print("|---|---|---|---|---|---|")
    for st, n, s, md, x in RR:
        L = [q["rc"] for q in s9(n) if "rc" in q]
        print(f"| {n} | {res(n)} | {sum(q['cont'] for q in L)} | {sum(q['fb'] for q in L)} | {x3c(n)} | {x3g(n)} |")

DR = [r for r in runs if any("bar" in q for q in s9(r[1]))]
if DR:
    print("\n## (11) 진단 요약 (s9.jsonl; 프레임 3~10 평균 → 프레임 80~ 평균; 활성 = 걸음 시작 활성 장벽 표본점 (y0e 부름 합), 띠 = min(d0, d1) < d̂, 띠 블록 = 띠 점이 든 128 블록; 잔차 비 = 뉴턴 반복 시작 ‖g‖²/F²; PCG 비 = 뉴턴 반복 0·1 의 마지막 PCG r²/rf)\n")
    print("| 판 | 노드 | 표본점 | 활성 | 띠 | 띠 블록 | 반복 0 잔차 비 (중앙) | 반복 1 잔차 비 (중앙) | PCG 비 반복 0 · 1 (중앙) | EW η 반복 1 (중앙) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for st, n, s, md, x in DR:
        L = [q for q in s9(n) if "bar" in q]
        def win(a, b):
            return [q for q in L if a <= q["f"] <= b]
        def med(v):
            v = [x for x in v if x is not None and math.isfinite(x)]
            return stt.median(v) if v else None
        def col(fn):
            A, B = win(2, 9), win(79, 999)
            fa = mean([fn(q) for q in A]); fb = mean([fn(q) for q in B])
            return f"{fmt(fa, 0)} → {fmt(fb, 0)}"
        def lgv(code, last=False):
            out = []
            for q in L:
                v = [e[1] for e in q.get("lg", []) if e[0] == code]
                if v:
                    out.append(v[-1] if last else v[0])
            return out
        def pcg_last(it):
            out = []
            for q in L:
                lg = q.get("lg", [])
                seq, cur = [], None
                for c_, v in lg:
                    if c_ == 100 + it:
                        cur = []
                        seq.append(cur)
                    elif 200 <= c_ < 300 and cur is not None:
                        cur.append(v)
                    elif 100 <= c_ < 200:
                        cur = None
                if seq and seq[0]:
                    out.append(seq[0][-1])
            return out
        print(f"| {n} | {col(lambda q: q['ip']['nn'])} | {col(lambda q: q['ip']['nq'])} | {col(lambda q: q['bar']['act'])} | {col(lambda q: q['bar']['band'])} | {col(lambda q: q['bar']['bblk'])} | {fmt(med(lgv(100)), 3)} | {fmt(med(lgv(101)), 4)} | {fmt(med(pcg_last(0)), 3)} · {fmt(med(pcg_last(1)), 3)} | {fmt(med(lgv(301)), 3)} |")


def s10(n):
    return jl(n, "s10.jsonl") or []


NR = [r for r in runs if s10(r[1])]
if NR:
    print("\n## (12) 노드 상자 압축 N 확인 (s10.jsonl; 확인 판 @s10n 2: 고친 커널 부름마다 압축 판 (진짜 버퍼) 대 상자 전체 원래 판 (그림자 버퍼, 같은 앞 상태) 32 비트 낱말 비트 대조)\n")
    print("| 판 | 결과 | 고친 커널 수 | 대조 부름 | **다른 낱말** | 다름 난 커널 (부름 · 낱말) | 못 맞춘 부름 |")
    print("|---|---|---|---|---|---|---|")
    for st, n, s_, md, x in NR:
        L = s10(n)
        tx = next((q["tx"] for q in L if "tx" in q), [])
        ch = {}
        for q in L:
            for k, v in (q.get("chk") or {}).items():
                a = ch.setdefault(k, [0, 0]); a[0] += v[0]; a[1] += v[1]
        um = {}
        for q in L:
            for k, v in (q.get("unm") or {}).items():
                um[k] = um.get(k, 0) + v
        bad = {k: v for k, v in ch.items() if v[1]}
        print(f"| {n} | {res(n)} | {len(tx)} | {sum(v[0] for v in ch.values())} | **{sum(v[1] for v in ch.values())}** | {', '.join(f'{k} ({v[0]} · {v[1]})' for k, v in sorted(bad.items())) or '-'} | {um or '-'} |")
    alln = sorted({k for r in NR for q in s10(r[1]) for k in (q.get("chk") or {})})
    if alln:
        print("\n| 커널 | 대조 부름 (모든 확인 판) | 다른 낱말 |")
        print("|---|---|---|")
        for k in alln:
            a = b = 0
            for r in NR:
                for q in s10(r[1]):
                    v = (q.get("chk") or {}).get(k)
                    if v:
                        a += v[0]; b += v[1]
            print(f"| {k} | {a} | {b} |")

print("\n## (13) 넘침 다시 할당 프레임 (fem.jsonl redo · s6.jsonl 그래프 재생 ng · 호스트 host; 다시 할당 기록 = s10.jsonl ovf 또는 s7ovf.jsonl: 넘침 때 상자 노드 IP[9]·칸 IP[10]·점유 칸 IP[11] 대 용량)\n")
print("| 판 | 다시 한 프레임 | 그 프레임 ng · host ms | 다시 할당 기록 (IP 9·10·11 / capN·capC·capQc -> 새 용량) |")
print("|---|---|---|---|")
for st, n, s_, md, x in runs:
    if s_ != "M3":
        continue
    fj = jl(n, "fem.jsonl") or []
    rd = [q["t"] for q in fj if q.get("redo")]
    s6 = {q["t"]: q for q in (jl(n, "s6.jsonl") or [])}
    ov = [q for q in (jl(n, "s7ovf.jsonl") or [])] + [q for q in s10(n) if q.get("ovf")]
    ovs = []
    for q in ov:
        ip = q.get("ip")
        if ip:
            ovs.append(f"{ip[9]}·{ip[10]}·{ip[11]} / {'·'.join(str(v) for v in (q.get('old') or [q.get('capN'), q.get('capC'), q.get('capQc')]))} -> {'·'.join(str(v) for v in (q.get('new') or []))}")
        elif q.get("new"):
            ovs.append(f"첫 할당 -> {'·'.join(str(v) for v in q['new'])}")
    rs = "; ".join("%s: %s · %s" % (t, s6[t].get("ng"), s6[t].get("host")) for t in rd if t in s6) or "-"
    print(f"| {n} | {rd or '-'} | {rs} | {'; '.join(ovs) or '-'} |")

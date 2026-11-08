"""절 CM56 묶음 t2chain 표: python exe/t2chain_tab.py <결과 디렉토리> -> markdown (stdout). 판 목록 = exe/t2chain_spec.txt.
바탕 = 공통 후보 F + A3 (@s4w 1 @s4y 1 @s4x 2). 정의 (@tc): C0 대조 (진단만) · K1 누적 이월 · K4 한 프레임 이월 · K2 자른 걸음 몸 전체 후퇴 오일러.
시간 = 그래프 재생 GPU ms (첫 2 프레임 뺌), 60 fps 문턱 16.7 ms. 판마다 빈 카드 (계산 프로세스 0) 하나에 판 하나, 진단 E1 켬.
발산 = 판정 발산 + 메모리 넘침 (요약 없음). 자른 프레임 = 예산이 뉴턴을 자른 프레임 (s3.jsonl cut > 0), 덜 풀린 프레임 = 마지막 시도 걸음 덜 풀림 (fem.jsonl uf.fail),
연쇄 = 자르고 덜 풀린 프레임이 잇달아 이어진 최대 길이. 빠른 노드 = 속도 > 3 m/s 노드 (e1.jsonl nfast), 이탈 = 몸 상자 4 칸 밖 입자 (e1 nfar).
"""
import json
import math
import os
import sys

O = sys.argv[1]
HERE = os.path.dirname(os.path.abspath(__file__))
spec = [l.split(None, 4) for l in open(os.path.join(HERE, "t2chain_spec.txt")) if l.strip() and not l.startswith("#")]
spec = [s + [""] * (5 - len(s)) for s in spec]
LIM = 1000.0 / 60.0
SCN = {"L": "레고", "W": "wolf", "F": "ficus", "B": "bread", "M3": "수박"}
CFG = {"0": "C0 대조 (F + A3)", "1": "K1 누적 이월", "4": "K4 한 프레임 이월", "2": "K2 자른 걸음 몸 전체 후퇴 오일러", "3": "K1 + K2", "6": "K4 + K2"}


def jl(n, f):
    p = os.path.join(O, n, f)
    if not os.path.exists(p):
        return []
    out = []
    for l in open(p, errors="replace"):
        try:
            out.append(json.loads(l))
        except Exception:
            pass
    return out


def summ(n):
    try:
        return json.load(open(os.path.join(O, n, "summary.json")))
    except Exception:
        return None


def logtxt(n):
    p = os.path.join(O, n, "log.txt")
    return open(p, errors="replace").read() if os.path.exists(p) else ""


def res(n):
    s = summ(n)
    if s is None:
        lt = logtxt(n)
        if "OutOfMemoryError" in lt or "out of memory" in lt:
            return "발산 (메모리)"
        return "도는 중/실패" if jl(n, "fem.jsonl") else "없음"
    return ("발산 f%d" % s["diverged"]) if s.get("diverged") else "완주"


def fmt(x, d=2):
    return "-" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{d}f}"


def tc_of(x):
    t = x.split()
    return t[t.index("@tc") + 1] if "@tc" in t else "0"


def run_info(n, sc):
    F = jl(n, "fem.jsonl")
    S3 = jl(n, "s3.jsonl")
    T = jl(n, "t2c.jsonl")
    E = {q["f"]: q for q in jl(n, "e1.jsonl") if "f" in q}
    s = summ(n)
    r = dict(n=n, res=res(n), fr=len(F))
    e = (s or {}).get("err_mean_done", (s or {}).get("err_mean"))
    r["err"] = 100 * e if e is not None else None
    ms = [q["ms"] for q in F if q.get("t", 0) >= 2 and q.get("ms") is not None and math.isfinite(q["ms"])]
    r["mean"] = sum(ms) / len(ms) if ms else None
    r["max"] = max(ms) if ms else None
    r["over"] = sum(1 for m in ms if m > LIM)
    r["ms_n"] = len(ms)
    cut = [int((S3[i] if i < len(S3) else {}).get("cut", 0) or 0) > 0 for i in range(len(F))]
    fail = [bool(sum((q.get("uf") or {}).get("fail", [0]))) for q in F]
    r["ncut"] = sum(cut)
    r["nfail"] = sum(fail)
    best = cur = 0
    for c_, f_ in zip(cut, fail):
        cur = cur + 1 if (c_ and f_) else 0
        best = max(best, cur)
    r["chain"] = best
    r["nfast"] = max((q.get("nfast", 0) for q in E.values()), default=None)
    r["nfar"] = max((q.get("nfar", 0) for q in E.values()), default=None)
    r["nn"] = max((q.get("nn", 0) for q in F), default=None)
    r["bank_end"] = T[-1]["bank"] if T else None
    r["bank_min"] = min((q["bank"] for q in T), default=None)
    r["nall"] = sum(q.get("nall", 0) for q in T)
    r["ins"] = max((q.get("ins", 0) for q in T), default=None)
    if sc == "W" and len(F) >= 50 and "hp" in F[49]:
        r["f50"] = F[49]["hp"][0] * 1000
    if sc == "W":
        p = os.path.join(O, n, "bbox.txt")
        if os.path.exists(p):
            for l in open(p):
                if "pred.pt" in l:
                    try:
                        d_ = json.loads(l.split(" ", 1)[1]); r["bb"] = (max(z["nfar"] for z in d_.values()), max(z["dmax"] for z in d_.values()))
                    except Exception:
                        pass
    if sc == "F":
        try:
            d = json.load(open(os.path.join(O, n, "period.json")))
            r["per"] = d.get("ratio"); r["amp"] = d.get("amp_ratio")
        except Exception:
            pass
    if sc.startswith("M"):
        for q in reversed(F):
            v = (q.get("en") or {}).get("lj0")
            if v is not None and q.get("err") is not None and math.isfinite(q["err"]):
                r["lj"] = 100 * v
                break
    # 연쇄 기록: 빠른 노드 > 1000 첫 프레임 (없으면 가장 긴 연쇄 시작) 앞뒤
    on = next((f for f in sorted(E) if E[f].get("nfast", 0) > 1000), None)
    r["onset"] = on
    r["F"], r["S3"], r["T"], r["E"], r["cut_l"], r["fail_l"] = F, S3, T, E, cut, fail
    return r


def rng(v, d=2):
    v = [q for q in v if q is not None and (not isinstance(q, float) or math.isfinite(q))]
    return f"{min(v):.{d}f}~{max(v):.{d}f}" if v else "-"


def mean(v):
    v = [q for q in v if q is not None]
    return sum(v) / len(v) if v else None


rows = {}
for st, n, sc, md, x in spec:
    if st == "s0k" or not os.path.isdir(os.path.join(O, n)):
        continue
    rows.setdefault((sc, tc_of(x)), []).append(run_info(n, sc))

print("# 절 CM56 묶음 t2chain -- 시간 예산 T2 덜 풀린 걸음 연쇄 끊기, 공통 후보 F + A3 위\n")
print(__doc__.split("\n", 1)[1])
print("## 씬 × 정의 (판마다 빈 카드 하나)\n")
print("| 씬 | 정의 | 판 | 발산 | 오차 % 평균 [범위] | 씬 지표 | 평균 ms (판 평균의 범위) | 최악 ms | > 16.7 프레임 (판 합 / 판당 최대) | 자른 프레임 판당 평균 | 연쇄 최대 | 빠른 노드 최대 | 이탈 최대 | 노드 상자 최대 | 끝 잔고 ms (K1·K4) |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for sc in ("L", "W", "F", "B", "M3"):
    for tc in ("0", "1", "4", "2", "3", "6"):
        R = rows.get((sc, tc))
        if not R:
            continue
        nd = sum(1 for r in R if r["res"].startswith("발산"))
        ok = [r for r in R if not r["res"].startswith("발산")]
        errs = [r["err"] for r in ok]
        if sc == "W":
            met = f"f50 {rng([r.get('f50') for r in R], 0)} mm, 이탈 판 {sum(1 for r in R if r.get('bb') and r['bb'][0] > 20)}/{sum(1 for r in R if r.get('bb'))}"
        elif sc == "F":
            met = f"주기 비 {rng([r.get('per') for r in R])}, 진폭 비 {rng([r.get('amp') for r in R])}"
        elif sc == "M3":
            met = f"끝 손상 {fmt(mean([r.get('lj') for r in ok]))} [{rng([r.get('lj') for r in ok])}] %"
        else:
            met = "-"
        means = [r["mean"] for r in ok]
        mx = max((r["max"] for r in ok if r["max"] is not None), default=None)
        print(f"| {SCN[sc]} | {CFG.get(tc, tc)} | {len(R)} | {nd} | {fmt(mean(errs))} [{rng(errs)}] | {met} | {rng(means)} | {fmt(mx, 1)} | {sum(r['over'] for r in ok)} / {max((r['over'] for r in ok), default='-')} | "
              f"{fmt(mean([r['ncut'] for r in R]), 1)} | {max(r['chain'] for r in R)} | {max((r['nfast'] or 0) for r in R)} | {max((r['nfar'] or 0) for r in R)} | {max((r['nn'] or 0) for r in R)} | "
              f"{rng([r['bank_end'] for r in R], 0) if tc in ('1', '4', '3', '6') else '-'} |")
print()

print("## 연쇄 기록 (발산 판과 빠른 노드 > 1000 판: 첫 빠른 노드 프레임 앞 6 ~ 뒤 4; 프레임 번호 = fem t + 1)\n")
print("열: 프레임 · ms · 자름 · 뉴턴 nN · PCG nP · 덜 풀림 · 보폭 최소 · 마지막 반복 시작 ‖g‖²/F² (판정) · 반복 0 시작 ‖g‖² (다음 프레임 시작점) · 마지막 반복 ‖g‖²/F² (기록) · 예산 한도 ms · 잔고 ms · 빠른 노드 · 이탈 · 노드 상자\n")
for (sc, tc), R in sorted(rows.items()):
    for r in R:
        if not (r["res"].startswith("발산") or (r["nfast"] or 0) > 1000):
            continue
        on = r["onset"]
        F, S3, T, E = r["F"], r["S3"], r["T"], r["E"]
        if on is None:
            on = len(F)
        print(f"**{r['n']}** ({SCN[sc]}, {CFG.get(tc, tc)}; {r['res']}, 첫 빠른 노드 > 1000 프레임 {r['onset']})\n")
        print("| 프레임 | ms | 자름 | nN | nP | 덜 풀림 | 보폭 | 판정 g²/F² | 반복 0 ‖g‖² | 끝 반복 g²/F² | 한도 | 잔고 | 빠른 | 이탈 | 상자 |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for i in range(max(0, on - 7), min(len(F), on + 4)):
            q = F[i]; s3 = S3[i] if i < len(S3) else {}; t = T[i] if i < len(T) else {}; e = E.get(i + 1, {})
            it = t.get("it") or []
            g0 = it[0][0] if it else None
            gl = (it[-1][0] / it[-1][1]) if it and it[-1][1] else None
            uf = q.get("uf") or {}
            print(f"| {i + 1} | {fmt(q.get('ms'), 1)} | {s3.get('cut', '-')} | {q['it']['nN'] if 'it' in q else '-'} | {q['it']['nP'] if 'it' in q else '-'} | {uf.get('fail')} | "
                  f"{[round(v, 3) for v in uf.get('smin', [])]} | {['%.2g' % v for v in uf.get('g2', [])]} | {('%.3g' % g0) if g0 is not None else '-'} | {('%.3g' % gl) if gl is not None else '-'} | "
                  f"{t.get('lim', '-')} | {t.get('bank', '-')} | {e.get('nfast', '-')} | {e.get('nfar', '-')} | {q.get('nn')} |")
        print()

print("## 판별\n")
print("| 판 | 씬 | 정의 | 결과 | 프레임 | 오차 % | 지표 | 평균 ms | 최악 ms | > 16.7 | 자른 프레임 | 덜 풀린 프레임 | 연쇄 최대 | 몸 전체 BE 걸음 | 빠른 노드 최대 | 이탈 최대 | 상자 최대 | 잔고 끝 · 최소 ms | 시작 카드 다른 프로세스 |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for st, n, sc, md, x in spec:
    if not os.path.isdir(os.path.join(O, n)):
        continue
    r = run_info(n, sc)
    met = fmt(r.get("f50"), 0) if sc == "W" else (fmt(r.get("per")) if sc == "F" else (fmt(r.get("lj")) if sc.startswith("M") else "-"))
    cp = "?"
    p = os.path.join(O, n, "apps_at_start.txt")
    if os.path.exists(p):
        for l in open(p):
            if l.startswith("카드 "):
                cp = l.split("계산 프로세스")[-1].strip()
    print(f"| {n} | {SCN.get(sc, sc)} | {tc_of(x)} | {r['res']} | {r['fr']} | {fmt(r['err'])} | {met} | {fmt(r['mean'])} | {fmt(r['max'], 1)} | {r['over']} | {r['ncut']} | {r['nfail']} | {r['chain']} | {r['nall']} | "
          f"{r['nfast']} | {r['nfar']} | {r['nn']} | {fmt(r['bank_end'], 0)} · {fmt(r['bank_min'], 0)} | {cp} |")

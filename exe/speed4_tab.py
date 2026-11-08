"""절 CM50 묶음 speed4 표: 판마다 시간·정확도, 단계 기록 판의 프레임별 단계 시간 (16.7 ms 넘는 프레임), S1 확인 셈.
사용: python exe/speed4_tab.py <출력 디렉토리>  (stdout = markdown)"""
import glob
import json
import os
import sys

import numpy as np

O = sys.argv[1]
B = 16.7
CATS = ["P2G", "걸음 시작", "W̄", "R2·A2", "요소", "장벽 기울기", "기울기·판정", "PCG", "장벽·뒤집힘 보폭", "선탐색 에너지", "걸음 끝", "RM", "G2P", "구간 사이", "기타"]
NEWTON = ["요소", "장벽 기울기", "기울기·판정", "PCG", "장벽·뒤집힘 보폭", "선탐색 에너지"]


def jl(p):
    if not os.path.exists(p):
        return []
    out = []
    for l in open(p):
        try:
            out.append(json.loads(l))
        except Exception:
            pass
    return out


def card_ok(d):
    p = os.path.join(d, "apps_at_start.txt")
    if not os.path.exists(p):
        return "?"
    for l in open(p):
        if l.startswith("카드 "):
            return l.split("계산 프로세스")[-1].strip()
    return "?"


def run_row(d):
    n = os.path.basename(d)
    S = json.load(open(os.path.join(d, "summary.json"))) if os.path.exists(os.path.join(d, "summary.json")) else None
    L = jl(os.path.join(d, "fem.jsonl"))
    if not L:
        L2 = [json.loads(l) for l in open(os.path.join(d, "fem.jsonl"))] if os.path.exists(os.path.join(d, "fem.jsonl")) else []
        L = L2
    ms = [q.get("ms") for q in L if q.get("t", 0) >= 2 and q.get("ms") is not None]
    ms = [m for m in ms if m == m]
    redo = sum(1 for q in L if len((q.get("ks") or {}).get("tries", [])) > 1)
    en = [q["en"] for q in L if isinstance(q.get("en"), dict)]
    lj = en[-1].get("lj0") if en else None
    nn = max((q.get("nn", 0) for q in L), default=0)
    r = dict(n=n, fr=(S or {}).get("frames_done", len(L)), div=(S or {}).get("diverged"), err=(S or {}).get("err_mean_done"),
             mean=float(np.mean(ms)) if ms else None, mx=float(np.max(ms)) if ms else None, mxf=(int(np.argmax(ms)) + 3) if ms else None,
             over=sum(m > B for m in ms), redo=redo, lj=lj, nn=nn, sha=(S or {}).get("x_sha1", "")[:10], card=card_ok(d))
    S4 = jl(os.path.join(d, "s4.jsonl"))
    if S4 and "ws4" in S4[0]:
        r["ws4"] = [sum(q["ws4"][k] for q in S4) for k in ("round", "clean", "bad")]
    if S4 and "yl4" in S4[0]:
        r["ws4"] = (r.get("ws4") or []) + ["S2"] + [sum(q["yl4"][k] for q in S4) for k in ("chk", "clean", "bad")]
    if S4 and "x1" in S4[0]:
        v = [q["x1"] if isinstance(q["x1"], list) else [q["x1"], 0, 0] for q in S4]
        r["x1"] = [sum(a[i] for a in v) for i in range(3)]
    return r


def f(v, p=2):
    return "-" if v is None else (f"{v:.{p}f}" if isinstance(v, float) else str(v))


dirs = sorted(d for d in glob.glob(os.path.join(O, "*")) if os.path.isdir(d) and os.path.exists(os.path.join(d, "cmd.txt")))
print("# 절 CM50 speed4 표\n")
print("## 판별 (시간 = 그래프 재생 GPU ms, 프레임 3~; 16.7 넘는 프레임 수; 다시 한 프레임 = 시도 2 이상; 손상 = 끝 프레임 logJp > 0 몫)\n")
print("| 판 | 프레임 | 발산 | 오차 % | 평균 ms | 최대 ms (프레임) | > 16.7 | 다시 한 프레임 | 끝 손상 % | 노드 상자 최대 | 끝 해시 | 시작 카드 다른 프로세스 | S1 라운드·깨끗·어긋남 | 꼬리 건너뜀·X2·X3e4 |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
rows = []
for d in dirs:
    try:
        r = run_row(d)
    except Exception as e:
        print(f"| {os.path.basename(d)} | 읽기 실패 {e!r} |")
        continue
    rows.append(r)
    print(f"| {r['n']} | {r['fr']} | {r['div'] or 0} | {f(100 * r['err'] if r['err'] is not None else None)} | {f(r['mean'])} | {f(r['mx'], 1)} ({r['mxf']}) | {r['over']} | {r['redo']} | "
          f"{f(100 * r['lj'] if r['lj'] is not None else None)} | {r['nn']} | {r['sha']} | {r['card']} | {r.get('ws4', '-')} | {r.get('x1', '-')} |")

# ---- 단계 기록 판
for d in dirs:
    S4 = jl(os.path.join(d, "s4.jsonl"))
    if not S4 or "sp" not in S4[0]:
        continue
    L = {q["t"]: q for q in jl(os.path.join(d, "fem.jsonl"))}
    n = os.path.basename(d)
    print(f"\n## 단계 기록 {n} (ms, 프레임 안 모든 시도 합; 구간 사이 = 나눈 그래프 경계의 호스트 읽기·띄우기; 표식 커널 몫 포함)\n")
    agg = {"보통": [], "다시 함": []}
    rowsF = []
    for q in S4:
        t = q["t"]
        sp = q.get("sp") or {}
        tr = sp.get("tries") or []
        fl = L.get(t, {})
        tot = {}
        its = []
        for tq in tr:
            for k, v in tq["cat"].items():
                tot[k] = tot.get(k, 0.0) + v
            its.append(tq.get("it", {}))
        kind = "다시 함" if len((fl.get("ks") or {}).get("tries", [])) > 1 else "보통"
        if t >= 2:
            agg[kind].append(tot)
        ms = fl.get("ms")
        rowsF.append((t, ms, tr, tot, fl, its))
    print("| 프레임 종류 | 수 | " + " | ".join(CATS) + " | 뉴턴 합 |")
    print("|---|---|" + "---|" * (len(CATS) + 1))
    for kind, A in agg.items():
        if not A:
            continue
        m = {c: float(np.mean([a.get(c, 0.0) for a in A])) for c in CATS}
        print(f"| {kind} (평균) | {len(A)} | " + " | ".join(f"{m[c]:.2f}" for c in CATS) + f" | {sum(m[c] for c in NEWTON):.2f} |")
    print(f"\n### {n} 16.7 ms 넘는 프레임 (재생 GPU ms) -- 원인 = 단계 합이 가장 큰 몫\n")
    print("| 프레임 | ms | 시도 (K, 무너짐, ms) | 노드 상자 | 가우스 점 | 뉴턴 반복 수 | " + " | ".join(CATS) + " | 반복별 뉴턴 ms (시도별) | 원인 |")
    print("|---|---|---|---|---|---|" + "---|" * len(CATS) + "---|---|")
    med = {c: float(np.median([a.get(c, 0.0) for a in agg["보통"]])) if agg["보통"] else 0.0 for c in CATS}
    for t, ms, tr, tot, fl, its in rowsF:
        if t < 2 or ms is None or not (ms > B):
            continue
        tries = (fl.get("ks") or {}).get("tries", [])
        ex = {c: tot.get(c, 0.0) - med[c] for c in CATS}
        why = []
        if len(tries) > 1:
            why.append(f"X3c·쪼개기 다시 하기 (시도 {len(tries)})")
        big = max(ex, key=ex.get)
        why.append(f"보통 프레임 중앙보다 많은 몫 1 위 {big} +{ex[big]:.1f}")
        itx = "; ".join(",".join(f"{k}:{v:.1f}" for k, v in it.items()) for it in its)
        print(f"| {t + 1} | {ms:.1f} | {tries} | {fl.get('nn')} | {fl.get('nq')} | {(fl.get('it') or {}).get('nN')} | " + " | ".join(f"{tot.get(c, 0.0):.2f}" for c in CATS) + f" | {itx} | {'; '.join(why)} |")
    unk = sorted({u for q in S4 for u in q.get("unk", [])})
    if unk:
        print(f"\n(기타 단계로 간 뉴턴 안 커널: {unk})")

# ---- 커널 프로파일 판 (speed1_prof, 동기 벽시계)
for d in dirs:
    L = jl(os.path.join(d, "fem.jsonl"))
    P = [q for q in L if "kernel_ms" in q]
    if not P:
        continue
    n = os.path.basename(d)
    print(f"\n## 커널 프로파일 {n} (그래프 밖 한 프레임, 커널마다 동기 벽시계 ms -- 실행 지연 포함, 모든 구간을 돌림; 프레임 평균 상위 30)\n")
    tot = {}
    for q in P:
        for k, (ms, c) in q["kernel_ms"].items():
            a = tot.setdefault(k, [0.0, 0])
            a[0] += ms; a[1] += c
    print("| 커널 | 프레임 평균 ms | 프레임 평균 호출 |")
    print("|---|---|---|")
    for k, (ms, c) in sorted(tot.items(), key=lambda kv: -kv[1][0])[:30]:
        print(f"| {k} | {ms / len(P):.3f} | {c / len(P):.1f} |")

# ---- X2 확인 판 (--s4x 3)
for d in dirs:
    S4 = jl(os.path.join(d, "s4.jsonl"))
    R = [r for q in S4 for r in q.get("x2c", [])]
    if not R:
        continue
    n = os.path.basename(d)
    same = [r for r in R if r["kind"] == "같은 시도"]
    redo = [r for r in R if r["kind"] == "다시"]
    keys = [k for k in (same or redo)[0] if k not in ("kind", "coll", "IP", "bx_snap", "y_snap", "dq_snap")]
    print(f"\n## X2 확인 {n}: (가) 같은 첫 시도의 걸음 시작 앞 대 uf_end 뒤 (비트 같아야, bx 는 걸음 안에서 옮겨짐 -> 사본이 걸음 시작 앞 값과 같은지 bx_snap) {len(same)} 프레임 · (나) X3c 로 프레임 처음부터 다시 한 시도의 걸음 시작 앞 대 첫 시도 값 (P2G 원자 합 순서만 다름) {len(redo)} 프레임\n")
    print("| 배열 | (가) 최대 |차| | (가) 다른 원소 프레임 수 | (나) 최대 |차| | (나) 최대 상대 차 |")
    print("|---|---|---|---|---|")
    for k in keys + ["bx_snap", "y_snap", "dq_snap"]:
        a = [r.get(k) for r in same if r.get(k) is not None]
        b = [r.get(k) for r in redo if r.get(k) is not None]
        fa = lambda v: v[0] if isinstance(v, list) else v
        print(f"| {k} | {max((fa(v) for v in a), default='-')} | {sum(1 for v in a if fa(v) != 0)} | {max((fa(v) for v in b), default='-')} | {max((v[1] for v in b if isinstance(v, list)), default='-')} |")

"""절 CM101 묶음 q27 표 (exe/gpl_tab.py 복사; (4) 를 수박 꿈틀거림 (jit.json) 으로, (6) 모래시계 모드 (eig/*.json) 더함) -- 이하 절 CM100 설명: 절 CM100 묶음 gpl 표 (json 만 읽음 -- torch 없이 로컬에서; exe/speed14_tab.py · exe/lightg4_tab.py 함수 가져옴): python exe/gpl_tab.py <결과 디렉토리> -> markdown.
(1) 수박 비결정 (발산·오차·손상·이탈·튐·바닥 뚫음) (2) 다른 씬 판정 (오차·씬 지표) (3) 렌더 포함 벽시계 단독 (4) P1X 면 가벼움 (gpl.jsonl 가운데 프레임) (5) 연기 판.
오차 = 끝난 프레임 평균 %. 이탈 = 몸 상자 (0.1~99.9 백분위) 4 칸 밖 입자 (e1.jsonl; wolf 는 덤프 bbox), 이탈 판 = > 20 개. 튐 프레임 = 진단 D2 이탈 입자의 앞 5 프레임 중
입자 속도가 한 프레임에 1 m/s 넘게 는 프레임 (절 CM73 정의). 수박 손상 = logJp > 0 몫 (정답 f5 4.41 · f6 6.94 · f8 7.08 · 끝 6.51 %).
"""
import json
import math
import os
import re
import statistics as stt
import sys

O = sys.argv[1]
HERE = os.path.dirname(os.path.abspath(__file__))
spec = [l.split(None, 4) for l in open(os.path.join(HERE, "q27_spec.txt")) if l.strip() and not l.startswith("#")]
spec = [[q.strip() for q in s] + [""] * (5 - len(s)) for s in spec]
LIM = 1000.0 / 60.0
SCN = {"L": "레고", "W": "wolf", "F": "ficus", "B": "bread", "M3": "수박"}
NF = {"L": 80, "W": 50, "F": 125, "B": 300, "M3": 100}


def lab(x):
    q27 = re.search(r"@q27 ([01])", x)
    z = " H5z " in f" {x} "
    if q27 and q27.group(1) == "1":
        return "3x3x3 + 벌점 끔" if z else "3x3x3 + H5"
    return "H5 2x2x2 (대조)" if q27 else x


def jl(n, f):
    p = os.path.join(O, n, f)
    try:
        return [json.loads(l) for l in open(p) if l.strip()] if os.path.exists(p) else None
    except Exception:
        return None


def jf(n, f):
    try:
        return json.load(open(os.path.join(O, n, f)))
    except Exception:
        return None


def summ(n):
    return jf(n, "summary.json")


def oom(n):
    p = os.path.join(O, n, "log.txt")
    return os.path.exists(p) and "OutOfMemoryError" in open(p, errors="replace").read()


def fmt(v, d=2):
    return "-" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{v:.{d}f}"


def res(n):
    s = summ(n)
    if s is None:
        return "발산 (메모리)" if oom(n) else "없음/실패"
    if s.get("diverged") or s.get("div_at") or s.get("oom"):
        return "발산 f%s" % (s.get("div_at") or s.get("diverged"))
    return "완주 %d" % s["frames_done"]


def dv(n, nf):
    s = summ(n)
    return bool(s is None or s.get("diverged") or s.get("div_at") or s.get("oom") or (s.get("frames_done") or 0) < nf)


def err(n):
    s = summ(n)
    e = None if s is None else s.get("err_mean_done", s.get("err_mean"))
    return 100 * e if e is not None else None


def mean(v):
    v = [x for x in v if x is not None and math.isfinite(x)]
    return stt.mean(v) if v else None


def rng(v, d=2):
    v = [q for q in v if q is not None and math.isfinite(q)]
    return (f"{stt.mean(v):.{d}f} [{min(v):.{d}f}~{max(v):.{d}f}]" if len(v) > 1 else f"{v[0]:.{d}f}") if v else "-"


def dmg(n):
    L = [q for q in (jl(n, "fem.jsonl") or []) if "t" in q]
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


def e2s(n):
    L = jl(n, "e2.jsonl") or []
    E = [q for q in L if "esc" in q]
    seen, kk = set(), []
    for e in E:
        for fr in e["frames"]:
            if fr.get("v1") is not None and fr.get("v0") is not None and fr["v1"] - fr["v0"] > 1.0 and "vn_end_max_sol" in fr and fr["f"] not in seen:
                seen.add(fr["f"]); kk.append(fr)
    return dict(nk=len(kk), vsol=max((q["vn_end_max_sol"] for q in kk), default=None)) if L else None


def e5zmin(n):
    L = [q for q in (jl(n, "e5.jsonl") or []) if "nq" in q and q.get("zmin") is not None]
    return min((q["zmin"] for q in L), default=None)


def bbox(n):
    p = os.path.join(O, n, "bbox.txt")
    if not os.path.isfile(p):
        return None
    for ln in open(p):
        if "pred.pt" in ln:
            try:
                D = json.loads(ln[ln.index("{"):])
            except Exception:
                return None
            return max((v["nfar"], v["dmax"]) for v in D.values())
    return None


def period(n):
    p = os.path.join(O, n, "period.txt")
    if not os.path.isfile(p):
        return None
    for ln in open(p):
        m = re.search(r"정답 대비 ([0-9.]+)", ln)
        if m and not ln.startswith("정답"):
            return float(m.group(1))
    return None


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


def wst(v):
    v = [x for x in v if x is not None and math.isfinite(x)]
    return f"{stt.mean(v):.2f} / {max(v):.1f} / {sum(1 for x in v if x > LIM)}" if v else "-"


runs = []
for st, n, s, md, x in spec:
    if md == "z":
        for r in ("a", "b"):
            if os.path.isdir(os.path.join(O, f"{n}_{r}")):
                runs.append((st, f"{n}_{r}", s, md, x))
    elif os.path.isdir(os.path.join(O, n)):
        runs.append((st, n, s, md, x))
G = {}
for st, n, s, md, x in runs:
    if md in ("z", "v") or st.endswith("k"):
        continue
    G.setdefault((s, lab(x)), []).append(n)

print("# 절 CM101 q27 표 -- 칸마다 가우스 3x3x3 (27 점) 적분; 바탕 H5 (확장 없음, κ 강성 빼기 없음); 벌점 끔 = H5 − P1X − 가벼움 벌점\n")
print("## (1) 수박 비결정 (진단 E1·D1·D2·D5; 정답 손상 f5 4.41 · f6 6.94 · f8 7.08 · 끝 6.51 %)\n")
print("| 판 | 판 수 | 발산 | 오차 % 평균 [범위] | 손상 f5 · f6 · f8 % | 끝 손상 평균 [범위] % | 이탈 판 · 최대 수 · 거리 m | 튐 프레임 (판 합 / 판당) · 튐 노드 끝 속도 최대 m/s | 바닥 뚫음 z 최소 m |")
print("|---|---|---|---|---|---|---|---|---|")
for (s, lb), ns in G.items():
    if s != "M3":
        continue
    nd = sum(1 for n in ns if dv(n, 100))
    D = [dmg(n) for n in ns]
    d3 = [mean([100 * d[0][i] for d in D if d[0][i] is not None]) for i in range(3)]
    fa = [far(n) for n in ns]
    K = [e2s(n) for n in ns]
    K = [k for k in K if k]
    nk = sum(k["nk"] for k in K)
    vs = [k["vsol"] for k in K if k["vsol"] is not None]
    print(f"| {lb} | {len(ns)} | {nd}/{len(ns)} | {rng([err(n) for n in ns])} | {' · '.join(fmt(v) for v in d3)} | {rng([100 * d[1] if d[1] is not None else None for d in D])} | "
          f"{sum(1 for f in fa if (f[0] or 0) > 20)}/{len(ns)} · {max((f[0] or 0) for f in fa)} · {fmt(max((f[1] or 0) for f in fa))} | "
          f"{nk} / {fmt(nk / len(K) if K else None, 1)} · {fmt(max(vs) if vs else None, 0)} | {fmt(min([z for z in (e5zmin(n) for n in ns) if z is not None], default=None), 3)} |")
print()
print("## (2) 다른 씬 판정 (비결정; 이탈 = wolf 덤프 몸 상자 4 칸 밖 > 20 개 판 (bbox), 나머지 e1; ficus 주기 비 = 잎 끝 주기 / 정답)\n")
print("| 씬 | 판 | 판 수 | 발산 | 오차 % 평균 [범위] | 씬 지표 | 이탈 판 · 최대 수 · 거리 m |")
print("|---|---|---|---|---|---|---|")
for (s, lb), ns in G.items():
    if s == "M3":
        continue
    nd = sum(1 for n in ns if dv(n, NF[s]))
    if s == "W":
        bb = [bbox(n) for n in ns]
        ec = f"{sum(1 for b in bb if b and b[0] > 20)}/{len(ns)} · {max((b[0] for b in bb if b), default=0)} · {fmt(max((b[1] for b in bb if b), default=None))}"
        sm = "(heap 표 따로)"
    else:
        fa = [far(n) for n in ns]
        ec = f"{sum(1 for f in fa if (f[0] or 0) > 20)}/{len(ns)} · {max((f[0] or 0) for f in fa)} · {fmt(max((f[1] or 0) for f in fa))}"
        sm = f"주기 비 {rng([period(n) for n in ns])}" if s == "F" else "-"
    print(f"| {SCN[s]} | {lb} | {len(ns)} | {nd}/{len(ns)} | {rng([err(n) for n in ns])} | {sm} | {ec} |")
print()
print("## (3) 렌더 포함 벽시계 단독 (카드 하나에 차례로 a 다 돌고 b; W = 평균 / 최악 / > 16.7 프레임 수, 프레임 3~; ms)\n")
print("| 판 | 씬 | 판 종류 | GPU 합 a+b+c | **벽시계 W** | 오차 % | 결과 | 시작 부하 |")
print("|---|---|---|---|---|---|---|---|")
for st, n, s, md, x in runs:
    if md != "z" or st.endswith("k"):
        continue
    F = frames(n)
    tot = [q.get("a", float("nan")) + q.get("b", 0.0) + q.get("c", 0.0) if "b" in q else None for q in F]
    try:
        ld = open(os.path.join(O, n, "apps_at_start.txt")).read().strip().split("\n")[-1].split()[0]
    except Exception:
        ld = "-"
    print(f"| {n} | {SCN[s]} | {lab(x)} | {wst(tot)} | **{wst([q.get('W') for q in F])}** | {fmt(err(n))} | {res(n)} | {ld} |")
print()
print("## (4) 수박 꿈틀거림 (덤프 위치 프레임 사이 속도에서 강체 운동 뺀 RMS, 질량 가중, 창 f30~f100; exe/q27_jit.py)\n")
print("| 판 | 판 수 | 강체 뺀 속도 RMS mm/s 평균 [범위] | 판별 최대 평균 |")
print("|---|---|---|---|")
try:
    gj = json.load(open(os.path.join(O, "gt_jit.json")))
    print(f"| 정답 (GF) | 1 | {fmt(gj['vdef_rms_mean'], 1)} | {fmt(gj['vdef_rms_max'], 1)} |")
except Exception:
    pass
for (s_, lb), ns in G.items():
    if s_ != "M3":
        continue
    J = [jf(n, "jit.json") for n in ns]
    J = [q for q in J if q and q.get("vdef_rms_mean") is not None]
    print(f"| {lb} | {len(J)} | {rng([q['vdef_rms_mean'] for q in J], 1)} | {fmt(mean([q['vdef_rms_max'] for q in J]), 1)} |")
print()
print("## (5) 연기 판\n")
print("| 판 | 씬 | 플래그 | 결과 | 오차 % |")
print("|---|---|---|---|---|")
for st, n, s, md, x in runs:
    if st.endswith("k"):
        print(f"| {n} | {SCN[s]} | `{x}` | {res(n)} | {fmt(err(n))} |")
print()
print("## (6) 모래시계 모드 (s0d 덤프 프레임의 첫 반복 탄성 헤시안, 점유 가중 포함, 벌점 없음; exe/q27_eig.py; 영 = λ̂ < 1e-5 λ̂_max, 근사 영 = < 1e-3; 강체 6 포함)\n")
print("| 씬 | 프레임 | 적분 | 칸 · 가우스 점 (V>0) · 노드 | 영 (수렴) | 근사 영 | 영 모드 몫: 지지 칸 점유 1-8 · 9-17 · 18-26 · 27 (노드 몫) | λ̂ 7·31·101 번째 / max |")
print("|---|---|---|---|---|---|---|---|")
import glob
for p in sorted(glob.glob(os.path.join(O, "eig", "*.json"))):
    try:
        E = json.load(open(p))
    except Exception:
        continue
    sc = {"dW": "wolf", "dL": "레고", "dM": "수박"}.get(os.path.basename(p).split("_")[0], "?")
    for c in E["cases"]:
        lz = c["loc"]["zero"]
        bins = " · ".join(f"{fmt(lz.get(b, {}).get('mode_frac'), 2)} ({fmt(lz.get(b, {}).get('node_frac'), 2)})" for b in ("1-8", "9-17", "18-26", "27-27")) if lz.get("n") else "-"
        ea = c.get("ev_at") or []
        print(f"| {sc} | {c['frame_label']} | {E['q'] if E['q'] == 27 else 8} 점 | {c['nc']} · {c['nq']} · {c['nn']} | {c['n_zero']} ({c['n_zero_conv']}){' ≥' if c['all_kb_small'] else ''} | {c['n_near']} | {bins} | {' · '.join(f'{v:.1e}' for v in ea[:3])} |")
print()

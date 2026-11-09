"""절 CM83 묶음 ext1 표 (exe/speed11_tab.py · exe/wolfh8_tab.py 도움 함수 복사; json 만 읽음 -- torch 없이 로컬에서도): python exe/ext1_tab.py <결과 디렉토리> -> markdown.
(1) 수박 비결정 (절 CM78.4·CM82 형식) (2) 다른 씬 판정 (오차·씬 지표: wolf f50·이탈, ficus 주기 비) (3) wolf 절 CM81 형식 (4) 렌더 포함 벽시계 단독 (절 CM82 (3) 형식)
(5) 확장 분류 (w8.jsonl) (6) 옛 실행기 이식 검사 (x1ck.jsonl) (7) 연기 판.
"""
import json
import math
import os
import re
import statistics as stt
import sys

O = sys.argv[1]
HERE = os.path.dirname(os.path.abspath(__file__))
spec = [l.split(None, 4) for l in open(os.path.join(HERE, "ext1_spec.txt")) if l.strip() and not l.startswith("#")]
spec = [[q.strip() for q in s] + [""] * (5 - len(s)) for s in spec]
LIM = 1000.0 / 60.0
SCN = {"L": "레고", "W": "wolf", "F": "ficus", "B": "bread", "M3": "수박"}
VAR = {"A": "H6", "N": "H6np"}


def jl(n, f):
    p = os.path.join(O, n, f)
    try:
        return [json.loads(l) for l in open(p) if l.strip()] if os.path.exists(p) else None
    except Exception:
        L = []
        for l in open(p):
            try:
                L.append(json.loads(l))
            except Exception:
                pass
        return L


def jf(n, f):
    try:
        return json.load(open(os.path.join(O, n, f)))
    except Exception:
        return None


def summ(n):
    return jf(n, "summary.json")


def fmt(v, d=2):
    return "-" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{v:.{d}f}"


def res(n):
    s = summ(n)
    if s is None:
        return "없음/실패"
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


def stat(v):
    v = [x for x in v if x is not None and math.isfinite(x)]
    return f"{stt.mean(v):.2f} / {max(v):.1f} / {sum(1 for x in v if x > LIM)}" if v else "-"


def mean(v):
    v = [x for x in v if x is not None and math.isfinite(x)]
    return stt.mean(v) if v else None


def rng(v, d=2):
    v = [q for q in v if q is not None and math.isfinite(q)]
    return (f"{stt.mean(v):.{d}f} [{min(v):.{d}f}~{max(v):.{d}f}]" if len(v) > 1 else f"{v[0]:.{d}f}") if v else "-"


def load(n):
    try:
        return open(os.path.join(O, n, "apps_at_start.txt")).read().strip().split("\n")[-1].split()[0]
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
    d = {q["t"]: (q.get("en") or {}).get("lj0") for q in L if "t" in q}
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


def com(n, t):
    r = [q for q in (jl(n, "fem.jsonl") or []) if "t" in q]
    return r[t]["hp"][0] * 1000 if r and t < len(r) and "hp" in r[t] else None


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


def relfall(n):
    p = os.path.join(O, n, "relfall.md")
    if not os.path.isfile(p):
        return None
    for ln in open(p):
        c = [x.strip() for x in ln.strip().strip("|").split("|")]
        if c and c[0] == n:
            return [float(re.findall(r"\(([-0-9.]+)\)", x)[0]) if "(" in x else None for x in c[1:]]
    return None


def relfall_gt():
    for st, n, s, md, x in spec:
        p = os.path.join(O, n, "relfall.md")
        if os.path.isfile(p):
            for ln in open(p):
                c = [x_.strip() for x_ in ln.strip().strip("|").split("|")]
                if c and c[0] == "정답":
                    return [float(re.findall(r"\(([-0-9.]+)\)", x_)[0]) for x_ in c[1:]]
    return None


AN, GT = {}, None
for st, n, s, md, x in spec:
    J = jf(n, "an.json")
    if J:
        GT = GT or J["gt"]
        for k_, v_ in J["runs"].items():
            AN[k_] = v_


def an(n, f, k):
    r = AN.get(n)
    if not r:
        return None
    s = r["series"].get(str(f)) or r["series"].get(f)
    return None if s is None else s.get(k)


def w8(n):
    L = jl(n, "w8.jsonl") or []
    L = [q for q in L if q.get("ks") == 0]
    if not L:
        return None
    return dict(n=len(L), outer=mean([q["outer"] for q in L]), iso=mean([q["iso"] for q in L]), inner=mean([q["inner"] for q in L]),
                mfo=mean([q["mf_outer"] for q in L if q.get("mf_outer") is not None]), mfi=mean([q["mf_iso"] for q in L if q.get("mf_iso") is not None]),
                mfo_max=max((q["mf_outer"] or 0) for q in L), mfi_max=max((q["mf_iso"] or 0) for q in L))


def grp(n):
    return n.split("_")[0]


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
    if md == "z" or st.endswith("k") or st == "s0m":
        continue
    G.setdefault((st, grp(n), s, x), []).append(n)

print("# 절 CM83 ext1 표 -- 공통 후보 H6 (= H5 + S_구속 + C + P1X 강성 × κ + T1E2) 와 H6np (H6 − 가벼움 벌점 × κ)\n")
print("## (1) 수박 비결정 (절 CM78.4 형식; 정답 손상 f5 4.41 · f6 6.94 · f8 7.08 · 끝 6.51 %; 이탈 판 = 4 칸 밖 > 20 개)\n")
print("| 묶음 | 판 | 판 수 | 발산 | 오차 % 평균 [범위] | 손상 f5 · f6 · f8 % | 끝 손상 평균 [범위] % | 이탈 판 · 최대 수 · 거리 m | 두 시도 (X3c) 프레임 (판당) · 그중 X3g | 두 시도 프레임 시뮬 ms 평균 / 최대 |")
print("|---|---|---|---|---|---|---|---|---|---|")
for (st, g, s, x), ns in G.items():
    if s != "M3":
        continue
    nd = sum(1 for n in ns if dv(n, 100))
    D = [dmg(n) for n in ns]
    d3 = [mean([100 * d[0][i] for d in D if d[0][i] is not None]) for i in range(3)]
    fa = [far(n) for n in ns]
    ms = [x3ms(n) for n in ns]
    print(f"| {g} | {VAR.get(g[1:2], x)} | {len(ns)} | {nd}/{len(ns)} | {rng([err(n) for n in ns])} | {' · '.join(fmt(v) for v in d3)} | {rng([100 * d[1] if d[1] is not None else None for d in D])} | "
          f"{sum(1 for f in fa if (f[0] or 0) > 20)} · {max((f[0] or 0) for f in fa)} · {fmt(max((f[1] or 0) for f in fa))} | {fmt(mean([x3c(n) for n in ns]), 1)} · {fmt(mean([x3g(n) for n in ns]), 1)} | "
          f"{fmt(mean([m[0] for m in ms]), 1)} / {fmt(max([m[1] for m in ms if m[1] is not None], default=None), 1)} |")
print()
print("## (2) 다른 씬 판정 (비결정; 이탈 = wolf 덤프 몸 상자 4 칸 밖 > 20 개 판 (bbox), 나머지 e1; ficus 주기 비 = 잎 끝 주기 / 정답 1.398 s)\n")
print("| 묶음 | 씬 | 판 | 판 수 | 발산 | 오차 % 평균 [범위] | 씬 지표 | 이탈 판 · 최대 수 · 거리 m | 외부 노드 질량 % 평균 (최대) · 고립 % |")
print("|---|---|---|---|---|---|---|---|---|")
NF = {"L": 80, "W": 50, "F": 125, "B": 300}
for (st, g, s, x), ns in G.items():
    if s == "M3":
        continue
    nd = sum(1 for n in ns if dv(n, NF.get(s, 1)))
    if s == "W":
        f50 = [com(n, 49) for n in ns if not dv(n, 50)]
        sm = f"f50 {rng(f50, 0)} mm (정답 533)"
        bb = [bbox(n) for n in ns]
        ec = f"{sum(1 for b in bb if b and b[0] > 20)} · {max((b[0] for b in bb if b), default=0)} · {fmt(max((b[1] for b in bb if b), default=None))}"
    else:
        sm = f"주기 비 {rng([period(n) for n in ns])}" if s == "F" else "-"
        fa = [far(n) for n in ns]
        ec = f"{sum(1 for f in fa if (f[0] or 0) > 20)} · {max((f[0] or 0) for f in fa)} · {fmt(max((f[1] or 0) for f in fa))}"
    W = [w8(n) for n in ns]; W = [w for w in W if w]
    ws = f"{fmt(mean([100 * w['mfo'] for w in W]), 1)} ({fmt(max([100 * w['mfo_max'] for w in W], default=None), 1)}) · {fmt(mean([100 * w['mfi'] for w in W]), 1)}" if W else "-"
    print(f"| {g} | {SCN.get(s, s)} | {VAR.get(g[1:2], x)} | {len(ns)} | {nd}/{len(ns)} | {rng([err(n) for n in ns])} | {sm} | {ec} | {ws} |")
for (st, g, s, x), ns in G.items():
    if s != "M3":
        continue
    W = [w8(n) for n in ns]; W = [w for w in W if w]
    if W:
        print(f"| {g} | 수박 | {VAR.get(g[1:2], x)} | {len(ns)} | (위 (1)) | - | - | - | {fmt(mean([100 * w['mfo'] for w in W]), 1)} ({fmt(max([100 * w['mfo_max'] for w in W], default=None), 1)}) · {fmt(mean([100 * w['mfi'] for w in W]), 1)} |")
print()

print("## (3) wolf 절 CM81 형식 (정답 f10 860 · f20 758 · f30 573 · f50 533)\n")
print("| 후보 | 판 | 발산 | f10 · f20 mm | f30 mm | f50 mm [범위] | 오차 % [범위] | 낙하 몫 τ [.04,.08) · [.16,.24) · [.32,.48) | 면적 f30 · f50 cm² | 회전 반경 f50 mm | 이탈 판 · 최대 m | 동시 ms 최대 |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
rg_ = relfall_gt()
if GT:
    print(f"| 정답 | - | - | {480 + GT['10']['com']:.0f} · {480 + GT['20']['com']:.0f} | {480 + GT['30']['com']:.0f} | 533 | - | "
          f"{'-' if not rg_ else f'{rg_[1]:.2f} · {rg_[4]:.2f} · {rg_[6]:.2f}'} | {GT['30']['area_cm2']:.0f} · {GT['50']['area_cm2']:.0f} | {GT['50']['rg_mm']:.0f} | - | - |")
for (st, g, s, x), ns in G.items():
    if s != "W":
        continue
    f10, f20, f30, f50, er, a30, a50, rg, nd, ne, dm, mx, R1, R4, R6 = [], [], [], [], [], [], [], [], 0, 0, [], [], [], [], []
    for n in ns:
        if dv(n, 50):
            nd += 1
        else:
            f50.append(com(n, 49)); er.append(err(n)); a50.append(an(n, 50, "area_cm2")); rg.append(an(n, 50, "rg_mm"))
        f10.append(com(n, 9)); f20.append(com(n, 19)); f30.append(com(n, 29)); a30.append(an(n, 30, "area_cm2"))
        rf = relfall(n)
        if rf:
            R1.append(rf[1]); R4.append(rf[4]); R6.append(rf[6])
        b = bbox(n)
        if b:
            dm.append(b[1]); ne += b[0] > 20
        q = [x_["ms"] for x_ in (jl(n, "fem.jsonl") or []) if x_.get("t", 0) >= 2 and x_.get("ms") is not None and math.isfinite(x_["ms"])]
        if q:
            mx.append(max(q))
    print(f"| {g} ({VAR.get(g[1:2], x)}) | {len(ns)} | {nd} | {rng(f10, 0)} · {rng(f20, 0)} | {rng(f30, 0)} | {rng(f50, 0)} | {rng(er)} | {rng(R1)} · {rng(R4)} · {rng(R6)} | {rng(a30, 0)} · {rng(a50, 0)} | {rng(rg, 0)} | "
          f"{ne}/{len(ns)} · {fmt(max(dm) if dm else None)} | {fmt(max(mx) if mx else None, 1)} |")
print()

print("## (4) 렌더 포함 벽시계 단독 (카드 하나에 차례로 a 다 돌고 b; GPU 합 a+b+c · 벽시계 W = 평균 / 최악 / > 16.7 프레임 수, 프레임 3~; ms)\n")
print("| 판 | 씬 | 판 종류 | GPU 합 a+b+c | **벽시계 W** | W 중앙 | 오차 % | 두 시도 프레임 수 · 시뮬 평균 / 최대 ms | 결과 | 부하 |")
print("|---|---|---|---|---|---|---|---|---|---|")
for st, n, s, md, x in runs:
    if md != "z":
        continue
    F = frames(n)
    tot = [q.get("a", float("nan")) + q.get("b", 0.0) + q.get("c", 0.0) if "b" in q else None for q in F]
    W = [q.get("W") for q in F]
    Wv = [w for w in W if w is not None]
    xm = x3ms(n)
    print(f"| {n} | {SCN.get(s, s)} | {VAR.get(n[1:2], x)} | {stat(tot)} | **{stat(W)}** | {fmt(stt.median(Wv) if Wv else None)} | {fmt(err(n))} | {x3c(n)} · {fmt(xm[0], 1)} / {fmt(xm[1], 1)} | {res(n)} | {load(n)} |")
print()

print("## (5) 옛 실행기 이식 검사 (x1ck.jsonl; 그래프 밖 데움 record = 첫 프레임 K 1 과 처음 캡처되는 쪼갠 걸음·X3c 그래프의 프레임; 값은 최대 [중앙])\n")


def mx_md(v):
    v = [q for q in v if q is not None and isinstance(q, (int, float)) and math.isfinite(q)]
    return "-" if not v else f"{max(v):.2e} [{stt.median(v):.2e}]"


print("| 판 | 반복 검사 수 · 외부 노드 수 | C1 펼치기 · 접기 커널 대 파이썬 · 수반 | C2 접은 잔차 대 Eᵀ r0 · r_o 최대 · SC[1] | C3 z 대 D⁻¹r · z_o · r·z · p_o 펼침 | C4 대칭 · min uᵀAu · Ap_o | C5 차분 대 곱 (줄어든 e1e-3 · e1e-4 / 원래 e1e-3) | C6 확장 관계 어긋남 |y_o − y0_o − Σc(y_j − y0_j)| 최대 [중앙] / 걸음 증분 최대 (검사 수) | C7 MG r_o · min r·z · p_o | 오류 |")
print("|---|---|---|---|---|---|---|---|---|---|")
for st, n, s, md, x in runs:
    if "@x1ck" not in x:
        continue
    L = jl(n, "x1ck.jsonl") or []
    It = [q for q in L if q.get("kind") == "iter" and "C1" in q]
    Iv = [q for q in L if q.get("kind") == "inv"]
    Mg = [q for q in L if q.get("kind") == "mg"]
    g = lambda c, k: [q[c].get(k) for q in It if isinstance(q.get(c), dict)]
    c5 = lambda e, k: [q["C5"][e].get(k) for q in It if isinstance(q.get("C5"), dict) and isinstance(q["C5"].get(e), dict)]
    xi = jf(n, "x1info.json") or {}
    print(f"| {n} | {len(It)} · {fmt(mean([q.get('next') for q in It]), 0)} | {mx_md(g('C1', 'expand_rel'))} · {mx_md(g('C1', 'fold_rel'))} · {mx_md(g('C1', 'adj'))} | "
          f"{mx_md(g('C2', 'fold_r0_rel'))} · {mx_md(g('C2', 'r_ext_max'))} · {mx_md(g('C2', 'gg_rel'))} | {mx_md(g('C3', 'z_rel'))} · {mx_md(g('C3', 'z_ext_max'))} · {mx_md(g('C3', 'rz_rel'))} · {mx_md(g('C3', 'p_ext_rel'))} | "
          f"{mx_md(g('C4', 'sym'))} · {fmt(min(g('C4', 'uAu') + g('C4', 'vAv'), default=None), 3)} · {mx_md(g('C4', 'Ap_ext_max'))} | {mx_md(c5('e0.001', 'red_rel'))} · {mx_md(c5('e0.0001', 'red_rel'))} / {mx_md(c5('e0.001', 'full_rel'))} | "
          f"{mx_md([q.get('dev') for q in Iv])} m / 증분 최대 {fmt(max([q.get('dmax') or 0 for q in Iv], default=None), 4)} m ({len(Iv)}) | {mx_md([q.get('r_ext_max') for q in Mg])} · {fmt(min([q.get('rz') for q in Mg], default=None), 3)} · {mx_md([q.get('p_ext_rel') for q in Mg])} | {xi.get('ck_err') or xi.get('err') or '-'} |")
print()

print("## (6) 확장 분류 (w8.jsonl, 걸음 시작마다; 판 평균)\n")
print("| 판 | 걸음 | 내부 · 외부 · 고립 노드 | 외부 질량 % 평균 (최대) | 고립 질량 % 평균 (최대) |")
print("|---|---|---|---|---|")
for st, n, s, md, x in runs:
    w = w8(n)
    if w and (md != "z"):
        print(f"| {n} | {w['n']} | {fmt(w['inner'], 0)} · {fmt(w['outer'], 0)} · {fmt(w['iso'], 0)} | {fmt(100 * w['mfo'], 1)} ({fmt(100 * w['mfo_max'], 1)}) | {fmt(100 * w['mfi'], 1)} ({fmt(100 * w['mfi_max'], 1)}) |")
print()

print("## (7) 연기·검사 판\n")
print("| 판 | 씬 | 플래그 | 결과 | 오차 % |")
print("|---|---|---|---|---|")
for st, n, s, md, x in runs:
    if st.endswith("k") or st == "s0m":
        print(f"| {n} | {SCN.get(s, s)} | `{x}` | {res(n)} | {fmt(err(n))} |")
print()

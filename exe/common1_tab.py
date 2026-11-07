"""절 CM31 묶음 common1 표: python exe/common1_tab.py <결과 디렉토리> <단계...> -> 표 (stdout, markdown).

판 묶음 = 결정적 판 X (d), 비결정 X_n<i> (nd·n), 단독 시간 zX_a·zX_b (z). 오차 = 끝난 프레임 평균 %, 시간 = 첫 2 프레임 뺀 재생 GPU ms 평균/최대.
레고: 질량중심 z 정답 대비 (f10 낙하 늦음, f31·f43), 덜 풀린 걸음 수, off 와 처음 갈리는 프레임 (프레임 해시). wolf: 질량중심 z f30·f50 (정답 572.6·533.3 mm).
"""
import json
import math
import os
import re
import sys

O = sys.argv[1]
STG = sys.argv[2:]
spec = [l.split(None, 4) for l in open(os.path.join(os.path.dirname(__file__), "common1_spec.txt")) if l.strip() and not l.startswith("#")]
spec = [s + [""] * (5 - len(s)) for s in spec if s[0] in STG]


def rd(n):
    p = os.path.join(O, n, "fem.jsonl")
    if not os.path.exists(p):
        return None
    return [json.loads(l) for l in open(p)]


def summ(n):
    p = os.path.join(O, n, "summary.json")
    return json.load(open(p)) if os.path.exists(p) else None


def oom(n):
    p = os.path.join(O, n, "log.txt")
    return os.path.exists(p) and "OutOfMemoryError" in open(p, errors="replace").read()


def err(r, s, n=None):
    if s is None:
        return (None, "메모리") if (n and oom(n)) else (None, None)
    e = s.get("err_mean_done", s.get("err_mean"))
    return (e * 100 if e is not None else None), s.get("diverged")


def ms(r):
    v = [x["ms"] for x in (r or []) if x["t"] >= 2 and x.get("ms") is not None and math.isfinite(x["ms"])]
    return (sum(v) / len(v), max(v)) if v else (None, None)


def fmt(x, d=2):
    return "-" if x is None else f"{x:.{d}f}"


def nuf(r):
    return sum(sum(x["uf"]["fail"]) for x in r if "uf" in x) if r else None


def fdiff(r, ref):
    if not r or not ref:
        return None
    for x, y in zip(r, ref):
        if x.get("xh") != y.get("xh"):
            return x["t"] + 1
    return "같음" if len(r) == len(ref) else None


def com(r, t):
    if not r or t >= len(r) or "hp" not in r[t]:
        return None, None
    return r[t]["hp"][0] * 1000, r[t]["hg"][0] * 1000


def ksf(r):
    if not r or "ks" not in r[0]:
        return None
    return sum(1 for x in r if x["ks"]["K"] > 1)


names = {}
for st, n, sc, md, fl in spec:
    b = re.sub(r"_n\d+$", "", n)
    b = re.sub(r"^z", "", b) if md == "z" else b
    if md in ("n", "nd") and re.fullmatch(r"[LW](_nd\d+)?", n.rsplit("_n", 1)[0] if re.search(r"_n\d+$", n) else n):
        b = n[0] + "_off"                                                             # s1 이름 (L_n1·L_nd1) = 기본 판
    names.setdefault((sc, b), dict(fl=fl.strip(), d=None, n=[], z=[]))
    if md == "d":
        names[(sc, b)]["d"] = n
    elif md in ("n", "nd"):
        names[(sc, b)]["n"].append(n)
    else:
        names[(sc, b)]["z"] += [n + "_a", n + "_b"]

refL = rd("L_off") or rd("L_off2")
refW = rd("W_off") or rd("W_off2")
for sc, title in (("L", "레고 (80 프레임; 정답 질량중심 z f10 · f31 · f43 = 940 · 832 · 692 mm)"), ("W", "wolf (50 프레임; 정답 질량중심 z f30 · f50 = 572.6 · 533.3 mm)")):
    rows = [(b, v) for (s, b), v in names.items() if s == sc]
    if not rows:
        continue
    print(f"## {title}\n")
    if sc == "L":
        print("| 판 | 플래그 | 결정적 오차 % | 덜 풀린 걸음 | off 와 갈림 (프레임) | 질량중심 z f10 · f31 · f43 mm | 쪼갠 프레임 | 비결정 오차 % 평균 [최소~최대] (n, 발산) | 단독 ms a · b (평균/최대) |")
    else:
        print("| 판 | 플래그 | 결정적 오차 % | 덜 풀린 걸음 | off 와 갈림 (프레임) | 질량중심 z f30 · f50 mm | 쪼갠 프레임 | 비결정 오차 % 평균 [최소~최대] (n, 발산) | 단독 ms a · b (평균/최대) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for b, v in rows:
        r = rd(v["d"]) if v["d"] else None
        e, dv = err(r, summ(v["d"]), v["d"]) if v["d"] else (None, None)
        ed = ("발산 (메모리 폭증)" if dv == "메모리" else "-") if e is None else (f"{e:.2f}" + (f" (발산 f{dv})" if dv else ""))
        if sc == "L":
            cz = " · ".join(fmt(com(r, t)[0], 0) for t in (9, 30, 42))
        else:
            cz = " · ".join(fmt(com(r, t)[0], 1) for t in (29, 49))
        ne = [(err(None, summ(n), n)) for n in v["n"]]
        ok = [x for x, d in ne if x is not None and not d and math.isfinite(x)]
        ndv = [d for x, d in ne if d]
        nd = "-" if not ne else (f"{sum(ok) / len(ok):.2f} [{min(ok):.2f}~{max(ok):.2f}] ({len(ne)}, {len(ndv)}{(' ' + ','.join(('f' + str(q)) if q != '메모리' else '메모리 폭증' for q in ndv)) if ndv else ''})" if ok else f"- ({len(ne)}, {len(ndv)})")
        zz = []
        for z in v["z"]:
            zr = rd(z); zs = summ(z)
            if zs is None:
                zz.append("발산 (메모리 폭증)" if oom(z) else "-"); continue
            a_, m_ = ms(zr)
            zz.append(f"{fmt(a_)}/{fmt(m_)}" + (f" (발산 f{zs.get('diverged')})" if zs.get("diverged") else ""))
        print(f"| {b} | `{v['fl'] or '-'}` | {ed} | {fmt(nuf(r), 0) if r else '-'} | {fdiff(r, refL if sc == 'L' else refW) if r else '-'} | {cz} | {ksf(r) if r and ksf(r) is not None else '-'} | {nd} | {' · '.join(zz) if zz else '-'} |")
    print()

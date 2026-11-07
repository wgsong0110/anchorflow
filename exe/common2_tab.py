"""절 CM35 묶음 common2 표: python exe/common2_tab.py <결과 디렉토리> -> 표 (stdout, markdown). 판 목록 = exe/common2_spec.txt (모든 단계).

레고·wolf: 결정적 판 X (d), 비결정 X_n<i> (nd), 단독 시간 zX_a·zX_b (z). 오차 = 끝난 프레임 평균 %, 시간 = 첫 2 프레임 뺀 재생 GPU ms 평균/최대.
수박: 판마다 비결정 a (진단 --enrec, 결과 불변) · b (--erec 만). 손상 = 위치 유한 입자 중 logJp > 0 몫 (%, 마지막 프레임; 정답 6.51 %),
생긴 E = 전 구간 최대 (운동 + 중력 위치 + SNH 탄성) − 처음 7.13 J. 단독 시간 = zX_a·zX_b (--erec 만, 카드 하나에 차례로).
"""
import json
import math
import os
import re
import sys

O = sys.argv[1]
spec = [l.split(None, 4) for l in open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "common2_spec.txt")) if l.strip() and not l.startswith("#")]
spec = [s + [""] * (5 - len(s)) for s in spec]
E0 = 7.13


def rd(n):
    p = os.path.join(O, n, "fem.jsonl")
    if not os.path.exists(p):
        return None
    return [json.loads(l) for l in open(p) if l.strip()]


def summ(n):
    p = os.path.join(O, n, "summary.json")
    try:
        return json.load(open(p))
    except Exception:
        return None


def oom(n):
    p = os.path.join(O, n, "log.txt")
    return os.path.exists(p) and "OutOfMemoryError" in open(p, errors="replace").read()


def err(n):
    s = summ(n)
    if s is None:
        return (None, "메모리") if oom(n) else (None, "-" if not os.path.exists(os.path.join(O, n)) else "도는 중")
    e = s.get("err_mean_done", s.get("err_mean"))
    return (e * 100 if e is not None else None), s.get("diverged")


def ms(r):
    v = [x["ms"] for x in (r or []) if x["t"] >= 2 and x.get("ms") is not None and math.isfinite(x["ms"])]
    return (sum(v) / len(v), max(v)) if v else (None, None)


def fmt(x, d=2):
    return "-" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{d}f}"


def nuf(r):
    return sum(sum(x["uf"]["fail"]) for x in r if "uf" in x) if r else None


def com(r, t):
    if not r or t >= len(r) or "hp" not in r[t]:
        return None
    return r[t]["hp"][0] * 1000


def flags(f):
    return f.replace("--y0loc 3 --ue be --uf_res", "P").replace("--newton 6 --xit_from 2 --en_mo", "X")


def zcol(n, sc):
    zz = []
    for r in "ab":
        z = f"z{n}_{r}"
        zr = rd(z); zs = summ(z)
        if zs is None:
            zz.append("발산 (메모리 폭증)" if oom(z) else "-"); continue
        a_, m_ = ms(zr)
        zz.append(f"{fmt(a_)}/{fmt(m_)}" + (f" (발산 f{zs.get('diverged')})" if zs.get("diverged") else ""))
    return " · ".join(zz)


groups = {}
for st, n, sc, md, fl in spec:
    if st == "s0":
        continue
    b = re.sub(r"_n\d+$", "", n) if md == "nd" else n
    if md in ("a", "b"):
        b = n[:-2]
    b = b[1:] if md == "z" else b
    g = groups.setdefault((sc, b), dict(fl=re.sub(r"\s*--frames \d+", "", fl.strip()), d=None, n=[], a=None, b=None))
    if md == "d":
        g["d"] = n
    elif md in ("nd", "n"):
        g["n"].append(n)
    elif md in ("a", "b"):
        g[md] = n

print("# 절 CM35 묶음 common2 -- 공통 후보 B\n")
print(__doc__.split("\n", 1)[1])
print("플래그 줄임: P = `--y0loc 3 --ue be --uf_res`, X = `--newton 6 --xit_from 2 --en_mo` (뉴턴 상한: 반복 2 시작에서 덜 풀림 판정이 아니면 2 번에서 멈춤, 덜 풀린 걸음만 6 번까지; 새 실행기 --en_mo = 선탐색이 이미 함).\n")
for sc, title, ts in (("L", "레고 (80 프레임; 정답 질량중심 z f10 · f31 · f43 = 940 · 832 · 692 mm; 기본 판 결정적 5.45 %, A2k 5.67 %)", (9, 30, 42)),
                      ("W", "wolf (50 프레임; 정답 질량중심 z f30 · f50 = 572.6 · 533.3 mm; 무너짐 기준 f50 ≤ 650 mm)", (29, 49))):
    rows = [(b, v) for (s, b), v in groups.items() if s == sc]
    if not rows:
        continue
    print(f"## {title}\n")
    print("| 판 | 플래그 | 결정적 오차 % | 덜 풀린 걸음 | 질량중심 z mm | 뉴턴 평균 | 비결정 오차 % 평균 [최소~최대] (n, 발산) | 단독 ms a · b (평균/최대) |")
    print("|---|---|---|---|---|---|---|---|")
    for b, v in rows:
        r = rd(v["d"]) if v["d"] else None
        e, dv = err(v["d"]) if v["d"] else (None, None)
        ed = (("발산 (메모리 폭증)" if dv == "메모리" else (dv or "-")) if e is None else (f"{e:.2f}" + (f" (발산 f{dv})" if dv else "")))
        cz = " · ".join(fmt(com(r, t), 1 if sc == "W" else 0) for t in ts)
        nN = [((x.get("it") or {}).get("nN")) for x in (r or [])]
        nN = [x for x in nN if x is not None]
        ne = [err(n) for n in v["n"]]
        ok = [x for x, d in ne if x is not None and not d]
        ndv = [d for x, d in ne if d]
        nd = "-" if not ne else (f"{sum(ok) / len(ok):.2f} [{min(ok):.2f}~{max(ok):.2f}] ({len(ne)}, {len(ndv)}{(' ' + ','.join(('f' + str(q)) if isinstance(q, int) else str(q) for q in ndv)) if ndv else ''})" if ok else f"- ({len(ne)}, {len(ndv)} {ndv})")
        print(f"| {b} | `{flags(v['fl']) or '-'}` | {ed} | {fmt(nuf(r), 0) if r else '-'} | {cz} | {fmt(sum(nN) / len(nN)) if nN else '-'} | {nd} | {zcol(b, sc)} |")
    print()


def E(q):
    e = (q or {}).get("en") or {}
    return None if e.get("Eel") is None else e["KE"] + e["PE"] + e["Eel"]


def emax(rows):
    v = [E(q) for q in rows or [] if E(q) is not None and math.isfinite(E(q))]
    return max(v) if v else None


def lj_last(rows):
    for q in reversed(rows or []):
        v = (q.get("en") or {}).get("lj0")
        if v is not None and q.get("err") is not None and math.isfinite(q["err"]):
            return v
    return None


def res(n):
    s = summ(n); r = rd(n)
    if s is None:
        return ("발산 (메모리)" if oom(n) else ("도는 중" if r else "없음"))
    return ("발산 f%d" % s["diverged"]) if s.get("diverged") else ("완주 %d" % s["frames_done"])


print("## 수박 (옛 실행기, 절 CM33 (2) 판 바탕 + 판 플래그; 비결정 a · b)\n")
print("| 셀 | 판 | 플래그 | 결과 a · b | 오차 % a · b | 손상 % a · b | 생긴 E J a · b | 덜 풀린 걸음 a · b | 뉴턴 평균 a | ms 평균 · 최대 (b, 다른 카드 동시) | 단독 ms a · b (평균/최대) | 단독 손상 % · 생긴 E a · b |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for (sc, b), v in groups.items():
    if not sc.startswith("M"):
        continue
    R = {r: (v[r], rd(v[r]) if v[r] else None) for r in "ab"}
    c = [" · ".join(res(R[r][0]) if R[r][0] else "-" for r in "ab"),
         " · ".join(fmt(100 * summ(R[r][0])["err_mean_done"]) if R[r][0] and summ(R[r][0]) else "-" for r in "ab"),
         " · ".join(fmt(100 * lj_last(R[r][1]) if lj_last(R[r][1]) is not None else None) for r in "ab"),
         " · ".join(fmt(emax(R[r][1]) - E0 if emax(R[r][1]) is not None else None) for r in "ab"),
         " · ".join(str(nuf(R[r][1])) if R[r][1] else "-" for r in "ab")]
    nN = [((x.get("it") or {}).get("nN")) for x in (R["a"][1] or [])]
    nN = [x for x in nN if x is not None]
    c.append(fmt(sum(nN) / len(nN)) if nN else "-")
    m_, x_ = ms(R["b"][1])
    c.append(f"{fmt(m_, 1)} · {fmt(x_, 0)}")
    c.append(zcol(b, sc))
    zz = []
    for r in "ab":
        zr = rd(f"z{b}_{r}")
        zz.append(f"{fmt(100 * lj_last(zr) if lj_last(zr) is not None else None)} · {fmt(emax(zr) - E0 if emax(zr) is not None else None)}" if zr else "-")
    c.append(" / ".join(zz))
    print(f"| {sc[1:].replace('15', '1.5')} | {b.split('_', 1)[1]} | `{flags(v['fl'])}` | " + " | ".join(c) + " |")
print("| - | 정답 | - | - | - | 6.51 | - | - | - | - | - | - |\n")

# 진단: S 나눔 (a 판 --enrec)
SEG = [(4, 4), (5, 10), (11, 30), (31, 100), (1, 100)]
print("## 진단: 덜 풀린 걸음 걸음 끝 갱신 노드 나눔 (수박 a 판 --enrec; S = 후퇴 오일러 노드 집합, S 밖 = HHT 갱신; 자른 = 이 걸음 노드별 국소 보폭 k > 0)\n")
print("덜 풀린 걸음만 모음. ρ = Σ m v1·v0 / Σ m |v0|² (1 = 속도 그대로, < 0 = 뒤집힘), 뒤집힘 = v1·v0 < 0 인 질량 몫, ΔKE_HHT−BE = 실제 KE1 − 후퇴 오일러였다면의 KE (½ Σ m |y/dt|²), 갱신식 R_upd·잔차 W_res (J, 구간 합).\n")
print("| 셀 | 판 | 구간 | 덜 풀린 걸음 | S 질량 % | S 밖 자른 질량 % | ρ S · S밖자른 · S밖안자른 | 뒤집힘 % S밖자른 · S밖안자른 | ΔKE_HHT−BE S밖 J | R_upd S · S밖 J | W_res S · S밖 J | 입자 ΔE J |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for (sc, b), v in groups.items():
    if not sc.startswith("M") or not v["a"]:
        continue
    rows = rd(v["a"]) or []
    if not any("S2" in ((q.get("enb") or {}).get("split") or {}) for q in rows):
        continue
    for a_, b_ in SEG:
        Q = [q for q in rows if a_ <= q["t"] + 1 <= b_ and (q.get("enb") or {}).get("uf") and "S2" in ((q["enb"].get("split")) or {})]
        Qa = [q for q in rows if a_ <= q["t"] + 1 <= b_ and (q.get("enb") or {}).get("dEp") is not None]
        if not Q:
            print(f"| {sc[1:].replace('15', '1.5')} | {b.split('_', 1)[1]} | f{a_}~{b_} | 0 | - | - | - | - | - | - | - | {fmt(sum(q['enb']['dEp'] for q in Qa), 2) if Qa else '-'} |")
            continue
        S2 = [q["enb"]["split"]["S2"] for q in Q]
        sm = lambda k, f: sum(x[k][f] for x in S2)
        mt = sum(x["S"]["m"] + x["nS"]["m"] for x in S2)
        rho = lambda k: sum(x[k]["rho"] * max(x[k]["KE0"], 0) for x in S2) / max(sum(x[k]["KE0"] for x in S2), 1e-30)
        flp = lambda k: 100 * sum(x[k]["mflip"] * x[k]["m"] for x in S2) / max(sum(x[k]["m"] for x in S2), 1e-30)
        print(f"| {sc[1:].replace('15', '1.5')} | {b.split('_', 1)[1]} | f{a_}~{b_} | {len(Q)} | {fmt(100 * sm('S', 'm') / max(mt, 1e-30), 1)} | {fmt(100 * sm('nS_cut', 'm') / max(mt, 1e-30), 1)} | "
              f"{fmt(rho('S'))} · {fmt(rho('nS_cut'))} · {fmt(rho('nS_unc'))} | {fmt(flp('nS_cut'), 1)} · {fmt(flp('nS_unc'), 1)} | {fmt(sm('nS', 'KE1') - sm('nS', 'KEbe'), 3)} | "
              f"{fmt(sm('S', 'Rupd'), 3)} · {fmt(sm('nS', 'Rupd'), 3)} | {fmt(sm('S', 'Wres'), 3)} · {fmt(sm('nS', 'Wres'), 3)} | {fmt(sum(q['enb']['dEp'] for q in Qa), 2)} |")

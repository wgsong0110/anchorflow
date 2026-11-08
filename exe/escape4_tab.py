"""절 CM49 묶음 escape4 표: python exe/escape4_tab.py <결과 디렉토리> -> markdown (stdout). 판 목록 = exe/escape4_spec.txt.
@e3w 1 = 절 CM45 W̄, @e5r 1 = R (걸음 끝 장벽 기준 높이 이어 받기), @e5a 1 = A (걸음 시작 기준 높이 ≤ 0 표본점 상대 장벽, 기준 d̂). E·V2NF 정의는 spec 머리말.
오차 = 끝난 프레임 평균 %, 시간 = 첫 2 프레임 뺀 재생 GPU ms (쪼갠·다시 한 프레임은 모든 시도 합), 60 fps 문턱 16.7 ms.
이탈 = 몸 상자 (0.1~99.9 백분위) 4 칸 밖 입자 (e1.jsonl). 바닥 뚫음 판 = 진단 D2 (이탈 입자 중 바닥 밑 한 칸 넘는 입자 > 0). z 최소 = D5 전 입자 (바닥 z 0.02).
수박 손상 = logJp > 0 몫 (fem.jsonl en.lj0; 정답 f5 4.41 · f6 6.94 · f8 7.08 · 끝 6.51 %). 발산 = 판정 + 단독 (메모리 넘침 포함).
"""
import collections
import json
import math
import os
import sys

O = sys.argv[1]
spec = [l.split(None, 4) for l in open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "escape4_spec.txt")) if l.strip() and not l.startswith("#")]
spec = [s + [""] * (5 - len(s)) for s in spec]
LIM = 1000.0 / 60.0
SCN = {"L": "레고", "W": "wolf", "F": "ficus", "B": "bread", "M3": "수박"}
ZF = 0.02


def jl(n, f):
    p = os.path.join(O, n, f)
    try:
        return [json.loads(l) for l in open(p) if l.strip()] if os.path.exists(p) else None
    except Exception:
        return None


def rd(n):
    return jl(n, "fem.jsonl")


def summ(n):
    try:
        return json.load(open(os.path.join(O, n, "summary.json")))
    except Exception:
        return None


def oom(n):
    p = os.path.join(O, n, "log.txt")
    return os.path.exists(p) and "OutOfMemoryError" in open(p, errors="replace").read()


def fmt(x, d=2):
    return "-" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{d}f}"


def res(n):
    s = summ(n)
    if s is None:
        return "발산 (메모리)" if oom(n) else ("도는 중/실패" if rd(n) else "없음")
    return ("발산 f%d" % s["diverged"]) if s.get("diverged") else ("완주 %d" % s["frames_done"])


def div(n):
    return res(n).startswith("발산")


def err(n):
    s = summ(n)
    if s is None:
        return None
    e = s.get("err_mean_done", s.get("err_mean"))
    return 100 * e if e is not None else None


def msl(r):
    return [(x["t"], x["ms"]) for x in (r or []) if x["t"] >= 2 and x.get("ms") is not None and math.isfinite(x["ms"])]


def tm(r):
    m = msl(r)
    if not m:
        return "-"
    q = [b for _, b in m]
    t, mx = max(m, key=lambda z: z[1])
    return f"{sum(q) / len(q):.2f} / {mx:.1f} (f{t + 1}) / {sum(1 for b in q if b > LIM)}"


def com(r, t):
    return r[t]["hp"][0] * 1000 if r and t < len(r) and "hp" in r[t] else None


def lj_at(r, t):
    if not r or t >= len(r):
        return None
    v = (r[t].get("en") or {}).get("lj0")
    return None if v is None or r[t].get("err") is None or not math.isfinite(r[t]["err"]) else 100 * v


def lj_last(r):
    for q in reversed(r or []):
        v = (q.get("en") or {}).get("lj0")
        if v is not None and q.get("err") is not None and math.isfinite(q["err"]):
            return 100 * v
    return None


def e1s(n):
    e = jl(n, "e1.jsonl") or []
    if not e:
        return None
    return dict(nfar=max(q.get("nfar", 0) for q in e), dmax=max((q.get("dmax") or 0) for q in e))


def e2below(n):
    L = [q for q in (jl(n, "e2.jsonl") or []) if "esc" not in q and "nbelow" in q]
    return max((q["nfar_below"] for q in L), default=None) if L else None


def e5(n):
    return [q for q in (jl(n, "e5.jsonl") or []) if "nq" in q]


def bbox(n):
    p = os.path.join(O, n, "bbox.txt")
    if os.path.exists(p):
        for l in open(p):
            if "pred.pt" in l:
                try:
                    d_ = json.loads(l.split(" ", 1)[1])
                    return max(z["nfar"] for z in d_.values()), max(z["dmax"] for z in d_.values())
                except Exception:
                    pass
    return None


def period(n):
    p = os.path.join(O, n, "period.txt")
    if not os.path.exists(p):
        return "-"
    t_ = open(p).read().strip().split("\n")[-1]
    return t_.split(":", 1)[1].strip() if ":" in t_ else t_


def rng(v, d=2):
    v = [q for q in v if q is not None]
    return f"{min(v):.{d}f}~{max(v):.{d}f}" if v else "-"


def mean(v):
    v = [q for q in v if q is not None]
    return sum(v) / len(v) if v else None


print("# 절 CM49 묶음 escape4 -- 장벽 표본점 걸음 끝 기준 높이 이어 받기 R · 상대 장벽 A (수박 E·V2NF + W̄) + 다른 씬 회귀\n")
print(__doc__.split("\n", 1)[1])

# ---- 판별 (판정·비결정)
for stg in sorted(set(s[0] for s in spec), key=lambda z: (len(z), z)):
    rows = [s for s in spec if s[0] == stg and s[3] not in ("z", "p")]
    if not rows or not any(rd(n) is not None or oom(n) for _, n, *_r in rows):
        continue
    print(f"## {stg} 판별 (시간은 다른 카드 동시라 참고)\n")
    print("| 판 | 씬 | 플래그 | 결과 | 오차 % | 손상 f5 · f6 · f8 · 끝 % | wolf f50 mm · ficus 주기 | 이탈 최대 수 · 거리 m | 바닥 뚫음 (D2) · z 최소 m | 노드 상자 최대 | ms 평균 / 최대 / >16.7 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for st, n, sc, md, fl in rows:
        r = rd(n)
        if r is None and not oom(n):
            continue
        e = e1s(n)
        E5 = e5(n)
        zm = min((q["zmin"] for q in E5 if q.get("zmin") is not None), default=None)
        dmg = " · ".join(fmt(lj_at(r, t)) for t in (4, 5, 7)) + " · " + fmt(lj_last(r)) if sc.startswith("M") else "-"
        met = f"{fmt(com(r, 49), 0)}" if sc == "W" else (period(n) if sc == "F" else "-")
        bl = e2below(n)
        print(f"| {n} | {SCN.get(sc, sc)} | `{fl.strip()}` | {res(n)} | {fmt(err(n))} | {dmg} | {met} | {(str(e['nfar']) + ' · ' + fmt(e['dmax'])) if e else '-'} | "
              f"{'-' if bl is None else bl} · {fmt(zm + ZF if zm is not None else None, 3)} | {(max(x.get('nn', 0) for x in r) / 1e6) if r else 0:.2f} 백만 | {tm(r)} |")
    print()

# ---- 단독 시간
zr = [s for s in spec if s[3] == "z"]
if any(rd(f"{n}_a") or rd(f"{n}_b") or oom(f"{n}_a") for _, n, *_r in zr):
    print("## 단독 시간 (계산 프로세스 0 인 카드 하나에 차례로 a · b; 진단 끔)\n")
    print("| 판 | 씬 | 플래그 | 결과 a · b | 오차 % a · b | 평균 / 최대 (프레임) / >16.7 ms a | b |")
    print("|---|---|---|---|---|---|---|")
    for st, n, sc, md, fl in zr:
        ra, rb = rd(f"{n}_a"), rd(f"{n}_b")
        if ra is None and rb is None and not oom(n + "_a"):
            continue
        print(f"| {n} | {SCN.get(sc, sc)} | `{fl.strip()}` | {res(n + '_a')} · {res(n + '_b')} | {fmt(err(n + '_a'))} · {fmt(err(n + '_b'))} | {tm(ra)} | {tm(rb)} |")
    print()


# ---- 수박 후보별 (절 CM46 표 형식)
def zrow(fl, sc):
    for st, n, s_, md, f_ in zr:
        if s_ == sc and f_.strip() == fl:
            return n
    return None


G = collections.OrderedDict()
for st, n, sc, md, fl in spec:
    if md == "g" and not st.endswith("k"):
        G.setdefault(fl.strip(), []).append(n)
if G:
    print("## 수박 후보별 (절 CM46 표 형식; 비결정, 진단 D1·D2·D5 켬; 정답 손상 f5 4.41 · f6 6.94 · f8 7.08 · 끝 6.51 %; 발산 = 판정 + 단독)\n")
    print("| 후보 | 판 | 발산 (판정 + 단독) | 바닥 뚫음 판 (D2) · z 최소 m | 이탈 판 (> 20 개) · 최대 수 · 거리 m | 노드 상자 최대 (> 5 백만 판) | 끝 손상 % [범위] | f5 · f6 · f8 손상 % (평균) | 오차 % | 단독 ms 평균/최대 a · b |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for fl, ns in G.items():
        rs = [(n, rd(n)) for n in ns if rd(n) is not None or oom(n)]
        if not rs:
            continue
        dv = sum(1 for n, _ in rs if div(n))
        zn = zrow(fl, "M3")
        zd, zt = 0, 0
        zs = []
        if zn:
            for k_ in ("a", "b"):
                nz = f"{zn}_{k_}"
                if rd(nz) is not None or oom(nz):
                    zt += 1
                    zd += int(div(nz))
                    m = msl(rd(nz))
                    zs.append("메모리 발산" if oom(nz) else (res(nz) if div(nz) else (f"{sum(b for _, b in m) / len(m):.1f}/{max(b for _, b in m):.0f}" if m else "-")))
        bl = [e2below(n) for n, _ in rs]
        bl = [b for b in bl if b is not None]
        zm = [min((q["zmin"] for q in e5(n) if q.get("zmin") is not None), default=None) for n, _ in rs]
        zm = [z for z in zm if z is not None]
        es = [e1s(n) for n, _ in rs]
        es = [e for e in es if e]
        far = [e["nfar"] for e in es]
        box = [max(x.get("nn", 0) for x in r) for _, r in rs if r]
        ljl = [lj_last(r) for _, r in rs if r]
        f5 = mean([lj_at(r, 4) for _, r in rs if r])
        f6 = mean([lj_at(r, 5) for _, r in rs if r])
        f8 = mean([lj_at(r, 7) for _, r in rs if r])
        E_ = [err(n) for n, _ in rs if err(n) is not None and not div(n)]
        lv = [q for q in ljl if q is not None]
        print(f"| `{fl}` | {len(rs)} | {dv}/{len(rs)} + {zd}/{zt} | {sum(1 for b in bl if b > 0)}/{len(bl)} · {fmt(min(zm) + ZF if zm else None, 3)} | "
              f"{(str(sum(1 for q in far if q > 20)) + '/' + str(len(far)) + ' · ' + str(max(far)) + ' · ' + fmt(max(e['dmax'] for e in es))) if far else '-'} | "
              f"{(max(box) / 1e6) if box else 0:.2f} 백만 ({sum(1 for b in box if b > 5e6)}/{len(box)}) | {fmt(mean(lv))} [{rng(lv)}] | {fmt(f5)} · {fmt(f6)} · {fmt(f8)} | {rng(E_)} | {' · '.join(zs) or '-'} |")
    print()

# ---- 진단 D5
out = []
for st, n, sc, md, fl in spec:
    E5 = e5(n)
    if not E5:
        continue
    sm = lambda k: sum(q.get(k) or 0 for q in E5)
    ca = collections.Counter()
    cc = collections.Counter()
    for q in E5:
        for k_, v in ((q.get("cat") or {}).get("act") or {}).items():
            ca[k_] += v
        for k_, v in ((q.get("cat") or {}).get("car") or {}).items():
            cc[k_] += v
    kc = [q.get("kcnt") for q in E5 if q.get("kcnt")]
    fo = [q.get("floor_oct_off") for q in E5 if q.get("floor_oct_off") is not None]
    bem = max((q["bar_end_max"] for q in E5 if q.get("bar_end_max") is not None), default=None)
    dmx = max((q["dmax"] for q in E5 if q.get("dmax") is not None), default=None)
    dsm = max((abs(q["dsum"]) for q in E5 if q.get("dsum") is not None), default=None)
    out.append(f"| {n} | `{fl.strip()}` | {res(n)} | {sm('nv_act')} ({ca['regroup']} · {ca['inherit']} · {ca['mismatch']}) | {sm('nv_car')} ({cc['regroup']} · {cc['inherit']} · {cc['mismatch']}) | "
               f"{sum(k[0] for k in kc)} · {sum(k[1] for k in kc)} | {sm('act_end_le0')} · {sm('bar_pos_act_le0')} · {'-' if bem is None else '%.1e' % bem} | {max((q.get('inact') or 0) for q in E5)} | "
               f"{'-' if dmx is None else '%.4f' % dmx} · {'-' if dsm is None else '%.1e' % dsm} | {fo[0] if fo else '-'} | {max((q.get('nbelow') or 0) for q in E5)} |")
if out:
    print("## 진단 D5: 걸음 시작 바닥 밑 표본점 가르기 (전 프레임 합)\n")
    print("실제 = 이 프레임 칸 입자 질량 평균 시작 높이 d_q ≤ 0 인 표본점 (원래·W̄ 판이 장벽에서 빼는 점), 이어 받음 = 앞 프레임 끝 기준 높이 h_p 의 질량 평균 ≤ 0 (R 판이 A 없이는 빼는 점). "
          "괄호 = (다시 모음 · 이어짐 · 끝 어긋남): 다시 모음 = 칸 입자 모두 앞 프레임 활성·끝 높이 > 0 표본점에서 옴, 이어짐 = 앞 프레임 비활성 표본점 소속 입자 있음, 끝 어긋남 = 앞 표본점 활성인데 끝 높이 ≤ 0. "
          "커널 셈 = e5_sh 의 d̃ ≤ 0 수 · A 로 옮긴 수. 끝 높이 ≤ 0 활성 점 · 그중 장벽이 본 끝 (x̄ + Σ W̄ thinv y) > 0 · 둘 차 최대 m (K 1 프레임). 바닥-팔분 경계 어긋남 = 바닥에서 가장 가까운 팔분 z 경계까지 (팔분 높이 단위, 0 = 겹침).\n")
    print("| 판 | 플래그 | 결과 | 실제 d_q ≤ 0 표본점 (모음 · 이어짐 · 어긋남) | 이어 받음 ≤ 0 (모음 · 이어짐 · 어긋남) | 커널 셈 d̃ ≤ 0 · 옮김 | 끝 ≤ 0 활성 · 장벽 > 0 · 차 | 비활성 최대 | 최대 |δ| m · |Σ m δ / Σ m| | 바닥-팔분 경계 | 바닥 밑 입자 최대 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    print("\n".join(out))
    print()

# ---- 다른 씬 회귀 (E 대비)
RG = collections.OrderedDict()
for st, n, sc, md, fl in spec:
    if sc in ("L", "W", "F", "B") and md == "nd" and not st.endswith("k"):
        RG.setdefault((sc, fl.strip()), []).append(n)
if RG:
    print("## 다른 씬 회귀 (E 대비; 비결정 판 + 단독 a · b)\n")
    print("| 씬 | 플래그 | 판 | 발산 | 오차 % 평균 [범위] | 씬 지표 | 단독 평균 / 최악 ms a · b |")
    print("|---|---|---|---|---|---|---|")
    for (sc, fl), ns in RG.items():
        rs = [(n, rd(n)) for n in ns if rd(n) is not None or oom(n)]
        if not rs:
            continue
        dv = sum(1 for n, _ in rs if div(n))
        E_ = [err(n) for n, _ in rs if err(n) is not None and not div(n)]
        met = "-"
        if sc == "W":
            f50 = [com(r, 49) for _, r in rs if r and com(r, 49) is not None]
            bb = [bbox(n) for n, _ in rs]
            bb = [b for b in bb if b]
            met = (f"f50 {min(f50):.0f}~{max(f50):.0f} mm" if f50 else "f50 -") + (f", 이탈 판 {sum(1 for b in bb if b[0] > 20)}/{len(bb)} (최대 {max(b[0] for b in bb)} 개·{max(b[1] for b in bb):.2f} m)" if bb else "")
        elif sc == "F":
            met = "; ".join(sorted(set(period(n) for n, _ in rs)))
        zn = zrow(fl, sc)
        zs = []
        if zn:
            for k_ in ("a", "b"):
                nz = f"{zn}_{k_}"
                m = msl(rd(nz))
                zs.append(("발산" if div(nz) else "-") if not m else f"{sum(b for _, b in m) / len(m):.2f} / {max(b for _, b in m):.1f}")
        print(f"| {SCN[sc]} | `{fl}` | {len(rs)} | {dv} | {fmt(mean(E_))} [{rng(E_)}] | {met} | {' · '.join(zs) or '-'} |")
    print()

"""절 CM52 묶음 wolfh1 표: 판별 결과 (오차·발산·f50 높이·이탈·시간) + 무리별 요약 + 정답 대 무리 평균 프레임별 곡선 (exe/wolfh1_an.py 출력) + 단독 시간.

사용: python exe/wolfh1_tab.py <출력 디렉토리>   (판 목록 exe/wolfh1_spec.txt)
"""
import glob
import json
import math
import os
import re
import sys

O = sys.argv[1]
HERE = os.path.dirname(os.path.abspath(__file__))
LIM = 1000.0 / 60.0
SPEC = {}
for ln in open(os.path.join(HERE, "wolfh1_spec.txt")):
    if ln.startswith("#") or not ln.strip():
        continue
    t = ln.split()
    SPEC[t[1]] = dict(stg=t[0], sc=t[2], md=t[3], fl=" ".join(t[4:]))


def summ(n):
    p = os.path.join(O, n, "summary.json")
    try:
        return json.load(open(p))
    except Exception:
        return None


def rec(n):
    p = os.path.join(O, n, "fem.jsonl")
    if not os.path.isfile(p):
        return None
    L = []
    for l_ in open(p):
        try:
            q = json.loads(l_)
        except Exception:
            continue
        if "t" in q:
            L.append(q)
    return L


def err(s):
    if s is None:
        return None
    e = s.get("err_mean_done", s.get("err_mean"))
    return 100 * e if e is not None else None


def tm(r):
    m = [(x["t"], x["ms"]) for x in (r or []) if x["t"] >= 2 and x.get("ms") is not None and math.isfinite(x["ms"])]
    if not m:
        return None
    q = [b for _, b in m]
    return sum(q) / len(q), max(q), sum(1 for b in q if b > LIM)


def com(r, t):
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


def res(n, s, r):
    if s is None:
        return "없음/실패"
    if s.get("diverged") or s.get("div_at") or s.get("oom"):
        return f"발산 f{s.get('div_at') or s.get('diverged')}" + (" (메모리)" if s.get("oom") else "")
    return f"완주 {s.get('frames_done')}"


AN = {}
for p in sorted(glob.glob(os.path.join(O, "wolfh1_an_*.json"))):
    try:
        J = json.load(open(p))
    except Exception:
        continue
    AN.setdefault("gt", J["gt"])
    AN.update({k: v for k, v in J["runs"].items()})
GT = AN.get("gt")


def an(n, f, k):
    r = AN.get(n)
    if not r:
        return None
    s = r["series"].get(str(f)) or r["series"].get(f)
    return None if s is None else s.get(k)


def fm(v, d=1):
    return "-" if v is None else f"{v:.{d}f}"


print("# 절 CM52 묶음 wolfh1 -- wolf 가 정답보다 높게 쌓이는 원인·고침 (공통 후보 F = E + W̄ + R2 + A2)\n")
print("정의: f50 = 50 프레임 질량중심 높이 (바닥 법선, mm; 정답 533). 갈라짐 = |질량중심 우리 − 정답| 5·10·20 mm 처음 넘는 프레임. 면적 = 바닥 1 cm 안 입자 5 mm 칸 점유 넓이 (cm²). "
      "남는 몫 = 위치 에너지 감소 − 운동 − 탄성 (J). 이탈 = 몸 상자 (0.1~99.9 백분위) 4 칸 밖 입자 최대 수·거리. 시간 = 재생 GPU ms (첫 2 프레임 뺌), 동시 판이라 참고 (단독은 아래).\n")
if GT:
    g = GT
    print(f"정답: f30 질량중심 {g['30']['com']:.1f} · f50 {g['50']['com']:.1f} mm, 최대 f30 {g['30']['zmax']:.0f} · f50 {g['50']['zmax']:.0f}, 면적 f30 {g['30']['area_cm2']:.0f} · f50 {g['50']['area_cm2']:.0f} cm², 회전 반경 f50 {g['50']['rg_mm']:.0f} mm\n")
STG = sorted({v["stg"] for v in SPEC.values()})
for st in STG:
    ns = [n for n, v in SPEC.items() if v["stg"] == st and v["md"] != "z" and os.path.isdir(os.path.join(O, n))]
    if not ns:
        continue
    print(f"## 단계 {st} 판별\n")
    print("| 판 | 씬 | 플래그 | 결과 | 오차 % | f30 · f50 mm | 최대 f30 · f50 | 갈라짐 5·10·20 | 면적 f30 · f50 | 회전 반경 f50 | 남는 몫 f50 J | 이탈 | ms 평균 / 최대 / >16.7 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for n in ns:
        v = SPEC[n]; s = summ(n); r = rec(n); t = tm(r); b = bbox(n)
        dv = (AN.get(n) or {}).get("diverge") or {}
        print(f"| {n} | {v['sc']} | `{v['fl']}` | {res(n, s, r)} | {fm(err(s), 2)} | {fm(com(r, 29), 0)} · {fm(com(r, 49), 0)} | {fm(an(n, 30, 'zmax'), 0)} · {fm(an(n, 50, 'zmax'), 0)} | "
              f"{dv.get('5', '-')}·{dv.get('10', '-')}·{dv.get('20', '-')} | {fm(an(n, 30, 'area_cm2'), 0)} · {fm(an(n, 50, 'area_cm2'), 0)} | {fm(an(n, 50, 'rg_mm'), 0)} | {fm(an(n, 50, 'rest'), 2)} | "
              f"{'-' if b is None else f'{b[0]}·{b[1]:.2f}'} | {'-' if t is None else f'{t[0]:.2f} / {t[1]:.1f} / {t[2]}'} |")
    print()
# 무리 요약 (이름 끝 _숫자 뗀 것)
print("## 무리별 요약 (wolf 비결정 판)\n")
print("| 무리 | 플래그 | 판 | 발산 | f50 mm [범위] | 오차 % [범위] | 이탈 판 (>20) | 동시 ms 최대 |")
print("|---|---|---|---|---|---|---|---|")
GR = {}
for n, v in SPEC.items():
    if v["sc"] != "W" or v["md"] == "z" or not os.path.isdir(os.path.join(O, n)):
        continue
    GR.setdefault(re.sub(r"_\d+$", "", n), []).append(n)
for gname, ns in GR.items():
    f50, er, nd, ne, mx = [], [], 0, 0, []
    for n in ns:
        s = summ(n); r = rec(n)
        if s is None:
            continue
        if s.get("diverged") or s.get("div_at") or s.get("oom"):
            nd += 1
        else:
            c = com(r, 49)
            if c is not None:
                f50.append(c)
            if err(s) is not None:
                er.append(err(s))
        b = bbox(n)
        if b and b[0] > 20:
            ne += 1
        t = tm(r)
        if t:
            mx.append(t[1])
    fl = SPEC[ns[0]]["fl"]
    print(f"| {gname} | `{fl}` | {len(ns)} | {nd} | {(sum(f50) / len(f50)):.0f} [{min(f50):.0f}~{max(f50):.0f}] | " if f50 else f"| {gname} | `{fl}` | {len(ns)} | {nd} | - | ", end="")
    print(f"{(sum(er) / len(er)):.2f} [{min(er):.2f}~{max(er):.2f}] | {ne} | {max(mx):.1f} |" if er else f"- | {ne} | - |")
print()
# 다른 씬
OT = [n for n, v in SPEC.items() if v["sc"] != "W" and v["md"] != "z" and os.path.isdir(os.path.join(O, n))]
if OT:
    print("## 다른 씬 회귀\n")
    print("| 판 | 씬 | 플래그 | 결과 | 오차 % | 씬 지표 | ms 평균 / 최대 / >16.7 |")
    print("|---|---|---|---|---|---|---|")
    for n in OT:
        v = SPEC[n]; s = summ(n); r = rec(n); t = tm(r)
        met = "-"
        pp = os.path.join(O, n, "period.txt")
        if os.path.isfile(pp):
            met = " ".join(open(pp).read().split()[-8:])
        if v["sc"].startswith("M") and r:
            lj = [((q.get("en") or {}).get("lj0")) for q in r]
            lj = [x for x in lj if x is not None]
            met = f"끝 손상 {100 * lj[-1]:.2f} %" if lj else "-"
        print(f"| {n} | {v['sc']} | `{v['fl']}` | {res(n, s, r)} | {fm(err(s), 2)} | {met} | {'-' if t is None else f'{t[0]:.2f} / {t[1]:.1f} / {t[2]}'} |")
    print()
# 단독
Z = [n for n, v in SPEC.items() if v["md"] == "z"]
if any(os.path.isdir(os.path.join(O, f"{n}_a")) for n in Z):
    print("## 단독 시간 (계산 프로세스 0 인 카드 하나에 a · b 차례로; 평균 / 최대 / >16.7 프레임 수)\n")
    print("| 판 | 씬 | 플래그 | a | b | 결과 a · b |")
    print("|---|---|---|---|---|---|")
    for n in Z:
        v = SPEC[n]
        cs = []
        for k in "ab":
            r = rec(f"{n}_{k}"); t = tm(r)
            cs.append("-" if t is None else f"{t[0]:.2f} / {t[1]:.1f} / {t[2]}")
        print(f"| {n} | {v['sc']} | `{v['fl']}` | {cs[0]} | {cs[1]} | {res(n, summ(n + '_a'), None)} · {res(n, summ(n + '_b'), None)} |")
    print()
# 곡선
if GT and GR:
    FR = [5, 8, 10, 12, 15, 18, 22, 26, 30, 35, 40, 45, 50]
    for key, lab, dd in (("com", "질량중심 높이 mm", 0), ("zmax", "최대 높이 mm", 0), ("area_cm2", "바닥 접촉 면적 cm²", 0), ("rg_mm", "평면 회전 반경 mm", 0),
                         ("ke", "운동 에너지 J", 2), ("el", "탄성 에너지 J", 3), ("rest", "남는 몫 (소산) J", 2)):
        if GT["50"].get(key) is None and not any(an(ns[0], 50, key) is not None for ns in GR.values()):
            continue
        print(f"## 곡선: {lab} (무리 평균)\n")
        print("| 무리 | " + " | ".join(f"f{f}" for f in FR) + " |")
        print("|---|" + "---|" * len(FR))
        print("| 정답 | " + " | ".join(fm(GT[str(f)].get(key), dd) for f in FR) + " |")
        for gname, ns in GR.items():
            row = []
            for f in FR:
                vv = [an(n, f, key) for n in ns]
                vv = [x for x in vv if x is not None]
                row.append(fm(sum(vv) / len(vv), dd) if vv else "-")
            print(f"| {gname} | " + " | ".join(row) + " |")
        print()
    print("## 곡선: 높이 층별 접선 속력 m/s (층 [0,2)·[2,4)·[4,8) cm, 무리 평균)\n")
    print("| 무리 | " + " | ".join(f"f{f}" for f in FR) + " |")
    print("|---|" + "---|" * len(FR))

    def lay(get):
        o = []
        for f in FR:
            v = get(f)
            o.append("-" if not v else "·".join("-" if x is None else f"{x:.2f}" for x in v[:3]))
        return o
    print("| 정답 | " + " | ".join(lay(lambda f: GT[str(f)].get("vt_layers"))) + " |")
    for gname, ns in GR.items():
        def gm(f, ns=ns):
            L = [an(n, f, "vt_layers") for n in ns]
            L = [x for x in L if x]
            if not L:
                return None
            return [None if any(q[i] is None for q in L) else sum(q[i] for q in L) / len(L) for i in range(3)]
        print(f"| {gname} | " + " | ".join(lay(gm)) + " |")
    print()

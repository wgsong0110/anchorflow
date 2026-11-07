"""절 CM37 묶음 common3 표: python exe/common3_tab.py <결과 디렉토리> -> 표 (stdout, markdown). 판 목록 = exe/common3_spec.txt.

BG = `--y0loc 3 --ue be --uf_res --newton 6 --xit_from 2 --en_mo --ls0_keep --ue_loc 50`, C = BG + `--ksplit 3 --ks_big --xg 1` (수박은 + `--sp2 31`, 셀 3).
오차 = 끝난 프레임 평균 %, 시간 = 첫 2 프레임 뺀 재생 GPU ms (쪼갠 프레임은 모든 시도 합). 쪼갠 프레임 = 마지막 시도 K > 1.
수박 손상 = 위치 유한 입자 중 logJp > 0 몫 (마지막 프레임, 정답 6.51 %), 생긴 E = 전 구간 최대 (운동 + 중력 위치 + SNH 탄성) − 처음 7.13 J,
튐 = f5 뒤 질량중심 법선 속도 최대 (m/s, 양수 = 위로).
"""
import json
import math
import os
import re
import sys

O = sys.argv[1]
spec = [l.split(None, 4) for l in open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "common3_spec.txt")) if l.strip() and not l.startswith("#")]
spec = [s + [""] * (5 - len(s)) for s in spec]
E0 = 7.13
LIM = 1000.0 / 60.0


def rd(n):
    p = os.path.join(O, n, "fem.jsonl")
    return [json.loads(l) for l in open(p) if l.strip()] if os.path.exists(p) else None


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
        return "발산 (메모리)" if oom(n) else ("도는 중" if rd(n) else "없음")
    return ("발산 f%d" % s["diverged"]) if s.get("diverged") else ("완주 %d" % s["frames_done"])


def err(n):
    s = summ(n)
    if s is None:
        return None, ("메모리" if oom(n) else None)
    e = s.get("err_mean_done", s.get("err_mean"))
    return (100 * e if e is not None else None), s.get("diverged")


def msl(r):
    return [(x["t"], x["ms"]) for x in (r or []) if x["t"] >= 2 and x.get("ms") is not None and math.isfinite(x["ms"])]


def splits(r):
    return [(x["t"] + 1, x["ks"]["K"], round(sum(q[2] for q in x["ks"]["tries"]), 1)) for x in (r or []) if x.get("ks") and x["ks"]["K"] > 1]


def g2max(r):
    v = [g for x in (r or []) if x.get("uf") and x["uf"].get("g2") for f, g in zip(x["uf"]["fail"], x["uf"]["g2"]) if f]
    return max(v) if v else None


def com(r, t):
    return r[t]["hp"][0] * 1000 if r and t < len(r) and "hp" in r[t] else None


def hsh(r):
    return [x.get("xh") for x in (r or [])]


print("# 절 CM37 묶음 common3 -- 공통 후보 C\n")
print(__doc__.split("\n", 1)[1])

# ---- 결정적 판: 해시 대조 (BG 같은 코드 · C · C --xg 0 · C --xg 2)
print("## 결정적 판 (레고·wolf·ficus·bread): 해시 대조\n")
print("| 씬 | 판 | 결과 | 오차 % | 쪼갠 프레임 (프레임, K, ms) | 덜 풀린 걸음 ‖g‖²/F² 최대 | 질량중심 z mm (레고 f10·f31·f43, wolf f30·f50) | 끝 해시 | BG 와 다른 프레임 수 |")
print("|---|---|---|---|---|---|---|---|---|")
for sc, ts in (("L", (9, 30, 42)), ("W", (29, 49)), ("F", ()), ("B", ())):
    hb = hsh(rd(f"{sc}_BG"))
    for v in ("BG", "C", "Cx0", "Cx2"):
        n = f"{sc}_{v}"; r = rd(n)
        if r is None:
            continue
        e, dv = err(n); h = hsh(r)
        nd = sum(1 for a, b in zip(h, hb) if a != b) + abs(len(h) - len(hb)) if hb else None
        per = ""
        p = os.path.join(O, n, "period.txt")
        if os.path.exists(p):
            per = " " + " ".join(open(p).read().split()[-6:])
        print(f"| {sc} | {v} | {res(n)} | {fmt(e)}{per} | {splits(r) or '없음'} | {fmt(g2max(r), 1) if g2max(r) is None or g2max(r) < 1e4 else f'{g2max(r):.2e}'} | "
              f"{' · '.join(fmt(com(r, t), 1) for t in ts) or '-'} | {h[-1] if h else '-'} | {nd} |")
print()

# ---- 비결정
print("## 비결정 판 (C)\n")
print("| 씬 | 판 | 결과 | 오차 % | 쪼갠 프레임 | 질량중심 z f50 mm | 입자 이탈 (몸 상자 4 칸 밖 개수, 최대 거리 m) | 노드 상자 최대 | 최악 프레임 ms (다른 카드 동시) |")
print("|---|---|---|---|---|---|---|---|---|")
for sc, k in (("L", 6), ("W", 10)):
    E_ = []
    for i in range(1, k + 1):
        n = f"{sc}_C_n{i}"; r = rd(n)
        if r is None:
            continue
        e, dv = err(n)
        if e is not None and not dv:
            E_.append(e)
        far = "-"
        p = os.path.join(O, n, "bbox.txt")
        if os.path.exists(p):
            for l in open(p):
                if "pred.pt" in l:
                    d_ = json.loads(l.split(" ", 1)[1]); far = f"{max(z['nfar'] for z in d_.values())}, {max(z['dmax'] for z in d_.values())}"
        m = msl(r)
        print(f"| {sc} | n{i} | {res(n)} | {fmt(e)} | {splits(r) or '없음'} | {fmt(com(r, 49), 1) if sc == 'W' else '-'} | {far} | {max(x.get('nn', 0) for x in r) / 1e3:.0f} k | {fmt(max(q for _, q in m), 1) if m else '-'} |")
    if E_:
        print(f"| {sc} | 평균 [최소~최대] | {len(E_)} 완주 | {sum(E_) / len(E_):.2f} [{min(E_):.2f}~{max(E_):.2f}] | | | | | |")
print()


def Ef(q):
    e = (q or {}).get("en") or {}
    return None if e.get("Eel") is None else e["KE"] + e["PE"] + e["Eel"]


def lj_last(r):
    for q in reversed(r or []):
        v = (q.get("en") or {}).get("lj0")
        if v is not None and q.get("err") is not None and math.isfinite(q["err"]):
            return v
    return None


print("## 수박 셀 3 --sp2 31 (옛 실행기, 비결정 b = --erec)\n")
print("| 판 | 결과 | 오차 % | 손상 % | 생긴 E J | 튐 m/s | 쪼갠 프레임 (프레임, K, ms) | 쪼갠 뒤에도 무너진 걸음 (마지막 시도) | ms 평균 · 최대 (다른 카드 동시) |")
print("|---|---|---|---|---|---|---|---|---|")
for st, n, sc, md, fl in spec:
    if st != "s1" or sc != "M3":
        continue
    r = rd(n)
    if r is None:
        continue
    e, dv = err(n)
    Ev = [Ef(q) for q in r if Ef(q) is not None and math.isfinite(Ef(q))]
    vz = [q["en"]["vz"] for q in r if q["t"] >= 5 and q.get("en") and q["en"].get("vz") is not None and math.isfinite(q["en"]["vz"])]
    big = [x["t"] + 1 for x in r if x.get("uf") and x["uf"].get("g2") and any(f and not (g <= 1e6) for f, g in zip(x["uf"]["fail"], x["uf"]["g2"]))]
    m = [q for _, q in msl(r)]
    lj = lj_last(r)
    print(f"| {n} | {res(n)} | {fmt(e)} | {fmt(100 * lj if lj is not None else None)} | {fmt(max(Ev) - E0 if Ev else None)} | {fmt(max(vz) if vz else None, 3)} | {splits(r) or '없음'} | {big or '없음'} | "
          f"{fmt(sum(m) / len(m), 1) if m else '-'} · {fmt(max(m), 1) if m else '-'} |")
print("| 정답 | - | - | 6.51 | - | - | - | - | - |\n")

# ---- 단독 시간
print("## 단독 시간 (카드 하나에 차례로 a · b; 판마다 노드의 다른 계산 프로세스 0)\n")
print("xg = 조건 구간 [건너뜀, 돎] 합 (--xg 1·2); --xg 2 (프로파일, 모든 구간 늘 재생 = 원래 그래프와 같은 일) 는 GPU ms [늘 도는 구간, 조건 구간 중 돈 것, 건너뛸 조건 구간] 프레임 평균.\n")
print("| 판 | 플래그 | 결과 a · b | 평균 ms a · b | 최대 ms a · b (프레임) | > 16.7 ms 프레임 a · b (프레임: ms) | 쪼갠 프레임 a | xg 건너뜀/돎 a | --xg 2 구간 ms 평균 a |")
print("|---|---|---|---|---|---|---|---|---|")
for st, n, sc, md, fl in spec:
    if md != "z":
        continue
    R = [(rd(f"{n}_{s}"), f"{n}_{s}") for s in "ab"]
    if all(r is None for r, _ in R):
        continue
    cells = [[], [], [], []]
    for r, nm in R:
        m = msl(r)
        cells[0].append(res(nm) if r is not None else "-")
        cells[1].append(fmt(sum(q for _, q in m) / len(m)) if m else "-")
        if m:
            t, q = max(m, key=lambda z: z[1]); cells[2].append(f"{q:.2f} (f{t + 1})")
        else:
            cells[2].append("-")
        ov = [(t + 1, round(q, 1)) for t, q in m if q > LIM]
        cells[3].append(f"{len(ov)}" + (f" ({', '.join(f'f{a}: {b}' for a, b in ov[:8])}{' …' if len(ov) > 8 else ''})" if ov else ""))
    ra = R[0][0] or []
    xs = [x["xg"] for x in ra if x.get("xg")]
    xsk = f"{sum(z['sk'] for z in xs)}/{sum(z['run'] for z in xs)}" if xs else "-"
    xm = [z["ms"] for z in xs if z.get("ms")]
    xms = " · ".join(fmt(sum(z[i] for z in xm) / len(xm)) for i in range(3)) if xm else "-"
    fl_ = re.sub(r"--y0loc 3 --ue be --uf_res --newton 6 --xit_from 2 --en_mo --ls0_keep --ue_loc 50", "BG", fl.strip())
    print(f"| {n} | `{fl_}` | {' · '.join(cells[0])} | {' · '.join(cells[1])} | {' · '.join(cells[2])} | {' · '.join(cells[3])} | {splits(ra) or '없음'} | {xsk} | {xms} |")
print()

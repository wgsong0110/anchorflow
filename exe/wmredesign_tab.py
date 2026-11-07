"""절 CM38 묶음 wmredesign 표: python exe/wmredesign_tab.py <결과 디렉토리> [단계,..] -> markdown (stdout). 판 목록 = exe/wmredesign_spec.txt (방식 b·z 수박 판).
지표는 절 CM37 common3_tab.py 와 같은 정의: 손상 = 마지막 유한 프레임 logJp > 0 몫 (정답 6.51 %), 생긴 E = 전 구간 최대 (운동 + 중력 위치 + SNH 탄성) − 처음 7.13 J,
튐 = f5 뒤 질량중심 법선 속도 최대, 무너진 걸음 = 마지막 시도에서 덜 풀림 ∧ ‖g‖²/F² > 1e6 인 걸음이 있는 프레임, 시간 = 첫 2 프레임 뺀 재생 GPU ms (쪼갠 프레임은 시도 합)."""
import json
import math
import os
import sys

O = sys.argv[1]
STG = set(sys.argv[2].split(",")) if len(sys.argv) > 2 else None
E0 = 7.13
spec = [l.split(None, 4) for l in open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "wmredesign_spec.txt")) if l.strip() and not l.startswith("#")]
spec = [s + [""] * (5 - len(s)) for s in spec]


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


def row(n):
    r = rd(n); s = summ(n)
    if r is None:
        return None
    if s is None:
        resu = "발산 (메모리)" if oom(n) else "도는 중/죽음"; e = None
    else:
        resu = ("발산 f%d" % s["diverged"]) if s.get("diverged") else ("완주 %d" % s["frames_done"])
        e = s.get("err_mean_done", s.get("err_mean")); e = 100 * e if e is not None else None
    Ef = lambda q: None if not (q.get("en") or {}).get("Eel") is not None else q["en"]["KE"] + q["en"]["PE"] + q["en"]["Eel"]
    Ev = [Ef(q) for q in r if Ef(q) is not None and math.isfinite(Ef(q))]
    vz = [q["en"]["vz"] for q in r if q["t"] >= 5 and q.get("en") and q["en"].get("vz") is not None and math.isfinite(q["en"]["vz"])]
    lj = None
    for q in reversed(r):
        v = (q.get("en") or {}).get("lj0")
        if v is not None and q.get("err") is not None and math.isfinite(q["err"]):
            lj = v; break
    big = [x["t"] + 1 for x in r if x.get("uf") and x["uf"].get("g2") and any(f and not (g <= 1e6) for f, g in zip(x["uf"]["fail"], x["uf"]["g2"]))]
    spl = [(x["t"] + 1, x["ks"]["K"]) for x in r if x.get("ks") and x["ks"]["K"] > 1]
    on = [x["t"] + 1 for x in r if x.get("ks") and any(x["ks"].get("on") or [])]
    m = [x["ms"] for x in r if x["t"] >= 2 and x.get("ms") is not None and math.isfinite(x["ms"])]
    return dict(n=n, res=resu, err=e, dmg=100 * lj if lj is not None else None, dE=(max(Ev) - E0) if Ev else None, vz=max(vz) if vz else None,
                big=big, spl=spl, on=on, ms=(sum(m) / len(m)) if m else None, mx=max(m) if m else None, n16=sum(1 for q in m if q > 16.67))


print("# 절 CM38 묶음 wmredesign -- 수박 재설계 후보\n")
print(__doc__.split("\n", 1)[1].strip() + "\n")
print("| 판 | 결과 | 오차 % | 손상 % | 생긴 E J | 튐 m/s | 무너진 걸음 프레임 (마지막 시도) | 쪼갠 프레임 수 (K 최대) | 후보 켠 프레임 | ms 평균 · 최대 · >16.7 프레임 |")
print("|---|---|---|---|---|---|---|---|---|---|")
for st, n, sc, md, fl in spec:
    if (STG and st not in STG) or not sc.startswith("M") or md not in ("b", "z"):
        continue
    for nm in ([n] if md == "b" else [n + "_a", n + "_b"]):
        q = row(nm)
        if q is None:
            continue
        print(f"| {nm} | {q['res']} | {fmt(q['err'])} | {fmt(q['dmg'])} | {fmt(q['dE'])} | {fmt(q['vz'], 3)} | {len(q['big'])} {q['big'][:12]} | {len(q['spl'])} ({max([k for _, k in q['spl']], default=1)}) | "
              f"{len(q['on'])} | {fmt(q['ms'], 1)} · {fmt(q['mx'], 1)} · {q['n16']} |")
print("| 정답 | - | - | 6.51 | (정답 첫 접촉 +16.7) | - | - | - | - | - |")

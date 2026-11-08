"""절 CM39 묶음 speed3 표: python exe/speed3_tab.py <결과 디렉토리> -> markdown (stdout). 판 목록 = exe/speed3_spec.txt.

D = `--y0loc 3 --ue be --uf_res --newton 6 --xit_from 2 --en_mo --ls0_keep --ue_loc 50 --ksplit 3 --ks_big --xg 1 --wrx 3 --wrx_col 1` (수박 + `--sp2 31`, 셀 3).
오차 = 끝난 프레임 평균 %, 시간 = 첫 2 프레임 뺀 재생 GPU ms (쪼갠·다시 한 프레임은 모든 시도 합), 60 fps 문턱 16.7 ms. nN = 프레임별 뉴턴 반복 수 분포 (마지막 시도).
수박 손상 = 마지막 유한 프레임 logJp > 0 몫 (정답 6.51 %), 생긴 E = 전 구간 최대 (운동 + 중력 위치 + SNH 탄성) − 처음 7.13 J, 튐 = f5 뒤 질량중심 법선 속도 최대 (m/s).
wolf 이탈 = 몸 상자 (0.1~99.9 백분위) 4 칸 밖 입자 수 최대·거리 최대 (exe/speed1_bbox.py), 노드 상자 = 프레임 조밀 노드 수 최대.
"""
import collections
import glob
import json
import math
import os
import re
import sys

O = sys.argv[1]
spec = [l.split(None, 4) for l in open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "speed3_spec.txt")) if l.strip() and not l.startswith("#")]
spec = [s + [""] * (5 - len(s)) for s in spec]
E0 = 7.13
LIM = 1000.0 / 60.0
DFL = "--y0loc 3 --ue be --uf_res --newton 6 --xit_from 2 --en_mo --ls0_keep --ue_loc 50 --ksplit 3 --ks_big --xg 1 --wrx 3 --wrx_col 1"


def rd(n):
    p = os.path.join(O, n, "fem.jsonl")
    try:
        return [json.loads(l) for l in open(p) if l.strip()] if os.path.exists(p) else None
    except Exception:
        return None


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


def err(n):
    s = summ(n)
    if s is None:
        return None
    e = s.get("err_mean_done", s.get("err_mean"))
    return 100 * e if e is not None else None


def msl(r):
    return [(x["t"], x["ms"]) for x in (r or []) if x["t"] >= 2 and x.get("ms") is not None and math.isfinite(x["ms"])]


def nN(r):
    c = collections.Counter(x["it"]["nN"] for x in (r or []) if x.get("it"))
    return " ".join(f"{k}:{c[k]}" for k in sorted(c))


def com(r, t):
    return r[t]["hp"][0] * 1000 if r and t < len(r) and "hp" in r[t] else None


def Ef(q):
    e = (q or {}).get("en") or {}
    return None if e.get("Eel") is None else e["KE"] + e["PE"] + e["Eel"]


def lj_last(r):
    for q in reversed(r or []):
        v = (q.get("en") or {}).get("lj0")
        if v is not None and q.get("err") is not None and math.isfinite(q["err"]):
            return v
    return None


def scene_metric(n, sc, r):
    if sc == "L":
        return "z " + " · ".join(fmt(com(r, t), 1) for t in (9, 30, 42))
    if sc == "W":
        far = "-"
        p = os.path.join(O, n, "bbox.txt")
        if os.path.exists(p):
            for l in open(p):
                if "pred.pt" in l:
                    try:
                        d_ = json.loads(l.split(" ", 1)[1]); far = f"이탈 {max(z['nfar'] for z in d_.values())}·{max(z['dmax'] for z in d_.values())} m"
                    except Exception:
                        pass
        return f"f50 {fmt(com(r, 49), 1)} mm, {far}, 상자 {max(x.get('nn', 0) for x in r) / 1e3:.0f} k"
    if sc == "F":
        p = os.path.join(O, n, "period.txt")
        return (open(p).read().strip().split("\n")[-1].split(":", 1)[-1].strip()) if os.path.exists(p) else "-"
    if sc.startswith("M"):
        Ev = [Ef(q) for q in r if Ef(q) is not None and math.isfinite(Ef(q))]
        vz = [q["en"]["vz"] for q in r if q["t"] >= 5 and q.get("en") and q["en"].get("vz") is not None and math.isfinite(q["en"]["vz"])]
        lj = lj_last(r)
        big = sum(1 for x in r if x.get("uf") and x["uf"].get("g2") and any(f and not (g <= 1e6) for f, g in zip(x["uf"]["fail"], x["uf"]["g2"])))
        on = sum(1 for x in r if x.get("ks") and any(x["ks"].get("on") or []))
        return f"손상 {fmt(100 * lj if lj is not None else None)} %, 생긴 E {fmt(max(Ev) - E0 if Ev else None)}, 튐 {fmt(max(vz) if vz else None, 2)}, 무너진 {big}, 켠 {on}"
    return "-"


def short(fl):
    fl = " " + fl.strip() + " "
    fl = fl.replace(" D ", " D ")
    return fl.strip()


def tm(r):
    m = msl(r)
    if not m:
        return "-"
    q = [b for _, b in m]
    t, mx = max(m, key=lambda z: z[1])
    return f"{sum(q) / len(q):.2f} / {mx:.1f} (f{t + 1}) / {sum(1 for b in q if b > LIM)}"


print("# 절 CM39 묶음 speed3 -- 공통 후보 D 60 fps\n")
print(__doc__.split("\n", 1)[1])
stages = sorted(set(s[0] for s in spec), key=lambda z: (len(z), z))
for stg in stages:
    rows = [s for s in spec if s[0] == stg and s[3] not in ("z", "p")]
    if not rows or all(rd(n) is None for _, n, *_r in rows):
        continue
    print(f"## {stg} 판정·비결정 판 (시간은 다른 카드 동시라 참고)\n")
    print("| 판 | 씬 | 방식 | 플래그 | 결과 | 오차 % | 씬 지표 | ms 평균 / 최대 / >16.7 | nN 분포 | 끝 해시 |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for st, n, sc, md, fl in rows:
        r = rd(n)
        if r is None:
            continue
        h = [x.get("xh") for x in r if x.get("xh")]
        print(f"| {n} | {sc} | {md} | `{short(fl)}` | {res(n)} | {fmt(err(n))} | {scene_metric(n, sc, r)} | {tm(r)} | {nN(r)} | {h[-1] if h else '-'} |")
    print()
    zr = [s for s in spec if s[0] == stg and s[3] == "z"]
    if zr and any(rd(f"{n}_a") for _, n, *_r in zr):
        print(f"## {stg} 단독 시간 (노드 7 장 모두 계산 프로세스 0 일 때 카드 하나에 차례로 a · b)\n")
        print("| 판 | 씬 | 플래그 | 결과 a · b | 오차 % a · b | 평균 / 최대 (프레임) / >16.7 ms a | b | nN 분포 a | 씬 지표 a |")
        print("|---|---|---|---|---|---|---|---|---|")
        for st, n, sc, md, fl in zr:
            ra, rb = rd(f"{n}_a"), rd(f"{n}_b")
            if ra is None and rb is None:
                continue
            print(f"| {n} | {sc} | `{short(fl)}` | {res(n + '_a')} · {res(n + '_b')} | {fmt(err(n + '_a'))} · {fmt(err(n + '_b'))} | {tm(ra)} | {tm(rb)} | {nN(ra)} | {scene_metric(n + '_a', sc, ra) if ra else '-'} |")
        print()
    pr = [s for s in spec if s[0] == stg and s[3] == "p"]
    for st, n, sc, md, fl in pr:
        r = rd(f"{n}_a")
        if not r:
            continue
        K = [(x["t"], x["kernel_ms"], x.get("it", {})) for x in r if x.get("kernel_ms")]
        if not K:
            continue
        tot = lambda d: sum(v[0] for k, v in d.items() if not k.startswith("_"))
        K.sort(key=lambda z: -tot(z[1]))
        print(f"## {stg} 프로파일 {n} (그래프 밖 한 프레임 커널별 ms -- 이벤트·동기 비용 포함, 몫 비교용; 가장 느린 프레임 3 개와 평범한 프레임 중앙)\n")
        med = K[len(K) // 2]
        for t, d, it in K[:3] + [med]:
            top = sorted(((k, v) for k, v in d.items() if not k.startswith("_")), key=lambda kv: -kv[1][0])[:14]
            print(f"- f{t + 1} 합 {tot(d):.1f} ms (nN {it.get('nN')}, nP {it.get('nP')}): " + ", ".join(f"{k} {v[0]:.2f}/{v[1]}" for k, v in top))
        print()
    ur = [s for s in spec if s[0] == stg and s[3] == "u"]
    if ur:
        print(f"## {stg} wolf 입자 이탈 진단 (exe/speed3_wolfdiag.py)\n")
        for st, n, sc, md, fl in ur:
            p = os.path.join(O, n, "wolfdiag.json")
            if os.path.exists(p):
                for l in open(p):
                    if l.startswith("{"):
                        d = json.loads(l)
                        d.pop("E_step_m_med_by_frame", None)
                        print(f"- {n} ({res(n)}, {scene_metric(n, 'W', rd(n) or [{}]) if rd(n) else '-'}): `{json.dumps(d, ensure_ascii=False)[:1500]}`")
        print()

# ---- 후보별 묶음 (같은 단계·씬·플래그의 비결정 판; 이름 끝 _숫자 뺌)
print("## 후보별 묶음 (비결정 판; 발산 = 메모리 넘침 포함)\n")
print("| 단계 | 씬 | 플래그 | 판 수 | 발산 | 오차 % 평균 [범위] | 씬 지표 범위 | 최악 ms (동시) |")
print("|---|---|---|---|---|---|---|---|")
G = collections.OrderedDict()
for st, n, sc, md, fl in spec:
    if md in ("z", "p", "d", "u") or st.endswith("k"):
        continue
    G.setdefault((st, sc, fl.strip()), []).append(n)
for (st, sc, fl), ns in G.items():
    rs = [(n, rd(n)) for n in ns if rd(n) is not None]
    if not rs:
        continue
    div = sum(1 for n, _ in rs if res(n).startswith("발산"))
    E_ = [err(n) for n, _ in rs if err(n) is not None and not res(n).startswith("발산")]
    mx = [max(q for _, q in msl(r)) for _, r in rs if msl(r)]
    met = "-"
    if sc == "W":
        f50 = [com(r, 49) for _, r in rs if com(r, 49) is not None]
        far, dm, box = [], [], []
        for n, r in rs:
            p = os.path.join(O, n, "bbox.txt")
            if os.path.exists(p):
                for l in open(p):
                    if "pred.pt" in l:
                        try:
                            d_ = json.loads(l.split(" ", 1)[1]); far.append(max(z["nfar"] for z in d_.values())); dm.append(max(z["dmax"] for z in d_.values()))
                        except Exception:
                            pass
            box.append(max(x.get("nn", 0) for x in r))
        met = (f"f50 {min(f50):.0f}~{max(f50):.0f} mm, " if f50 else "") + (f"이탈 판 {sum(1 for q in far if q > 20)}/{len(far)} (최대 {max(far)} 개·{max(dm):.2f} m), " if far else "") + f"상자 최대 {max(box) / 1e3:.0f} k"
    elif sc.startswith("M"):
        lj = [lj_last(r) for _, r in rs]
        lj = [100 * q for q in lj if q is not None]
        Ev = []
        for _, r in rs:
            e_ = [Ef(q) for q in r if Ef(q) is not None and math.isfinite(Ef(q))]
            if e_:
                Ev.append(max(e_) - E0)
        met = (f"손상 {min(lj):.1f}~{max(lj):.1f} % (중앙 {sorted(lj)[len(lj) // 2]:.1f}), " if lj else "") + (f"생긴 E {min(Ev):.1f}~{max(Ev):.1f} J" if Ev else "")
    elif sc == "F":
        ps = []
        for n, _ in rs:
            p = os.path.join(O, n, "period.txt")
            if os.path.exists(p):
                t_ = open(p).read().strip().split("\n")[-1]
                ps.append(t_.split(":", 1)[1].strip() if ":" in t_ else t_)
        met = "; ".join(sorted(set(ps))) or "-"
    print(f"| {st} | {sc} | `{fl}` | {len(rs)} | {div} | " + (f"{sum(E_) / len(E_):.2f} [{min(E_):.2f}~{max(E_):.2f}]" if E_ else "-") + f" | {met} | {max(mx):.1f} |" if mx else " | - |")
print()

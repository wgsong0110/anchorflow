"""절 CM42 묶음 escape1 표: python exe/escape1_tab.py <결과 디렉토리> -> markdown (stdout). 판 목록 = exe/escape1_spec.txt.

E = D + 시간 예산 T2 (감싸개 `--tb 16.7 --tb_m 0.8 --tb_w 2`), 수박 M3 은 + 가속 I (`--inv 1`) + `--sp2 31`, 셀 3. D = `--y0loc 3 --ue be --uf_res --newton 6 --xit_from 2 --en_mo --ls0_keep --ue_loc 50 --ksplit 3 --ks_big --xg 1 --wrx 3 --wrx_col 1`.
--gq_mass: 레고·wolf·ficus·bread (새 실행기 exe/fem_an_rt_sim.py) 는 이 묶음에서 켬; 수박 (옛 실행기 exe/fem_ao_rt_sim.py) 바탕 명령에는 절 CM9 이후 원래 --gq_mass 가 들어 있음 (절 CM37~CM41 의 수박 판도 모두 켬) -> 수박 E + --gq_mass = E.
오차 = 끝난 프레임 평균 %, 시간 = 첫 2 프레임 뺀 재생 GPU ms (쪼갠·다시 한 프레임은 모든 시도 합), 60 fps 문턱 16.7 ms.
질량 비 = 노드 m_n / 그 프레임 m_n > 0 중앙값 (e1.jsonl, 전 프레임 최소). 이탈 = 몸 상자 (0.1~99.9 백분위) 4 칸 밖 입자 (프레임마다 e1.jsonl; wolf 는 덤프 bbox.txt 도 -- exe/speed1_bbox.py, 5 프레임마다 + 끝).
수박 손상 = 마지막 유한 프레임 logJp > 0 몫 (정답 6.51 %), 생긴 E = 전 구간 최대 (운동 + 중력 위치 + SNH 탄성) − 처음 7.13 J; 메모리 = 노드 상자 커짐으로 메모리 넘침 (발산).
"""
import collections
import json
import math
import os
import sys

O = sys.argv[1]
spec = [l.split(None, 4) for l in open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "escape1_spec.txt")) if l.strip() and not l.startswith("#")]
spec = [s + [""] * (5 - len(s)) for s in spec]
E0 = 7.13
LIM = 1000.0 / 60.0
SCN = {"L": "레고", "W": "wolf", "F": "ficus", "B": "bread", "M3": "수박"}


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


def eg(x):
    return "-" if x is None else f"{x:.1e}"


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


def e1s(n):
    e = jl(n, "e1.jsonl") or []
    if not e:
        return None
    rmin = [q["rmin"] for q in e if q.get("rmin") is not None]
    on = [q["f"] for q in e if q.get("nfar", 0) > 0]
    return dict(rmin=min(rmin) if rmin else None, n4=max(q.get("n4", 0) for q in e), n6=max(q.get("n6", 0) for q in e),
                nfar=max(q.get("nfar", 0) for q in e), dmax=max((q.get("dmax") or 0) for q in e), f0=on[0] if on else None, e=e)


def tm(r):
    m = msl(r)
    if not m:
        return "-"
    q = [b for _, b in m]
    t, mx = max(m, key=lambda z: z[1])
    return f"{sum(q) / len(q):.2f} / {mx:.1f} (f{t + 1}) / {sum(1 for b in q if b > LIM)}"


def scene_metric(n, sc, r):
    if sc == "L":
        return "-"
    if sc == "W":
        bb = bbox(n)
        return f"f50 {fmt(com(r, 49), 1)} mm" + (f", 덤프 이탈 {bb[0]}·{bb[1]:.3f} m" if bb else "")
    if sc == "F":
        p = os.path.join(O, n, "period.txt")
        return (open(p).read().strip().split("\n")[-1].split(":", 1)[-1].strip()) if os.path.exists(p) else "-"
    if sc.startswith("M"):
        Ev = [Ef(q) for q in r if Ef(q) is not None and math.isfinite(Ef(q))]
        lj = lj_last(r)
        return f"손상 {fmt(100 * lj if lj is not None else None)} %, 생긴 E {fmt(max(Ev) - E0 if Ev else None)} J"
    return "-"


print("# 절 CM42 묶음 escape1 -- 공통 후보 E + --gq_mass, 입자 이탈\n")
print(__doc__.split("\n", 1)[1])
stages = sorted(set(s[0] for s in spec), key=lambda z: (len(z), z))
for stg in stages:
    rows = [s for s in spec if s[0] == stg and s[3] not in ("z", "p")]
    if rows and any(rd(n) is not None for _, n, *_r in rows):
        print(f"## {stg} 판정·비결정 판 (시간은 다른 카드 동시라 참고)\n")
        print("| 판 | 씬 | 플래그 | 결과 | 오차 % | 씬 지표 | 최소 질량 비 · <1e-4 노드 최대 · <1e-6 | 이탈 (프레임마다) 최대 수 · 거리 m · 첫 프레임 | 노드 상자 최대 | ms 평균 / 최대 / >16.7 | nN 분포 |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        for st, n, sc, md, fl in rows:
            r = rd(n)
            if r is None:
                continue
            e = e1s(n)
            mr = f"{eg(e['rmin'])} · {e['n4']} · {e['n6']}" if e else "-"
            es = f"{e['nfar']} · {e['dmax']:.3f} · {('f%d' % e['f0']) if e['f0'] else '-'}" if e else "-"
            print(f"| {n} | {SCN.get(sc, sc)} | `{fl.strip()}` | {res(n)} | {fmt(err(n))} | {scene_metric(n, sc, r)} | {mr} | {es} | {max(x.get('nn', 0) for x in r) / 1e3:.0f} k | {tm(r)} | {nN(r)} |")
        print()
        # 이탈 시작 진단: 새 이탈 입자가 생긴 프레임 (판마다 처음 3 개)
        dl = []
        for st, n, sc, md, fl in rows:
            e = e1s(n)
            if not e:
                continue
            r = {x["t"] + 1: x for x in (rd(n) or [])}
            s3 = {}
            for i, q in enumerate(jl(n, "s3.jsonl") or []):
                s3[i + 1] = q
            fr = [q for q in e["e"] if q.get("new", {}).get("n", 0) > 0 and q["f"] > 1][:3]
            for q in fr:
                w = q["new"]
                x = r.get(q["f"], {})
                uf = x.get("uf") or {}
                stp = q.get("step") or {}
                dl.append(f"| {n} | f{q['f']} | {w['n']} | {eg(w.get('rmin_med'))} · {eg(w.get('rmin_min'))} | {fmt(w.get('frac_lt1e4'))} · {fmt(w.get('frac_lt1e2'))} | {fmt(w.get('vn_med'), 1)} · {fmt(w.get('vp_med'), 1)} | "
                          f"{eg(q.get('rmin'))} · {q.get('nfast')} · {eg(q.get('fast_rmed'))} | {stp.get('fail', uf.get('fail'))} · {[float(f'{g:.3g}') for g in (stp.get('g2') or uf.get('g2') or [])]} | {uf.get('smin')} | {(x.get('it') or {}).get('nN')} · {s3.get(q['f'], {}).get('cut')} |")
        if dl:
            print(f"## {stg} 이탈 시작 진단 (새 이탈 입자 = 그 프레임에 처음 몸 상자 4 칸 밖; 판마다 처음 3 프레임)\n")
            print("| 판 | 프레임 | 새 이탈 수 | 그 입자 둘레 27 노드 질량 비 최소: 중앙 · 최소 | <1e-4 몫 · <1e-2 몫 | 둘레 노드 속도 최대 중앙 · 입자 속도 중앙 m/s | 그 프레임 노드 질량 비 최소 · 3 m/s 넘는 노드 수 · 그 질량 비 중앙 | 걸음 덜 풀림 · ‖g‖²/F² | 보폭 최소 | 뉴턴 수 · 예산 자름 |")
            print("|---|---|---|---|---|---|---|---|---|---|")
            print("\n".join(dl))
            print()
    zr = [s for s in spec if s[0] == stg and s[3] == "z"]
    if zr and any(rd(f"{n}_a") or rd(f"{n}_b") for _, n, *_r in zr):
        print(f"## {stg} 단독 시간 (노드 7 장 모두 계산 프로세스 0 일 때 카드 하나에 차례로 a · b; 진단 끔)\n")
        print("| 판 | 씬 | 플래그 | 결과 a · b | 오차 % a · b | 평균 / 최대 (프레임) / >16.7 ms a | b | 최악 ≤ 16.7 | nN 분포 a | 씬 지표 a |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for st, n, sc, md, fl in zr:
            ra, rb = rd(f"{n}_a"), rd(f"{n}_b")
            if ra is None and rb is None:
                continue
            mx = [max(b for _, b in msl(q)) for q in (ra, rb) if msl(q)]
            ok = ("예" if mx and max(mx) <= LIM and not div(n + "_a") and not div(n + "_b") else "아니오") if mx else "-"
            print(f"| {n} | {SCN.get(sc, sc)} | `{fl.strip()}` | {res(n + '_a')} · {res(n + '_b')} | {fmt(err(n + '_a'))} · {fmt(err(n + '_b'))} | {tm(ra)} | {tm(rb)} | {ok} | {nN(ra)} | {scene_metric(n + '_a', sc, ra) if ra else '-'} |")
        print()

print("## 후보·씬별 묶음 (비결정 판; 발산 = 메모리 넘침 포함)\n")
print("| 단계 | 씬 | 플래그 | 판 수 | 발산 | 오차 % 평균 [범위] | 이탈 판 (> 20 개) · 최대 수 · 거리 m | 최소 질량 비 | 노드 상자 최대 | 씬 지표 범위 | 최악 ms (동시) |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
G = collections.OrderedDict()
for st, n, sc, md, fl in spec:
    if md in ("z", "p", "d", "u") or st.endswith("k"):
        continue
    G.setdefault((st, sc, fl.strip()), []).append(n)
for (st, sc, fl), ns in G.items():
    rs = [(n, rd(n)) for n in ns if rd(n) is not None]
    if not rs:
        continue
    dv = sum(1 for n, _ in rs if div(n))
    E_ = [err(n) for n, _ in rs if err(n) is not None and not div(n)]
    mx = [max(q for _, q in msl(r)) for _, r in rs if msl(r)]
    es = [e1s(n) for n, _ in rs]
    es = [e for e in es if e]
    far = [e["nfar"] for e in es]
    rmin = [e["rmin"] for e in es if e["rmin"] is not None]
    box = [max(x.get("nn", 0) for x in r) for _, r in rs]
    met = "-"
    if sc == "W":
        f50 = [com(r, 49) for _, r in rs if com(r, 49) is not None]
        bb = [bbox(n) for n, _ in rs]
        bb = [b for b in bb if b]
        met = (f"f50 {min(f50):.0f}~{max(f50):.0f} mm (≤ 650: {sum(1 for v in f50 if v <= 650)}/{len(f50)})" if f50 else "") + (f"; 덤프 이탈 판 {sum(1 for b in bb if b[0] > 20)}/{len(bb)} (최대 {max(b[0] for b in bb)} 개·{max(b[1] for b in bb):.2f} m)" if bb else "")
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
    ef = f"{sum(E_) / len(E_):.2f} [{min(E_):.2f}~{max(E_):.2f}]" if E_ else "-"
    ft = f"{sum(1 for q in far if q > 20)}/{len(far)} · {max(far)} · {max(e['dmax'] for e in es):.2f}" if far else "-"
    print(f"| {st} | {SCN.get(sc, sc)} | `{fl}` | {len(rs)} | {dv} | {ef} | {ft} | {eg(min(rmin)) if rmin else '-'} | {max(box) / 1e3:.0f} k | {met} | {fmt(max(mx), 1) if mx else '-'} |")
print()

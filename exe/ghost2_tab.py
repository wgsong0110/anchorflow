"""절 CM48 묶음 ghost2 표: python exe/ghost1_tab.py <결과 디렉토리> -> markdown (stdout). 판 목록 = exe/ghost2_spec.txt (exe/escape3_tab.py 바탕).
@e3w 1 = 절 CM45 W̄ (원래 G2P 기저; ficus·bread 해당 없음), @g1 1 = 절 CM47 ghost penalty: 경계 칸 (26 이웃 중 빈 칸이 있는 찬 칸) 의 내부 면에서 2 차 법선 미분 점프 벌점,
½ γ_a h^-2 h^5 ∫_F |[∂²_n y]|², γ_a = (2μ+λ)·1e-4 (Hansbo·Larson·Larsson 2017).

E = D + 시간 예산 T2 (감싸개 `--tb 16.7 --tb_m 0.8 --tb_w 2`), 수박 M3 은 + 가속 I (`--inv 1`) + `--sp2 31`, 셀 3. D = `--y0loc 3 --ue be --uf_res --newton 6 --xit_from 2 --en_mo --ls0_keep --ue_loc 50 --ksplit 3 --ks_big --xg 1 --wrx 3 --wrx_col 1`.
오차 = 끝난 프레임 평균 %, 시간 = 첫 2 프레임 뺀 재생 GPU ms (쪼갠·다시 한 프레임은 모든 시도 합), 60 fps 문턱 16.7 ms.
질량 비 = 노드 m_n / 그 프레임 m_n > 0 중앙값 (e1.jsonl, 전 프레임 최소). 이탈 = 몸 상자 (0.1~99.9 백분위) 4 칸 밖 입자 (프레임마다 e1.jsonl; wolf 는 덤프 bbox.txt 도).
수박 손상 = 마지막 유한 프레임 logJp > 0 몫 (정답 6.51 %), 생긴 E = 전 구간 최대 (운동 + 중력 위치 + SNH 탄성) − 처음 7.13 J; 메모리 = 노드 상자 커짐으로 메모리 넘침 (발산).
"""
import collections
import json
import math
import os
import sys

O = sys.argv[1]
spec = [l.split(None, 4) for l in open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ghost2_spec.txt")) if l.strip() and not l.startswith("#")]
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


print("# 절 CM48 묶음 ghost2 -- 절 CM47 ghost penalty 검사 + γ 바꿈 (수박 E + W̄ + ghost) + 다른 씬 회귀\n")
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
        print(f"## {stg} 단독 시간 (계산 프로세스 0 인 카드 하나에 차례로 a · b; 진단 끔)\n")
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


# ---- 절 CM43 진단 D2 요약 (방식 g 판; e2.jsonl)
def e2s(n):
    L = jl(n, "e2.jsonl") or []
    F = [q for q in L if "esc" not in q]
    E = [q for q in L if "esc" in q]
    if not F:
        return None
    kick = []
    for e in E:
        for fr in e["frames"]:
            if fr.get("v1") is not None and fr.get("v0") is not None and fr["v1"] - fr["v0"] > 1.0 and "vn_end_max_sol" in fr:
                kick.append(fr)
    seen, kk = set(), []
    for fr in kick:                                                                      # 같은 프레임은 한 번
        if fr["f"] not in seen:
            seen.add(fr["f"]); kk.append(fr)
    zb = [q for q in F if "nbelow" in q]
    return dict(nk=len(kk), vsol=max((q["vn_end_max_sol"] for q in kk), default=None), vpin=max((q["vn_end_max_pin"] for q in kk), default=None),
                share=sorted(q["vpin_share"] for q in kk)[len(kk) // 2] if kk else None, mnmin=sorted(q["mn_min_med"] for q in kk)[len(kk) // 2] if kk else None,
                lj=sorted(q["lj1_max"] for q in kk)[len(kk) // 2] if kk else None, J=sorted(q["J1_med"] for q in kk)[len(kk) // 2] if kk else None,
                ulm=sorted(q.get("ulm_w", 0) for q in kk)[len(kk) // 2] if kk else None,
                below=max((q["nfar_below"] for q in zb), default=None) if zb else None, zmin=min((q["zmin"] for q in zb), default=None) if zb else None,
                nn=max(q["nn"] for q in F))


rows = [s_ for s_ in spec if s_[3] == "g" and not s_[0].endswith("k")]
out = []
for st, n, sc, md, fl in rows:
    e = e2s(n)
    if not e:
        continue
    out.append(f"| {st} | {n} | `{fl.strip()}` | {res(n)} | {e['nk']} | {fmt(e['vsol'], 1)} · {fmt(e['vpin'], 1)} | {fmt(e['share'])} | {eg(e['mnmin'])} | {fmt(e['J'])} · {fmt(e['lj'])} | {fmt(e['ulm'])} | "
               f"{'-' if e['below'] is None else e['below']} · {fmt(e['zmin'])} | {e['nn'] / 1e3:.0f} k |")
if out:
    print("## 진단 D2: 튐 프레임 (이탈 입자의 앞 5 프레임 중 입자 속도가 한 프레임에 1 m/s 넘게 는 프레임, 판마다 프레임 단위) · 바닥 뚫음\n")
    print("튐 노드 = 그 입자 27 노드 (가중 > 1e-3) 중 걸음 끝 속도 최대. 풀린 = 고정 (m_n < mthr, 관성 예측에 묶임) 아닌 노드. 고정 몫 = 입자 속도 Σ w v_n 중 고정 노드 몫 (중앙). "
          "J·logJp = 튐 걸음 끝 입자 (중앙). 후퇴 오일러 무게 = 27 노드 가중 중 ulm 노드 몫 (중앙). 바닥 뚫음 = 이탈 입자 중 바닥 (z 0.02) 아래 한 칸 넘는 입자 최대 수 · 전 입자 z 최소 (h1 부터 기록).\n")
    print("| 단계 | 판 | 플래그 | 결과 | 튐 프레임 수 | 튐 노드 끝 속도 최대 m/s: 풀린 · 고정 | 고정 몫 | 27 노드 최소 질량 비 (중앙) | 입자 J · logJp 최대 (중앙) | 후퇴 오일러 무게 | 바닥 뚫음 수 · z 최소 m | 노드 상자 최대 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    print("\n".join(out))
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


# ---- 절 CM45 진단 D3 (방식 g 판; e3.jsonl)
def e3s(n):
    L = jl(n, "e3.jsonl") or []
    L = [q for q in L if "nq" in q]
    if not L:
        return None
    mx = lambda k: max((q[k] for q in L if q.get(k) is not None), default=None)
    sm = lambda k: sum(q.get(k) or 0 for q in L)
    new = [z for q in L for z in (q.get("new") or [])]
    return dict(dbw=mx("dbw_max"), dbw999=mx("dbw_p999"), dwq=mx("dwq_max"), g2p=mx("g2p_err"), pmb=sm("nq_pm_below"), miss=sm("nq_pm_below_bw_above"), nbel=mx("nbelow"),
                nnew=sm("nnew_below"), zmin=min(q["zmin"] for q in L), new=new, fnew=[q["f"] for q in L if q.get("nnew_below")][:6])


out = []
dl = []
for st, n, sc, md, fl in [s_ for s_ in spec if s_[3] == "g"]:
    e = e3s(n)
    if not e:
        continue
    nw = e["new"]
    abv = sum(1 for z in nw if z["d_bw"] >= 0)
    dwq = "-" if e["dwq"] is None else f"{e['dwq']:.1e}"
    g2p = "-" if e["g2p"] is None else f"{e['g2p']:.1e}"
    out.append(f"| {st} | {n} | `{fl.strip()}` | {res(n)} | {fmt(e['dbw'], 4)} · {fmt(e['dbw999'], 4)} | {dwq} | {g2p} | "
               f"{e['pmb']} · {e['miss']} | {e['nbel']} · {e['nnew']} ({' '.join('f%d' % f for f in e['fnew'])}) | {abv}/{len(nw)} | {fmt(e['zmin'])} |")
    for z in nw[:4]:
        dl.append(f"| {n} | {z['z0']:.4f} → {z['z1']:.4f} | {z['d_bw']:+.4f} | {z['d_pm']:+.4f} | {'-' if z.get('d_wq') is None else '%+.4f' % z['d_wq']} | {z.get('bact')} | {z['mq']:.3g} | {z['vz']:.2f} |")
if out:
    print("## 진단 D3: 장벽 표본점 높이 -- 가우스 점 기저 BW (원래) 대 칸 입자 G2P 끝 높이 질량 평균 (d = z − 0.02 m, 걸음 끝)\n")
    print("|d_bw − d_pm| = 장벽이 본 표본점 높이와 그 칸 입자들이 실제로 간 높이 (질량 평균) 의 차 (전 프레임 최대 · 99.9 백분위 최대, m). |d_wq − d_pm| = W̄ 로 잰 높이와의 차 (W̄ 판만 -- 0 이어야 함). "
          "G2P 재구성 = 입자 끝 z 와 x0 + Σ ŵ U 의 차 (U·기저 확인). 질량 평균 바닥 밑 표본점 = d_pm < 0 인 (프레임, 표본점) 수 · 그중 장벽은 바닥 위 (d_bw ≥ 0) 로 본 수. "
          "새 바닥 밑 입자 = 이 프레임에 z < 0.02 로 넘어간 입자 (전 프레임 합), 그중 자기 표본점 d_bw ≥ 0 = 장벽이 바닥 위로 본 몫.\n")
    print("| 단계 | 판 | 플래그 | 결과 | |d_bw − d_pm| 최대 · p99.9 m | |d_wq − d_pm| 최대 | G2P 재구성 최대 | 질량 평균 바닥 밑 표본점 · 그중 장벽 바닥 위 | 바닥 밑 입자 최대 · 새로 (첫 프레임) | 새 바닥 밑 입자 중 d_bw ≥ 0 | z 최소 m |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    print("\n".join(out))
    print()
if dl:
    print("### D3 새 바닥 밑 입자 (판마다 처음 4 개): 입자 z 시작 → 끝, 자기 표본점 d_bw (장벽이 본 높이) · d_pm (칸 입자 질량 평균) · d_wq (W̄), 활성 (AF_BACT), 칸 입자 질량 / 중앙, 입자 v_z m/s\n")
    print("| 판 | z m | d_bw | d_pm | d_wq | 활성 | 칸 질량 비 | v_z |")
    print("|---|---|---|---|---|---|---|---|")
    print("\n".join(dl))
    print()

# ---- 절 CM43 표 형식 (수박 후보별)
print("## 수박 후보별 (절 CM43 표 형식; 비결정, 진단 D2·D3 켬, 이탈 판 = 4 칸 밖 > 20 개, 시간은 동시 판이라 참고)\n")
print("| 후보 | 판 | 발산 | 오차 % | 이탈 판 · 최대 수 · 거리 m | 노드 상자 최대 (> 5 백만 판) | 바닥 뚫음 판 (D2: 이탈 입자 중 바닥 밑 한 칸 넘는 입자 > 0) | 손상 % (정답 6.51) · 생긴 E J | 튐 노드 끝 속도 최대 m/s |")
print("|---|---|---|---|---|---|---|---|---|")
GM = collections.OrderedDict()
for st, n, sc, md, fl in spec:
    if sc.startswith("M") and md == "g" and not st.endswith("k"):
        GM.setdefault(fl.strip(), []).append(n)
for fl, ns in GM.items():
    rs = [(n, rd(n)) for n in ns if rd(n) is not None]
    if not rs:
        continue
    dv = sum(1 for n, _ in rs if div(n))
    E_ = [err(n) for n, _ in rs if err(n) is not None and not div(n)]
    es = [e1s(n) for n, _ in rs]
    es = [e for e in es if e]
    far = [e["nfar"] for e in es]
    box = [max(x.get("nn", 0) for x in r) for _, r in rs]
    e2 = [e2s(n) for n, _ in rs]
    bel = [q["below"] for q in e2 if q and q.get("below") is not None]
    vs = [q["vsol"] for q in e2 if q and q.get("vsol") is not None]
    lj, Ev = [], []
    for _, r in rs:
        v_ = lj_last(r)
        if v_ is not None:
            lj.append(100 * v_)
        e_ = [Ef(q) for q in r if Ef(q) is not None and math.isfinite(Ef(q))]
        if e_:
            Ev.append(max(e_) - E0)
    ft = f"{sum(1 for q in far if q > 20)}/{len(far)} · {max(far)} · {max(e['dmax'] for e in es):.2f}" if far else "-"
    print(f"| `{fl}` | {len(rs)} | {dv} | {('%.2f~%.2f' % (min(E_), max(E_))) if E_ else '-'} | {ft} | {max(box) / 1e6:.1f} 백만 ({sum(1 for b in box if b > 5e6)}/{len(box)}) | "
          f"{sum(1 for b in bel if b > 0)}/{len(bel)} | {('%.1f~%.1f' % (min(lj), max(lj))) if lj else '-'} · {('%.1f~%.1f' % (min(Ev), max(Ev))) if Ev else '-'} | {('%.0f~%.0f' % (min(vs), max(vs))) if vs else '-'} |")
print()


# ---- 절 CM47 ghost 계수 (g1.jsonl)
out = []
for st, n, sc, md, fl in spec:
    L = jl(n, "g1.jsonl") or []
    if not L:
        continue
    out.append(f"| {st} | {n} | {SCN.get(sc, sc)} | {res(n)} | {L[0]['k']:.4g} | {min(q['ncut'] for q in L)}~{max(q['ncut'] for q in L)} | {min(q['nface'] for q in L)}~{max(q['nface'] for q in L)} | {min(q['nc'] for q in L)}~{max(q['nc'] for q in L)} |")
if out:
    print("## 절 CM47 ghost 계수 (프레임별 범위): k = γ_a h, 경계 칸 수, 벌점 면 수, 찬 칸 수\n")
    print("| 단계 | 판 | 씬 | 결과 | k | 경계 칸 | 벌점 면 | 찬 칸 |")
    print("|---|---|---|---|---|---|---|---|")
    print("\n".join(out))
    print()

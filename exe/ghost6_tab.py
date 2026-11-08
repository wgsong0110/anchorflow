"""절 CM57 묶음 ghost6 표 (ghost5_tab 복사 + 보존 판 검사·에너지 절): python exe/ghost6_tab.py <결과 디렉토리> -> markdown (stdout). 판 목록 = exe/ghost6_spec.txt.
@g6c 1 = 보존 판 (강성 벌점 인자 = 기준 형상 변위 u0 + 걸음 변위 y), 없으면 증분 판 (절 CM47~CM55 그대로, 인자 = y).
F = 공통 후보 F (E + W̄ @e3w 1 + R2 @e5r 2 + A2 @e5a 2, 절 CM49), V2NF 바탕 = 절 CM46 V2NF + 같은 W̄·R2·A2. @g4 g = ghost penalty (γ̄ = g, 칸별 커널), @g4m 0|1 고스트 질량, @g4t 0~3 면 한정 방식 (0 없음 · 1 3x3x3 열림 · 2 5x5x5 열림 · 3 내부 칸-경계 칸 면).
오차 = 끝난 프레임 평균 %, 시간 = 첫 2 프레임 뺀 재생 GPU ms, 60 fps 문턱 16.7 ms. 이탈 = 몸 상자 (0.1~99.9 백분위) 4 칸 밖 입자 (e1.jsonl), 이탈 판 = > 20 개.
튐 프레임 = 진단 D2 이탈 입자의 앞 5 프레임 중 입자 속도가 한 프레임에 1 m/s 넘게 는 프레임 (판마다 프레임 단위, 판 합). 바닥 뚫음 판 = D2 이탈 입자 중 바닥 밑 한 칸 넘는 입자 > 0, z 최소 = D5 전 입자 (바닥 0.02).
수박 손상 = logJp > 0 몫 (정답 f5 4.41 · f6 6.94 · f8 7.08 · 끝 6.51 %). 발산 = 판정 + 단독 (메모리 넘침 포함). 단독 = 계산 프로세스 0 인 카드 하나에 a · b 차례로, 진단 끔.
"""
import collections
import json
import math
import os
import sys

O = sys.argv[1]
spec = [l.split(None, 4) for l in open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ghost6_spec.txt")) if l.strip() and not l.startswith("#")]
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


def tz(n):
    """단독 판 한 개: '평균 / 최악 (> 16.7 프레임 수)' 또는 발산."""
    r = rd(n)
    if r is None and not oom(n):
        return None
    if oom(n):
        return "메모리 발산"
    m = msl(r)
    if div(n):
        return res(n)
    if not m:
        return "-"
    q = [b for _, b in m]
    return f"{sum(q) / len(q):.2f} / {max(q):.1f} ({sum(1 for b in q if b > LIM)})"


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


def e2s(n):
    L = jl(n, "e2.jsonl") or []
    F = [q for q in L if "esc" not in q]
    E = [q for q in L if "esc" in q]
    if not F:
        return None
    seen, kk = set(), []
    for e in E:
        for fr in e["frames"]:
            if fr.get("v1") is not None and fr.get("v0") is not None and fr["v1"] - fr["v0"] > 1.0 and "vn_end_max_sol" in fr and fr["f"] not in seen:
                seen.add(fr["f"]); kk.append(fr)
    zb = [q for q in F if "nbelow" in q]
    return dict(nk=len(kk), vsol=max((q["vn_end_max_sol"] for q in kk), default=None),
                below=max((q["nfar_below"] for q in zb), default=None) if zb else None)


def e5zmin(n):
    L = [q for q in (jl(n, "e5.jsonl") or []) if "nq" in q and q.get("zmin") is not None]
    return min((q["zmin"] for q in L), default=None)


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
    p = os.path.join(O, n, "period.json")
    try:
        d = json.load(open(p))
        r = d.get("ratio")
        return (r if r is not None and math.isfinite(r) else None), d.get("amp_ratio"), d.get("nzc")
    except Exception:
        return None


def rng(v, d=2):
    v = [q for q in v if q is not None]
    return f"{min(v):.{d}f}~{max(v):.{d}f}" if v else "-"


def mean(v):
    v = [q for q in v if q is not None]
    return sum(v) / len(v) if v else None


print("# 절 CM57 묶음 ghost6 -- 보존 ghost penalty (기준 형상 변위) 를 공통 후보 F 위에\n")
print(__doc__.split("\n", 1)[1])

zr = [s for s in spec if s[3] in ("z", "zd")]


def zrow(fl, sc):
    for st, n, s_, md, f_ in zr:
        if s_ == sc and f_.strip() == fl:
            return n
    return None


def zcol(fl, sc):
    zn = zrow(fl, sc)
    if not zn:
        return "-"
    v = [tz(f"{zn}_{k_}") for k_ in ("a", "b")]
    v = [q for q in v if q is not None]
    return " · ".join(v) if v else "-"


# ---- 검사 (g4fd.jsonl)
out = []
for st, n, sc, md, fl in spec:
    for q in jl(n, "g4fd.jsonl") or []:
        u = q.get("us") or {}
        hk = q.get("hook") or {}
        hmax = []
        herr = []
        for nm, d in hk.items():
            if isinstance(d, dict):
                if "err" in d:
                    herr.append(nm)
                hmax += [v for k_, v in d.items() if k_.endswith("_rel") or k_ == "ls_rel_max"]
        out.append(f"| {n} | {SCN.get(sc, sc)} | {q['f']} | {q.get('nact')} · {q.get('ncut')} · {q.get('nthick')} | {q.get('nface_old')} · {q.get('nface_all')} · {q.get('nface_thin')} | "
                   f"{int(bool(q.get('set_all_eq_old')))}{int(bool(q.get('set_all_eq_torch')))}{int(bool(q.get('set_thin_eq_torch')))}{int(bool(q.get('mask_restored')))} | "
                   f"{q.get('mv_old_old', 0):.1e} · {q.get('mv_new_old', 0):.1e} · {q.get('mv_new_ref', 0):.1e} | {q.get('f_new_old', 0):.1e} · {q.get('B_new_old', 0):.1e} · {q.get('E_new_old', 0):.1e} | "
                   f"{q.get('fm_rel', 0):.1e} · {q.get('Em_rel', 0):.1e} · {fmt(q.get('diag_rel'), 7) if q.get('diag_rel') is not None else '-'} | {fmt(q.get('ls_rel_max'), 9)} · {q.get('phi_rel', 0):.1e} | "
                   f"{(max(hmax) if hmax else 0):.1e}{(' 오류 ' + ','.join(herr)) if herr else ''} ({len(hk) - 1 if hk else 0}) | "
                   f"{fmt(u.get('mv_old'), 1)} · {fmt(u.get('mv_new'), 1)} · {fmt(u.get('mv_new_thin'), 1)} | {fmt(u.get('force_old'), 1)} · {fmt(u.get('force_new'), 1)} · {fmt(u.get('force_new_mass_thin'), 1)} |")
if out:
    print("## 검사·비교 (그래프 밖, 결과 불변; fem_ghost3.check)\n")
    print("집합 4 자리 = (빼기 끔 칸별 표지 = 옛 면 목록 커널 gp_face) (빼기 끔 = torch) (빼기 켬 = torch) (실행 표지 되돌림). 곱 상대 차: 옛 커널 두 번끼리 (원자 합 순서 잡음) · 칸별 대 옛 · 칸별 대 torch f64 참값. "
          "힘·대각·에너지: 칸별 대 옛 (질량 0, 같은 면). 질량 몫 = 칸별 G (k y + k_M/τ² (y − ŷ)) 와 에너지 대 torch 참값, 대각 = 단위 벡터 곱 대 B. 선탐색 후보 (강성 + 질량) · Φ = 칸별 대 참값. "
          "훅 = '훅 켬 − 훅 끔' 대 참값 최대 (훅 자리 수). 시간 µs (CUDA 이벤트 200 회 평균, 같은 상태): 곱 옛 · 칸별 (같은 면) · 칸별 (빼기 켬), 힘 (+대각) 옛 · 칸별 · 칸별 (질량 + 빼기 켬).\n")
    print("| 판 | 씬 | 프레임 | 활성 · 경계 · 두꺼운 칸 | 면 옛 · 빼기 끔 · 켬 | 집합 | 곱 옛-옛 · 칸별-옛 · 칸별-참값 | 힘 · 대각 · E 칸별-옛 | 질량 f · E · 대각 | 선탐색 · Φ | 훅 | 곱 µs | 힘 µs |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    print("\n".join(out))
    print()

# ---- 결정적 해시
out = []
for st, n, sc, md, fl in spec:
    if md != "d":
        continue
    s = summ(n)
    if s is None and rd(n) is None:
        continue
    out.append(f"| {n} | {SCN.get(sc, sc)} | `{fl.strip()}` | {res(n)} | {(s or {}).get('x_sha1', '-')} | {fmt(err(n), 4)} |")
if out:
    print("## 결정적 해시 (끝 상태 입자 위치 sha1 앞 10 자리; γ̄ = 0 은 칸별 커널이 모두 돌고 더하는 값 0)\n")
    print("| 판 | 씬 | 플래그 | 결과 | 해시 | 오차 % |")
    print("|---|---|---|---|---|---|")
    print("\n".join(out))
    print()

# ---- 수박 후보별 (절 CM48 표 형식)
G = collections.OrderedDict()
for st, n, sc, md, fl in spec:
    if sc.startswith("M") and md == "g" and not st.endswith("k"):
        G.setdefault(fl.strip(), []).append(n)
rows = []
for fl, ns in G.items():
    rs = [(n, rd(n)) for n in ns if rd(n) is not None or oom(n)]
    if not rs:
        continue
    dv = sum(1 for n, _ in rs if div(n))
    zn = zrow(fl, "M3")
    zd = zt = 0
    if zn:
        for k_ in ("a", "b"):
            nz = f"{zn}_{k_}"
            if rd(nz) is not None or oom(nz):
                zt += 1
                zd += int(div(nz))
    E_ = [err(n) for n, _ in rs if err(n) is not None and not div(n)]
    es = [e for e in (e1s(n) for n, _ in rs) if e]
    far = [e["nfar"] for e in es]
    box = [max(x.get("nn", 0) for x in r) for _, r in rs if r]
    e2 = [e2s(n) for n, _ in rs]
    bel = [q["below"] for q in e2 if q and q.get("below") is not None]
    nk = sum(q["nk"] for q in e2 if q)
    vs = [q["vsol"] for q in e2 if q and q.get("vsol") is not None]
    zm = [z for z in (e5zmin(n) for n, _ in rs) if z is not None]
    lv = [lj_last(r) for _, r in rs if r]
    lv = [q for q in lv if q is not None]
    f5 = mean([lj_at(r, 4) for _, r in rs if r])
    f6 = mean([lj_at(r, 5) for _, r in rs if r])
    f8 = mean([lj_at(r, 7) for _, r in rs if r])
    rows.append(f"| `{fl}` | {len(rs)} | {dv}/{len(rs)} + {zd}/{zt} | {sum(1 for b in bel if b > 0)}/{len(bel)} · {fmt(min(zm) + ZF if zm else None, 3)} | "
                f"{(str(sum(1 for q in far if q > 20)) + '/' + str(len(far)) + ' · ' + str(max(far)) + ' · ' + fmt(max(e['dmax'] for e in es))) if far else '-'} | "
                f"{(max(box) / 1e6) if box else 0:.2f} 백만 ({sum(1 for b in box if b > 5e6)}/{len(box)}) | {fmt(f5)} · {fmt(f6)} · {fmt(f8)} · {fmt(mean(lv))} [{rng(lv)}] | {rng(E_)} | "
                f"{nk} | {('%.0f~%.0f' % (min(vs), max(vs))) if vs else '-'} | {zcol(fl, 'M3')} |")
if rows:
    print("## 수박 후보별 (절 CM48 표 형식; 비결정, 진단 D1·D2·D5 켬)\n")
    print("| 후보 | 판 | 발산 (판정 + 단독) | 바닥 뚫음 판 · z 최소 m | 이탈 판 · 최대 수 · 거리 m | 노드 상자 최대 (> 5 백만 판) | 손상 f5 · f6 · f8 (평균) · 끝 (평균 [범위]) % | 오차 % | 튐 프레임 (판 합) | 튐 노드 끝 속도 최대 m/s | 단독 평균 / 최악 ms (> 16.7 프레임) a · b |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    print("\n".join(rows))
    print()

# ---- 다른 씬 회귀
RG = collections.OrderedDict()
for st, n, sc, md, fl in spec:
    if sc in ("L", "W", "F", "B") and md == "nd" and not st.endswith("k"):
        RG.setdefault((sc, fl.strip()), []).append(n)
rows = []
for (sc, fl), ns in RG.items():
    rs = [(n, rd(n)) for n in ns if rd(n) is not None or oom(n)]
    if not rs:
        continue
    dv = sum(1 for n, _ in rs if div(n))
    E_ = [err(n) for n, _ in rs if err(n) is not None and not div(n)]
    met = "-"
    if sc == "W":
        f50 = [com(r, 49) for _, r in rs if r and com(r, 49) is not None]
        bb = [b for b in (bbox(n) for n, _ in rs) if b]
        met = (f"f50 {min(f50):.0f}~{max(f50):.0f} mm" if f50 else "f50 -") + (f", 이탈 판 {sum(1 for b in bb if b[0] > 20)}/{len(bb)} (최대 {max(b[0] for b in bb)} 개·{max(b[1] for b in bb):.2f} m)" if bb else "")
    elif sc == "F":
        pp = [p for p in (period(n) for n, _ in rs) if p]
        met = ("주기 비 " + rng([p[0] for p in pp]) + ", 진폭 비 " + rng([p[1] for p in pp]) + ", 영점 " + ",".join(str(p[2]) for p in pp)) if pp else "-"
    elif sc == "L":
        es = [e for e in (e1s(n) for n, _ in rs) if e]
        met = f"이탈 판 {sum(1 for e in es if e['nfar'] > 20)}/{len(es)} (최대 {max(e['nfar'] for e in es)} 개)" if es else "-"
    g1 = [jl(n, "g1.jsonl") or [] for n, _ in rs]
    g1 = [q for L in g1 for q in L]
    fc = f"{min(q['nface'] for q in g1) / 1e3:.1f}~{max(q['nface'] for q in g1) / 1e3:.1f} 천 / {min(q['nface_all'] for q in g1) / 1e3:.1f}~{max(q['nface_all'] for q in g1) / 1e3:.1f} 천" if g1 else "-"
    rows.append(f"| {SCN[sc]} | `{fl}` | {len(rs)} | {dv} | {fmt(mean(E_))} [{rng(E_)}] | {met} | {fc} | {zcol(fl, sc)} |")
if rows:
    print("## 다른 씬 회귀 (비결정 판 + 단독 a · b)\n")
    print("| 씬 | 플래그 | 판 | 발산 | 오차 % 평균 [범위] | 씬 지표 | 벌점 면 (빼기 뒤 / 앞) | 단독 평균 / 최악 ms (> 16.7 프레임) a · b |")
    print("|---|---|---|---|---|---|---|---|")
    print("\n".join(rows))
    print()

# ---- 판별
print("## 판별 (판마다; 시간은 동시 판이라 참고)\n")
print("| 판 | 씬 | 플래그 | 결과 | 오차 % | 손상 f5 · f6 · f8 · 끝 % | wolf f50 mm · ficus 주기 비 | 이탈 최대 수 · 거리 m | 튐 프레임 · 바닥 뚫음 · z 최소 m | 노드 상자 최대 | 벌점 면 (빼기 뒤 · 앞) · k · k_M |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for st, n, sc, md, fl in spec:
    if md in ("z",):
        continue
    r = rd(n)
    if r is None and not oom(n):
        continue
    e = e1s(n)
    e2 = e2s(n)
    zm = e5zmin(n)
    dmg = " · ".join(fmt(lj_at(r, t)) for t in (4, 5, 7)) + " · " + fmt(lj_last(r)) if sc.startswith("M") else "-"
    pp = period(n) if sc == "F" else None
    met = fmt(com(r, 49), 0) if sc == "W" else ((fmt(pp[0]) if pp else "-") if sc == "F" else "-")
    g1 = jl(n, "g1.jsonl") or []
    gc = f"{max(q['nface'] for q in g1)} · {max(q['nface_all'] for q in g1)} · {g1[0]['k']:.3g} · {g1[0]['kM']:.3g}" if g1 else "-"
    print(f"| {n} | {SCN.get(sc, sc)} | `{fl.strip()}` | {res(n)} | {fmt(err(n))} | {dmg} | {met} | {(str(e['nfar']) + ' · ' + fmt(e['dmax'])) if e else '-'} | "
          f"{(str(e2['nk']) + ' · ' + str(e2['below'])) if e2 else '-'} · {fmt(zm + ZF if zm is not None else None, 3)} | {(max(x.get('nn', 0) for x in r) / 1e6) if r else 0:.2f} 백만 | {gc} |")
print()

# ---- 절 CM55 φ·가중 검사 (g4fd.jsonl 의 ghost5 키)
out = []
for st, n, sc, md, fl in spec:
    for q in jl(n, "g4fd.jsonl") or []:
        if "phi_ker_ref" not in q:
            continue
        hk = q.get("hook") or {}
        hmax = [v for d in hk.values() if isinstance(d, dict) for k_, v in d.items() if k_.endswith("_rel") or k_ == "ls_rel_max"]
        out.append(f"| {n} | {SCN.get(sc, sc)} | {q['f']} | {q.get('phi_ker_ref', 0):.1e} · {q.get('rho_ker_ref', 0):.1e} · {q.get('phi_run_eq_now', 0):.1e} | "
                   f"{' '.join(fmt(v, 3) for v in q.get('phi_q') or [])} | {q.get('w_ker_ref', float('nan')):.1e} · {fmt(q.get('w_mean'), 3)} · {' '.join(fmt(v, 3) for v in q.get('w_q') or [])} | "
                   f"{q.get('nface_thin')} | {q.get('mv_new_ref', 0):.1e} · {q.get('fm_rel', 0):.1e} · {q.get('Em_rel', 0):.1e} · {fmt(q.get('diag_rel'), 7) if q.get('diag_rel') is not None else '-'} · {fmt(q.get('ls_rel_max'), 9)} · {q.get('phi_rel', 0):.1e} | "
                   f"{(max(hmax) if hmax else 0):.1e} ({len(hk) - 1 if hk else 0}) |")
if out:
    print("## 절 CM55 검사 -- 칸 점유 φ·면 가중 (그래프 밖, 결과 불변)\n")
    print("φ: 커널 대 torch (지금 C·A) 최대 차 · 칸 밀도 상대 차 · 실행 φ (rt_gauss 때) 대 지금. 가중: 커널 대 torch (실행 φ) 최대 차 · 평균 · 10/50/90 %. 값 (가중 켠 면 집합): 질량 몫 f · E · 대각 · 선탐색 · Φ 대 torch 가중 참값, 훅 = '훅 켬 − 훅 끔' 대 가중 참값 최대.\n")
    print("| 판 | 씬 | 프레임 | φ 커널-참값 · ρ_c · 실행-지금 | φ 10/50/90 % | 가중 커널-참값 · 평균 · 10/50/90 % | 벌점 면 | 곱 · 힘 · E · 대각 · 선탐색 · Φ | 훅 |")
    print("|---|---|---|---|---|---|---|---|---|")
    print("\n".join(out))
    print()

# ---- 절 CM55 헤시안 몫 (g5ed.jsonl)
out = []
for st, n, sc, md, fl in spec:
    for q in jl(n, "g5ed.jsonl") or []:
        qe, qM = q.get("q_e"), q.get("q_M")
        qg = q.get("q_g") or {}
        cells = []
        for m_ in ("0", "1", "2", "3", "4", "5", "6"):
            d = qg.get(m_) or {}
            a = d.get("all")
            cells.append(f"{(a / qe) if (a is not None and qe) else float('nan'):.3g} ({(d.get('low', 0) / qe) if qe else float('nan'):.3g} / {(d.get('high', 0) / qe) if qe else float('nan'):.3g}) w̄ {fmt(d.get('w_mean'), 2)}")
        nf = q.get("nface") or {}
        out.append(f"| {n} | {SCN.get(sc, sc)} | {q['f']} | {qe if qe is not None else float('nan'):.3g} · {qM if qM is not None else float('nan'):.3g} | {nf.get('all')} ({nf.get('low')} / {nf.get('high')}) | " + " | ".join(cells) + " |")
if out:
    print("## 절 CM55 헤시안 몫 -- 걸음 끝 y 방향 (q_g / q_e, 괄호 = 칸 밀도 < ρ̄ 면 / ≥ ρ̄ 면; 실행 면 표지에 가중 방식 0~4)\n")
    print("| 판 | 씬 | 프레임 | q_e = yᵀK_e y · q_M = Σ m|y|²/τ² | 면 (밀도 낮음 / 높음) | 방식 0 (가중 없음) | 1 φmax(1−φmin) | 2 Frei w_max | 3 1−φ̄ | 4 φmax | 5 1−φmin | 6 1−φmax |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    print("\n".join(out))
    print()


# ---- 절 CM57 보존 판 검사 (g6fd.jsonl)
out = []
for st, n, sc, md, fl in spec:
    for q in jl(n, "g6fd.jsonl") or []:
        if "err" in q:
            out.append(f"| {n} | {SCN.get(sc, sc)} | {q['f']} | 오류 {q['err'][:120]} |||||")
            continue
        out.append(f"| {n} | {SCN.get(sc, sc)} | {q['f']} | {q.get('u0_ker_ref_abs', float('nan')):.1e} / {fmt(q.get('u0_scale'), 3)} · {q.get('den_rel', float('nan')):.1e} · {q.get('nbad_detF')} | "
                   f"{q.get('rigid_u0_err_abs', float('nan')):.1e} / {fmt(q.get('rigid_u0_scale'), 3)} (참값 {q.get('rigid_u0_ref_err_abs', float('nan')):.1e}) | "
                   f"{q.get('affine_u0_err_abs', float('nan')):.1e} / {fmt(q.get('affine_u0_scale'), 3)} (참값 {q.get('affine_u0_ref_err_abs', float('nan')):.1e}) | "
                   f"{q.get('fu_rel', float('nan')):.1e} · {q.get('Eu_rel', float('nan')):.1e} · {q.get('ls_u0_rel_max', float('nan')):.1e} | {q.get('Eu', float('nan')):.3g} · {q.get('E_u0only', float('nan')):.3g} · {q.get('E_yonly', float('nan')):.3g} |")
if out:
    print("## 절 CM57 검사 -- 보존 판 u0 (그래프 밖, 결과 불변)\n")
    print("u0 커널 대 torch f64 참값 (지금 입자 상태, 같은 식) 최대 절대 차 / |u0| 최대 · 노드 가중 합 상대 차 · det F ≤ 0 입자 수. 강체 (축 회전 0.4 rad + 이동) · 아핀 (회전 x 늘림 1.1·0.9·1.05 + 이동), 입자 = 첫 프레임 위치 X0 를 그 운동으로 옮김, F = 그 행렬: "
          "커널 u0 대 정확한 아핀 장 ξ − M⁻¹(ξ − c) 최대 절대 차 / 크기 (괄호 = torch 참값 대 정확한 장). 실행 면 집합 위: 힘 k G (y + u0) · 에너지 · 선탐색 후보 (ŷ + T(η + cδ) + u0) 대 torch 참값 상대 차. 에너지 J: ½k (y+u0)ᵀG(y+u0) · u0 만 · y 만.\n")
    print("| 판 | 씬 | 프레임 | u0 커널-참값 · 가중 합 · det≤0 | 강체 | 아핀 | 힘 · E · 선탐색 | E(y+u0) · E(u0) · E(y) |")
    print("|---|---|---|---|---|---|---|---|")
    print("\n".join(out))
    print()

# ---- 절 CM57 에너지 진단 (g6en.jsonl; ficus 자유 진동)
out = []
curves = {}
for st, n, sc, md, fl in spec:
    L = [q for q in (jl(n, "g6en.jsonl") or []) if "err" not in q]
    if not L:
        continue
    L.sort(key=lambda q: q["f"])
    cons = "@g6c 1" in fl
    tot = [q["KE"] + q["Eel"] + q["PE"] + (q.get("E_end", 0.0) if cons else 0.0) for q in L]
    curves[n] = [(q["f"], t) for q, t in zip(L, tot)]
    d_all = tot[0] - tot[-1]
    winc = sum(q.get("W_inc", 0.0) for q in L)
    if cons:
        wst = sum(q["E_end"] - q["E_start"] for q in L if "E_end" in q)
        rg = sum(L[i]["E_end"] - L[i + 1]["E_start"] for i in range(len(L) - 1) if "E_end" in L[i] and "E_start" in L[i + 1])
        es = max(q.get("E_end", 0.0) for q in L)
    else:
        wst = rg = es = None
    pp = period(n) if sc == "F" else None
    ke = [q["KE"] for q in L]
    out.append(f"| {n} | {SCN.get(sc, sc)} | `{fl.strip()}` | {len(L)} | {L[0]['k']:.3g}{'' if L[0].get('k_run') else ' (참고 k)'} | {tot[0]:.4g} → {tot[-1]:.4g} (최소 {min(tot):.4g}) | {d_all:.4g} | {winc:.4g} ({(100 * winc / d_all) if d_all > 0 else float('nan'):.0f} %) | "
               f"{fmt(wst, 4) if wst is not None else '-'} · {fmt(rg, 4) if rg is not None else '-'} · {fmt(es, 4) if es is not None else '-'} | {max(ke):.3g} · {ke[-1]:.3g} | {(fmt(pp[0]) + ' · ' + fmt(pp[1])) if pp else '-'} |")
if out:
    print("## 절 CM57 에너지 진단 (프레임마다 걸음 뒤, 결과 불변; J)\n")
    print("총 에너지 = 입자 KE + 탄성 Σ V ψ_FCR(F) + 중력 위치 에너지 (+ 보존 판은 걸음 끝 벌점 에너지 E_end). 소산 = 첫 프레임 총 − 끝 프레임 총. 증분 꼴 벌점 일 Σ W_inc = Σ_f ½ k U_fᵀ G U_f (U = 프레임 끝 노드 변위; 증분 꼴은 이 에너지를 걸음 끝에 버림 -- k 없는 판 (γ̄ 0) 은 같은 면·k = 0.1 (2μ+λ) h 로 잰 참고 값). "
          "보존 판: Σ (E_end − E_start) = 걸음 안 벌점 일 (되돌려 받는 몫) · Σ (E_end(f) − E_start(f+1)) = 다시 모으기 (격자·면 집합 바뀜) 로 잃는 몫 · E_end 최대.\n")
    print("| 판 | 씬 | 플래그 | 프레임 | k | 총 에너지 처음 → 끝 | 소산 | Σ W_inc (소산 대비) | 보존: 걸음 안 일 · 다시 모으기 손실 · E_end 최대 | KE 최대 · 끝 | 주기 비 · 진폭 비 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    print("\n".join(out))
    print()
    for n, c in curves.items():
        print(f"- {n} 총 에너지 (10 프레임마다): " + " ".join(f"{f}:{t:.4g}" for f, t in c if f % 10 == 0 or f == 1))
    print()

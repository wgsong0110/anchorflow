"""절 S.11 표: lego L000 발산 판 + (B) K 2 에 --gq_mass 끔 (기존) | 켬 (exe/stab_gqm.sh 출력 디렉토리 B -> gqm_table.md 를 stdout 으로).
판 디렉토리: <판>_off_w = 원래 명령 + 진단, <판>_on_w = + --gq_mass --gq_chk + 진단, <판>_on_o = + --gq_mass --fhash (진단 불변 확인), z_{off,on}_{a,b} = K 2 단독 시간 (비결정).
노드 질량 비 = m_n / 그 프레임 m_n > 0 노드 질량 중앙값 (wd_f<t>.npz n_m = k4why_diag 와 같은 정의; 셈은 m_n > 0 노드 중).
"""
import json
import os
import sys

import numpy as np

B = sys.argv[1]
RUNS = [("K4", "(B) 장벽 K 4"), ("K6", "(B) 장벽 K 6"), ("K8", "(B) 장벽 K 8"), ("K12", "(B) 장벽 K 12"), ("K16", "(B) 장벽 K 16"),
        ("fa", "(A) 원래 고착"), ("fu1", "(A)(i) 위 허용"), ("fu2", "(A)(ii) 위 허용"), ("vs", "−z 저항 강 k 9"), ("vw", "−z 저항 약 k 0.9"), ("nc", "노드 −z 클램프"),
        ("K2", "(B) 장벽 K 2 (채택)")]
# 기존 기록 (발산 프레임, 끝난 프레임 평균 %): stab_k4diag (K 지도·k4a), stab_k4why k2o, fem_cm astick·vres·nclamp (= divwhy <판>o 재현; nc 만 divwhy 재현이 54 · 26.37)
REC = {"K4": (68, 3.83), "K6": (48, 6.75), "K8": (44, 10.40), "K12": (10, 7.35), "K16": (39, 6.81), "K2": (None, 3.62),
       "fa": (23, 1.46), "fu1": (26, 1.92), "fu2": (33, 11.92), "vs": (29, 9.69), "vw": (28, 9.20), "nc": (53, 25.72)}
TH = (1e-2, 1e-4)


def jl(d):
    p = os.path.join(B, d, "fem.jsonl")
    out = []
    if os.path.exists(p):
        for l_ in open(p):
            try:
                out.append(json.loads(l_))
            except Exception:
                pass
    return out


def sm(d):
    try:
        return json.load(open(os.path.join(B, d, "summary.json")))
    except Exception:
        return None


def npz_list(d):
    p = os.path.join(B, d)
    if not os.path.isdir(p):
        return []
    fs = sorted([f for f in os.listdir(p) if f.startswith("wd_f") and f.endswith(".npz")], key=lambda f: int(f[4:-4]))
    return [(int(f[4:-4]), os.path.join(p, f)) for f in fs]


def snapname(si, nt):
    if si < 0:
        return "F_q"
    k, ph = si // (nt + 2), si % (nt + 2)
    return f"걸음 {k + 1} " + ("시작점" if ph == 0 else (f"뉴턴 {ph} 끝" if ph <= nt else "끝 U"))


def nodepos(nidx, ids):
    o_ = np.argsort(nidx); ns_ = nidx[o_]
    ids = np.asarray(ids, dtype=np.int64)
    pos = np.clip(np.searchsorted(ns_, ids), 0, len(ns_) - 1)
    return np.where(ns_[pos] == ids, o_[pos], -1)


def scan(d):
    """프레임별 거의 빈 노드 수·최소 질량 비, 처음 뒤집힌 가우스 점 (빈 포함·찬 팔분)."""
    per, first_any, first_fill = [], None, None
    for t, p in npz_list(d):
        try:
            z = np.load(p)
            m = z["n_m"].astype(np.float64)
        except Exception:
            continue
        per.append((t, int(len(m)), int((m < TH[0]).sum()), int((m < TH[1]).sum()), float(m.min()) if len(m) else None))
        if first_fill is not None:
            continue
        J = np.nan_to_num(z["q_J"], nan=-1e30); J0 = np.nan_to_num(z["q_J0"], nan=-1e30)
        bad = np.minimum(J.min(1), J0) <= 0
        if not bad.any():
            continue
        nt = int(z["nt"][0])
        for filled in (False, True):
            if filled and first_fill is not None or (not filled and first_any is not None):
                continue
            b = bad & (z["q_np"] > 0) if filled else bad
            q = np.nonzero(b)[0]
            if not len(q):
                continue
            fs = np.where(J0[q] <= 0, -1, np.argmax(J[q] <= 0, 1))
            s0 = int(fs.min()); qq = q[fs == s0]
            i0 = qq[np.argmin(J[qq, s0] if s0 >= 0 else J0[qq])]
            nid = nodepos(z["n_idx"].astype(np.int64), z["q_ids"][i0])
            nm = np.where(nid >= 0, z["n_m"][np.maximum(nid, 0)], 0.0)
            rec = dict(t=t, snap=snapname(s0, nt), n=int(len(qq)), cell=z["q_cell"][i0].tolist(), oc=int(z["q_oc"][i0]), x=[round(float(v), 3) for v in z["q_x"][i0]],
                       h=float(z["q_h"][i0]), np=int(z["q_np"][i0]), nc=int(z["q_nc"][i0]), V=float(z["q_V"][i0]), J0=float(J0[i0]),
                       J=float(J[i0, s0]) if s0 >= 0 else float(J0[i0]), nmin=float(nm.min()), nfr=int(b.sum()))
            if filled:
                first_fill = rec
            else:
                first_any = rec
    return per, first_any, first_fill


def f3(v, n=2):
    if v is None:
        return "-"
    v = float(v)
    if not np.isfinite(v):
        return str(v)
    if v != 0 and (abs(v) >= 1e4 or abs(v) < 1e-2):
        return f"{v:.1e}"
    return f"{v:.{n}f}"


def res(s):
    if s is None:
        return "없음"
    dv = s.get("diverged")
    e = s.get("err_mean_done")
    return (f"발산 {dv}" if dv else "완주") + f" · {f3(100 * e) if e is not None else '-'} %"


def inv(r):
    if r is None:
        return "없음"
    return (f"프레임 {r['t']} {r['snap']} (점 {r['n']}, 프레임 합 {r['nfr']}) 칸 {tuple(r['cell'])}·{r['oc']} 높이 {r['h']:.3f} m, 칸 입자 {r['nc']} · 팔분 {r['np']}, "
            f"부피 {f3(r['V'])} x, det {f3(r['J0'])} -> {f3(r['J'])}, 27 노드 최소 질량 비 {f3(r['nmin'])}")


def summ_per(per, upto=None):
    if not per:
        return "-", "-", "-"
    if upto:
        per = [p for p in per if p[0] <= upto] or per
    a1 = [p[2] for p in per]; a4 = [p[3] for p in per]; mn = [p[4] for p in per if p[4] is not None]
    f1 = per[0]
    return (f"{f1[2]} / {int(np.median(a1))} / {max(a1)}", f"{f1[3]} / {int(np.median(a4))} / {max(a4)}", f"{f3(f1[4])} / {f3(min(mn)) if mn else '-'}")


print("# 절 S.11 lego L000 발산 판에 --gq_mass (질량·운동량을 입자 -> 팔분 가우스 점 -> 노드, 절 CM9) 끔 (기존) | 켬 (exe/stab_gqm.sh, 모두 결정적 --det)\n")
print("끔 = 원래 명령 그대로 다시 (진단 --wdiag 1 --pdiag --fhash, 결과 불변), 켬 = 같은 명령 + --gq_mass (+ --gq_chk 진단). 노드 질량 비 = m_n / 그 프레임 m_n>0 노드 질량 중앙값, 셈은 m_n>0 노드 중.")
print("거의 빈 노드 칸 = 프레임 1 / 프레임 중앙값 / 최대 (발산 전 프레임까지). 처음 뒤집힌 가우스 점 = det X ≤ 0 (X = F_q + ∇y F_q; 걸음 시작점·뉴턴 끝·걸음 끝 U 스냅샷), 찬 팔분 = 프레임 시작에 입자가 든 팔분.\n")

ROWS = {}
for r, nm in RUNS:
    ROWS[r] = dict(off=sm(r + "_off_w"), on=sm(r + "_on_w"), ono=sm(r + "_on_o"), offj=jl(r + "_off_w"), onj=jl(r + "_on_w"), onoj=jl(r + "_on_o"))
    ROWS[r]["offs"] = scan(r + "_off_w")
    ROWS[r]["ons"] = scan(r + "_on_w")

print("## 1. 판별 끔 (기존) | 켬\n")
print("| 판 | 끔: 결과 (기록: 발산 · %) | 켬: 결과 | 질량 비 < 1e-2 노드 끔 | 켬 | < 1e-4 끔 | 켬 | 최소 질량 비 (프레임 1 / 전체) 끔 | 켬 |")
print("|---|---|---|---|---|---|---|---|---|")
for r, nm in RUNS:
    R = ROWS[r]
    rec = REC.get(r)
    rs = f" (기록 {('발산 ' + str(rec[0])) if rec[0] else '완주'} · {rec[1]:.2f})" if rec else ""
    o1, o4, om = summ_per(R["offs"][0]); n1, n4, nmn = summ_per(R["ons"][0])
    print(f"| {r} {nm} | {res(R['off'])}{rs} | {res(R['on'])} | {o1} | {n1} | {o4} | {n4} | {om} | {nmn} |")

print("\n## 2. 처음 뒤집힌 가우스 점 (찬 팔분; 괄호 빈 포함)\n")
print("| 판 | 끔 | 켬 |")
print("|---|---|---|")
for r, nm in RUNS:
    R = ROWS[r]
    oa, of_ = R["offs"][1], R["offs"][2]; na, nf = R["ons"][1], R["ons"][2]
    sa = lambda a: f" (빈 포함 프레임 {a['t']} {a['snap']})" if a else " (빈 포함 없음)"
    print(f"| {r} | {inv(of_)}{sa(oa)} | {inv(nf)}{sa(na)} |")

print("\n## 3. 검사 -- 켬 판 진단 불변 (on_o 대 on_w 프레임별 해시), 질량·운동량 합 차 (gqchk 최대), 끔 판 기록 대조\n")
print("| 판 | 켬 해시 같음 (o 대 w) | 켬 o 결과 | gqchk 질량 · 운동량 · 가속도 몫 최대 상대 차 | 끔 끝 해시 | 켬 끝 해시 |")
print("|---|---|---|---|---|---|")
for r, nm in RUNS:
    R = ROWS[r]
    a, b = R["onoj"], R["onj"]
    n = min(len(a), len(b))
    eq = sum(1 for i in range(n) if a[i].get("xh") == b[i].get("xh"))
    gq = [x.get("gqchk") for x in b if x.get("gqchk")]
    gm = [max((g[i] for g in gq if g[i] is not None and np.isfinite(g[i])), default=None) for i in range(3)] if gq else [None] * 3
    print(f"| {r} | {eq} / {n} (길이 {len(a)}·{len(b)}) | {res(R['ono'])} | {' · '.join(f3(v) for v in gm)} | {(R['off'] or {}).get('x_sha1', '-')} | {(R['on'] or {}).get('x_sha1', '-')} |")

print("\n## 4. (B) K 2 단독 시간 (비결정 = --det 뺌, 카드 0, 노드에 다른 판 없음; 프레임당 ms 재생 GPU 시간)\n")
print("| 판 | 결과 | 프레임당 ms 평균 · p90 · 최소 | 그 카드 계산 프로세스 수 최대 (10 초마다) |")
print("|---|---|---|---|")
for n in ("z_off_a", "z_on_a", "z_off_b", "z_on_b"):
    s = sm(n)
    gp = os.path.join(B, n, "gpu_procs.txt")
    mx = max([int(l_.split()[1]) for l_ in open(gp) if len(l_.split()) > 1] or [0]) if os.path.exists(gp) else "-"
    print(f"| {n} | {res(s)} | {f3(s['ms_mean']) if s else '-'} · {f3(s.get('ms_p90')) if s else '-'} · {f3(s.get('ms_min')) if s else '-'} | {mx} |")

print("\n## 5. 프레임별 거의 빈 노드 수 (< 1e-2 / < 1e-4) · 최소 질량 비 -- 끔 | 켬\n")
for r, nm in RUNS:
    R = ROWS[r]
    don = {p[0]: p for p in R["ons"][0]}; dof = {p[0]: p for p in R["offs"][0]}
    ts = sorted(set(don) | set(dof))
    if not ts:
        continue
    print(f"- {r} {nm}: " + ", ".join(f"{t}: " + (f"{dof[t][2]}/{dof[t][3]}/{f3(dof[t][4])}" if t in dof else "-") + " | " + (f"{don[t][2]}/{don[t][3]}/{f3(don[t][4])}" if t in don else "-") for t in ts))

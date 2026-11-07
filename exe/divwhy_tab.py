"""절 S.10 표: lego L000 에서 발산한 판 전부 -- "발산 = (1) 아주 가벼운 노드에서 (2) 칸이 뒤집히고 (3) 큰 복원력을 받아 (4) 발산" 인가 (exe/stab_divwhy.sh 출력 디렉토리 B 를 받아 divwhy_table.md 를 stdout 으로).
판 디렉토리: <판>o = 원래 명령 + --fhash (진단 끔), <판>w = + --wdiag 1 (모든 프레임 원자료 wd_f<t>.npz) --pdiag --diag --fhash, ref/<판> = 기존 기록 (R2 의 fem.jsonl·summary.json).
"""
import json
import os
import sys

import numpy as np

B = sys.argv[1]
RUNS = [("k4", "(B) K 4"), ("fa", "(A) 원래 FCR"), ("fu1", "(A)(i) FCR"), ("fu2", "(A)(ii) FCR"), ("sa", "(A) SNH"), ("su1", "(A)(i) SNH"), ("su2", "(A)(ii) SNH"),
        ("vs", "−z 저항 강 k 9"), ("vw", "−z 저항 약 k 0.9"), ("nc", "노드 법선 클램프")]
RUNS = [r for r in RUNS if os.path.isdir(os.path.join(B, r[0] + "w")) or os.path.isdir(os.path.join(B, r[0] + "o"))]
LIGHT, VLIGHT, HEAVY = 1e-2, 1e-3, 0.1                                                 # 노드 질량 / 중앙값 문턱 (아주 가벼움 1e-2, 극히 1e-3, 무거움 0.1)
DIAGF = ("wdiag", "wdiag_end", "wdiag_cap", "pdiag", "diag", "fhash", "out", "gfin")


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
    p = os.path.join(B, d, "summary.json")
    try:
        return json.load(open(p))
    except Exception:
        return None


def f3(v, n=3):
    if v is None:
        return "-"
    v = float(v)
    if not np.isfinite(v):
        return str(v)
    if v != 0 and (abs(v) >= 1e4 or abs(v) < 1e-3):
        return f"{v:.2e}"
    return f"{v:.{n}f}"


def npz(r, t):
    p = os.path.join(B, r + "w", f"wd_f{t}.npz")
    if not os.path.exists(p):
        return None
    try:
        return {k: v for k, v in np.load(p).items()}
    except Exception:
        return None


def snapname(si, nt):
    if si < 0:
        return "F_q (프레임 시작)"
    k, ph = si // (nt + 2), si % (nt + 2)
    return f"걸음 {k + 1} " + ("시작점 (관성 예측)" if ph == 0 else (f"뉴턴 {ph} 끝" if ph <= nt else "끝 U"))


def cfg(z, si):
    """스냅샷 si 의 노드 변위 (det 계산과 같은 y)."""
    nt = int(z["nt"][0]); k, ph = si // (nt + 2), si % (nt + 2)
    if ph == 0:
        return z["n_Y0"][k]
    if ph <= nt:
        return z["n_Y"][k * nt + ph - 1]
    return z["n_U"][k]


def prev_cfg(z, si):
    nt = int(z["nt"][0]); k, ph = si // (nt + 2), si % (nt + 2)
    if ph == 0:
        return z["n_U"][k - 1] if k > 0 else np.zeros_like(z["n_U"][0])
    return cfg(z, si - 1)


def nodepos(z, ids):
    """노드 상자 번호 -> n_idx 안 자리 (-1 = 질량 0)."""
    nidx = z["n_idx"].astype(np.int64)
    o_ = np.argsort(nidx); ns_ = nidx[o_]
    ids = np.asarray(ids, dtype=np.int64)
    pos = np.clip(np.searchsorted(ns_, ids), 0, len(ns_) - 1)
    ok = ns_[pos] == ids
    return np.where(ok, o_[pos], -1)


def first_inv(z, filled):
    """프레임 원자료에서 det ≤ 0 이 되는 점: (점 번호 배열, 점별 처음 스냅샷, 가장 이른 스냅샷)."""
    J = z["q_J"]; J0 = z["q_J0"]
    bad = (np.minimum(np.nan_to_num(J, nan=-1e30).min(1), np.nan_to_num(J0, nan=-1e30)) <= 0)
    if filled:
        bad &= z["q_np"] > 0
    q = np.nonzero(bad)[0]
    if not len(q):
        return q, np.array([], int), None
    Jn = np.nan_to_num(J[q], nan=-1e30)
    fs = np.where(np.nan_to_num(J0[q], nan=-1e30) <= 0, -1, np.argmax(Jn <= 0, 1))
    return q, fs, int(fs.min())


print("# 절 S.10 lego L000 발산 판 전부 -- '가벼운 노드 -> 칸 뒤집힘 -> 큰 복원력 -> 발산' 검증 (exe/stab_divwhy.sh, 모두 결정적 --det)\n")
print(f"문턱: 노드 질량 / 그 프레임 노드 질량 중앙값 < {LIGHT} = 아주 가벼움 (< {VLIGHT} 극히), ≥ {HEAVY} = 무거움. 이동·변위는 칸 (h) 단위. 찬 팔분 = 프레임 시작에 입자가 든 팔분 가우스 점, 빈 포함 = 부피 > 0 인 모든 점.\n")
print("힘: f = ∂Ψ/∂y (뉴턴 반복마다 rt_elem 직후 탄성 기울기, 복원력 = −f), Mi = meff/τ² (rt_grad 관성 대각), B = 탄성 헤시안 노드 블록 (고윳값 최대). |f|/Mi = 관성만 있을 때 그 힘이 내는 y 이동, λmax(B)/Mi = 강성 대 관성.\n")

# ---- 0. 재현·결과 불변
print("## 0. 재현 (끔 판 대 기존 기록) · 결과 불변 (켬 판 대 끔 판)\n")
print("| 판 | 기존: 프레임 · 발산 · 평균 % | 끔: 프레임 · 발산 · 평균 % | 프레임별 오차·노드 상자 같음 (끔 대 기존) | 명령 인자 차 (기존 대 끔) | 켬: 프레임 · 발산 | 해시 같음 (켬 대 끔) | 처음 다른 프레임 | 진단 끈 프레임 (용량) |")
print("|---|---|---|---|---|---|---|---|---|")
REP = {}
for r, nm in RUNS:
    rf, of, wf = jl(os.path.join("ref", r)), jl(r + "o"), jl(r + "w")
    rs, os_, ws = sm(os.path.join("ref", r)), sm(r + "o"), sm(r + "w")
    n = min(len(rf), len(of))
    eqe = sum(1 for i in range(n) if rf[i].get("err") == of[i].get("err") and rf[i].get("nn") == of[i].get("nn"))
    ad = []
    if rs and os_:
        a1, a2 = rs["args"], os_["args"]
        for k in sorted(set(a1) | set(a2)):
            if k in DIAGF:
                continue
            if a1.get(k) != a2.get(k):
                ad.append(f"{k}: {a1.get(k)}→{a2.get(k)}")
    n2 = min(len(wf), len(of))
    eqh = sum(1 for i in range(n2) if wf[i].get("xh") is not None and wf[i].get("xh") == of[i].get("xh"))
    fd = next((i + 1 for i in range(n2) if wf[i].get("xh") != of[i].get("xh")), None)
    offf = [i + 1 for i, e in enumerate(wf) if (e.get("wd") or {}).get("off")]
    g = lambda s, k: (s or {}).get(k)
    pm = lambda s: f3(100 * s["err_mean_done"], 2) if s and s.get("err_mean_done") is not None else "-"
    REP[r] = dict(same=(n > 0 and eqe == n and len(rf) == len(of) and g(rs, "diverged") == g(os_, "diverged")), inv=(n2 > 0 and eqh == n2 and len(wf) == len(of)))
    print(f"| {r} {nm} | {len(rf)} · {g(rs, 'diverged')} · {pm(rs)} | {len(of)} · {g(os_, 'diverged')} · {pm(os_)} | {eqe} / {n} | {'; '.join(ad) or '없음'} | {len(wf)} · {g(ws, 'diverged')} | {eqh} / {n2} | {fd or ''} | {(str(offf[0]) + '~' + str(offf[-1])) if offf else ''} |")
print()

VER = {}
for r, nm in RUNS:
    wf = jl(r + "w"); ws = sm(r + "w") or {}
    of = jl(r + "o"); os_ = sm(r + "o") or {}
    print(f"## {r} {nm}\n")
    if not wf:
        print("켬 판 없음\n")
        continue
    # 발산 형태
    div = os_.get("diverged") or ws.get("diverged")
    last = (of or wf)[-1]
    if os_.get("oom") or ws.get("oom"):
        form = f"노드 상자 폭증 -> GPU 메모리 부족 (프레임 {div}; 마지막 끝난 프레임 노드 상자 {last.get('nn')})"
    elif div and not np.isfinite(last.get("err", np.nan) if last.get("err") is not None else np.nan):
        form = f"비유한 (프레임 {div})"
    elif div:
        form = f"오차 > 2 (프레임 {div}, {f3(100 * last['err'], 1)} %)"
    else:
        form = "완주"
    print(f"- 발산 형태: {form}\n")
    T = sorted(int(f[4:-4]) for f in os.listdir(os.path.join(B, r + "w")) if f.startswith("wd_f") and f.endswith(".npz"))
    # 프레임별 뒤집힌 점 (찬·빈) 수
    cnt = {}
    Z = {}
    for t in T:
        z = npz(r, t)
        if z is None or "n_F" not in z:
            continue
        Jm = np.minimum(np.nan_to_num(z["q_J"], nan=-1e30).min(1), np.nan_to_num(z["q_J0"], nan=-1e30))
        inv = Jm <= 0
        fl = z["q_np"] > 0
        cnt[t] = (int((inv & fl).sum()), int((inv & ~fl).sum()))
    t_any = next((t for t in sorted(cnt) if cnt[t][0] + cnt[t][1] > 0), None)
    t_fill = next((t for t in sorted(cnt) if cnt[t][0] > 0), None)
    tl = max(cnt) if cnt else None
    t_on = None                                                                        # 마지막 사슬 시작: 그 뒤 끝까지 찬 팔분 뒤집힘이 끊이지 않는 첫 프레임
    if cnt and cnt[tl][0] > 0:
        t_on = tl
        while (t_on - 1) in cnt and cnt[t_on - 1][0] > 0:
            t_on -= 1
    print(f"- 처음 뒤집힘: 빈 포함 프레임 {t_any}, 찬 팔분 프레임 {t_fill}; 마지막 사슬 (끝까지 찬 팔분 뒤집힘 연속) 시작 프레임 {t_on}; 원자료 마지막 프레임 {tl}\n")
    res = {}
    for lab, t0, filled in (("빈 포함 처음", t_any, False), ("찬 팔분 처음", t_fill, True), ("사슬 시작 (찬 팔분)", t_on, True)):
        if t0 is None:
            continue
        if lab == "사슬 시작 (찬 팔분)" and t0 == t_fill:
            res[lab] = res.get("찬 팔분 처음")
            print(f"### {lab}: 프레임 {t0} = 찬 팔분 처음과 같음\n")
            continue
        z = npz(r, t0)
        nt = int(z["nt"][0]); K = int(z["K"][0]); h = float(z["cellh"][0]); mmed = float(z["mmed"][0])
        q, fs, s0 = first_inv(z, filled)
        sel = q[fs == s0]
        Js = np.nan_to_num(z["q_J"][sel, s0], nan=-1e30) if s0 >= 0 else z["q_J0"][sel]
        q1 = int(sel[np.argmin(Js)])
        k1 = s0 // (nt + 2) if s0 >= 0 else 0; ph1 = s0 % (nt + 2) if s0 >= 0 else -1
        ids = z["q_ids"][q1]; pos = nodepos(z, ids)
        mr = np.where(pos >= 0, z["n_m"][np.maximum(pos, 0)], 0.0)
        ok = pos >= 0
        P = pos[ok]
        # 이동: 이 스냅샷으로 들어오는 이동 (칸)
        mv_in = np.linalg.norm(cfg(z, s0) - prev_cfg(z, s0), axis=1)[P] / h if s0 >= 0 else np.zeros(len(P))
        pred = np.linalg.norm(z["n_Y0"][k1] - (z["n_U"][k1 - 1] if k1 > 0 else 0), axis=1)[P] / h
        stepd = np.linalg.norm(z["n_U"][k1] - (z["n_U"][k1 - 1] if k1 > 0 else 0), axis=1)[P] / h
        il = int(np.argmin(mr[ok])); imv = int(np.argmax(mv_in)) if len(mv_in) else 0
        allpred = np.linalg.norm(z["n_Y0"][k1] - (z["n_U"][k1 - 1] if k1 > 0 else 0), axis=1) / h
        Lm_all = z["n_m"] < LIGHT
        print(f"### {lab}: 프레임 {t0}, {snapname(s0, nt)} (같은 시점 뒤집힌 점 {len(sel)} 개, 이 프레임 {('찬 팔분' if filled else '빈 포함')} 뒤집힘 {len(q)} 개)\n")
        print(f"- 점: 칸 {tuple(int(v) for v in z['q_cell'][q1])} 팔분 {int(z['q_oc'][q1])}, x {tuple(round(float(v), 3) for v in z['q_x'][q1])} m, 바닥 위 {f3(z['q_h'][q1])} m ({f3(z['q_h'][q1] / h, 1)} 칸), "
              f"팔분 입자 {int(z['q_np'][q1])} · 칸 입자 {int(z['q_nc'][q1])}, 부피/중앙 {f3(z['q_V'][q1])}, F_q det {f3(z['q_J0'][q1])}")
        print(f"- 그 점 det (스냅샷 차례): " + ", ".join(f"{snapname(i, nt)} {f3(v)}" for i, v in enumerate(z["q_J"][q1])))
        qs = np.quantile(mr[ok], [0.1, 0.5, 0.9]) if ok.any() else [np.nan] * 3
        print(f"- 27 노드 질량 / 중앙값: 최소 {f3(mr[ok].min())} · 10 % {f3(qs[0])} · 중앙 {f3(qs[1])} · 90 % {f3(qs[2])} · 최대 {f3(mr[ok].max())}; < {VLIGHT} {int((mr[ok] < VLIGHT).sum())} 개 · < {LIGHT} {int((mr[ok] < LIGHT).sum())} · < {HEAVY} {int((mr[ok] < HEAVY).sum())} / {int(ok.sum())} (질량 0 노드 {int((~ok).sum())})")
        print(f"- 27 노드 이동 (칸): 이 스냅샷으로 들어온 이동 최대 {f3(mv_in.max())} (그 노드 질량 {f3(mr[ok][imv])}) · 가장 가벼운 노드 {f3(mv_in[il])}; 걸음 {k1 + 1} 시작점 예측 이동 최대 {f3(pred.max())} (가장 가벼운 노드 {f3(pred[il])}, 무거운 노드 (≥ {HEAVY}) 최대 {f3(pred[mr[ok] >= HEAVY].max() if (mr[ok] >= HEAVY).any() else None)}); "
              f"걸음 {k1 + 1} 변위 최대 {f3(stepd.max())} (가장 가벼운 {f3(stepd[il])})")
        print(f"- 비교: 이 프레임 걸음 {k1 + 1} 시작점 예측 이동 -- 전체 노드 중앙 {f3(np.median(allpred))} · 99 % {f3(np.percentile(allpred, 99))}, 아주 가벼운 노드 (< {LIGHT}) 중앙 {f3(np.median(allpred[Lm_all]) if Lm_all.any() else None)} · 최대 {f3(allpred[Lm_all].max() if Lm_all.any() else None)}, 무거운 노드 최대 {f3(allpred[z['n_m'] >= HEAVY].max())}")
        # 같은 시점 뒤집힌 점 전부: 이웃 최소 질량 분포
        mins = []
        for qq in q:
            pp = nodepos(z, z["q_ids"][qq]); mm = np.where(pp >= 0, z["n_m"][np.maximum(pp, 0)], 0.0)
            mins.append(mm.min())
        mins = np.array(mins)
        print(f"- 이 프레임 뒤집힌 점 {len(q)} 개의 27 노드 최소 질량: < {LIGHT} {int((mins < LIGHT).sum())} 개 · {LIGHT}~{HEAVY} {int(((mins >= LIGHT) & (mins < HEAVY)).sum())} · ≥ {HEAVY} {int((mins >= HEAVY).sum())}; 처음 시점이 걸음 시작점 {int(sum(1 for v in fs if v >= 0 and v % (nt + 2) == 0))} · 뉴턴 끝 {int(sum(1 for v in fs if v >= 0 and 0 < v % (nt + 2) <= nt))} · 끝 U {int(sum(1 for v in fs if v >= 0 and v % (nt + 2) == nt + 1))} · F_q {int((fs < 0).sum())}")
        # ---- (2) 뒤집힌 상태에서 받는 힘
        Fr = {}
        if s0 >= 0 and ph1 < nt:
            j = k1 * nt + ph1                                                          # 뒤집힌 상태 y 에서 계산된 힘 = 다음 뉴턴 반복 (k1, ph1+1) 의 rt_elem
            jb = j - 1 if ph1 > 0 else (k1 - 1) * nt + nt - 1 if k1 > 0 else None     # 그 앞 힘 계산 (뒤집히기 전)
            me = z["n_me"][k1][P]; tau = float(z["tau"][k1]); Mi = me / tau ** 2
            fv = z["n_F"][j][P].astype(np.float64); fn = np.linalg.norm(fv, axis=1)
            fb = np.linalg.norm(z["n_F"][jb][P].astype(np.float64), axis=1) if jb is not None else None
            mraw = z["n_mraw"][P].astype(np.float64)
            acc = fn / np.maximum(mraw, 1e-30)
            kr = z["n_Bmax"][j][P] / Mi
            yin = fn / Mi / h                                                          # 관성만이면 이 힘이 내는 y 이동 (칸)
            nxt = np.linalg.norm(cfg(z, s0 + 1) - cfg(z, s0), axis=1)[P] / h            # 이 힘으로 한 뉴턴 갱신 이동 (다음 스냅샷)
            nst = np.linalg.norm(z["n_U"][k1 + 1] - z["n_U"][k1], axis=1)[P] / h if k1 + 1 < K else None
            npred = np.linalg.norm(z["n_Y0"][k1 + 1] - z["n_U"][k1], axis=1)[P] / h if k1 + 1 < K else None
            # 전체 노드 같은 반복의 힘 (비교)
            Pall = np.arange(len(z["n_idx"]))
            fall = np.linalg.norm(z["n_F"][j].astype(np.float64), axis=1)
            Miall = z["n_me"][k1] / tau ** 2
            light = mr[ok] < LIGHT
            Jn = z["q_J"][q1]
            Fr = dict(fl=float(fn[light].max()) if light.any() else None, fmax=float(fn.max()), fmed_all=float(np.median(fall)), fb=float(fb.max()) if fb is not None else None,
                      acc_l=float(acc[light].max()) if light.any() else None, acc_h=float(acc[mr[ok] >= HEAVY].max()) if (mr[ok] >= HEAVY).any() else None,
                      yin_l=float(yin[light].max()) if light.any() else None, yin_max=float(yin.max()), kr_l=float(np.nanmax(kr[light])) if light.any() else None, kr_h=float(np.nanmedian(kr[mr[ok] >= HEAVY])) if (mr[ok] >= HEAVY).any() else None,
                      mv_in=float(mv_in.max()), nxt=float(nxt.max()), nxt_l=float(nxt[light].max()) if light.any() else None, nst=float(nst.max()) if nst is not None else None, npred=float(npred.max()) if npred is not None else None,
                      jnext=float(Jn[s0 + 1]) if s0 + 1 < len(Jn) else None)
            g9 = 9.81
            print(f"- **뒤집힌 상태에서 받는 힘** (걸음 {k1 + 1} 뉴턴 {ph1 + 1} 의 rt_elem = 이 y 에서; τ {f3(tau, 5)} s):")
            print(f"  |f| 27 노드 최대 {f3(fn.max())} (아주 가벼운 노드 최대 {f3(Fr['fl'])}), 그 앞 힘 계산 (뒤집히기 전) 최대 {f3(Fr['fb'])}, 이 반복 전체 노드 |f| 중앙 {f3(Fr['fmed_all'])};"
                  f" 가속도 |f|/m 가벼운 노드 최대 {f3(Fr['acc_l'])} m/s² ({f3((Fr['acc_l'] or 0) / g9, 1)} g) · 무거운 노드 최대 {f3(Fr['acc_h'])} ({f3((Fr['acc_h'] or 0) / g9, 1)} g)")
            print(f"  강성/관성 λmax(B)/Mi: 가벼운 노드 최대 {f3(Fr['kr_l'])} · 무거운 노드 중앙 {f3(Fr['kr_h'])}; 관성만일 때 이동 |f|/Mi: 가벼운 노드 최대 {f3(Fr['yin_l'])} 칸 · 27 최대 {f3(Fr['yin_max'])} 칸")
            print(f"  실제 이동: 뒤집히게 한 이동 최대 {f3(Fr['mv_in'])} 칸 -> 이 힘으로 한 뉴턴 갱신 이동 최대 {f3(Fr['nxt'])} 칸 (가벼운 노드 {f3(Fr['nxt_l'])}), 그 뒤 점 det {f3(Fr['jnext'])}; 다음 걸음 시작점 예측 이동 {f3(Fr['npred'])} · 다음 걸음 변위 {f3(Fr['nst'])} 칸")
            # 27 노드 표
            print("\n| 노드 (상대 i,j,k) | 질량/중앙 | 들어온 이동 칸 | 걸음 시작 예측 이동 칸 | 걸음 변위 칸 | |f| | |f|/m m/s² | λmax(B)/Mi | |f|/Mi 칸 | 다음 갱신 이동 칸 | 다음 걸음 변위 칸 |")
            print("|---|---|---|---|---|---|---|---|---|---|---|")
            oi = np.argsort(mr[ok])
            okn = np.nonzero(ok)[0]
            for ii in oi[:8]:
                o27 = int(okn[ii])
                print(f"| {o27 // 9},{(o27 // 3) % 3},{o27 % 3} | {f3(mr[ok][ii])} | {f3(mv_in[ii])} | {f3(pred[ii])} | {f3(stepd[ii])} | {f3(fn[ii])} | {f3(acc[ii])} | {f3(kr[ii])} | {f3(yin[ii])} | {f3(nxt[ii])} | {f3(nst[ii]) if nst is not None else '-'} |")
            print()
        elif s0 >= 0:
            print(f"- 뒤집힌 시점이 {snapname(s0, nt)} 라 같은 y 에서 계산된 힘이 없음 (다음 힘 계산은 다음 걸음 시작점)\n")
        res[lab] = dict(t=t0, si=s0, ph=ph1, k=k1, nt=nt, mmin=float(mr[ok].min()), mvnode_m=float(mr[ok][imv]), mv_in=float(mv_in.max()), pred=float(pred.max()), pred_l=float(pred[il]),
                        h=float(z["q_h"][q1]), hc=float(z["q_h"][q1] / h), np=int(z["q_np"][q1]), nc=int(z["q_nc"][q1]), frac_light=float((mins < LIGHT).mean()), nq=len(q), F=Fr)
    # ---- (3) 번짐
    print("### 번짐 (프레임별)\n")
    print("| 프레임 | 오차 % | 노드 상자 | 찬 팔분 뒤집힘 | 빈 팔분 뒤집힘 | 찬 뒤집힘 점 중심 x (m) | 높이 최소·중앙·최대 (m) | 첫 점에서 거리 중앙·최대 (m) | 그 점들 27 노드 최소 질량 < 1e-2 비율 | 입자 det ≤ 0 (pd) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    x1 = None
    if t_fill is not None:
        z1 = npz(r, t_fill); q_, fs_, s_ = first_inv(z1, True)
        x1 = z1["q_x"][q_[fs_ == s_][0]]
    tshow = [t for t in sorted(cnt) if t >= (t_any or 1) - 1]
    if len(tshow) > 40:
        keep = set(tshow[:5]) | set(tshow[-25:]) | set(t for t in tshow if cnt[t][0] > 0)
        tshow = sorted(keep)[-45:]
    for t in tshow:
        e = wf[t - 1] if t - 1 < len(wf) else {}
        z = npz(r, t)
        Jm = np.minimum(np.nan_to_num(z["q_J"], nan=-1e30).min(1), np.nan_to_num(z["q_J0"], nan=-1e30))
        qq = np.nonzero((Jm <= 0) & (z["q_np"] > 0))[0]
        if len(qq):
            xs = z["q_x"][qq]; hs = z["q_h"][qq]
            dd = np.linalg.norm(xs - x1, axis=1) if x1 is not None else np.array([np.nan])
            mins = []
            for qv in qq[:3000]:
                pp = nodepos(z, z["q_ids"][qv]); mins.append(np.where(pp >= 0, z["n_m"][np.maximum(pp, 0)], 0.0).min())
            fr = f"{f3(100 * float((np.array(mins) < LIGHT).mean()), 0)} %"
            c_ = f"{tuple(round(float(v), 2) for v in xs.mean(0))} | {f3(hs.min(), 2)}·{f3(np.median(hs), 2)}·{f3(hs.max(), 2)} | {f3(np.median(dd), 2)}·{f3(dd.max(), 2)} | {fr}"
        else:
            c_ = "| | |"
        pdn = (e.get("pd") or {}).get("nJneg")
        print(f"| {t} | {f3(100 * e.get('err', np.nan), 2)} | {e.get('nn')} | {cnt[t][0]} | {cnt[t][1]} | {c_} | {pdn if pdn is not None else '-'} |")
    # 원자료 없는 뒤 프레임 (진단 끈 프레임)
    for t in range((tl or 0) + 1, len(wf) + 1):
        e = wf[t - 1]
        w = e.get("wd") or {}
        print(f"| {t} | {f3(100 * e.get('err', np.nan), 2)} | {e.get('nn')} | (진단 끔: 용량 {w.get('capN')}) | | | | | | {(e.get('pd') or {}).get('nJneg', '-')} |")
    print()
    pdf = next((i + 1 for i, e in enumerate(wf) if (e.get("pd") or {}).get("nJneg", 0) > 0), None)
    VER[r] = dict(nm=nm, form=form, div=div, res=res, t_any=t_any, t_fill=t_fill, t_on=t_on, pdf=pdf, cnt=cnt)

# ---- 4. 판별
print("## 판별 (판마다 네 단계)\n")
print(f"(1) 가벼운 노드: 사슬 시작 점의 27 노드 최소 질량 < {LIGHT} x 중앙 이고 뒤집히게 한 가장 큰 이동이 가벼운 노드 (< {HEAVY}) 에서 · (2) 칸 뒤집힘이 입자 det ≤ 0 보다 먼저 · (3) 뒤집힌 y 에서 가벼운 노드 |f|/Mi > 1 칸 (관성으로 못 버팀) · (4) 그 힘으로 한 갱신 이동 > 뒤집히게 한 이동 (더 크게 튐) 그리고 사슬이 끊기지 않고 발산까지\n")
print("| 판 | 발산 | 사슬 시작 프레임 · 시점 | (1) 27 노드 최소 질량 · 큰 이동 노드 질량 · 예측 이동 칸 | (2) 점 뒤집힘 → 입자 det≤0 프레임 | (3) 가벼운 노드 |f|/Mi 칸 · λmax(B)/Mi · 가속도 g | (4) 뒤집게 한 이동 → 갱신 이동 칸 (갱신 뒤 점 det) · 다음 걸음 변위 | 판정 (1)(2)(3)(4) | 다른 경로 |")
print("|---|---|---|---|---|---|---|---|---|")
for r, v in VER.items():
    R = v["res"].get("사슬 시작 (찬 팔분)") or v["res"].get("찬 팔분 처음")
    if not R:
        print(f"| {r} {v['nm']} | {v['form']} | 찬 팔분 뒤집힘 없음 | | | | | | |")
        continue
    F = R["F"] or {}
    c1 = R["mmin"] < LIGHT and R["mvnode_m"] < HEAVY
    c2 = v["pdf"] is None or v["t_fill"] <= v["pdf"]
    c3 = (F.get("yin_l") or 0) > 1.0
    c4 = F.get("nxt") is not None and F["nxt"] > R["mv_in"] and v["t_on"] is not None
    other = []
    if R["mmin"] >= HEAVY:
        other.append("무거운 노드에서 시작")
    if R["hc"] < 2:
        other.append(f"바닥 가까이 ({f3(R['hc'], 1)} 칸)")
    if R["ph"] is not None and R["ph"] > 0:
        other.append("뉴턴 갱신 중 뒤집힘")
    yn = lambda b: "예" if b else "아니오"
    print(f"| {r} {v['nm']} | {v['form']} | {R['t']} · {snapname(R['si'], R['nt'])} | {f3(R['mmin'])} · {f3(R['mvnode_m'])} · {f3(R['pred'])} | {v['t_fill']} → {v['pdf']} | {f3(F.get('yin_l'))} · {f3(F.get('kr_l'))} · {f3((F.get('acc_l') or 0) / 9.81, 1)} | "
          f"{f3(R['mv_in'])} → {f3(F.get('nxt'))} (뒤 det {f3(F.get('jnext'))}) · {f3(F.get('nst'))} | {yn(c1)} {yn(c2)} {yn(c3)} {yn(c4)} | {', '.join(other) or '-'} |")
print()
print("재현·불변: " + ", ".join(f"{r} 재현 {'같음' if REP[r]['same'] else '다름'} · 불변 {'같음' if REP[r]['inv'] else '다름'}" for r, _ in RUNS))

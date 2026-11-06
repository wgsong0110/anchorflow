"""절 S.9 표: 우리 (B) L000 K 4 발산 -- 몸통 안 가우스 점이 왜 뒤집히나 (exe/stab_k4why.sh 출력 디렉토리 하나를 받아 k4why_table.md 를 stdout 으로).
판 (디렉토리 이름): k4w·k2w·hsw·mvw = 진단 켬 (--wdiag --gfin --diag --pdiag --fhash --dump --dump_part), k4o·k2o·hso·mvo = 진단 끔 (--fhash 만, 호스트 해시).
"""
import glob
import json
import os
import sys

import numpy as np

B = sys.argv[1]
W0 = int(sys.argv[2]) if len(sys.argv) > 2 else 40
RUNS = [("k4w", "K 4 (B) 기준"), ("k2w", "K 2 (완주)"), ("hsw", "K 4 --hspec --ncomp 끔 (반올림만)"), ("mvw", "K 4 --mv2 까지 끔 (반올림만)")]
OFFS = {"k4w": "k4o", "k2w": "k2o", "hsw": "hso", "mvw": "mvo"}


def jl(r):
    p = os.path.join(B, r, "fem.jsonl")
    if not os.path.exists(p):
        return []
    out = []
    for l_ in open(p):
        try:
            out.append(json.loads(l_))
        except Exception:
            pass
    return out


def sm(r):
    p = os.path.join(B, r, "summary.json")
    return json.load(open(p)) if os.path.exists(p) else None


def f3(v, n=3):
    if v is None:
        return ""
    if isinstance(v, float) and not np.isfinite(v):
        return str(v)
    if abs(v) >= 1e4 or (v != 0 and abs(v) < 1e-3):
        return f"{v:.2e}"
    return f"{v:.{n}f}"


J = {r: jl(r) for r, _ in RUNS}
for r in OFFS.values():
    J[r] = jl(r)
print("# 절 S.9 우리 (B) lego L000 K 4 -- 몸통 안 가우스 점 뒤집힘의 원인 (exe/stab_k4why.sh, 모두 결정적 --det)\n")
# ---- 0. 결과 불변 확인
print("## 0. 진단 켬·끔 대조 (프레임별 입자 위치 해시 xh)\n")
print("| 판 | 켬 프레임 · 발산 | 끔 프레임 · 발산 | 같은 해시 프레임 / 비교 | 처음 다른 프레임 | 오차 % (끝난 프레임 평균) 켬 / 끔 | 끝 해시 켬 / 끔 |")
print("|---|---|---|---|---|---|---|")
for r, nm in RUNS:
    o = OFFS[r]
    a_, b_ = J[r], J[o]
    sa, sb = sm(r), sm(o)
    n = min(len(a_), len(b_))
    eq = sum(1 for i in range(n) if a_[i].get("xh") == b_[i].get("xh"))
    fd = next((i + 1 for i in range(n) if a_[i].get("xh") != b_[i].get("xh")), None)
    g = lambda s, k: (s or {}).get(k)
    print(f"| {r} {nm} | {len(a_)} · {g(sa, 'diverged')} | {len(b_)} · {g(sb, 'diverged')} | {eq} / {n} | {fd or ''} | {f3(100 * g(sa, 'err_mean_done')) if sa else ''} / {f3(100 * g(sb, 'err_mean_done')) if sb else ''} | {g(sa, 'x_sha1')} / {g(sb, 'x_sha1')} |")
print()


def wd(r, t):
    a_ = J[r]
    return a_[t - 1].get("wd") if 0 < t <= len(a_) else None


# ---- 1. 프레임별 요약 (K 4 대 K 2 대 반올림 판)
print(f"## 1. 프레임별 (K 4 기준 · K 2 · 반올림 판) -- 프레임 {W0}~끝\n")
print("열: 오차 % / 가우스 점 det 최소 (프레임 안 모든 시점) / F_q det 최소 (프레임 시작, P2G 평균) / det ≤ 0 점 수 / 입자 det 최소 (프레임 시작 → 끝) / 수렴한 걸음 (걸음 끝 |g|/F ≤ εN) / 걸음 끝 |g|/F 최대 / PCG 상한 도달 풀이 / 선탐색 보폭 최소\n")
for r, nm in RUNS:
    if not J[r]:
        continue
    print(f"### {r} {nm}\n")
    print("| 프레임 | 오차 % | 점 det 최소 | F_q det 최소 | det≤0 점 | det<0.1 점 | 입자 det 시작 → 끝 (≤0 수) | 수렴 걸음 | 걸음 끝 |g|/F (걸음별) | 뉴턴 처음 |g|/F (걸음별) | PCG 상한 / 풀이 | 보폭 최소 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for t in range(W0, len(J[r]) + 1):
        e = J[r][t - 1]; w = e.get("wd")
        if not w:
            continue
        caps = sum(q.get("cap", 0) for q in w["it"] if q["ran"]); ns = sum(1 for q in w["it"] if q["ran"])
        g0 = [q["gF"] for q in w["it"] if q["i"] == 0]
        print(f"| {t} | {f3(100 * e['err'])} | {f3(w['Jqmin'])} | {f3(w['Jq0min'])} | {w['nqinv']} | {w['nqlt01']} | {f3(w['Jp0min'])} → {f3(w['Jp1min'])} ({w['nJp0le0']}→{w['nJp1le0']}) | {w['nconv']}/{len(w['gend'])} | "
              f"{' '.join(f3(q, 2) for q in w['gend'])} | {' '.join(f3(q, 2) for q in g0)} | {caps}/{ns} | {f3(w['st_min'])} |")
    print()
# ---- 1b. 앞 구간 (프레임 1 ~ W0−1) 요약: 덜 수렴이 언제부터 쌓이나
print(f"## 1b. 프레임 1~{W0 - 1} 평균 (같은 열, 비교용)\n")
print("| 판 | 프레임 | 걸음 끝 |g|/F 중앙값 · 최대 | 수렴 걸음 비율 | PCG 상한 비율 | 보폭 < 1 반복 비율 | 점 det 최소 (최소 프레임) | det≤0 점 생긴 프레임 |")
print("|---|---|---|---|---|---|---|---|---|")
for r, nm in RUNS:
    for lo_, hi_ in ((1, W0 - 1), (W0, 10 ** 6)):
        ws = [(i + 1, e["wd"]) for i, e in enumerate(J[r]) if e.get("wd") and lo_ <= i + 1 <= hi_]
        if not ws:
            continue
        ge = [q for _, w in ws for q in w["gend"] if q is not None]
        nc = sum(w["nconv"] for _, w in ws); nst = sum(len(w["gend"]) for _, w in ws)
        its = [q for _, w in ws for q in w["it"] if q["ran"]]
        cap = sum(q.get("cap", 0) for q in its)
        sl = sum(1 for q in its if q.get("st") is not None and q["st"] < 1)
        jm = min(ws, key=lambda x: x[1]["Jqmin"])
        inv = [t for t, w in ws if w["nqinv"] > 0]
        print(f"| {r} | {lo_}~{min(hi_, len(J[r]))} | {f3(float(np.median(ge)), 2) if ge else ''} · {f3(max(ge), 2) if ge else ''} | {nc}/{nst} | {cap}/{len(its)} | {sl}/{len(its)} | {f3(jm[1]['Jqmin'])} ({jm[0]}) | {inv[:12]}{' …' if len(inv) > 12 else ''} |")
print()


# ---- 2. 처음 뒤집히는 가우스 점 (K 4)
_NZ = {}


def npz(r, t):
    if (r, t) not in _NZ:
        p = os.path.join(B, r, f"wd_f{t}.npz")
        _NZ[(r, t)] = ({k: v for k, v in np.load(p).items()} if os.path.exists(p) else None)
        if len(_NZ) > 40:                                                              # 메모리: 오래된 것 버림
            _NZ.pop(next(iter(_NZ)))
    return _NZ[(r, t)]


_KZ = {}


def kz_of(r, t, zz):
    if (r, t) not in _KZ:
        _KZ[(r, t)] = skey(zz["q_cell"], zz["q_oc"])
    return _KZ[(r, t)]


def skey(c, o):
    c = c.astype(np.int64) + 4096
    return ((c[:, 0] * 8192 + c[:, 1]) * 8192 + c[:, 2]) * 8 + o.astype(np.int64)


def snapname(si, nt):
    k, ph = si // (nt + 2), si % (nt + 2)
    return f"걸음 {k + 1} " + ("시작점" if ph == 0 else (f"뉴턴 {ph} 끝" if ph <= nt else "끝 U"))


fs = sorted(int(os.path.basename(p)[4:-4]) for p in glob.glob(os.path.join(B, "k4w", "wd_f*.npz")))
tstar = None
for t in fs:
    z = npz("k4w", t)
    if ((np.minimum(z["q_J"].min(1), z["q_J0"]) <= 0) & (z["q_np"] > 0)).any():
        tstar = t
        break
print("## 2a. 뒤집힌 점: 입자 있는 팔분 (찬) 대 빈 팔분 -- 판별 창 프레임 (찬 팔분 뒤집힘만 G2P 로 입자 F 에 옮겨짐)\n")
print("| 판 | 찬 팔분 뒤집힘 (프레임: 수) | 빈 팔분 뒤집힘 합 |")
print("|---|---|---|")
for r, nm in RUNS:
    fr, emp = [], 0
    for t in range(W0, 69):
        zz = npz(r, t)
        if zz is None:
            continue
        Jm_ = np.minimum(zz["q_J"].min(1), zz["q_J0"]); oc_ = zz["q_np"] > 0
        n_ = int(((Jm_ <= 0) & oc_).sum()); emp += int(((Jm_ <= 0) & ~oc_).sum())
        if n_:
            fr.append(f"{t}: {n_}")
    print(f"| {r} | {', '.join(fr[:8])}{' …' if len(fr) > 8 else ''} | {emp} |")
print()
print("## 2b. 가벼운 노드 (질량 < 1e-3 x 중앙값) 대 무거운 노드 (≥ 0.1 x) -- 걸음 2.. 시작점 (관성 예측) 이동 |y0_k − U_k−1|·걸음 증분 |U_k − U_k−1| 최대 (칸)\n")
print("| 판 | 프레임 | 시작점 이동 가벼운 · 무거운 | 걸음 증분 가벼운 · 무거운 | 1 칸 넘게 움직인 가벼운 노드 |")
print("|---|---|---|---|---|")
for r, nm in RUNS[:3]:
    for t in range(max(W0, 46), 57):
        zz = npz(r, t)
        if zz is None:
            continue
        K_ = int(zz["K"][0]); h_ = float(zz["cellh"][0]); U_ = zz["n_U"]; Y_ = zz["n_Y0"]; m_ = zz["n_m"]
        pm_ = np.max([np.linalg.norm(Y_[k] - U_[k - 1], axis=1) / h_ for k in range(1, K_)], 0)
        im_ = np.max([np.linalg.norm(U_[k] - (U_[k - 1] if k else 0), axis=1) / h_ for k in range(K_)], 0)
        Lm, Hm = m_ < 1e-3, m_ >= 0.1
        print(f"| {r} | {t} | {f3(float(pm_[Lm].max()), 2)} · {f3(float(pm_[Hm].max()), 2)} | {f3(float(im_[Lm].max()), 2)} · {f3(float(im_[Hm].max()), 2)} | {int((im_[Lm] > 1).sum())} |")
print()
print("## 2. K 4 에서 처음 뒤집히는 찬 팔분 가우스 점 (창 안 첫 프레임; 빈 팔분 점은 2a 에만)\n")
if tstar is None:
    print("창 안에 det ≤ 0 점 없음\n")
else:
    z = npz("k4w", tstar)
    nt = int(z["nt"][0]); K = int(z["K"][0]); NS = z["q_J"].shape[1]
    Jm = np.minimum(z["q_J"].min(1), z["q_J0"])
    bad = np.nonzero((Jm <= 0) & (z["q_np"] > 0))[0]
    first_si = [int(np.argmax(z["q_J"][q] <= 0)) if (z["q_J"][q] <= 0).any() else -1 for q in bad]
    s_first = min(first_si) if first_si else -1
    print(f"- 프레임 {tstar}: det ≤ 0 점 {len(bad)} 개 (부피 > 0 점 {len(Jm)} 중), F_q (프레임 시작) det ≤ 0 {int((z['q_J0'] <= 0).sum())}, 가장 이른 시점 = {snapname(s_first, nt) if s_first >= 0 else 'F_q'}")
    cnt = {}
    for s_ in first_si:
        cnt[s_] = cnt.get(s_, 0) + 1
    print("- 점별 처음 det ≤ 0 시점 분포: " + ", ".join(f"{snapname(k_, nt) if k_ >= 0 else 'F_q'} {v}" for k_, v in sorted(cnt.items())))
    hgt = z["q_h"][bad]
    print(f"- 높이 (바닥 위, 프레임 시작 점 위치) 최소 / 중앙 / 최대 {f3(float(hgt.min()))} / {f3(float(np.median(hgt)))} / {f3(float(hgt.max()))} m, 칸 수 {len(np.unique(z['q_cell'][bad], axis=0))}")
    # 노드: 증분·질량
    nidx = z["n_idx"]; o_ = np.argsort(nidx); ns_ = nidx[o_]
    U = z["n_U"]; Y0 = z["n_Y0"]; h = float(z["cellh"][0])
    inc = np.stack([np.linalg.norm(U[k] - (U[k - 1] if k else 0), axis=1) for k in range(K)], 0) / h      # 걸음 증분 (칸)
    tot = np.linalg.norm(U[K - 1], axis=1) / h
    incm = inc.max(0)
    def nodes_of(qs):
        ii = z["q_ids"][qs].reshape(-1)
        pos = np.searchsorted(ns_, ii); pos = np.clip(pos, 0, len(ns_) - 1); ok = ns_[pos] == ii
        return o_[pos[ok]]
    nb_ = np.unique(nodes_of(bad))
    pr = lambda a, v: float((a < v).mean() * 100)
    print(f"- 그 점들의 이웃 노드 {len(nb_)} 개: 질량 / 중앙값 최소 {f3(float(z['n_m'][nb_].min()))} · 중앙 {f3(float(np.median(z['n_m'][nb_])))} (전체 노드 중앙 1, 전체 최소 {f3(float(z['n_m'].min()))}; 질량 < 0.1 x 중앙 노드 비율 이웃 {f3(100 * float((z['n_m'][nb_] < 0.1).mean()), 1)} % · 전체 {f3(100 * float((z['n_m'] < 0.1).mean()), 1)} %)")
    print(f"- 걸음 증분 |U_k − U_k−1| 최대 (칸): 이웃 노드 중앙 {f3(float(np.median(incm[nb_])))} · 최대 {f3(float(incm[nb_].max()))} (전체 노드 중앙 {f3(float(np.median(incm)))} · 99 % {f3(float(np.percentile(incm, 99)))} · 최대 {f3(float(incm.max()))}; 이웃 최대의 전체 백분위 {f3(pr(incm, incm[nb_].max()), 1)})")
    print(f"- 프레임 변위 |U_K| (칸): 이웃 중앙 {f3(float(np.median(tot[nb_])))} (전체 중앙 {f3(float(np.median(tot)))})\n")
    order = bad[np.argsort(Jm[bad])][:12]
    print("| 점 (칸·팔분) | x (m) | 높이 m | V/중앙 | 팔분 입자 · 칸 입자 | 팔분 입자 det 최소 (시작) | 회전 일치 σmin | 유클리드 평균 | F_q det | " + " | ".join(snapname(i, nt) for i in range(NS)) + " | 이웃 질량/중앙 최소 | 이웃 증분 최대 (칸) |")
    print("|" + "---|" * (11 + NS))
    for q in order:
        nn_ = nodes_of(np.array([q]))
        print(f"| {tuple(int(v) for v in z['q_cell'][q])}·{int(z['q_oc'][q])} | {tuple(round(float(v), 3) for v in z['q_x'][q])} | {f3(float(z['q_h'][q]))} | {f3(float(z['q_V'][q]))} | {int(z['q_np'][q])} · {int(z['q_nc'][q])} | {f3(float(z['q_jp'][q]))} | {f3(float(z['q_rsv'][q]))} | {int(z['q_bad'][q])} | {f3(float(z['q_J0'][q]))} | "
              + " | ".join(f3(float(v)) for v in z["q_J"][q]) + f" | {f3(float(z['n_m'][nn_].min()))} | {f3(float(incm[nn_].max()))} |")
    print()
    # J 이력 (같은 칸·팔분 열쇠): K 4, K 2, 반올림 판
    keys = skey(z["q_cell"][order], z["q_oc"][order])
    print(f"### 그 점들 (위 표 앞 6 개) 의 det 이력 -- 프레임별 [F_q det, 프레임 안 최소 det] (같은 칸·팔분; 빈칸 = 그 프레임에 그 점 없음)\n")
    for r, nm in RUNS[:3]:
        print(f"- {r} {nm}:")
        for kk, q in zip(keys[:6], order[:6]):
            row = []
            for t in range(max(W0, tstar - 12), min(tstar + 3, 69)):
                zz = npz(r, t)
                if zz is None:
                    continue
                kz = kz_of(r, t, zz)
                m_ = np.nonzero(kz == kk)[0]
                row.append(f"{t}: {f3(float(zz['q_J0'][m_[0]]))}/{f3(float(zz['q_J'][m_[0]].min()))}" if len(m_) else f"{t}: -")
            print(f"  - {tuple(int(v) for v in z['q_cell'][q])}·{int(z['q_oc'][q])}: " + ", ".join(row))
    print()
    # 그 점들의 입자 (프레임 tstar 시작에 그 팔분에 있던 입자) det 이력
    pq = z["p_q"]; qi = z["q_qi"]
    selq = set(int(qi[q]) for q in bad)
    pp = np.nonzero(np.isin(pq, list(selq)))[0]
    print(f"### 그 점들 팔분의 입자 {len(pp)} 개 (프레임 {tstar} 시작) det F 이력 -- 프레임별 [시작 최소, 끝 최소, 끝 ≤ 0 수]\n")
    row = []
    for t in range(max(W0, tstar - 12), min(tstar + 3, 69)):
        zz = npz("k4w", t)
        if zz is None:
            continue
        if not len(pp):
            break
        row.append(f"{t}: {f3(float(zz['p_J0'][pp].min()))} → {f3(float(zz['p_J1'][pp].min()))} ({int((zz['p_J1'][pp] <= 0).sum())})")
    print("- " + ", ".join(row) + "\n")
# ---- 3. 뉴턴·PCG 반복별 (K 4 기준, 뒤집힘 프레임 앞뒤) 와 K 2 같은 프레임
print("## 3. 뉴턴·PCG 반복별 (걸음 k·반복 i: |g|/F · PCG 반복 (상한 도달 *) · √(r·z 끝/처음) · 보폭)\n")
for r in ("k4w", "k2w", "hsw"):
    if not J[r]:
        continue
    print(f"### {r}\n")
    t0 = (tstar or 52) - 6
    for t in range(max(1, t0), min(len(J[r]), (tstar or 52) + 3) + 1):
        w = wd(r, t)
        if not w:
            continue
        s_ = []
        for q in w["it"]:
            if not q["ran"]:
                s_.append(f"{q['k'] + 1}.{q['i'] + 1} 멈춤 ({f3(q['gF'], 3) if q['gF'] is not None else ''})" if q["stop"] else f"{q['k'] + 1}.{q['i'] + 1} -")
            else:
                s_.append(f"{q['k'] + 1}.{q['i'] + 1} {f3(q['gF'], 3)}·{q['nP']}{'*' if q['cap'] else ''}·{f3(q['rz'], 2)}·{f3(q['st'], 3)}")
        print(f"- 프레임 {t}: " + " | ".join(s_) + f" | 걸음 끝 |g|/F {' '.join(f3(q, 3) for q in w['gend'])}")
    print()
# ---- 4. 위치 차 (판 사이, 창 프레임 시작 위치 p_x0)
print("## 4. 판 사이 입자 위치 차 (프레임 시작 위치, 창 npz) -- 평균 / 최대 (mm)\n")
pairs = [("k4w", "k2w"), ("k4w", "hsw"), ("k4w", "mvw"), ("hsw", "k2w")]
print("| 프레임 | " + " | ".join(f"{a} − {b}" for a, b in pairs) + " |")
print("|---|" + "---|" * len(pairs))
for t in (40, 45, 48, 50, 51, 52, 53, 54, 56, 60, 65, 68):
    row = []
    for a_, b_ in pairs:
        za, zb = npz(a_, t), npz(b_, t)
        if za is None or zb is None:
            row.append(""); continue
        d = np.linalg.norm(za["p_x0"] - zb["p_x0"], axis=1) * 1000
        d = d[np.isfinite(d)]
        row.append(f"{f3(float(d.mean()), 1)} / {f3(float(d.max()), 0)}")
    print(f"| {t} | " + " | ".join(row) + " |")
print()

"""절 CM131 분석 2 (로컬): 끝 평가 잔차가 몰린 노드 (상위) 를 자세히 -- 접은 잔차 중 외부 노드에서 접혀 들어온 몫 (RF − R0), 그 외부 노드들의 질량비·받침 입자·꼭짓점 몫,
주인 노드 하중 Σ c², 근사 줄어든 대각 D̃_j = Mi_j + λmax(B_j) + Σ_o c_oj² (Mi_o + λmax(B_o)) 대 할선 응답, 반복 사이 방향 코사인.
사용: python3 exe/cm131_top.py <판> <프레임들 쉼표> [--top 10]"""
import os, sys, subprocess
import numpy as np
run = sys.argv[1]; frames = [int(q) for q in sys.argv[2].split(",")]; top = int(sys.argv[4]) if len(sys.argv) > 4 else 10
sys.argv = ["x", "--out", "/dev/null", "--runs", ""]
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cm131_an.py")).read().split("res = {}")[0])
for f in frames:
    pe, pr = fetch(run, "esc", f), fetch(run, "rm", f)
    z, m = np.load(pe), np.load(pr)
    k = 0
    C = classes(z, m, k)
    IP = z["IP"]; nb1, nb2 = int(IP[7]), int(IP[8]); nid = z["nid"].astype(np.int64); nn = len(nid)
    nrel = np.stack([nid // (nb1 * nb2), (nid // nb2) % nb1, nid % nb2], 1); sidx = np.argsort(nid)
    def idx_of(g):
        j = np.clip(np.searchsorted(nid, g, sorter=sidx), 0, nn - 1); jj = sidx[j]; return np.where(nid[jj] == g, jj, -1)
    OK = m["OK"][k]; ev = [e for e in range(OK.shape[0]) if OK[e, 0] == 1]; eL = ev[-1]
    R0, RF, Y = m["R0"][k].astype(np.float64), m["RF"][k].astype(np.float64), m["Y"][k].astype(np.float64)
    Mi = m["ME"][k].astype(np.float64) * float(m["IT2"][k]); B = m["B"][k].astype(np.float64).reshape(-1, 3, 3)
    lB = np.array([np.linalg.eigvalsh(0.5 * (b + b.T)).max() for b in B])
    W8 = z["W8"][k].astype(np.int64); ext = W8 >= 0
    # 주인 j 에 묶인 외부 노드 목록과 c
    bound = {}
    for o in np.nonzero(ext)[0]:
        bb = W8[o]; bc = np.array([bb // (nb1 * nb2), (bb // nb2) % nb1, bb % nb2]); xs = nrel[o] - bc - 1
        if np.any(np.abs(xs) > 2): continue
        w = [ext_weights(int(x)) for x in xs]
        for p_ in range(3):
            for q_ in range(3):
                for r_ in range(3):
                    c = w[0][p_] * w[1][q_] * w[2][r_]
                    if c != 0.0:
                        j = int(idx_of(np.array([bb + (p_ * nb1 + q_) * nb2 + r_]))[0])
                        if j >= 0: bound.setdefault(j, []).append((o, c))
    Dt = Mi + lB
    for j, L in bound.items():
        Dt[j] += sum(c * c * (Mi[o] + lB[o]) for o, c in L)
    gF = (RF[eL] ** 2).sum(1); g0F = (RF[ev[0]] ** 2).sum(1)
    o_ = np.argsort(-gF)[:top]
    print(f"== {run} f{f} 평가 {ev} 접은 |r| 처음 {np.sqrt(g0F.sum()):.3e} 끝 {np.sqrt(gF.sum()):.3e} | 상위 {top} 노드 끝 |r|² 몫 {gF[o_].sum() / gF.sum():.3f} (처음 평가에서 같은 노드 몫 {g0F[o_].sum() / g0F.sum():.3f})")
    for j in o_:
        L = bound.get(j, [])
        fin = RF[eL][j] - R0[eL][j]
        oo = np.array([o for o, c in L], dtype=np.int64); cc = np.array([c for o, c in L])
        dy = Y[eL][j] - Y[ev[0]][j]
        rs = [np.linalg.norm(RF[e][j]) for e in ev]
        cs = [float(RF[e][j] @ RF[e2][j] / max(np.linalg.norm(RF[e][j]) * np.linalg.norm(RF[e2][j]), 1e-300)) for e, e2 in zip(ev[:-1], ev[1:])]
        s = (f"  노드 {nid[j]} 질량비 {C['mr'][j]:.2f} 받침 {int(C['cnt'][j])} 꼭짓점 {C['fap'][j]:.2f} 높이 {C['hz'][j]:+.3f} 고스트 {int(C['gadj'][j])} 장벽 {int(C['bar'][j])} 외부 {int(ext[j])}"
             f" | 묶인 외부 {len(L)} Σc² {C['load'][j]:.0f} |c| 최대 {np.abs(cc).max() if len(cc) else 0:.0f} | |r| 평가별 {[f'{q:.2e}' for q in rs]} 방향 코사인 {[round(q, 3) for q in cs]}"
             f" | 접혀 온 몫 |RF−R0|/|RF| {np.linalg.norm(fin) / max(np.linalg.norm(RF[eL][j]), 1e-300):.3f} | Mi {Mi[j]:.3g} λB {lB[j]:.3g} D̃ {Dt[j]:.3g} (D̃/(Mi+λB) {Dt[j] / (Mi[j] + lB[j]):.0f}) |Δy| {np.linalg.norm(dy):.2e} |r|/D̃ {rs[-1] / Dt[j]:.2e}")
        if len(L):
            contrib = np.array([c * R0[eL][o] for o, c in L]); cn = np.linalg.norm(contrib, axis=1); w_ = np.argsort(-cn)[:4]
            s += "\n     접혀 온 상위 외부 노드: " + "; ".join(f"c {cc[i]:+.0f} 질량비 {C['mr'][oo[i]]:.1e} 받침 {int(C['cnt'][oo[i]])} 꼭짓점 {C['fap'][oo[i]]:.2f} 높이 {C['hz'][oo[i]]:+.3f} |c r_o| {cn[i]:.2e} λB/Mi {lB[oo[i]] / max(Mi[oo[i]], 1e-30):.1e}" for i in w_)
            s += f"\n     외부 노드 전체: 질량비 중앙 {np.median(C['mr'][oo]):.1e} 받침 중앙 {np.median(C['cnt'][oo]):.0f} 꼭짓점 중앙 {np.median(C['fap'][oo]):.2f} | Σ|c r_o| {cn.sum():.2e} 대 |Σ c r_o| {np.linalg.norm(contrib.sum(0)):.2e}"
        print(s)
    os.remove(pe); os.remove(pr)

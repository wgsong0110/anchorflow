#!/usr/bin/env python3
"""절 CM193 진단 (결과 불변, 실행 뒤 분석): 덤프 pred.pt 에서 '튀어 나간 입자' (정답 같은 프레임 상자 밖으로 dz 넘게) 를 프레임마다 세고,
그 입자들의 처음 위치·det F·프레임 변위 (속도) 를 찍음. 사용: wfn_ej.py <pred.pt> <정답 wolf.pt> [여유 m (기본 0.05)]"""
import sys, torch
P = torch.load(sys.argv[1], map_location="cpu"); G = torch.load(sys.argv[2], map_location="cpu")
mg = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05
X = P["pred"].float(); F = P.get("F"); gx = G["x"].float()
L = X.shape[0]; print("프레임", L, "입자", X.shape[1], "정답", tuple(gx.shape))
x0 = gx[0]
prev = None
for f in range(L):
    g = gx[f + 1]; lo = g.min(0).values - mg; hi = g.max(0).values + mg
    out = ((X[f] < lo) | (X[f] > hi)).any(-1)
    J = torch.linalg.det(F[f].double()) if F is not None else None
    sp = (X[f] - (X[f - 1] if f else x0)).norm(dim=-1)
    n = int(out.sum())
    s = f"f{f+1:02d} 밖 {n:6d}  det<=0 {int((J <= 0).sum()) if J is not None else -1:5d}  Jmin {float(J.min()) if J is not None else 0:.3g} Jmax {float(J.max()) if J is not None else 0:.3g}  최대 프레임 변위 {float(sp.max()):.3f} m"
    if n:
        idx = out.nonzero()[:, 0]
        s += f"  | 밖 입자: z {float(X[f, idx, 2].min()):.2f}~{float(X[f, idx, 2].max()):.2f} 변위 중앙 {float(sp[idx].median()):.3f}"
        if J is not None:
            s += f" J 중앙 {float(J[idx].median()):.3g} 최소 {float(J[idx].min()):.3g} 최대 {float(J[idx].max()):.3g}"
        s += f" 처음 위치 중앙 {[round(float(q), 2) for q in x0[idx].median(0).values]}"
        if prev is not None:
            s += f" 앞 프레임에도 밖 {int((prev & out).sum())}"
    prev = out
    print(s, flush=True)

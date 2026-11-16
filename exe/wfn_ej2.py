#!/usr/bin/env python3
"""절 CM193 진단: 덤프 pred.pt 에서 프레임 f 에 정답 상자 밖 (여유 m) 으로 나간 입자의 앞 프레임 (f-1) 이웃 밀도 (반경 r 안 입자 수) 와 바닥 높이·J 를,
같은 높이 띠 (z < zb) 전체 입자와 비교. 사용: wfn_ej2.py <pred.pt> <wolf.pt> <f (1 부터)> [r 0.02] [zb 0.56]"""
import sys, torch
P = torch.load(sys.argv[1], map_location="cpu", weights_only=False); G = torch.load(sys.argv[2], map_location="cpu", weights_only=False)
f = int(sys.argv[3]) - 1; r = float(sys.argv[4]) if len(sys.argv) > 4 else 0.02; zb = float(sys.argv[5]) if len(sys.argv) > 5 else 0.56
X = P["pred"].float(); gx = G["x"].float()
g = gx[f + 1]; lo = g.min(0).values - 0.05; hi = g.max(0).values + 0.05
out = ((X[f] < lo) | (X[f] > hi)).any(-1)
xp = X[f - 1]
band = xp[:, 2] < zb
def dens(idx):
    q = xp[idx]; c = torch.zeros(len(idx), dtype=torch.long)
    for i in range(0, len(idx), 2048):
        d = torch.cdist(q[i:i + 2048], xp); c[i:i + 2048] = (d < r).sum(1) - 1
    return c.float()
io = out.nonzero()[:, 0]; ib = band.nonzero()[:, 0]
ib = ib[torch.randperm(len(ib))[:4000]]
do, db = dens(io), dens(ib)
print(f"프레임 {f+1}: 밖 {len(io)} 입자, 앞 프레임 반경 {r} 이웃 수 중앙 {float(do.median()) if len(io) else -1:.0f} (하위 10% {float(do.quantile(0.1)) if len(io) else -1:.0f}); "
      f"z<{zb} 띠 입자 표본 {len(ib)} 이웃 중앙 {float(db.median()):.0f} (하위 10% {float(db.quantile(0.1)):.0f}), 띠 안 이웃 ≤ 2 비율 {float((db <= 2).float().mean()):.3f} / 밖 입자 {float((do <= 2).float().mean()) if len(io) else -1:.3f}")

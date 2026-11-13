"""절 CM126: 관성 예측에서 1.5 칸 넘게 벗어난 노드 (|y−Ũ| > 0.03 m) 의 바닥 위 높이·둘레 판 두께 (같은 x·y 기둥의 질량 있는 노드 높이 범위) 분포, 걸음별."""
import numpy as np, sys, os
run = sys.argv[1]; f0, f1 = [int(q) for q in sys.argv[2].split(':')]; PN = 0.48
for f in range(f0, f1 + 1):
    p = os.path.join(run, 'esc', f'f{f:03d}.npz')
    if not os.path.exists(p): continue
    z = np.load(p)
    if 'nid' not in z.files: continue
    IP, A = z['IP'], z['A']; nb1, nb2 = int(IP[7]), int(IP[8]); nid = z['nid'].astype(np.int64)
    g = np.stack([nid // (nb1 * nb2), (nid // nb2) % nb1, nid % nb2], 1)
    pos = (g + IP[0:3]) @ A.T; med = np.median(z['mn']); heavy = z['mn'] > 0.1 * med
    col = g[:, 0] * 100000 + g[:, 1]
    top = {}
    for c, h, hv in zip(col, pos[:, 2], heavy):
        if hv: top[c] = max(top.get(c, -1), h)
    for k in range(int(z['K'])):
        dy = np.linalg.norm(z['Y'][k] - z['Ut'][k], axis=1); s = dy > 0.03
        if not s.any(): continue
        h = pos[s, 2] - PN; th = np.array([top.get(c, PN) - PN for c in col[s]])
        print(f'f{f} k{k}: 노드 {s.sum()} 바닥 위 높이 중앙 {np.median(h):.3f} m (10~90% {np.quantile(h,.1):.3f}~{np.quantile(h,.9):.3f}), 그 기둥 무거운 노드 꼭대기 높이 중앙 {np.median(th):.3f} m, 바닥 5 cm 안 몫 {(h < 0.05).mean():.2f}')

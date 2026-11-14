"""절 CM162 분석 (CPU, numpy): escwhy 기록 (f28~49) 에서 (1) 프레임별 몸 밖 입자 수·속도 분포, 정답 (wolf.pt) 대조
(2) 선두 이탈 입자의 걸음 안 받침 노드 (질량·바닥 높이·고착 몫 sk·an·구속 kb·속도·가속·잔차 |g0|·가우스 점 꼭짓점 몫·뒤집힘)를 표로.
사용: python3 exe/cm162_an.py <run 디렉토리> <gt_x.npy> [--fr 46] [--pn 0.48]"""
import argparse, glob, os
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("gt"); ap.add_argument("--fr", type=int, default=46); ap.add_argument("--pn", type=float, default=0.48); ap.add_argument("--xlim", type=float, default=0.5)
a = ap.parse_args()
GT = np.load(a.gt); np.set_printoptions(precision=3, suppress=True, linewidth=200)
fs = sorted(glob.glob(os.path.join(a.run, "esc", "f*.npz")))
print("프레임 | 노드 상자 | x<%.2f 입자 (우리/정답) | 속력>1 (우리/정답) | 속력>2 (우리) | 위치 0.1%% x 최소 (우리/정답)" % a.xlim)
for f in fs:
    z = np.load(f); fr = int(os.path.basename(f)[1:4]); x = z["p1_x"]; v = z["p1_v"]; sp = np.linalg.norm(v, axis=1); g = GT[fr]; gv = np.linalg.norm(GT[fr] - GT[fr - 1], axis=1) / 0.04
    print(f"f{fr} | {int(z['IP'][9])} | {(x[:,0]<a.xlim).sum()} / {(g[:,0]<a.xlim).sum()} | {(sp>1).sum()} / {(gv>1).sum()} | {(sp>2).sum()} | {np.quantile(x[:,0],0.001):.3f} / {np.quantile(g[:,0],0.001):.3f}")
fr = a.fr; z = np.load(os.path.join(a.run, "esc", "f%03d.npz" % fr)); x0, x1 = z["p0_x"], z["p1_x"]
P = np.argsort(x1[:, 0])[:3]
nid = z["nid"].astype(np.int64); IP = z["IP"]; A = z["A"]; Ai = z["Ai"]; nb1, nb2 = IP[7], IP[8]
pos = (np.stack([nid // (nb1 * nb2), (nid // nb2) % nb1, nid % nb2], 1) + IP[0:3]) @ A.T
mn = z["mn"]; mm = np.median(mn); sidx = np.argsort(nid)
y = x0 @ Ai.T; b = np.floor(y - 0.5).astype(np.int64); fx = y - b; b = b - IP[0:3]
print(f"\n== f{fr} 선두 입자 (x 최소 3개) 와 받침 노드 (가중 상위 6)")
for p in P:
    print(f"입자 {p} x0 {x0[p]} 바닥 위 {x0[p][2]-a.pn:+.3f} x1 {x1[p]} v0 {z['p0_v'][p]} v1 {z['p1_v'][p]} detF0 {np.linalg.det(z['p0_F'][p].reshape(3,3)):.2f} detF1 {np.linalg.det(z['p1_F'][p].reshape(3,3)):.2f} | 정답 같은 인덱스 x {GT[fr][p]} z {GT[fr-1][p]}")
    w = [np.stack([0.5*(1.5-fx[p][k])**2, 0.75-(fx[p][k]-1)**2, 0.5*(fx[p][k]-0.5)**2]) for k in range(3)]
    rows = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                gid = ((b[p][0]+i)*nb1 + (b[p][1]+j))*nb2 + (b[p][2]+k); ww = w[0][i]*w[1][j]*w[2][k]
                jx = np.searchsorted(nid, gid, sorter=sidx); jx = sidx[jx] if jx < len(nid) else -1
                if jx < 0 or nid[jx] != gid: rows.append((ww, None)); continue
                rows.append((ww, jx))
    rows.sort(key=lambda r: -r[0])
    tot = sum(r[0] for r in rows); lost = sum(r[0] for r in rows if r[1] is None)
    print(f"   질량 0 (없는) 받침 가중 몫 {lost/tot:.3f}")
    for ww, jx in rows[:6]:
        if jx is None: continue
        k = 0
        print(f"   노드 w {ww:.3f} m x{mn[jx]/mm:.1e} 바닥 위 {pos[jx][2]+z['U0'][k][jx][2]-a.pn:+.3f} sk {z['SK'][k][jx]:.2f} an {z['AN'][k][jx]:.1e} kb {z['KB'][k][jx]:.2f} pm {z['PM'][k][jx]} bK {z['BK'][k][jx]:.1e} v0 {z['V0'][k][jx]} v1 {z['V1'][k][jx]} a1 {z['A1'][k][jx]} |g0| {np.linalg.norm(z['G0'][k][jx]):.1e} (노드 무게 {mn[jx]*9.8:.1e})")
    cc = b[p]; ci = np.where((z["cb"] == cc).all(1))[0]
    if len(ci):
        c = ci[0]; st = z["QS"][0][c*8:c*8+8]
        print(f"   칸 가우스 점 8개: 꼭짓점 상태 몫 {(st[:,3]>0).mean():.2f} · detX<0 {(st[:,4]<0).sum()} · detX 최소 {st[:,4].min():.3f} 최대 {st[:,4].max():.3f} · 점 질량 합 {z['mq'][c*8:c*8+8].sum():.2e}")

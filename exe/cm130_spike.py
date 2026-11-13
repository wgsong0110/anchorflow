"""절 CM130 (로컬 numpy): 한 프레임 기록에서 입자 psi 가 자리 가우스 점 psi 가중평균보다 크게 (기본 10 배·psi>1e5) 튄 입자의 둘레 가우스 점 자세히.
사용: python exe/cm130_spike.py <esc/fNNN.npz> [배 10]"""
import sys, numpy as np
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
MU, LAM, CDP = 1.9230769e7, 2.8846154e7, 1.0614514; G = 0.28867513459481287
z = np.load(sys.argv[1]); R = float(sys.argv[2]) if len(sys.argv) > 2 else 10
K = int(z["K"]); QS = z["QS"][K - 1]; st = QS[:, 3].astype(int); xq = z["xq0"].astype(np.float64); cb = z["cb"].astype(np.int64)
def hen(e): return MU * (e ** 2).sum(-1) + 0.5 * LAM * e.sum(-1) ** 2
e = np.log(np.maximum(np.abs(QS[:, :3].astype(np.float64)), 1e-14)); tr = e.sum(1, keepdims=True); eh = e - tr / 3; rho = np.linalg.norm(eh, axis=1, keepdims=True)
eE = np.where(st[:, None] == 1, tr / 3 + np.where(rho > 0, -CDP * tr / np.maximum(rho, 1e-30), 0) * eh, e); eE = np.where(st[:, None] == 2, 0, eE); pq = hen(eE)
F1 = z["p1_F"].reshape(-1, 3, 3).astype(np.float64); s1 = np.linalg.svd(F1, compute_uv=False); pp = hen(np.log(np.maximum(np.abs(s1), 1e-14))); d1 = np.linalg.det(F1)
F0 = z["p0_F"].reshape(-1, 3, 3).astype(np.float64); pp0 = hen(np.log(np.maximum(np.abs(np.linalg.svd(F0, compute_uv=False)), 1e-14)))
nc = len(cb); xc = xq.reshape(nc, 8, 3).mean(1); co = np.linalg.lstsq(np.c_[cb, np.ones(nc)], xc, rcond=None)[0]
lat = np.linalg.solve(co[:3].T, (z["p0_x"].astype(np.float64) - co[3]).T).T; pb = np.rint(lat).astype(np.int64); l = lat - pb
key = lambda a: (a[:, 0] * 200003 + a[:, 1]) * 200003 + a[:, 2]; kc = key(cb); o = np.argsort(kc); ks = kc[o]; kp = key(pb)
ii = np.clip(np.searchsorted(ks, kp), 0, nc - 1); pc = np.where(ks[ii] == kp, o[ii], -1)
t = np.clip((l + G) / (2 * G), 0, 1); W = np.stack([(t[:, 0] if oc & 4 else 1 - t[:, 0]) * (t[:, 1] if oc & 2 else 1 - t[:, 1]) * (t[:, 2] if oc & 1 else 1 - t[:, 2]) for oc in range(8)], 1)
I = np.maximum(pc, 0)[:, None] * 8 + np.arange(8); loc = (W * pq[I]).sum(1)
sp = np.where((pp > 1e5) & (pp > R * np.maximum(loc, 1)))[0]
print(f"{sys.argv[1]}: 튄 입자 {len(sp)} (psi>1e5, 자리 가중의 {R} 배 넘음); 입자 psi>1e5 전체 {(pp > 1e5).sum()}")
for p in sp[:12]:
    print(f" p{p} psi {pp[p]:.3g} (시작 {pp0[p]:.3g}) 자리 {loc[p]:.3g} sig {np.round(s1[p], 4)} det {d1[p]:.3g} z {z['p1_x'][p, 2] - 0.5:.3f} |v| {np.linalg.norm(z['p1_v'][p]):.2f} 칸 {pc[p]}")
    for k in range(8):
        q = I[p, k]
        if W[p, k] > 0: print(f"    점 {k} w {W[p, k]:.3f} 상태 {st[q]} sig {np.round(QS[q, :3], 4)} detX {QS[q, 4]:.3g} psi {pq[q]:.3g} Vq {z['Vq'][q]:.2e}")

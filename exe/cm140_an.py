"""절 CM140 집계 (CPU, 읽기만): 판마다 (1) 낙하 가속 (방출 뒤 tau 1~5 프레임 평균 v_z, 정답 대 판; 절 CM134.1 정의 그대로) (2) 운동에너지 (유한 차분 v 의 입자 합 1/2 v^2, 수직·수평, 정답 대 판)
(3) 바닥 근처 띠 수평 속력 (f26·f32) -- 절 exe/cm134_layer.py 와 같은 정의.  사용: python exe/cm140_an.py <wolf.pt> <O> <판 ...> -> 표준 출력 + <O>/cm140_an.json"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import cm130_ptload as L

d = L.load(sys.argv[1]); GT = np.array(d["x"][:51], dtype=np.float32)
O = sys.argv[2]; h = 0.04
runs = {"gt": GT}
for n in sys.argv[3:]:
    p = os.path.join(O, n, "pred.pt")
    if os.path.exists(p):
        Pp = np.array(L.load(p)["pred"], dtype=np.float32); runs[n] = np.concatenate([GT[:1], Pp], 0)
z0 = GT[0][:, 2]; dz = GT[:, :, 2] - z0[None]
on = np.where((dz < -5e-4).any(0), (dz < -5e-4).argmax(0), 999); sel = (on < 40) & (z0 > 0.75)
res = {}
for k, X in runs.items():
    r = {"tau_vz": [], "KE_v": {}, "KE_h": {}, "band": {}}
    for tau in range(1, 6):
        vs = 0.0; nn = 0
        for o in np.unique(on[sel]):
            f = o + tau
            if f >= len(X) or f < 1: continue
            m = sel & (on == o); vs += ((X[f][m, 2] - X[f - 1][m, 2]) / h).mean() * m.sum(); nn += m.sum()
        r["tau_vz"].append(round(float(vs / max(nn, 1)), 3))
    for f in (12, 20, 28, 36):
        if f < len(X):
            v = (X[f] - X[f - 1]) / h
            r["KE_v"][f] = round(float(0.5 * (v[:, 2] ** 2).sum() / 1e3), 2); r["KE_h"][f] = round(float(0.5 * (v[:, :2] ** 2).sum() / 1e3), 2)
    for f in (26, 32):
        if f < len(X):
            v = (X[f] - X[f - 1]) / h; z = X[f - 1][:, 2]; b = []
            for a_, c_ in ((0.48, 0.50), (0.50, 0.54), (0.54, 0.62), (0.62, 2)):
                m = (z >= a_) & (z < c_); b.append(round(float(np.linalg.norm(v[m, :2], axis=1).mean()) if m.any() else -1, 3))
            r["band"][f] = b
    res[k] = r
    print(f"{k:14s} tau1-5 vz {r['tau_vz']}  KE_v(천) {r['KE_v']}  KE_h {r['KE_h']}  띠 속력 {r['band']}", flush=True)
json.dump(res, open(os.path.join(O, "cm140_an.json"), "w"))

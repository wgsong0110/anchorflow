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
# (4) 압력 (절 CM70.1 정의): 입자 F 의 p = -K ln|det F| (K = E/(3(1-2nu)), wolf E 5e7 nu 0.3), 시작 높이 별 평균·rms kPa -- 구속 몸 (아직 안 풀린) 의 거짓 압축 쌓임 확인
K_ = 5e7 / (3 * (1 - 0.6)); rng = np.random.default_rng(0); idx = np.sort(rng.choice(GT.shape[1], 20000, replace=False))
for n in sys.argv[3:]:
    p_ = os.path.join(O, n, "pred.pt")
    if not os.path.exists(p_): continue
    FF = L.load(p_)["F"]; out = {}
    for f in (1, 3, 6, 12, 20, 30):
        if f - 1 < FF.shape[0]:
            Fm = np.array(FF[f - 1][idx], dtype=np.float32).reshape(-1, 3, 3); pk = -K_ * np.log(np.abs(np.linalg.det(Fm))) / 1e3
            a_, b_ = pk[z0[idx] > 1.0], pk[z0[idx] < 0.7]
            out[f] = [round(float(a_.mean()), 1), round(float(np.sqrt((a_ ** 2).mean())), 1), round(float(b_.mean()), 1), round(float(np.sqrt((b_ ** 2).mean())), 1)]
    res[n]["p_kPa"] = out
    print(f"{n:14s} 압력 kPa [z0>1 평균, rms, z0<0.7 평균, rms] (정답 F 는 반정밀 ~0~10) {out}", flush=True)
json.dump(res, open(os.path.join(O, "cm140_an.json"), "w"))

"""절 CM134 집계 (CPU, 읽기만): wolf 높이 과다의 몫 가르기 -- 정답과 판의 프레임별 질량중심 높이·속도, 운동 에너지, 소산, 방출 층별 낙하 속도.
사용: python exe/cm134_an.py <wolf.pt> <출력.json> <판 디렉토리 ...>   (판 디렉토리에 pred.pt; 없으면 건너뜀)
정의 (정답·판 같은 함수, 입자 질량 = DM.traj_mass):
- 높이 = 질량중심 z − 0.48 (바닥), mm. 속도 = (x_f − x_{f−1}) / frame_dt (유한 차분, 정답·판 같음). 운동 에너지 K = ½ Σ m |v|² (J).
- 위치 에너지 감소 ΔΠ = M g (z̄_0 − z̄_f), 소산 D = ΔΠ − K (J, 탄성 몫 무시 -- 정답 탄성 ≤ 0.2 J, 절 CM52), 걸음당 소산 dD = D_f − D_{f−1}.
- 방출 층: 처음 위치 x 0.05 m 띠 [1.40,1.45) [1.30,1.35) [1.20,1.25) [1.10,1.15) [1.00,1.05) 의 질량 가중 v_z (m/s), 층별 '처음 |v_z| > 0.3 m/s 프레임' 과 최대 |v_z|.
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from anchorflow import deform_model as DM                                           # noqa: E402

d = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
OUT = sys.argv[2]
cfg = d["cfg"]
h = float(cfg["frame_dt"])
m = DM.traj_mass(d, "cpu").double().numpy()
M = float(m.sum())
G = 9.8
X0 = d["x"][0].double().numpy()
SL = [(1.40, 1.45), (1.30, 1.35), (1.20, 1.25), (1.10, 1.15), (1.00, 1.05)]
SM = [(X0[:, 0] >= a) & (X0[:, 0] < b) for a, b in SL]


def curves(X):
    """X: [T+1, N, 3] (0 = 처음 상태)."""
    T = X.shape[0] - 1
    zc = np.array([(m * X[f][:, 2]).sum() / M for f in range(T + 1)])
    out = dict(h=[], vz=[], K=[], D=[], dD=[], sl=[])
    Dp = 0.0
    for f in range(1, T + 1):
        v = (X[f] - X[f - 1]) / h
        K = 0.5 * float((m * (v * v).sum(1)).sum())
        D = M * G * (zc[0] - zc[f]) - K
        out["h"].append(1000 * (zc[f] - 0.48)); out["vz"].append(float((m * v[:, 2]).sum() / M)); out["K"].append(K)
        out["D"].append(D); out["dD"].append(D - Dp); Dp = D
        out["sl"].append([float((m[s] * v[s, 2]).sum() / m[s].sum()) for s in SM])
    S = np.array(out["sl"])
    out["sl_on"] = [int(np.argmax(np.abs(S[:, j]) > 0.3)) + 1 if (np.abs(S[:, j]) > 0.3).any() else None for j in range(len(SL))]
    out["sl_max"] = [float(np.abs(S[:, j]).max()) for j in range(len(SL))]
    return out


R = {}
GT = d["x"].double().numpy()
R["gt"] = curves(GT[:51])
for rd in sys.argv[3:]:
    p = os.path.join(rd, "pred.pt")
    if not os.path.exists(p):
        print(f"[{rd}] pred.pt 없음", flush=True); continue
    z = torch.load(p, map_location="cpu", weights_only=False)
    P = z["pred"].double().numpy()
    X = np.concatenate([X0[None], P], 0)
    R[os.path.basename(rd.rstrip("/"))] = curves(X)
    R[os.path.basename(rd.rstrip("/"))]["part"] = bool(z.get("part", False))
json.dump(R, open(OUT, "w"))
g = R["gt"]
for k, o in R.items():
    n = len(o["h"])
    q = lambda a, f: (f"{a[f - 1]:.0f}" if f <= n else "-")
    print(f"[{k}] 높이 f15·20·25·30·50 {q(o['h'],15)}·{q(o['h'],20)}·{q(o['h'],25)}·{q(o['h'],30)}·{q(o['h'],50)} (정답 {q(g['h'],15)}·{q(g['h'],20)}·{q(g['h'],25)}·{q(g['h'],30)}·{q(g['h'],50)})"
          f" | K 최대 {max(o['K']):.1f} J (정답 {max(g['K']):.1f}) f20 {q(o['K'],20)} | 소산 f20·30 {q(o['D'],20)}·{q(o['D'],30)} (정답 {q(g['D'],20)}·{q(g['D'],30)})"
          f" | 층 처음 >0.3 m/s {o['sl_on']} (정답 {g['sl_on']}) 최대 {[round(v, 2) for v in o['sl_max']]} (정답 {[round(v, 2) for v in g['sl_max']]})", flush=True)

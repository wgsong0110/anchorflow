"""절 P: ficus 잎·가지 끝 진동 주기 (exe/fem_o_compose.py 4 번과 같은 분석 -- z 상위 5 % 입자 평균의 정답 주 방향 d 변위,
영점 교차 앞 8 개 간격 x 2, FFT 최대) 를 렌더 없이 pred.pt (fem_ex_sim --dump_pred) 에서 바로.
사용: python exe/fem_p_period.py --data <F_pg_v0.pt> --pred <dir>/pred.pt [...]  -> 각 <dir>/period.json, 표 출력"""
import argparse
import json
import os

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--pred", nargs="+", required=True)
a = ap.parse_args()
dt = 0.04
d = torch.load(a.data, map_location="cpu", weights_only=False)
X = d["x"][:126].float().numpy()
x0 = X[0]
sel = np.where(x0[:, 2] >= np.quantile(x0[:, 2], 0.95))[0]
D = np.array([0.977, -0.21, -0.016])
t = np.arange(126) * dt


def ana(u):
    s = u - u[63:].mean()
    zc = [t[i - 1] + s[i - 1] / (s[i - 1] - s[i]) * dt for i in range(1, len(s)) if s[i - 1] * s[i] < 0]
    T = 2 * np.mean(np.diff(zc[:8])) if len(zc) >= 3 else float("nan")
    n = 8192
    Fq = np.abs(np.fft.rfft(s * np.hanning(len(s)), n))
    f = np.fft.rfftfreq(n, dt)
    return float(T), float(1 / f[1:][np.argmax(Fq[1:])]), float(np.abs(u).max()), len(zc)


ug = (X[:, sel].mean(1) - x0[sel].mean(0)) @ D
g = ana(ug)
print(f"정답: 주기 영점 {g[0]:.3f} s FFT {g[1]:.3f} s 최대 진폭 {g[2]:.4f} m", flush=True)
for p in a.pred:
    if not os.path.exists(p):
        print(f"{p}: 없음 (발산?)")
        continue
    P = torch.load(p, map_location="cpu", weights_only=False)["pred"][:, sel].float().numpy()
    Xs = np.concatenate([x0[sel][None], P], 0)[:126]
    u = (Xs.mean(1) - x0[sel].mean(0)) @ D
    r = ana(u)
    json.dump(dict(T_zc=r[0], T_fft=r[1], amp=r[2], nzc=r[3], gt=dict(T_zc=g[0], T_fft=g[1], amp=g[2]), ratio=r[0] / g[0], amp_ratio=r[2] / g[2]),
              open(os.path.join(os.path.dirname(p), "period.json"), "w"), indent=1)
    print(f"{os.path.basename(os.path.dirname(p))}: 주기 영점 {r[0]:.3f} s FFT {r[1]:.3f} s (정답 대비 {r[0] / g[0]:.2f}), 최대 진폭 {r[2]:.4f} m (비 {r[2] / g[2]:.2f}), 영점 {r[3]}", flush=True)

"""절 AE: ficus 진동 곡선 (fem_p_period.py 와 같은 분석 -- z 상위 5 % 입자 평균의 정답 주 방향 d 변위) 을 json 으로 (그림은 로컬 fem_ae_fig.py).
사용: python exe/fem_ae_curve.py --data <F_pg_v0.pt> --out <json> --pred 이름=<dir>/pred.pt ..."""
import argparse
import json

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--pred", nargs="+", required=True)
a = ap.parse_args()
dt = 0.04
d = torch.load(a.data, map_location="cpu", weights_only=False)
X = d["x"][:126].float().numpy()
x0 = X[0]
sel = np.where(x0[:, 2] >= np.quantile(x0[:, 2], 0.95))[0]
D = np.array([0.977, -0.21, -0.016])
R = dict(t=(np.arange(126) * dt).tolist(), gt=((X[:, sel].mean(1) - x0[sel].mean(0)) @ D).tolist(), pred={})
for q in a.pred:
    nm, p = q.split("=", 1)
    P = torch.load(p, map_location="cpu", weights_only=False)["pred"][:, sel].float().numpy()
    Xs = np.concatenate([x0[sel][None], P], 0)[:126]
    R["pred"][nm] = ((Xs.mean(1) - x0[sel].mean(0)) @ D).tolist()
json.dump(R, open(a.out, "w"))
print("[저장]", a.out, list(R["pred"]))

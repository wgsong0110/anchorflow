"""실시간 판 궤적의 퍼짐 진단: 프레임마다 입자 바운딩박스 대각선 / EXT, 정답 바운딩박스에서 0.05 EXT 넘게 벗어난 입자 수, 최대 |x - x_gt| / EXT.
사용: python exe/fem_rt_spread.py <폴더>...   (pred.pt 또는 roll_fem.pt; 정답은 --data)"""
import os
import sys

import torch

data = "/home/dkta/work/one/L000/L000.pt"
D0 = torch.load(data, map_location="cpu", weights_only=False)
for d in sys.argv[1:]:
    f = os.path.join(d, "pred.pt") if os.path.exists(os.path.join(d, "pred.pt")) else os.path.join(d, "roll_fem.pt")
    P = torch.load(f, map_location="cpu", weights_only=False)
    X, E = P["pred"].float(), float(P["EXT"])
    L = X.shape[0]
    GT = D0["x"][1:L + 1].float()
    out = []
    for t in list(range(0, L, 10)) + [L - 1]:
        lo, hi = GT[t].min(0).values - 0.05 * E, GT[t].max(0).values + 0.05 * E
        outside = int(((X[t] < lo) | (X[t] > hi)).any(-1).sum())
        diag = float((X[t].max(0).values - X[t].min(0).values).norm()) / E
        mx = float((X[t] - GT[t]).norm(dim=-1).max()) / E
        out.append(f"{t + 1}:{diag:.2f}/{outside}/{mx:.2f}")
    print(f"[퍼짐] {os.path.basename(d.rstrip('/'))} (프레임:대각선/EXT / 벗어난 입자 / 최대오차/EXT) " + " ".join(out), flush=True)

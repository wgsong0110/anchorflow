"""절 CM34 진단 (읽기만): 정답·예측 입자 상자 크기 -- wolf 최악 프레임이 조밀 노드 상자 (입자 최소·최대) 커짐에서 오는지, 정답도 그렇게 퍼지는지.

사용: python exe/speed1_bbox.py <자료.pt> <셀 m> [pred.pt ...]   -> 프레임별 상자 노드 수 (셀 간격, 축마다 +3), 몸 상자 (0.1~99.9 백분위) 밖 입자 수, 그 입자들의 몸 상자 밖 거리 최대
"""
import json
import sys

import torch


def stats(x, cell):
    ok = torch.isfinite(x).all(-1)
    x = x[ok].double()
    lo, hi = x.min(0).values, x.max(0).values
    nb = torch.floor((hi - lo) / cell) + 3
    q = torch.quantile(x[::7], torch.tensor([0.001, 0.999], dtype=torch.float64), dim=0)
    out = ((x < q[0]) | (x > q[1])).any(-1)
    d = torch.clamp(torch.maximum(q[0] - x, x - q[1]), min=0).norm(dim=-1)
    nbq = torch.floor((q[1] - q[0]) / cell) + 3
    far = d > 4 * cell
    return dict(nn=int(nb.prod()), nb=[int(v) for v in nb], nn_body=int(nbq.prod()), nfar=int(far.sum()), dmax=round(float(d.max()), 3))


d = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
cell = float(sys.argv[2])
X = d["x"]
print("정답", json.dumps({f: stats(X[f], cell) for f in range(0, X.shape[0], 5)}, ensure_ascii=False), flush=True)
for p in sys.argv[3:]:
    P = torch.load(p, map_location="cpu", weights_only=False)["pred"]
    print(p, json.dumps({f + 1: stats(P[f], cell) for f in list(range(0, P.shape[0], 5)) + [P.shape[0] - 1]}, ensure_ascii=False), flush=True)

"""절 CM109 이탈 입자 수 (읽기만): 판 덤프 pred.pt 의 프레임마다 몸 상자 (입자 0.1~99.9 백분위, 축별) 밖으로 4 칸 넘게 떨어진 입자 수
(exe/speed1_bbox.py 와 같은 정의, 모든 프레임). 이탈 판 = 어느 프레임이든 20 개 넘음 (절 CM53·wolfh10_heap 과 같은 판정).
사용: python exe/xfer_esc.py <pred.pt> <셀 m> <출력.json>
"""
import json
import sys

import torch

P = torch.load(sys.argv[1], map_location="cpu", weights_only=False)["pred"]
cell = float(sys.argv[2])
rows = []
for f in range(P.shape[0]):
    x = P[f]
    ok = torch.isfinite(x).all(-1)
    if not bool(ok.any()):
        rows.append(None); continue
    x = x[ok].double()
    q = torch.quantile(x[::7], torch.tensor([0.001, 0.999], dtype=torch.float64), dim=0)
    d = torch.clamp(torch.maximum(q[0] - x, x - q[1]), min=0).norm(dim=-1)
    rows.append(dict(f=f + 1, nfar=int((d > 4 * cell).sum()), dmax=round(float(d.max()), 4), nonfinite=int((~ok).sum())))
V = [r for r in rows if r]
o = dict(cell=cell, frames=len(rows), max_nfar=max((r["nfar"] for r in V), default=None), f_max=(max(V, key=lambda r: r["nfar"])["f"] if V else None),
         first_f_over20=next((r["f"] for r in V if r["nfar"] > 20), None), end_nfar=(V[-1]["nfar"] if V else None), dmax=max((r["dmax"] for r in V), default=None), rows=rows)
json.dump(o, open(sys.argv[3], "w"))
print(json.dumps({k: v for k, v in o.items() if k != "rows"}))

"""절 AK.16: 0 프레임 (쉼 모양) 입자의 얇음 지표 -- 입자 칸 (node_h 정육면체) 둘레 27 칸 중 입자가 있는 칸 비율 f. 형상마다 분포 출력.
사용: python exe/fem_ak_thin.py <data.pt> <node_h> [...] (짝으로)"""
import sys

import torch

for i in range(1, len(sys.argv), 2):
    d = torch.load(sys.argv[i], map_location="cpu", weights_only=False)
    nh = float(sys.argv[i + 1])
    x = d["x"][0].float()
    c = torch.floor(x / nh).long()
    c = c - c.min(0).values + 1
    B = int(c.max()) + 2
    key = (c[:, 0] * B + c[:, 1]) * B + c[:, 2]
    occ = torch.unique(key)
    cnt = torch.zeros(x.shape[0])
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                k2 = ((c[:, 0] + dx) * B + c[:, 1] + dy) * B + c[:, 2] + dz
                pos = torch.searchsorted(occ, k2).clamp(max=occ.numel() - 1)
                cnt += (occ[pos] == k2).float()
    f = cnt / 27
    q = torch.quantile(f, torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9]))
    print(sys.argv[i], "node_h", nh, "f 분위 10·25·50·75·90 %", [round(float(v), 3) for v in q], "f<0.5 비율", round(float((f < 0.5).float().mean()), 3), "f<0.7", round(float((f < 0.7).float().mean()), 3), flush=True)

"""절 AK.29: 정답 x240 에서 프레임 1/60 (drop_f1, 80)·1/120 (drop_f2, 160)·1/240 (drop_f4, 320) 자료 (fem_ex_sim 형식). 사용: python exe/fem_ak_drop_mk.py <블록 폴더>"""
import os
import sys

import torch

B = sys.argv[1]
d = torch.load(os.path.join(B, "init.pt"), weights_only=False)
X = torch.load(os.path.join(B, "mpm", "x240.pt"))
for k in (1, 2, 4):
    e = dict(d)
    c = dict(d["cfg"]); c["frame_dt"] = 1.0 / (60 * k); c["frame_num"] = 80 * k
    e["cfg"] = c
    e["x"] = X[:: 4 // k].contiguous()
    N = X.shape[1]
    e["v"] = d["v"].expand(e["x"].shape[0], N, 3)
    torch.save(e, os.path.join(B, f"drop_f{k}.pt"))
    print(k, tuple(e["x"].shape), flush=True)

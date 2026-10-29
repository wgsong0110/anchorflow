"""--lagdump 의 lagdump.pt (torch) 를 같은 내용 numpy npz 로 (torch 없는 로컬 그림용). 원본 .pt 는 그대로 둔다.
사용: python exe/fem_fly_npz.py <lagdump.pt>...   -> 같은 디렉토리 lagdump.npz (키 f<k>_<이름>, k = 덤프 안 순번, meta = json)"""
import json
import os
import sys

import numpy as np
import torch

for p in sys.argv[1:]:
    R = torch.load(p, map_location="cpu", weights_only=False)
    out, meta = {}, []
    for k, r in enumerate(R):
        mk = {}
        for n, v in r.items():
            if torch.is_tensor(v):
                out[f"f{k}_{n}"] = v.numpy()
            else:
                mk[n] = v
        meta.append(mk)
    out["meta"] = np.array(json.dumps(meta))
    q = os.path.join(os.path.dirname(p), "lagdump.npz")
    np.savez(q, **out)
    print(q, [m["t"] + 1 for m in meta], f"{os.path.getsize(q) / 1e6:.0f} MB", flush=True)

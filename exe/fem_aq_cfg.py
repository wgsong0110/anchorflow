"""절 AQ 준비: 씬 자료 (.pt) 의 정답 config·입자 수·프레임 수를 출력 (json 한 줄씩). 사용: python exe/fem_aq_cfg.py <이름>:<데이터.pt> ..."""
import json
import sys

import torch

for arg in sys.argv[1:]:
    nm, path = arg.split(":", 1)
    d = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    print(json.dumps(dict(scene=nm, keys=sorted(d.keys()), x=list(d["x"].shape), ext=float(d.get("_ext", 0)), cfg=d["cfg"]), default=str), flush=True)

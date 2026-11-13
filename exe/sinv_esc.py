"""절 CM128 이탈 집계 (CPU, 읽기만): 판 pred.pt 의 프레임별 입자 위치로 '몸 상자 4 칸 밖' 입자 수 (exe/escwhy_out.py 와 같은 정의:
그 프레임 입자 위치 0.1 %·99.9 % 분위 상자 ± 0.08 m 밖). 첫 이탈 프레임, 프레임별 수, 끝 프레임.
사용: python exe/sinv_esc.py <판 디렉토리>"""
import json
import os
import sys

import numpy as np
import torch

d = sys.argv[1]
z = torch.load(os.path.join(d, "pred.pt"), map_location="cpu", weights_only=False)
P = z["pred"]
P = P.numpy() if hasattr(P, "numpy") else np.asarray(P)
nf = P.shape[0]
cnt, first = [], None
for f in range(nf):
    x = P[f].astype(np.float64)
    lo = np.quantile(x, 0.001, axis=0); hi = np.quantile(x, 0.999, axis=0)
    n = int(((x < lo - 0.08) | (x > hi + 0.08)).any(1).sum())
    cnt.append(n)
    if n > 0 and first is None:
        first = f + 1
print("프레임별 4 칸 밖: " + " ".join(f"f{f + 1}:{c}" for f, c in enumerate(cnt) if c), flush=True)
r = dict(frames=nf, part=bool(z.get("part", False)), diverged=z.get("diverged"), first_escape=first, max_out=max(cnt) if cnt else 0, out=cnt)
json.dump(r, open(os.path.join(d, "esc.json"), "w"))
print(f"[이탈] 프레임 {nf} (부분 {r['part']}, 발산 {r['diverged']}) 첫 이탈 f{first} 최대 {r['max_out']} 개", flush=True)

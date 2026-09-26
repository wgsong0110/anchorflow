"""i-PG 가 낸 프레임별 h5 를 벤치 지표가 읽는 .pt 로 모은다."""
import argparse
import glob
import json
import os
import re

import h5py
import numpy as np
import torch

from anchorflow import phys_resid

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--cfg", required=True)
a = ap.parse_args()

fs = sorted(glob.glob(os.path.join(a.dir, "**", "*.h5"), recursive=True),
            key=lambda p: int(re.findall(r"(\d+)", os.path.basename(p))[-1]))
if not fs:
    raise SystemExit(f"h5 가 없다: {a.dir}")
cfg = json.load(open(a.cfg))
X, F = [], []
for f in fs:
    with h5py.File(f, "r") as d:
        k = "x" if "x" in d else list(d.keys())[0]
        X.append(np.asarray(d[k]).reshape(-1, 3).astype(np.float32))
        for fk in ("F", "particle_F", "def_grad"):
            if fk in d:
                F.append(np.asarray(d[fk]).reshape(-1, 3, 3).astype(np.float32))
                break
x = torch.from_numpy(np.stack(X))
out = dict(x=x.half(), cfg=cfg, n=x.shape[1])
if len(F) == len(X):
    out["F"] = torch.from_numpy(np.stack(F)).half()
else:
    # F 를 안 내놓으면 위치에서 되살린다 (프레임 해상도의 야코비안 누적)
    out["F"] = phys_resid.rebuild_F(x.cuda().float(), cfg,
                                    float(cfg["frame_dt"])).cpu()
torch.save(out, a.out)
print(f"[저장] {a.out} {x.shape[0]}프레임 x {x.shape[1]}입자")

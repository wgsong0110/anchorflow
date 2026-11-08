"""절 CM41: 정답 수박 프레임별 손상 입자 상태 (lib/anchorflow/fem_wmshape2.py frame() 의 입자 칸과 같은 정의).
lb = logJp 구간 (0~0.01·0.01~0.1·0.1~0.4·> 0.4) 몫, dJ·dJt = 손상 입자 탄성 J·전체 부피 비 J·exp(logJp + 0.04) 분위 (10·50·90 %), nJ·nJt = 새 손상 입자, dlq = 프레임 logJp 증분 분위 (1·50·99 %).
위치·F = 원래 정답 GF h5, logJp = 다시 돌린 정답 gt_jp (절 CM13.4).
사용: python exe/wmshape2_gt.py <h5 디렉토리> <gt_jp 디렉토리> <출력 json> [마지막 프레임 100]
"""
import json
import os
import sys

import h5py
import numpy as np
import torch

H5, JP, OUT = sys.argv[1:4]
NF = int(sys.argv[4]) if len(sys.argv) > 4 else 100
dev = "cuda" if torch.cuda.is_available() else "cpu"


def q(t, qs):
    if t.numel() == 0:
        return None
    t = t.float()
    if t.numel() > 8_000_000:
        t = t[torch.randperm(t.numel(), device=t.device)[:8_000_000]]
    return [round(float(v), 5) for v in torch.quantile(t, torch.tensor(qs, device=t.device))]


out, lprev = {}, None
for f in range(1, NF + 1):
    with h5py.File(os.path.join(H5, "sim_%010d.h5" % f), "r") as h:
        F = torch.from_numpy(np.ascontiguousarray(h["f_tensor"][:].T)).to(dev).reshape(-1, 3, 3)
        x = torch.from_numpy(np.ascontiguousarray(h["x"][:].T)).to(dev)
    lj_all = torch.from_numpy(np.load(os.path.join(JP, "sc_%04d.npy" % f))).to(dev).double()
    ok = torch.isfinite(x).all(-1) & torch.isfinite(F.flatten(1)).all(-1) & torch.isfinite(lj_all)
    lj = lj_all[ok]
    J = torch.linalg.det(F[ok].double())
    Jt = J * torch.exp(lj + 0.04)
    dm = lj > 0
    o = dict(lb=[float(((lj > a) & (lj <= b)).double().mean()) for a, b in ((0.0, 0.01), (0.01, 0.1), (0.1, 0.4), (0.4, 1e9))],
             dJ=q(J[dm], [0.1, 0.5, 0.9]), dJt=q(Jt[dm], [0.1, 0.5, 0.9]))
    if lprev is not None:
        lp = lprev[ok]
        nw = dm & ~(lp > 0)
        o.update(nJ=q(J[nw], [0.1, 0.5, 0.9]), nJt=q(Jt[nw], [0.1, 0.5, 0.9]), dlq=q(lj - lp, [0.01, 0.5, 0.99]))
    out[f] = o
    lprev = lj_all
    print(f"정답 f{f} lb {[round(100 * v, 2) for v in o['lb']]} dJt {o['dJt']} nJt {o.get('nJt')}", flush=True)
json.dump(out, open(OUT, "w"), indent=1)

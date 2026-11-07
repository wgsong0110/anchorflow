"""절 CM40: 정답 수박 프레임별 입자 통계 (lib/anchorflow/fem_wmshape.py pstats -- 우리 판 --wsh_rec 와 같은 함수).
위치·탄성 F_e = 원래 정답 GF h5 (x, f_tensor = particle_F, RM 뒤), logJp = 다시 돌린 정답 (lbours/fem_cm/wmJd/gt_jp, 절 CM13.4; 위치 차 f20 1.05 %).
사용: python exe/wmshape_gt.py <wm.pt> <h5 디렉토리> <gt_jp 디렉토리> <출력 json> [마지막 프레임 100]
"""
import json
import os
import sys

import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from anchorflow import fem_wmshape as WSH  # noqa: E402
from anchorflow import phys_resid as PR  # noqa: E402

WM, H5, JP, OUT = sys.argv[1:5]
NF = int(sys.argv[5]) if len(sys.argv) > 5 else 100
dev = "cuda" if torch.cuda.is_available() else "cpu"
d = torch.load(WM, map_location="cpu", weights_only=False)
cfg = d["cfg"]; mu, lam = PR.lame(cfg["E"], cfg["nu"])
ref = WSH.ref_of(d["x"][0].to(dev))
out, lprev = [], None
for f in range(1, NF + 1):
    with h5py.File(os.path.join(H5, "sim_%010d.h5" % f), "r") as h:
        F = torch.from_numpy(np.ascontiguousarray(h["f_tensor"][:].T)).to(dev).reshape(-1, 3, 3)
        x = torch.from_numpy(np.ascontiguousarray(h["x"][:].T)).to(dev)
    lj = torch.from_numpy(np.load(os.path.join(JP, "sc_%04d.npy" % f))).to(dev).float()
    r = WSH.pstats(x, F, lj, lprev, float(mu), float(lam), float(cfg["xi"]), ref)
    r["t"] = f
    out.append(r)
    lprev = lj
    print(f"정답 f{f} H {r['H']:.3f} r95 {r['r95']:.3f} 바닥 r95 {r['foot95']} zc {r['zc']:.3f} 손상 {100 * r['d0']:.2f} % 새 {r.get('nnew')}", flush=True)
json.dump(dict(ref=ref, frames=out), open(OUT, "w"), indent=1)

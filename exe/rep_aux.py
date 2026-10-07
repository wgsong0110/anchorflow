"""영상용: 형상의 3DGS 가우시안 중심을 목표 궤적과 **같은 정규화 좌표**로 옮긴다.

경로: 모델 좌표 -> (PG 러너와 같은 순서) 불투명도 거르기·transform2origin·
shift2center111 -> 시뮬 좌표 -> gauss_flow 의 정규화 (lo, s, 가운데 맞춤).
이렇게 만든 점을 rep_track.py --aux 로 주면 각 표현이 가우시안도 같이 옮긴다.
되돌릴 때 쓰는 값(scale_origin, mean, lo, s, off)도 함께 저장한다.

  cd i-physgaussian && python <anchorflow>/exe/rep_aux.py --shape wolf \
      --fill pgfill_wolf.npy --flow repflow/flow_wolf.npz --out repflow/aux_wolf.npz
"""
import argparse
import json
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--pg", default="/home/dkta/work/i-physgaussian")
ap.add_argument("--shape", required=True)
ap.add_argument("--fill", required=True)
ap.add_argument("--flow", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--max_g", type=int, default=200000, help="가우시안 상한 (불투명도 큰 순)")
a = ap.parse_args()
W = "/home/dkta/work"
MODEL = {"wolf": "wolf_whitebg-trained", "bread": "bread-trained",
         "ship": "ship_whitebg-trained", "lego": "lego_whitebg-trained"}
sys.path.append(a.pg)
sys.path.append(os.path.join(a.pg, "gaussian-splatting"))
os.chdir(a.pg)

import numpy as np                                               # noqa: E402
import torch                                                     # noqa: E402
from utils.decode_param import decode_param_json                 # noqa: E402
from utils.transformation_utils import (                          # noqa: E402
    generate_rotation_matrices, apply_rotations, transform2origin,
    shift2center111)
from scene.gaussian_model import GaussianModel                   # noqa: E402

cfgp = f"{W}/wmats/{a.shape}_fillonly.json"
(mp, bc, tp, pp, cp) = decode_param_json(cfgp)
mpth = f"{W}/pgmodel/{MODEL[a.shape]}"
gs = GaussianModel(3)
gs.load_ply(f"{mpth}/point_cloud/iteration_30000/point_cloud.ply")
op = gs.get_opacity.detach()[:, 0]
keep = op > pp["opacity_threshold"]
pos = gs.get_xyz.detach()[keep]
R = generate_rotation_matrices(torch.tensor(pp["rotation_degree"]),
                               pp["rotation_axis"])
rot = apply_rotations(pos, R)
tpos, so, omp = transform2origin(rot, pp["scale"])
tpos = shift2center111(tpos)                                      # 시뮬 좌표
# gauss_flow 와 같은 정규화
X = np.load(a.fill).astype(np.float64)
lo = X.min(0)
s = float((X.max(0) - lo).max())
off = (1.0 - ((X.max(0) - lo) / s)) / 2.0
G = (tpos.cpu().numpy().astype(np.float64) - lo) / s + off
kidx = torch.nonzero(keep).squeeze(1).cpu().numpy()
if len(G) > a.max_g:                                              # 불투명도 큰 순
    o = np.argsort(-op[keep].cpu().numpy())[:a.max_g]
    o.sort()
    G, kidx = G[o], kidx[o]
D = np.load(a.flow)
assert np.allclose(D["lo"], lo) and abs(float(D["s"]) - s) < 1e-9, "정규화가 다르다"
np.save(a.out.replace(".npz", "_pts.npy"), G.astype(np.float32))
np.savez(a.out, G=G.astype(np.float32), gidx=kidx, lo=lo, s=s, off=off,
         scale_origin=float(so), mean=omp.reshape(-1).cpu().numpy(),
         model=mpth, opacity_threshold=float(pp["opacity_threshold"]))
print(f"[aux] {a.shape}: 가우시안 {len(G)} 개  범위 {G.min(0).round(3)}~"
      f"{G.max(0).round(3)}  scale_origin {float(so):.4f}", flush=True)

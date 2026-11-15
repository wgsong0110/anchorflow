"""repflow A 새 장면 (PG 장면 ficus·vasedeck·plane) 입력 만들기 -- 채우기 없이 (사용자 11-15).

rep_aux.py 와 같은 변환 (PG 러너 순서: 불투명도 거르기 -> 회전 -> transform2origin -> shift2center111) 을 거친 가우시안 중심을
시뮬 좌표로 만들고, gauss_flow 와 같은 정규화 ([0,1]^3, 긴 축 1, 짧은 축 가운데; 기준 = 거른 가우시안 중심의 bbox -- 채우기 입자가 없어서)
로 옮겨 aux npz (rep_track --aux), 정규화 기준 점 (gauss_sim.npy, gauss_flow --fill), 정규화 점 (pts.npy, gauss_flow --pts) 을 낸다.
  cd i-physgaussian && python exe/rep_prep_scene.py --name ficus --cfg config/ficus_config.json --model /home/dkta/work/pgmodel/ficus_whitebg-trained --out_dir DIR
"""
import argparse
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--pg", default="/home/dkta/work/i-physgaussian")
ap.add_argument("--name", required=True)
ap.add_argument("--cfg", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--out_dir", required=True)
ap.add_argument("--max_g", type=int, default=200000, help="가우시안 상한 (불투명도 큰 순, rep_aux 와 같음)")
a = ap.parse_args()
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

(mp, bc, tp, pp, cp) = decode_param_json(a.cfg)
gs = GaussianModel(3)
gs.load_ply(f"{a.model}/point_cloud/iteration_30000/point_cloud.ply")
op = gs.get_opacity.detach()[:, 0]
keep = op > pp["opacity_threshold"]
pos = gs.get_xyz.detach()[keep]
R = generate_rotation_matrices(torch.tensor(pp["rotation_degree"]), pp["rotation_axis"])
rot = apply_rotations(pos, R)
tpos, so, omp = transform2origin(rot, pp["scale"])
tpos = shift2center111(tpos)                                      # 시뮬 좌표
kidx = torch.nonzero(keep).squeeze(1).cpu().numpy()
T = tpos.cpu().numpy().astype(np.float64)
if len(T) > a.max_g:                                              # 불투명도 큰 순
    o = np.argsort(-op[keep].cpu().numpy())[:a.max_g]
    o.sort()
    T, kidx = T[o], kidx[o]
lo = T.min(0)
s = float((T.max(0) - lo).max())
off = (1.0 - ((T.max(0) - lo) / s)) / 2.0
G = (T - lo) / s + off
os.makedirs(a.out_dir, exist_ok=True)
np.save(f"{a.out_dir}/gauss_sim_{a.name}.npy", T.astype(np.float64))
np.save(f"{a.out_dir}/aux_{a.name}_pts.npy", G.astype(np.float32))
np.savez(f"{a.out_dir}/aux_{a.name}.npz", G=G.astype(np.float32), gidx=kidx, lo=lo, s=s, off=off,
         scale_origin=float(so), mean=omp.reshape(-1).cpu().numpy(),
         model=a.model, opacity_threshold=float(pp["opacity_threshold"]))
print(f"[aux] {a.name}: 거른 전 {len(keep)} -> 가우시안 {len(G)} 개 (opacity > {pp['opacity_threshold']})  범위 {G.min(0).round(3)}~"
      f"{G.max(0).round(3)}  scale_origin {float(so):.4f}  n_grid {mp['n_grid']}", flush=True)

"""3DGS 에서 메시를 뽑는다 (PhysGaussian 전처리와 같은 좌표계).

가우시안 중심을 밀도로 격자에 담고, 가우시안 크기만큼 번지게 한 뒤 marching
cubes 로 등위면을 딴다. 시뮬과 같은 좌표계(회전·정규화·격자 중앙 이동)를 쓰므로
그대로 충돌체나 사면체 케이지의 씨앗으로 쓸 수 있다.

문턱은 **점유 칸의 밀도 분포**에서 잡는다. 0 을 쓰면 빈 칸까지 표면으로 잡혀
삼각형이 백만 개 넘게 나온다 (겪었다).

  python exe/gs_to_mesh.py --model <3DGS> --config <씬 config> --out out.obj
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--grid", type=int, default=160, help="격자 해상도")
ap.add_argument("--sigma", type=float, default=1.0, help="번지게 하는 정도 (칸)")
ap.add_argument("--level", type=float, default=0.5,
                help="등위면 높이 (점유 칸 밀도의 분위)")
ap.add_argument("--decimate", type=int, default=0, help="목표 삼각형 수 (0=안 줄임)")
a = ap.parse_args()

import mcubes                                                     # noqa: E402
from scipy.ndimage import gaussian_filter                         # noqa: E402

from scene.gaussian_model import GaussianModel                    # noqa: E402
from utils.decode_param import decode_param_json                  # noqa: E402
from utils.render_utils import load_params_from_gs                # noqa: E402
from utils.system_utils import searchForMaxIteration               # noqa: E402
from utils.transformation_utils import (apply_rotations,           # noqa: E402
                                        generate_rotation_matrices,
                                        shift2center111,
                                        transform2origin)


class _Pipe:
    convert_SHs_python = False
    compute_cov3D_python = True
    debug = False


(mat, bc, tp, pre, cam) = decode_param_json(a.config)
g = GaussianModel(3)
it = searchForMaxIteration(os.path.join(a.model, "point_cloud"))
g.load_ply(os.path.join(a.model, "point_cloud", f"iteration_{it}", "point_cloud.ply"))
p = load_params_from_gs(g, _Pipe())
pos, opa = p["pos"], p["opacity"]
keep = opa[:, 0] > pre["opacity_threshold"]
pos = pos[keep]
rots = generate_rotation_matrices(torch.tensor(pre["rotation_degree"]),
                                  pre["rotation_axis"])
pos = apply_rotations(pos, rots)
pos, scale_origin, mean_pos = transform2origin(pos, float(pre.get("scale", 1.0)))
pos = shift2center111(pos).detach().cpu().numpy().astype(np.float64)
if pre.get("sim_area") is not None:                     # 씬의 시뮬 영역만
    ar = pre["sim_area"]
    print(f"[주의] sim_area 가 있는 씬이다 -- 전체를 뽑는다 (필요하면 잘라 쓰라)",
          flush=True)
print(f"[입력] 가우시안 {pos.shape[0]}, 범위 {pos.min(0).round(3)} ~ "
      f"{pos.max(0).round(3)}", flush=True)

N = a.grid
lo, hi = pos.min(0) - 0.04, pos.max(0) + 0.04
h = (hi - lo) / N
idx = np.floor((pos - lo) / h).astype(int).clip(0, N - 1)
vol = np.zeros((N, N, N), np.float32)
np.add.at(vol, (idx[:, 0], idx[:, 1], idx[:, 2]), 1.0)
vol = gaussian_filter(vol, sigma=a.sigma)
occ = vol[vol > 1e-6]
thr = float(np.quantile(occ, a.level))
print(f"[격자] {N}^3, 번짐 {a.sigma} 칸, 문턱 {thr:.4f} "
      f"(점유 칸 {int((vol > thr).sum())})", flush=True)
v, f = mcubes.marching_cubes(vol, thr)
v = v / N * (hi - lo) + lo
print(f"[메시] 정점 {v.shape[0]}, 삼각형 {f.shape[0]}", flush=True)
mcubes.export_obj(v, f, a.out)
np.savez(os.path.splitext(a.out)[0] + ".npz", v=v, f=f,
         scale_origin=float(scale_origin), mean_pos=mean_pos.detach().cpu().numpy())
print(f"[저장] {a.out}", flush=True)
print("MESH_OK", flush=True)

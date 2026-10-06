"""i-PG 시연 실행과 **똑같은** 카메라와 좌표 변환을 뽑는다 (Spring-Gaus 렌더용).

Spring-Gaus 장면 좌표는 원래 3DGS 모델 좌표와 같다 (피팅용 카메라를 PG 에서 그대로
뽑았다). 그래서 i-PG 시연과 같은 카메라로 같은 원본 가우시안을 그리면 첫 프레임이
같아진다. 카메라는 그쪽 러너(gs_simulation.py)와 **같은 순서**로 만든다:
불투명도 거르기 -> 회전 -> sim_area -> transform2origin -> shift2center111.
(불투명도 거르기를 빼면 떠다니는 가우시안 때문에 scale_origin 이 달라진다.)

시뮬 공간 <-> 모델 공간 변환(scale_origin, original_mean_pos)도 같이 남긴다 --
바닥 높이처럼 시뮬 공간에서 정한 값을 모델 공간으로 옮길 때 쓴다:
    x_model = (x_sim - 1) / scale_origin + original_mean_pos   (회전이 항등일 때)

  cd i-physgaussian && python <anchorflow>/exe/dump_demo_cam.py \
      --model_path <3DGS> --config <시연 설정.json> --out cam.json
"""
import argparse
import json
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--pg", default="/home/dkta/work/i-physgaussian")
ap.add_argument("--model_path", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
sys.path.append(a.pg)
sys.path.append(os.path.join(a.pg, "gaussian-splatting"))
os.chdir(a.pg)

import numpy as np                                               # noqa: E402
import torch                                                     # noqa: E402
from utils.decode_param import decode_param_json                 # noqa: E402
from utils.transformation_utils import (                          # noqa: E402
    generate_rotation_matrices, apply_rotations, transform2origin,
    shift2center111, get_center_view_worldspace_and_observant_coordinate)
from utils.camera_view_utils import get_camera_view              # noqa: E402
from scene.gaussian_model import GaussianModel                   # noqa: E402

(mp, bc, tp, pp, cp) = decode_param_json(a.config)
gs = GaussianModel(3)
gs.load_ply(os.path.join(a.model_path, "point_cloud/iteration_30000/point_cloud.ply"))
pos = gs.get_xyz.detach()
op = gs.get_opacity.detach()
keep = op[:, 0] > pp["opacity_threshold"]
pos = pos[keep]
R = generate_rotation_matrices(torch.tensor(pp["rotation_degree"]),
                               pp["rotation_axis"])
rot = apply_rotations(pos, R)
if pp.get("sim_area") is not None:
    b = pp["sim_area"]
    m = torch.ones(rot.shape[0], dtype=torch.bool, device=rot.device)
    for i in range(3):
        m &= (rot[:, i] > b[2 * i]) & (rot[:, i] < b[2 * i + 1])
    rot = rot[m]
tpos, so, omp = transform2origin(rot, pp["scale"])
tpos = shift2center111(tpos)
center, obs = get_center_view_worldspace_and_observant_coordinate(
    torch.tensor(cp["mpm_space_viewpoint_center"]).reshape(1, 3).cuda(),
    torch.tensor(cp["mpm_space_vertical_upward_axis"]).reshape(1, 3).cuda(),
    R, so, omp)
cam = get_camera_view(
    a.model_path, default_camera_index=cp["default_camera_index"],
    center_view_world_space=center, observant_coordinates=obs, show_hint=False,
    init_azimuthm=cp["init_azimuthm"], init_elevation=cp["init_elevation"],
    init_radius=cp["init_radius"], move_camera=False, current_frame=0,
    delta_a=cp["delta_a"], delta_e=cp["delta_e"], delta_r=cp["delta_r"])
out = dict(R=np.asarray(cam.R).tolist(), T=np.asarray(cam.T).tolist(),
           FoVx=float(cam.FoVx), FoVy=float(cam.FoVy),
           width=int(cam.image_width), height=int(cam.image_height),
           opacity_threshold=float(pp["opacity_threshold"]),
           scale_origin=float(so), original_mean_pos=omp.reshape(-1).tolist(),
           rotation_identity=bool(len(pp["rotation_degree"]) == 0
                                  or all(abs(d) < 1e-9
                                         for d in pp["rotation_degree"])))
json.dump(out, open(a.out, "w"), indent=1)
print(f"[카메라] {a.out}  {out['width']}x{out['height']}  scale_origin "
      f"{out['scale_origin']:.5f}  mean {np.round(out['original_mean_pos'], 4)}")

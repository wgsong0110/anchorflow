"""PhysGaussian 공식 코드로 궤도 카메라를 프레임별로 뽑는다.

sim_area 로 자른 **뒤에** transform2origin 을 부르는 순서를 지켜야 한다 --
전체 클라우드로 계산하면 scale_origin 이 두 자릿수 달라져 카메라가 엉뚱한 곳에 선다.
"""
import sys, json, argparse
ap = argparse.ArgumentParser()
ap.add_argument("--pg", required=True)
ap.add_argument("--model_path", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--frames", type=int, default=40)
ap.add_argument("--out", required=True)
a = ap.parse_args()
sys.path.append(a.pg); sys.path.append(a.pg + "/gaussian-splatting")
import torch, numpy as np
from utils.decode_param import decode_param_json
from utils.transformation_utils import (generate_rotation_matrices, apply_rotations,
    transform2origin, shift2center111, get_center_view_worldspace_and_observant_coordinate)
from utils.camera_view_utils import get_camera_view
from scene.gaussian_model import GaussianModel
(mp, bc, tp, pp, cp) = decode_param_json(a.config)
gs = GaussianModel(3)
gs.load_ply(a.model_path + "/point_cloud/iteration_30000/point_cloud.ply")
R = generate_rotation_matrices(torch.tensor(pp["rotation_degree"]), pp["rotation_axis"])
rot = apply_rotations(gs.get_xyz.detach(), R)
if pp.get("sim_area") is not None:
    b = pp["sim_area"]
    m = torch.ones(rot.shape[0], dtype=torch.bool, device=rot.device)
    for i in range(3):
        m &= (rot[:, i] > b[2*i]) & (rot[:, i] < b[2*i+1])
    rot = rot[m, :]
    print("sim_area 뒤", int(m.sum()))
tpos, so, omp = transform2origin(rot, pp["scale"])
tpos = shift2center111(tpos)
center, obs = get_center_view_worldspace_and_observant_coordinate(
    torch.tensor(cp["mpm_space_viewpoint_center"]).reshape(1,3).cuda(),
    torch.tensor(cp["mpm_space_vertical_upward_axis"]).reshape(1,3).cuda(), R, so, omp)
out = []
for f in range(a.frames):
    cam = get_camera_view(a.model_path, default_camera_index=cp["default_camera_index"],
        center_view_world_space=center, observant_coordinates=obs, show_hint=False,
        init_azimuthm=cp["init_azimuthm"], init_elevation=cp["init_elevation"],
        init_radius=cp["init_radius"], move_camera=cp["move_camera"], current_frame=f,
        delta_a=cp["delta_a"], delta_e=cp["delta_e"], delta_r=cp["delta_r"])
    out.append(dict(R=np.asarray(cam.R).tolist(), T=np.asarray(cam.T).tolist(),
                    FoVx=float(cam.FoVx), FoVy=float(cam.FoVy),
                    width=int(cam.image_width), height=int(cam.image_height)))
json.dump(out, open(a.out, "w"))
print("카메라", len(out), "->", a.out, out[0]["width"], "x", out[0]["height"])

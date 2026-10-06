"""시연 카메라 고르기: 원본 3DGS 를 여러 궤도 카메라로 한 장씩 그린다.

PG 러너와 같은 경로로 카메라를 만든다 (default_camera_index=-1 이어야 방위각·
고도·반경이 먹는다 -- 0 이면 학습 카메라 0 번을 그대로 쓴다).

  cd i-physgaussian && python <anchorflow>/exe/try_demo_cams.py --config <설정.json> \
      --model_path <3DGS> --out <폴더> --azims 0 90 180 270 --elev 20 --radius 2.5
"""
import argparse
import json
import math
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--pg", default="/home/dkta/work/i-physgaussian")
ap.add_argument("--model_path", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--azims", type=float, nargs="+", default=[0, 90, 180, 270])
ap.add_argument("--elev", type=float, default=20.0)
ap.add_argument("--radius", type=float, default=2.5)
a = ap.parse_args()
sys.path.append(a.pg)
sys.path.append(os.path.join(a.pg, "gaussian-splatting"))
os.chdir(a.pg)

import numpy as np                                               # noqa: E402
import torch                                                     # noqa: E402
import imageio.v2 as imageio                                     # noqa: E402
from utils.decode_param import decode_param_json                 # noqa: E402
from utils.transformation_utils import (                          # noqa: E402
    generate_rotation_matrices, apply_rotations, transform2origin,
    shift2center111, get_center_view_worldspace_and_observant_coordinate)
from utils.camera_view_utils import get_camera_view              # noqa: E402
from scene.gaussian_model import GaussianModel                   # noqa: E402
from diff_gaussian_rasterization import (                         # noqa: E402
    GaussianRasterizationSettings, GaussianRasterizer)

os.makedirs(a.out, exist_ok=True)
(mp, bc, tp, pp, cp) = decode_param_json(a.config)
gs = GaussianModel(3)
gs.load_ply(os.path.join(a.model_path, "point_cloud/iteration_30000/point_cloud.ply"))
keep = gs.get_opacity[:, 0] > pp["opacity_threshold"]
pos = gs.get_xyz.detach()[keep]
R = generate_rotation_matrices(torch.tensor(pp["rotation_degree"]),
                               pp["rotation_axis"])
tpos, so, omp = transform2origin(apply_rotations(pos, R), pp["scale"])
center, obs = get_center_view_worldspace_and_observant_coordinate(
    torch.tensor(cp["mpm_space_viewpoint_center"]).reshape(1, 3).cuda(),
    torch.tensor(cp["mpm_space_vertical_upward_axis"]).reshape(1, 3).cuda(),
    R, so, omp)
cov = gs.get_covariance()[keep]
shs = gs.get_features[keep]
op = gs.get_opacity[keep]
for az in a.azims:
    cam = get_camera_view(
        a.model_path, default_camera_index=-1, center_view_world_space=center,
        observant_coordinates=obs, show_hint=False, init_azimuthm=az,
        init_elevation=a.elev, init_radius=a.radius, move_camera=False,
        current_frame=0, delta_a=0, delta_e=0, delta_r=0)
    st = GaussianRasterizationSettings(
        image_height=int(cam.image_height), image_width=int(cam.image_width),
        tanfovx=math.tan(cam.FoVx * 0.5), tanfovy=math.tan(cam.FoVy * 0.5),
        bg=torch.ones(3, device="cuda"), scale_modifier=1.0,
        viewmatrix=cam.world_view_transform, projmatrix=cam.full_proj_transform,
        sh_degree=3, campos=cam.camera_center, prefiltered=False, debug=False)
    with torch.no_grad():
        img = GaussianRasterizer(raster_settings=st)(
            means3D=gs.get_xyz[keep], means2D=torch.zeros_like(gs.get_xyz[keep]),
            shs=shs, colors_precomp=None, opacities=op, scales=None,
            rotations=None, cov3D_precomp=cov)[0]
    fn = f"{a.out}/az{int(az):03d}_el{int(a.elev)}_r{a.radius}.png"
    imageio.imwrite(fn, (img.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                         * 255).astype(np.uint8))
    print(f"[카메라] az {az} -> 중심 {cam.camera_center.tolist()}  {fn}", flush=True)

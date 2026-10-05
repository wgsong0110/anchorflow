"""학습된 3DGS 에서 **학습용 뷰를 다시 렌더**해 NeRF-synthetic 형식 데이터셋을 만든다.

왜: GASP 는 GaMeS(`--gs_type gs_flat`) 모델이 필요하고 GaMeS 는 이미지가 필요한데,
wolf·bread 는 원본 이미지가 남아 있지 않다 (다른 사용자 홈에 있었고 사라졌다).
모델에 동봉된 `cameras.json`(400 뷰)으로 그 시점들을 그대로 렌더하면 같은 카메라
배치의 데이터셋이 만들어진다. 원본 사진이 아니라 **3DGS 의 렌더**라는 점은 결과에
반드시 같이 적는다.

좌표 변환: `cameras.json` 의 (position, rotation) 은 COLMAP 규약의 camera-to-world
다 (gaussian-splatting 의 `camera_to_JSON`). Blender/NeRF 의 transform_matrix 는
y·z 축이 뒤집힌 규약이라 2·3 열의 부호를 바꾼다.

  python exe/render_views.py --model <pgmodel/x-trained> --out <데이터셋 경로>
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--pg", default="/home/dkta/work/PhysGaussian")
ap.add_argument("--iteration", type=int, default=30000)
ap.add_argument("--sh_degree", type=int, default=3)
ap.add_argument("--test_every", type=int, default=8, help="N 장마다 1 장을 test 로")
a = ap.parse_args()

sys.path.insert(0, a.pg)
sys.path.insert(0, os.path.join(a.pg, "gaussian-splatting"))
from scene.cameras import Camera                                # noqa: E402
from gaussian_renderer import render, GaussianModel             # noqa: E402
from utils.graphics_utils import focal2fov                      # noqa: E402
from argparse import Namespace                                  # noqa: E402
from PIL import Image                                           # noqa: E402

cams = json.load(open(os.path.join(a.model, "cameras.json")))
print(f"[카메라] {len(cams)} 뷰, {cams[0]['width']}x{cams[0]['height']}", flush=True)

gaussians = GaussianModel(a.sh_degree)
ply = os.path.join(a.model, "point_cloud", f"iteration_{a.iteration}",
                   "point_cloud.ply")
gaussians.load_ply(ply)
print(f"[모델] {ply}  가우시안 {gaussians.get_xyz.shape[0]}", flush=True)

bg = torch.tensor([1.0, 1.0, 1.0], device="cuda")      # 흰 배경 (학습도 흰 배경)
pipe = Namespace(convert_SHs_python=False, compute_cov3D_python=False,
                 debug=False)

splits = {"train": [], "test": []}
for sp in splits:
    os.makedirs(os.path.join(a.out, sp), exist_ok=True)

for i, c in enumerate(cams):
    R_c2w = np.array(c["rotation"], dtype=np.float64)       # [3,3]
    pos = np.array(c["position"], dtype=np.float64)         # 월드 좌표의 카메라 중심
    # gaussian-splatting 의 Camera 는 (R=c2w 회전, T=world2cam 평행이동) 을 받는다
    W2C_R = R_c2w.T
    T = -W2C_R @ pos
    fovx = focal2fov(c["fx"], c["width"])
    fovy = focal2fov(c["fy"], c["height"])
    cam = Camera(colmap_id=i, R=R_c2w, T=T, FoVx=fovx, FoVy=fovy,
                 image=torch.zeros(3, c["height"], c["width"]),
                 gt_alpha_mask=None, image_name=f"r_{i:04d}", uid=i,
                 data_device="cuda")
    with torch.no_grad():
        # PG 가 품은 gaussian-splatting 은 변형 인자(d_xyz, d_rotation,
        # d_scaling)를 요구하는 갈래다 -- 정적 렌더이므로 0 을 준다
        out = render(cam, gaussians, pipe, bg, 0.0, 0.0,
                     0.0)["render"].clamp(0.0, 1.0)
    sp = "test" if (i % a.test_every == 0) else "train"
    rel = f"./{sp}/r_{i:04d}"
    # NeRF-synthetic 리더는 **RGBA** 를 가정한다 (알파로 배경을 합성한다).
    # 우리 렌더는 흰 배경이 이미 구워져 있으므로 알파는 1 로 둔다 -- 흰 배경
    # 합성이 항등이 되어 그림이 그대로 남는다.
    rgb = (out.permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
    rgba = np.concatenate([rgb, np.full(rgb.shape[:2] + (1,), 255,
                                        dtype=np.uint8)], axis=-1)
    Image.fromarray(rgba, mode="RGBA").save(os.path.join(a.out, f"{rel}.png"))
    # blender 규약: c2w 의 y·z 축 부호 반전
    M = np.eye(4)
    M[:3, :3] = R_c2w
    M[:3, 3] = pos
    M[:3, 1] *= -1.0
    M[:3, 2] *= -1.0
    splits[sp].append(dict(file_path=rel, rotation=0.0,
                           transform_matrix=M.tolist()))
    if (i + 1) % 50 == 0:
        print(f"  {i + 1}/{len(cams)}", flush=True)

ang = 2.0 * math.atan(cams[0]["width"] / (2.0 * cams[0]["fx"]))
for sp, frames in splits.items():
    json.dump(dict(camera_angle_x=ang, frames=frames),
              open(os.path.join(a.out, f"transforms_{sp}.json"), "w"), indent=1)
    print(f"[저장] transforms_{sp}.json  {len(frames)} 장", flush=True)
# GaMeS 의 readNerfSyntheticInfo 는 val 도 읽을 수 있게 둔다
json.dump(dict(camera_angle_x=ang, frames=splits["test"]),
          open(os.path.join(a.out, "transforms_val.json"), "w"), indent=1)
print(f"[완료] {a.out}  camera_angle_x {ang:.5f} rad "
      f"({math.degrees(ang):.2f} 도)", flush=True)

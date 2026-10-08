"""Fracture-GS Teapot & Table 대체 장면: 학습한 3DGS 두 개를 실제 크기 비율로 한 장면에 놓고 첫 프레임을 렌더한다.

자산 (논문 장면은 미공개라 대체): Poly Haven CC0 tea_set_01 의 찻주전자(뚜껑 포함), wooden_table_02.
  - 각 3DGS 는 mesh_to_nerfsynth 로 렌더한 다시점(가장 긴 변 2.0)에서 학습했다 -> meta.json 의 배율로 실제 크기(m)로
  - 장면 좌표는 m, z 위. 탁자 다리 밑이 바닥 z = 0, 찻주전자는 탁자 상판 중앙 위 --drop_h 높이
  - 렌더: 공식 래스터라이저, 흰 배경, 비스듬한 위 시점

  cd i-physgaussian && python <anchorflow>/exe/fgs_scene.py --assets /home/dkta/work/fgs_assets --out DIR
"""
import argparse
import json
import math
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--pg", default="/home/dkta/work/i-physgaussian")
ap.add_argument("--assets", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--teapot_scale", type=float, default=1.0, help="찻주전자 실제 크기 배수")
ap.add_argument("--drop_h", type=float, default=0.3, help="상판 윗면에서 찻주전자 바닥까지 (m)")
ap.add_argument("--opacity", type=float, default=0.02)
ap.add_argument("--res", type=int, default=1000)
a = ap.parse_args()
sys.path.append(a.pg)
sys.path.append(os.path.join(a.pg, "gaussian-splatting"))

import numpy as np                                               # noqa: E402
import torch                                                     # noqa: E402
import imageio.v2 as imageio                                     # noqa: E402
from scene.gaussian_model import GaussianModel                   # noqa: E402
from utils.graphics_utils import getWorld2View2, getProjectionMatrix  # noqa: E402
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer  # noqa: E402

dev = torch.device("cuda")


def load(name):
    meta = json.load(open(f"{a.assets}/{name}_ns/meta.json"))
    gs = GaussianModel(3)
    gs.load_ply(f"{a.assets}/{name}_gs/point_cloud/iteration_30000/point_cloud.ply")
    keep = gs.get_opacity.detach()[:, 0] > a.opacity
    x = gs.get_xyz.detach()[keep] / meta["scale"]                 # 실제 크기 (m), 중심 원점
    c6 = gs.get_covariance()[keep].detach() / meta["scale"] ** 2
    return dict(x=x, c6=c6, shs=gs.get_features[keep].detach(), op=gs.get_opacity[keep].detach())


T, P = load("table"), load("teapot")
P["x"] = P["x"] * a.teapot_scale; P["c6"] = P["c6"] * a.teapot_scale ** 2
# 탁자: 다리 밑을 바닥 z = 0 에
T["x"] = T["x"] - torch.tensor([0.0, 0.0, float(T["x"][:, 2].min())], device=dev)
ztop = float(torch.quantile(T["x"][:, 2], 0.995))
# 찻주전자: 상판 중앙 위
pc = 0.5 * (P["x"].min(0).values + P["x"].max(0).values)
P["x"] = P["x"] - pc + torch.tensor([0.0, 0.0, ztop + a.drop_h + float(pc[2] - P["x"][:, 2].min())], device=dev)
print(f"[장면] 탁자 {T['x'].shape[0]} 가우시안, 크기 {np.round((T['x'].max(0).values - T['x'].min(0).values).cpu().numpy(), 3)} m, "
      f"상판 윗면 z {ztop:.3f}", flush=True)
print(f"[장면] 찻주전자 {P['x'].shape[0]} 가우시안, 크기 {np.round((P['x'].max(0).values - P['x'].min(0).values).cpu().numpy(), 3)} m, "
      f"바닥 z {float(P['x'][:, 2].min()):.3f}", flush=True)

X = torch.cat([T["x"], P["x"]]); C6 = torch.cat([T["c6"], P["c6"]])
SH = torch.cat([T["shs"], P["shs"]]); OP = torch.cat([T["op"], P["op"]])
# 카메라: 장면 중심을 비스듬히 위에서
ctr = 0.5 * (X.min(0).values + X.max(0).values).cpu().numpy()
ext = float((X.max(0).values - X.min(0).values).max())
fov = 0.6911
eye = ctr + np.array([0.55, -1.0, 0.45]) / np.linalg.norm([0.55, -1.0, 0.45]) * (0.75 * ext / math.tan(fov / 2))
f = (ctr - eye) / np.linalg.norm(ctr - eye); up = np.array([0, 0, 1.0])
r = np.cross(f, up); r /= np.linalg.norm(r); u = np.cross(r, f)
c2w = np.eye(4); c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = r, -u, f, eye     # COLMAP (x 오른, y 아래, z 앞)
W2C = np.linalg.inv(c2w)
wv = torch.tensor(getWorld2View2(W2C[:3, :3].T, W2C[:3, 3])).transpose(0, 1).to(dev).float()
pj = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=fov, fovY=fov).transpose(0, 1).to(dev).float()
RAST = GaussianRasterizer(raster_settings=GaussianRasterizationSettings(
    image_height=a.res, image_width=a.res, tanfovx=math.tan(fov / 2), tanfovy=math.tan(fov / 2),
    bg=torch.ones(3, device=dev), scale_modifier=1.0, viewmatrix=wv, projmatrix=(wv[None] @ pj[None])[0],
    sh_degree=3, campos=wv.inverse()[3, :3], prefiltered=False, debug=False))
with torch.no_grad():
    img = RAST(means3D=X, means2D=torch.zeros_like(X), shs=SH, colors_precomp=None, opacities=OP,
               scales=None, rotations=None, cov3D_precomp=C6)[0]
os.makedirs(a.out, exist_ok=True)
imageio.imwrite(f"{a.out}/frame0.png", (img.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
json.dump(dict(teapot_scale=a.teapot_scale, drop_h=a.drop_h, table_top_z=ztop, n_table=int(T["x"].shape[0]),
               n_teapot=int(P["x"].shape[0]), cam_c2w_colmap=c2w.tolist(), fov=fov),
          open(f"{a.out}/scene.json", "w"), indent=1)
print(f"[저장] {a.out}/frame0.png", flush=True)

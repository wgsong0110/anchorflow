"""Fracture-GS Teapot & Table 대체 장면의 시뮬 결과를 3DGS 로 렌더한다 (공식 래스터라이저, 흰 배경).

가우시안 = 각 물체 채우기 결과의 앞부분 입자 (fgs_init.py 의 gaussians.pt 의 gi). 위치는 그 입자 위치,
공분산은 F Σ Fᵀ (시뮬 h5 의 F). 카메라는 fgs_scene.py 와 같은 비스듬한 위 시점 (초기 장면 기준).

  cd i-physgaussian && python <anchorflow>/exe/fgs_render.py --run DIR --out DIR/video.mp4
"""
import argparse
import glob
import math
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--pg", default="/home/dkta/work/i-physgaussian")
ap.add_argument("--run", required=True, help="fgs_init 출력 (gaussians.pt) + sim/ (h5 프레임)")
ap.add_argument("--out", required=True)
ap.add_argument("--res", type=int, default=1000)
ap.add_argument("--frames", type=int, default=0, help="0 이면 있는 만큼")
ap.add_argument("--fps", type=int, default=30)
a = ap.parse_args()
sys.path.append(a.pg)
sys.path.append(os.path.join(a.pg, "gaussian-splatting"))

import h5py                                                      # noqa: E402
import imageio.v2 as imageio                                     # noqa: E402
import numpy as np                                               # noqa: E402
import torch                                                     # noqa: E402
from utils.graphics_utils import getWorld2View2, getProjectionMatrix  # noqa: E402
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer  # noqa: E402

dev = torch.device("cuda")
G = torch.load(f"{a.run}/gaussians.pt")
GI = G["gi"].to(dev).long(); SH = G["shs"].to(dev); OP = G["op"].to(dev)
c6 = G["c6"].to(dev)
C0 = torch.zeros(c6.shape[0], 3, 3, device=dev)
C0[:, 0, 0], C0[:, 0, 1], C0[:, 0, 2] = c6[:, 0], c6[:, 1], c6[:, 2]
C0[:, 1, 1], C0[:, 1, 2], C0[:, 2, 2] = c6[:, 3], c6[:, 4], c6[:, 5]
C0[:, 1, 0], C0[:, 2, 0], C0[:, 2, 1] = c6[:, 1], c6[:, 2], c6[:, 4]
fs = sorted(glob.glob(f"{a.run}/sim/sim_*.h5"))
if a.frames:
    fs = fs[:a.frames + 1]


def rd(f):
    with h5py.File(f, "r") as h:
        x = np.array(h["x"]); x = x.T if x.shape[0] == 3 else x
        F = np.array(h["F"]).reshape(-1, 3, 3) if "F" in h else None
    return x, F


x0, _ = rd(fs[0])
P0 = torch.as_tensor(x0, device=dev).float()[GI]
ctr = (0.5 * (P0.min(0).values + P0.max(0).values)).cpu().numpy()
ext = float((P0.max(0).values - P0.min(0).values).max())
fov = 0.6911
d = np.array([0.55, -1.0, 0.45]); d = d / np.linalg.norm(d)
eye = ctr + d * (0.75 * ext / math.tan(fov / 2))
f = (ctr - eye) / np.linalg.norm(ctr - eye); up = np.array([0, 0, 1.0])
r = np.cross(f, up); r /= np.linalg.norm(r); u = np.cross(r, f)
c2w = np.eye(4); c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = r, -u, f, eye     # COLMAP
W2C = np.linalg.inv(c2w)
wv = torch.tensor(getWorld2View2(W2C[:3, :3].T, W2C[:3, 3])).transpose(0, 1).to(dev).float()
pj = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=fov, fovY=fov).transpose(0, 1).to(dev).float()
RAST = GaussianRasterizer(raster_settings=GaussianRasterizationSettings(
    image_height=a.res, image_width=a.res, tanfovx=math.tan(fov / 2), tanfovy=math.tan(fov / 2),
    bg=torch.ones(3, device=dev), scale_modifier=1.0, viewmatrix=wv, projmatrix=(wv[None] @ pj[None])[0],
    sh_degree=3, campos=wv.inverse()[3, :3], prefiltered=False, debug=False))
WR = imageio.get_writer(a.out, fps=a.fps, codec="libx264", quality=8)
os.makedirs(os.path.splitext(a.out)[0] + "_frames", exist_ok=True)
for t, fp in enumerate(fs):
    x, F = rd(fp)
    X = torch.as_tensor(x, device=dev).float()[GI]
    ok = torch.isfinite(X).all(1)
    if F is not None:
        Fg = torch.as_tensor(F, device=dev).float()[GI]
        cov = Fg @ C0 @ Fg.transpose(1, 2)
    else:
        cov = C0
    c6_ = torch.stack([cov[:, 0, 0], cov[:, 0, 1], cov[:, 0, 2], cov[:, 1, 1], cov[:, 1, 2], cov[:, 2, 2]], 1)
    with torch.no_grad():
        img = RAST(means3D=X[ok], means2D=torch.zeros_like(X[ok]), shs=SH[ok], colors_precomp=None,
                   opacities=OP[ok], scales=None, rotations=None, cov3D_precomp=c6_[ok])[0]
    im = (img.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    WR.append_data(im)
    if t % 10 == 0:
        imageio.imwrite(f"{os.path.splitext(a.out)[0]}_frames/{t:03d}.png", im)
WR.close()
print(f"[영상] {a.out} ({len(fs)} 프레임)", flush=True)

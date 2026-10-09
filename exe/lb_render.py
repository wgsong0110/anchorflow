"""lb_gen.py 궤적을 3DGS 알파블렌딩으로 렌더 (rep_ip 와 같은 렌더러·카메라 규약).

가우시안 ↔ 입자: lego 는 채움 캐시 앞 NG 개가 불투명도 필터를 거친 가우시안 그대로라 **같은 번호**로 대응한다
(rep_ip 처럼 0 프레임 최근접으로 짝지으면 lb_gen 의 초기 회전 때문에 틀어진다). 시작 시 회전 전 입력 h5 로 대응을 검산한다.
공분산 = F C0 Fᵀ (F 에 초기 회전이 실려 있다).

  python exe/lb_render.py --sim DIR --config cfg.json --shape lego --out DIR/video.mp4
"""
import argparse
import glob
import json
import math
import os
import sys

W = "/home/dkta/work"
MODEL = {"lego": "lego_whitebg-trained", "mic": "mic_whitebg-trained", "ficus": "ficus_whitebg-trained"}
ap = argparse.ArgumentParser()
ap.add_argument("--sim", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--shape", default="lego")
ap.add_argument("--out", required=True)
ap.add_argument("--fps", type=int, default=30)
a = ap.parse_args()
import h5py                                                      # noqa: E402
import imageio.v2 as imageio                                     # noqa: E402
import numpy as np                                               # noqa: E402
import torch                                                     # noqa: E402
torch.set_grad_enabled(False)
dev = "cuda"
cfg = json.load(open(a.config))
meta = json.load(open(f"{a.sim}/meta.json"))
sys.path.append(f"{W}/i-physgaussian"); sys.path.append(f"{W}/i-physgaussian/gaussian-splatting")
_cwd = os.getcwd(); os.chdir(f"{W}/i-physgaussian")
from scene.gaussian_model import GaussianModel                       # noqa: E402
from utils.transformation_utils import transform2origin, shift2center111   # noqa: E402
from utils.graphics_utils import getWorld2View2, getProjectionMatrix      # noqa: E402
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer   # noqa: E402
os.chdir(_cwd)
mp = f"{W}/pgmodel/{MODEL[a.shape]}"
gs = GaussianModel(3)
gs.load_ply(f"{mp}/point_cloud/iteration_30000/point_cloud.ply")
KIDX = torch.nonzero(gs.get_opacity[:, 0] > float(cfg.get("opacity_threshold", 0.02))).squeeze(1)
TP, SO, MEAN = transform2origin(gs.get_xyz[KIDX], float(cfg.get("scale", 1.0)))
TP = shift2center111(TP)
NG = KIDX.numel()
with h5py.File(meta["h5"]) as h:
    x0 = np.array(h["x"]); x0 = x0.T if x0.shape[0] == 3 else x0
err = float((torch.as_tensor(x0[:NG], device=dev).float() - TP).norm(dim=1).max())
print(f"[대응] 가우시안 {NG}, 회전 전 입자 앞 {NG} 개와 위치 차이 최대 {err:.2e}", flush=True)
assert err < 1e-4, "채움 캐시 앞부분이 가우시안과 같은 번호가 아니다"
c6 = gs.get_covariance()[KIDX]
C0 = torch.zeros(NG, 3, 3, device=dev)
C0[:, 0, 0], C0[:, 0, 1], C0[:, 0, 2] = c6[:, 0], c6[:, 1], c6[:, 2]
C0[:, 1, 1], C0[:, 1, 2], C0[:, 2, 2] = c6[:, 3], c6[:, 4], c6[:, 5]
C0[:, 1, 0], C0[:, 2, 0], C0[:, 2, 1] = c6[:, 1], c6[:, 2], c6[:, 4]
SHS, OPA = gs.get_features[KIDX], gs.get_opacity[KIDX]
cam = json.load(open(f"{mp}/cameras.json"))[0]
Rw, pos = np.array(cam["rotation"]), np.array(cam["position"])
_ctr = MEAN.cpu().numpy().astype(np.float64)
pos = _ctr + 1.6 * (pos - _ctr)                                     # 회전·낙하·이동이 다 들어오게 뒤로
W2C = np.linalg.inv(np.block([[Rw, pos[:, None]], [np.zeros((1, 3)), np.ones((1, 1))]]))
fx_ = 2 * math.atan(cam["width"] / (2 * cam["fx"])); fy_ = 2 * math.atan(cam["height"] / (2 * cam["fy"]))
wv = torch.tensor(getWorld2View2(W2C[:3, :3].T, W2C[:3, 3])).transpose(0, 1).to(dev).float()
pj = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=fx_, fovY=fy_).transpose(0, 1).to(dev).float()
RAST = GaussianRasterizer(raster_settings=GaussianRasterizationSettings(
    image_height=int(cam["height"]), image_width=int(cam["width"]), tanfovx=math.tan(fx_ * 0.5), tanfovy=math.tan(fy_ * 0.5),
    bg=torch.ones(3, device=dev), scale_modifier=1.0, viewmatrix=wv, projmatrix=(wv[None] @ pj[None])[0], sh_degree=3,
    campos=wv.inverse()[3, :3], prefiltered=False, debug=False))
WR = imageio.get_writer(a.out, fps=a.fps, codec="libx264", quality=8)
files = sorted(glob.glob(f"{a.sim}/sim_*.h5"))
for p in files:
    with h5py.File(p) as h:
        x = torch.as_tensor(np.array(h["x"][:NG]), device=dev).float()
        F = torch.as_tensor(np.array(h["F"][:NG]), device=dev).float().reshape(-1, 3, 3)
    cov = F @ C0 @ F.transpose(1, 2)
    cc = torch.stack([cov[:, 0, 0], cov[:, 0, 1], cov[:, 0, 2], cov[:, 1, 1], cov[:, 1, 2], cov[:, 2, 2]], 1)
    Pm = (x - 1.0) / SO + MEAN
    img = RAST(means3D=Pm, means2D=torch.zeros_like(Pm), shs=SHS, colors_precomp=None, opacities=OPA, scales=None,
               rotations=None, cov3D_precomp=cc)[0]
    WR.append_data((img.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
WR.close()
print(f"[영상] {a.out} ({len(files)} 프레임)", flush=True)

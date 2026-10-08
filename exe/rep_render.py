"""표현력 비교 영상: 각 표현(또는 목표)이 옮긴 3DGS 가우시안을 공식 래스터라이저로 그린다.

- 위치: rep_track.py 가 --aux 로 함께 옮긴 가우시안 중심 (정규화 좌표) 을
  모델 좌표로 되돌린다:  x_sim = (x - off)·s + lo,  x_model = (x_sim - 1)/scale_origin + mean
- 공분산: Σ' = F Σ Fᵀ. 목표는 흐름의 해석적 F (gauss_flow 로 다시 적분),
  표현들은 정지 이웃 8 개 최소제곱 F (rep_track 과 같은 규약).
- 카메라: 그 모델의 학습 카메라 0 번 (cameras.json) -- 피팅 GT 와 같은 시점.

  cd i-physgaussian && python <anchorflow>/exe/rep_render.py --aux repflow/aux_wolf.npz \
      --res repflow/res_wolf_ours.npz --out repflow/vid_wolf_ours.mp4
  (--target 이면 목표 궤적을 그린다: --flow 필요)
"""
import argparse
import json
import math
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--pg", default="/home/dkta/work/i-physgaussian")
ap.add_argument("--aux", required=True)
ap.add_argument("--res", default="")
ap.add_argument("--target", action="store_true")
ap.add_argument("--flow", default="")
ap.add_argument("--out", required=True)
ap.add_argument("--cam", type=int, default=0)
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--label", default="")
ap.add_argument("--iso", type=float, default=0.0,
                help="공분산을 등방형 (iso × 가우시안 축 길이 중앙값)² I 로 (0 이면 원래 FΣFᵀ)")
ap.add_argument("--sub", action="store_true",
                help="목표 영상을 흐름의 부분표본 가우시안만으로 (추적과 같은 점 집합)")
a = ap.parse_args()
sys.path.append(a.pg)
sys.path.append(os.path.join(a.pg, "gaussian-splatting"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
os.chdir(a.pg)

import numpy as np                                               # noqa: E402
import torch                                                     # noqa: E402
import imageio.v2 as imageio                                     # noqa: E402
from scene.gaussian_model import GaussianModel                   # noqa: E402
from utils.graphics_utils import getWorld2View2, getProjectionMatrix  # noqa: E402
from diff_gaussian_rasterization import (                         # noqa: E402
    GaussianRasterizationSettings, GaussianRasterizer)

dev = "cuda"
X = np.load(a.aux, allow_pickle=True)
model = str(X["model"])
gs = GaussianModel(3)
gs.load_ply(f"{model}/point_cloud/iteration_30000/point_cloud.ply")
_sub = np.load(a.flow)["idx"] if (a.sub and a.flow) else None
gi = torch.as_tensor(X["gidx"] if _sub is None else X["gidx"][_sub], device=dev)
cov0 = gs.get_covariance()[gi].detach()                           # [G,6] (모델 좌표)
shs = gs.get_features[gi].detach()
op = gs.get_opacity[gi].detach()
G0 = torch.as_tensor(X["G"] if _sub is None else X["G"][_sub], device=dev, dtype=torch.float32)
lo = torch.as_tensor(X["lo"], device=dev, dtype=torch.float32)
s, so = float(X["s"]), float(X["scale_origin"])
off = torch.as_tensor(X["off"], device=dev, dtype=torch.float32)
mean = torch.as_tensor(X["mean"], device=dev, dtype=torch.float32)


def to_model(P):
    return ((P - off) * s + lo - 1.0) / so + mean


def sym6_to_mat(c):
    m = torch.zeros(c.shape[0], 3, 3, device=c.device)
    m[:, 0, 0], m[:, 0, 1], m[:, 0, 2] = c[:, 0], c[:, 1], c[:, 2]
    m[:, 1, 1], m[:, 1, 2], m[:, 2, 2] = c[:, 3], c[:, 4], c[:, 5]
    m[:, 1, 0], m[:, 2, 0], m[:, 2, 1] = c[:, 1], c[:, 2], c[:, 4]
    return m


def mat_to_sym6(m):
    return torch.stack([m[:, 0, 0], m[:, 0, 1], m[:, 0, 2],
                        m[:, 1, 1], m[:, 1, 2], m[:, 2, 2]], 1)


C0 = sym6_to_mat(cov0)
if a.iso > 0:
    _sm = float(gs.get_scaling[gi].detach().median())
    print(f"[등방] 축 길이 중앙값 {_sm:.5f} (모델 좌표) x {a.iso} -> 표준편차 {a.iso * _sm:.5f}", flush=True)

# 궤적 (정규화 좌표) 과 F
if a.target:
    import gauss_flow as gf
    D = np.load(a.flow)
    field = D["field"]
    T = D["traj"].shape[0] - 1
    period = int(round(T / max(len(field), 1))) if len(field) else 10
    period = 10 if T % len(field) else T // len(field)
    x = G0.double()
    F = torch.eye(3, dtype=torch.float64, device=dev).expand(x.shape[0], 3, 3).clone()
    Ps, Fs = [x.float()], [F.float()]
    for t in range(T):
        f = torch.as_tensor(field[t // period], dtype=torch.float64, device=dev)
        x, F, _, _ = gf.advance(x, F, f, 1.0 / 30.0, 0.5)
        Ps.append(x.float()); Fs.append(F.float())
    getP = lambda t: Ps[t]
    getF = lambda t: Fs[t]
    T1 = T + 1
else:
    R = np.load(a.res)
    if "AUXY" in R:
        AY = R["AUXY"]                                            # [T+1,G,3] float16
    else:                                                         # rep_track2 --save_traj: 1..T 프레임
        AY = np.concatenate([G0.cpu().numpy()[None].astype(np.float16), R["traj"]], 0)
    T1 = AY.shape[0]
    nb = None
    for c in range(0, 1):
        pass
    # 정지 이웃 8 개 (청크로)
    k = 8
    nbr = torch.empty(G0.shape[0], k, dtype=torch.long, device=dev)
    for i in range(0, G0.shape[0], 8192):
        nbr[i:i + 8192] = torch.cdist(G0[i:i + 8192], G0).topk(
            k + 1, largest=False).indices[:, 1:]
    dX = G0[nbr] - G0[:, None]
    Binv = torch.linalg.inv(dX.transpose(1, 2) @ dX + 1e-12 * torch.eye(3, device=dev))
    getP = lambda t: torch.as_tensor(AY[t], device=dev, dtype=torch.float32)

    def getF(t):
        P = getP(t)
        dY = P[nbr] - P[:, None]
        F = (dY.transpose(1, 2) @ dX) @ Binv
        U, S, Vh = torch.linalg.svd(F)                           # 렌더용 특잇값 묶기
        return U @ torch.diag_embed(S.clamp(1 / 3, 3)) @ Vh

# 카메라: 학습 카메라
cams = json.load(open(f"{model}/cameras.json"))
c = cams[a.cam]
Rw = np.array(c["rotation"]); pos = np.array(c["position"])
W2C = np.linalg.inv(np.block([[Rw, pos[:, None]], [np.zeros((1, 3)), np.ones((1, 1))]]))
Rm, Tm = W2C[:3, :3].T, W2C[:3, 3]
fovx = 2 * math.atan(c["width"] / (2 * c["fx"]))
fovy = 2 * math.atan(c["height"] / (2 * c["fy"]))
wv = torch.tensor(getWorld2View2(Rm, Tm)).transpose(0, 1).to(dev).float()
pj = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=fovx, fovY=fovy).transpose(0, 1).to(dev).float()
fp = (wv.unsqueeze(0).bmm(pj.unsqueeze(0))).squeeze(0)
campos = wv.inverse()[3, :3]
st = GaussianRasterizationSettings(
    image_height=int(c["height"]), image_width=int(c["width"]),
    tanfovx=math.tan(fovx * 0.5), tanfovy=math.tan(fovy * 0.5),
    bg=torch.ones(3, device=dev), scale_modifier=1.0, viewmatrix=wv,
    projmatrix=fp, sh_degree=3, campos=campos, prefiltered=False, debug=False)
rast = GaussianRasterizer(raster_settings=st)

wr = imageio.get_writer(a.out, fps=a.fps, codec="libx264", quality=8)
with torch.no_grad():
    for t in range(T1):
        P = to_model(getP(t))
        if a.iso > 0:
            cov = torch.zeros(P.shape[0], 6, device=dev)
            cov[:, 0] = cov[:, 3] = cov[:, 5] = (a.iso * _sm) ** 2
        else:
            F = getF(t)
            cov = mat_to_sym6(F @ C0 @ F.transpose(1, 2))
        img = rast(means3D=P, means2D=torch.zeros_like(P), shs=shs,
                   colors_precomp=None, opacities=op, scales=None,
                   rotations=None, cov3D_precomp=cov)[0]
        wr.append_data((img.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255)
                       .astype(np.uint8))
wr.close()
print(f"[영상] {a.out}  {T1} 프레임", flush=True)

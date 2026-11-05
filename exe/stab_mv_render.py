"""Spring-Gaus 입력 (공식 mpm_synthetic 형식 다시점 영상) 을 정답 궤적에서 렌더 (doc/pareto.md 절 S.3).

Spring-Gaus 는 다시점 영상에서 물체를 재구성하고 물성을 맞추는 방법이라 정답 궤적을 그쪽 데이터 형식 영상으로 만든다
(예전 벤치는 PG gs_simulation 에 다시점 렌더를 붙였다 -- exe/patch_pg_multiview.py; 여기서는 이미 있는 정답 h5 궤적을
exe/lb_render.py 와 같은 3DGS·같은 공분산 F C0 Fᵀ 로 렌더). 카메라 = 물체 중심을 도는 10 대 (공식 합성 데이터 N_CAM 10),
512x512 (공식 합성 데이터 H·W), 시야각 = 3DGS cameras.json[0], 거리 = lb_render 와 같게 cameras.json[0] 거리의 cam_back 배,
고도 = cameras.json[0] 고도와 그 +20°를 번갈아, 방위 36° 간격. 배경 검정 (공식 로더 MPM_Synthetic 의 bg = 0).
좌표 = 시뮬 좌표 (z 위). Spring-Gaus 좌표로 옮기는 것은 exe/stab_sg.py --mode prep (닮음 변환; 영상은 그대로).

출력: <out>/cam_<i>/<프레임:03>.png (프레임 0..L), <out>/cams_sim.json [{camera, K, c2w (blender 규약, 시뮬 좌표), width, height}],
<out>/mv_meta.json (중심·거리·프레임 수).

  python exe/stab_mv_render.py --sim <정답 궤적 폴더 (meta.json, sim_*.h5)> --config cfg.json --shape lego --frames 80 --out DIR
"""
import argparse
import glob
import json
import math
import os
import sys

W = "/home/dkta/work"
MODEL = {"lego": "lego_whitebg-trained", "ficus": "ficus_whitebg-trained", "mic": "mic_whitebg-trained", "hotdog": "hotdog_whitebg-trained"}
ap = argparse.ArgumentParser()
ap.add_argument("--sim", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--shape", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--frames", type=int, default=0)
ap.add_argument("--ncam", type=int, default=10)
ap.add_argument("--res", type=int, default=512)
ap.add_argument("--cam_back", type=float, default=1.6)
ap.add_argument("--iter", type=int, default=-1)
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
from utils.sh_utils import eval_sh                                    # noqa: E402  (gaussian-splatting/utils; utils 는 이름공간 꾸러미)
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer   # noqa: E402
os.chdir(_cwd)
mp = f"{W}/pgmodel/{MODEL[a.shape]}"
gs = GaussianModel(3)
_it = a.iter if a.iter >= 0 else max(int(q.split("_")[-1]) for q in os.listdir(f"{mp}/point_cloud") if q.startswith("iteration_"))
gs.load_ply(f"{mp}/point_cloud/iteration_{_it}/point_cloud.ply")
KIDX = torch.nonzero(gs.get_opacity[:, 0] > float(cfg.get("opacity_threshold", 0.02))).squeeze(1)
# PG 전처리 회전 (config rotation_degree/axis) -- exe/lb_render.py 절 AG 와 같은 식. L000·ficus config 는 [0.0] (항등) 이라 예전 판과 같은 그림.
_RT = torch.eye(3, device=dev)
for _dg, _ax in zip(cfg.get("rotation_degree") or [], cfg.get("rotation_axis") or []):
    _c, _s = math.cos(_dg / 180.0 * 3.1415926), math.sin(_dg / 180.0 * 3.1415926)
    _R = {0: [[1, 0, 0], [0, _c, -_s], [0, _s, _c]], 1: [[_c, 0, _s], [0, 1, 0], [-_s, 0, _c]], 2: [[_c, -_s, 0], [_s, _c, 0], [0, 0, 1]]}[int(_ax)]
    _RT = torch.tensor(_R, device=dev, dtype=torch.float32) @ _RT
ROT = bool((_RT - torch.eye(3, device=dev)).abs().max() > 0)
TP, SO, MEAN = transform2origin(gs.get_xyz[KIDX] @ _RT.T if ROT else gs.get_xyz[KIDX], float(cfg.get("scale", 1.0)))
TP = shift2center111(TP)
NG = KIDX.numel()
with h5py.File(meta["h5"]) as h:                                 # lb_render 와 같은 대응 검산
    x0 = np.array(h["x"]); x0 = x0.T if x0.shape[0] == 3 else x0
err = float((torch.as_tensor(x0[:NG], device=dev).float() - TP).norm(dim=1).max())
assert err < 1e-4, f"채움 캐시 앞부분이 가우시안과 같은 번호가 아님 ({err})"
c6 = gs.get_covariance()[KIDX]
C0 = torch.zeros(NG, 3, 3, device=dev)
C0[:, 0, 0], C0[:, 0, 1], C0[:, 0, 2] = c6[:, 0], c6[:, 1], c6[:, 2]
C0[:, 1, 1], C0[:, 1, 2], C0[:, 2, 2] = c6[:, 3], c6[:, 4], c6[:, 5]
C0[:, 1, 0], C0[:, 2, 0], C0[:, 2, 1] = c6[:, 1], c6[:, 2], c6[:, 4]
if ROT:
    C0 = _RT @ C0 @ _RT.T                                        # 회전 뒤 공간
C0 = C0 * float(SO) ** 2                                         # 시뮬 좌표 공분산 (시뮬 = (회전 월드 − MEAN) x SO + 1)
SHS, OPA = gs.get_features[KIDX], gs.get_opacity[KIDX]
files = sorted(glob.glob(f"{a.sim}/sim_*.h5"))
if a.frames:
    files = files[:a.frames + 1]
X = []
for p in files:
    with h5py.File(p) as h:
        X.append(np.array(h["x"][:NG]))
X = np.stack(X)
lo, hi = X.reshape(-1, 3).min(0), X.reshape(-1, 3).max(0)
ctr = 0.5 * (lo + hi)
cam = json.load(open(f"{mp}/cameras.json"))[0]
pos_w = _RT.cpu().numpy().astype(np.float64) @ np.array(cam["position"]); mean_w = MEAN.cpu().numpy().astype(np.float64)   # 둘 다 회전 뒤 공간
dist = a.cam_back * float(np.linalg.norm(pos_w - mean_w)) * float(SO)
el0 = math.degrees(math.asin(max(-1.0, min(1.0, (pos_w - mean_w)[2] / np.linalg.norm(pos_w - mean_w)))))
el0 = min(max(el0, 10.0), 60.0)
fov = 2 * math.atan(cam["width"] / (2 * cam["fx"]))
f_px = a.res / (2 * math.tan(fov / 2))
pj = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=fov, fovY=fov).transpose(0, 1).to(dev).float()
CAMS, RAST, CPOS = [], [], []
for i in range(a.ncam):
    az, el = math.radians(36.0 * i * 10 / a.ncam), math.radians(el0 + (20.0 if i % 2 else 0.0))
    p = ctr + dist * np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    fw = (ctr - p) / np.linalg.norm(ctr - p)
    rt = np.cross(fw, [0.0, 0.0, 1.0]); rt /= np.linalg.norm(rt)
    up = np.cross(rt, fw)
    c2w_b = np.eye(4); c2w_b[:3, 0], c2w_b[:3, 1], c2w_b[:3, 2], c2w_b[:3, 3] = rt, up, -fw, p     # blender (y 위, z 뒤)
    c2w_c = c2w_b.copy(); c2w_c[:3, 1:3] *= -1                                                    # COLMAP (공식 로더와 같은 변환)
    W2C = np.linalg.inv(c2w_c)
    wv = torch.tensor(getWorld2View2(W2C[:3, :3].T, W2C[:3, 3])).transpose(0, 1).to(dev).float()
    RAST.append(GaussianRasterizer(raster_settings=GaussianRasterizationSettings(
        image_height=a.res, image_width=a.res, tanfovx=math.tan(fov * 0.5), tanfovy=math.tan(fov * 0.5), bg=torch.zeros(3, device=dev),
        scale_modifier=1.0, viewmatrix=wv, projmatrix=(wv[None] @ pj[None])[0], sh_degree=3, campos=wv.inverse()[3, :3], prefiltered=False, debug=False)))
    CPOS.append(torch.tensor(p, device=dev).float())
    CAMS.append(dict(camera=f"cam_{i}", K=[[f_px, 0.0, a.res / 2.0], [0.0, f_px, a.res / 2.0], [0.0, 0.0, 1.0]], c2w=c2w_b.tolist(), width=a.res, height=a.res))
    os.makedirs(f"{a.out}/cam_{i}", exist_ok=True)
for fi, p in enumerate(files):
    with h5py.File(p) as h:
        x = torch.as_tensor(np.array(h["x"][:NG]), device=dev).float()
        F = torch.as_tensor(np.array(h["F"][:NG]), device=dev).float().reshape(-1, 3, 3)
    cov = F @ C0 @ F.transpose(1, 2)
    cc = torch.stack([cov[:, 0, 0], cov[:, 0, 1], cov[:, 0, 2], cov[:, 1, 1], cov[:, 1, 2], cov[:, 2, 2]], 1)
    for i, RS in enumerate(RAST):
        cp = None
        if ROT:                                                  # SH 는 원래 월드 방향 기준 -> 시점 방향을 월드로 되돌려 색을 미리 계산 (3DGS 파이썬 SH 경로와 같은 식)
            d = x - CPOS[i]
            d = (d / d.norm(dim=1, keepdim=True)) @ _RT
            cp = torch.clamp_min(eval_sh(3, SHS.transpose(1, 2), d) + 0.5, 0.0)
        img = RS(means3D=x, means2D=torch.zeros_like(x), shs=(SHS if cp is None else None), colors_precomp=cp, opacities=OPA, scales=None, rotations=None, cov3D_precomp=cc)[0]
        imageio.imwrite(f"{a.out}/cam_{i}/{fi:03d}.png", (img.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
json.dump(CAMS, open(f"{a.out}/cams_sim.json", "w"), indent=1)
json.dump(dict(ctr=ctr.tolist(), lo=lo.tolist(), hi=hi.tolist(), dist=dist, el0=el0, fov=fov, n_frames=len(files), NG=NG, SO=float(SO)),
          open(f"{a.out}/mv_meta.json", "w"), indent=1)
print(f"[다시점] {a.out}: 카메라 {a.ncam} x 프레임 {len(files)}, 가우시안 {NG}, 거리 {dist:.3f} 고도 {el0:.1f}°\nMV_OK", flush=True)

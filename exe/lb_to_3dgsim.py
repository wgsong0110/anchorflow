"""lbdata 궤적을 3DGSim(dataset_genesis) 형식으로 렌더한다.

출력 (궤적 하나당):
  <out>/metadata.json                       공식 bunny 샘플의 카메라 블록 그대로 (12 대, 480x480, fov 30)
  <out>/cam_{i}/rgb/{t:05d}.png             RGB 8bit, 검은 배경, lego 자체 SH(시점 방향 평가)
  <out>/cam_{i}/seg/{t:05d}.png             팔레트(P) PNG: 0 배경, 3 물체 (바닥 원판 없음 -> 1 은 쓰지 않는다)

좌표 변환 (모든 궤적에 같은 것 하나):
  w = S * (x_sim - (1, 1, 0.48))         바닥 z=0.48 -> 0, xy 중심 (1,1) -> (0,0)
  공분산_w = S^2 * SO^2 * F C0 F^T       (C0 는 모델 좌표 공분산, SO 는 모델->시뮬 배율)
  S = --diag / (lego 회전 전 가우시안 bbox 대각선). 기본 --diag 1.1 은 공식 bunny 샘플을
  두 시점(cam_0, cam_3) 실루엣 교차로 잰 bbox 대각선(0프레임 1.09, 정지 후 1.13)이다.
분할: 흰색을 검은 배경 위에 렌더한 알파 > --alpha_thr(0.5) 인 픽셀 = 물체(3).

  python exe/lb_to_3dgsim.py --traj /home/dkta/work/lbdata/train/0000 --out .../lb_train/r0000
  python exe/lb_to_3dgsim.py --split train --out_root .../lb_train --prefix r   (전부)
"""
import argparse
import glob
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

W = "/home/dkta/work"
ap = argparse.ArgumentParser()
ap.add_argument("--traj", default=None)
ap.add_argument("--out", default=None)
ap.add_argument("--split", default=None, help="train|test: lbdata/<split> 전부")
ap.add_argument("--out_root", default=None)
ap.add_argument("--prefix", default="r")
ap.add_argument("--lbdata", default=f"{W}/lbdata")
ap.add_argument("--meta_template", default=f"{W}/lb/3DGSim_syn/assets/data/soft_genesis_elastic_unseen/bunny/0/metadata.json")
ap.add_argument("--seg_template", default=f"{W}/lb/3DGSim_syn/assets/data/soft_genesis_elastic_unseen/bunny/0/cam_0/seg/00000.png")
ap.add_argument("--diag", type=float, default=1.1)
ap.add_argument("--alpha_thr", type=float, default=0.5)
ap.add_argument("--skip_done", action="store_true")
a = ap.parse_args()

import h5py                                                      # noqa: E402
import numpy as np                                               # noqa: E402
import torch                                                     # noqa: E402
from PIL import Image                                            # noqa: E402
torch.set_grad_enabled(False)
dev = "cuda"
cfg = json.load(open(f"{a.lbdata}/config.json"))
sys.path.append(f"{W}/i-physgaussian"); sys.path.append(f"{W}/i-physgaussian/gaussian-splatting")
_cwd = os.getcwd(); os.chdir(f"{W}/i-physgaussian")
from scene.gaussian_model import GaussianModel                       # noqa: E402
from utils.transformation_utils import transform2origin, shift2center111   # noqa: E402
from utils.graphics_utils import getProjectionMatrix                 # noqa: E402
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer   # noqa: E402
os.chdir(_cwd)

# ---- lego 가우시안 (lb_render.py 와 같은 적재·C0) ----
MP = f"{W}/pgmodel/lego_whitebg-trained"
gs = GaussianModel(3)
gs.load_ply(f"{MP}/point_cloud/iteration_30000/point_cloud.ply")
KIDX = torch.nonzero(gs.get_opacity[:, 0] > float(cfg.get("opacity_threshold", 0.02))).squeeze(1)
TP, SO, MEAN = transform2origin(gs.get_xyz[KIDX], float(cfg.get("scale", 1.0)))
TP = shift2center111(TP)
NG = KIDX.numel()
c6 = gs.get_covariance()[KIDX]
C0 = torch.zeros(NG, 3, 3, device=dev)
C0[:, 0, 0], C0[:, 0, 1], C0[:, 0, 2] = c6[:, 0], c6[:, 1], c6[:, 2]
C0[:, 1, 1], C0[:, 1, 2], C0[:, 2, 2] = c6[:, 3], c6[:, 4], c6[:, 5]
C0[:, 1, 0], C0[:, 2, 0], C0[:, 2, 1] = c6[:, 1], c6[:, 2], c6[:, 4]
SHS, OPA = gs.get_features[KIDX].contiguous(), gs.get_opacity[KIDX].contiguous()
SO = float(SO)

# ---- 고정 좌표 변환 ----
FLOOR = 0.48
ORIG = torch.tensor([1.0, 1.0, FLOOR], device=dev)
lo, hi = TP.min(0).values, TP.max(0).values
DIAG_SIM = float((hi - lo).norm())
S = a.diag / DIAG_SIM
print(f"[변환] 가우시안 {NG}, SO {SO:.5f}, lego 회전 전 bbox(시뮬) {(hi - lo).tolist()} 대각선 {DIAG_SIM:.4f}"
      f" -> S = {a.diag}/{DIAG_SIM:.4f} = {S:.5f}", flush=True)

# ---- 카메라: 공식 metadata 의 extrinsics = OpenCV world->camera ----
META_T = json.load(open(a.meta_template))
NCAM = META_T["n_cams"]
PAL = Image.open(a.seg_template).getpalette()
RASTS = []
for i in range(NCAM):
    c = META_T[f"cam_{i}"]
    E = torch.tensor(c["extrinsics"], dtype=torch.float32, device=dev)            # w2c
    K = np.array(c["intrinsics"]); Wd, Hd = c["resolution"]
    assert abs(K[0, 2] - Wd / 2) < 1e-6 and abs(K[1, 2] - Hd / 2) < 1e-6
    fovx = 2 * math.atan(Wd / (2 * K[0, 0])); fovy = 2 * math.atan(Hd / (2 * K[1, 1]))
    wv = E.T.contiguous()                                                          # 3DGS 행벡터 규약
    pj = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=fovx, fovY=fovy).transpose(0, 1).to(dev).float()
    campos = torch.linalg.inv(E)[:3, 3]
    mk = lambda bg: GaussianRasterizer(raster_settings=GaussianRasterizationSettings(   # noqa: E731
        image_height=int(Hd), image_width=int(Wd), tanfovx=math.tan(fovx * 0.5), tanfovy=math.tan(fovy * 0.5),
        bg=bg, scale_modifier=1.0, viewmatrix=wv, projmatrix=(wv[None] @ pj[None])[0], sh_degree=3,
        campos=campos, prefiltered=False, debug=False))
    RASTS.append(mk(torch.zeros(3, device=dev)))
    if i == 0:
        print(f"[카메라] {NCAM} 대, {Wd}x{Hd}, fov {math.degrees(fovx):.3f}, cam_0 중심 {campos.tolist()}", flush=True)
WHITE = torch.ones(NG, 3, device=dev)
POOL = ThreadPoolExecutor(16)


def save_rgb(p, arr):
    Image.fromarray(arr).save(p)


def save_seg(p, arr):
    im = Image.fromarray(arr, mode="P"); im.putpalette(PAL); im.save(p)


def do_traj(traj, out):
    files = sorted(glob.glob(f"{traj}/sim_*.h5"))
    nt = len(files)
    if a.skip_done and os.path.exists(f"{out}/metadata.json"):
        print(f"[건너뜀] {out}", flush=True); return None
    for i in range(NCAM):
        os.makedirs(f"{out}/cam_{i}/rgb", exist_ok=True); os.makedirs(f"{out}/cam_{i}/seg", exist_ok=True)
    t0 = time.time(); futs = []; cover = np.zeros(NCAM); border = 0; xyz_lo = np.full(3, 1e9); xyz_hi = -xyz_lo
    for t, p in enumerate(files):
        with h5py.File(p) as h:
            x = torch.as_tensor(np.array(h["x"][:NG]), device=dev).float()
            F = torch.as_tensor(np.array(h["F"][:NG]), device=dev).float().reshape(-1, 3, 3)
        Pw = ((x - ORIG) * S).contiguous()
        cov = (F @ C0 @ F.transpose(1, 2)) * (S * SO) ** 2
        cc = torch.stack([cov[:, 0, 0], cov[:, 0, 1], cov[:, 0, 2], cov[:, 1, 1], cov[:, 1, 2], cov[:, 2, 2]], 1).contiguous()
        xyz_lo = np.minimum(xyz_lo, Pw.min(0).values.cpu().numpy()); xyz_hi = np.maximum(xyz_hi, Pw.max(0).values.cpu().numpy())
        for i in range(NCAM):
            kw = dict(means3D=Pw, means2D=torch.zeros_like(Pw), opacities=OPA, scales=None, rotations=None, cov3D_precomp=cc)
            rgb = RASTS[i](shs=SHS, colors_precomp=None, **kw)[0]
            alp = RASTS[i](shs=None, colors_precomp=WHITE, **kw)[0][0]
            rgb8 = (rgb.clamp(0, 1).permute(1, 2, 0) * 255).round().to(torch.uint8).cpu().numpy()
            m = (alp > a.alpha_thr)
            seg = (m.to(torch.uint8) * 3).cpu().numpy()
            cover[i] += float(m.float().mean()) / nt
            border += int(m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any())
            futs.append(POOL.submit(save_rgb, f"{out}/cam_{i}/rgb/{t:05d}.png", rgb8))
            futs.append(POOL.submit(save_seg, f"{out}/cam_{i}/seg/{t:05d}.png", seg))
    for f in futs:
        f.result()
    meta = dict(META_T)
    meta["n_steps"] = nt
    meta["object"] = "lego"
    meta["scene_name"] = os.path.basename(out.rstrip("/"))
    meta["anchorflow"] = dict(src=traj, S=S, floor=FLOOR, origin=[1.0, 1.0, FLOOR], diag_target=a.diag,
                              alpha_thr=a.alpha_thr, n_gauss=NG)
    json.dump(meta, open(f"{out}/metadata.json", "w"), indent=1)       # 마지막에 써서 완료 표시로 쓴다
    print(f"[궤적] {traj} -> {out}: {nt} 프레임, {time.time() - t0:.1f}s, 물체 픽셀 비율(캠별 평균) "
          f"{np.round(cover, 4).tolist()}, 테두리 닿은 이미지 {border}/{nt * NCAM}, "
          f"world bbox {np.round(xyz_lo, 3).tolist()}..{np.round(xyz_hi, 3).tolist()}", flush=True)
    return border


if a.traj:
    do_traj(a.traj, a.out)
else:
    trajs = sorted(d for d in glob.glob(f"{a.lbdata}/{a.split}/*") if os.path.isdir(d))
    for d in trajs:
        do_traj(d, f"{a.out_root}/{a.prefix}{os.path.basename(d)}")
    print("LB_TO_3DGSIM_DONE", flush=True)

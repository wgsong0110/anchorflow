"""Spring-Gaus 의 스프링-질량 시뮬레이터를 우리 소성 씬에 돌린다.

Spring-Gaus(ECCV 2024, arXiv 2403.09434)는 가우시안마다 kNN 스프링을 달아 **탄성**
물체를 재구성·시뮬레이션한다. 소성은 다루지 않는다 -- 그래서 소성 씬에 돌리면
"할 수 있는데 못 한다"가 아니라 **구조적으로 영구 변형을 남길 수 없다**는 것이 보여야
한다. 스프링은 언제나 원래 길이로 되돌리려 하기 때문이다.

물성은 학습하지 않는다. 다른 베이스라인과 같은 조건에서 시뮬레이터만 재는 것이
목적이므로, 스프링 상수는 config 의 E 에서 유도한 고정값을 쓴다.

렌더는 PhysGaussian 궤도 카메라를 그대로 받아 같은 시점에서 비교할 수 있게 한다.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--ply", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--cameras", required=True)
ap.add_argument("--springgaus", required=True)
ap.add_argument("--scgs", required=True)
ap.add_argument("--anchorflow", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--cam_seq", default=None)
ap.add_argument("--frames", type=int, default=400)
ap.add_argument("--width", type=int, default=540)
ap.add_argument("--fps", type=int, default=25)
ap.add_argument("--n_sim", type=int, default=8000,
                help="시뮬할 가우시안 수. kNN 스프링이라 n x k 로 붙는다")
ap.add_argument("--k_neighbors", type=int, default=256,
                help="공식 default.yaml 의 K_NEIGHBORS")
ap.add_argument("--n_step", type=int, default=100,
                help="공식 default.yaml 의 N_STEP (프레임당 서브스텝)")
ap.add_argument("--bench", action="store_true", help="속도만 재고 끝낸다")
a = ap.parse_args()

sys.path.insert(0, a.anchorflow)
sys.path.insert(0, a.scgs)
dev = "cuda"
torch.set_grad_enabled(False)

import warp as wp  # noqa: E402

wp.init()
from anchorflow import scene_setup  # noqa: E402

sc = scene_setup.build(a.ply, a.config, 512, 8, device=dev,
                       frozen_weights=True, rot_fallback=True, eig_floor=0.02)
cfg = sc.cfg
FRAME_DT = float(cfg.get("frame_dt", 0.01))
FULL = torch.nonzero(sc.keep).squeeze(-1)
if a.n_sim > 0 and FULL.numel() > a.n_sim:
    g = torch.Generator(device="cpu").manual_seed(0)
    SUB = torch.randperm(FULL.numel(), generator=g)[:a.n_sim].sort().values.to(dev)
else:
    SUB = torch.arange(FULL.numel(), device=dev)
KEEP = torch.zeros_like(sc.keep)
KEEP[FULL[SUB]] = True
mat = sc.pos[KEEP].contiguous()
N = mat.shape[0]
print(f"[씬] 물질 {int(sc.keep.sum())} -> 시뮬 {N}, frame_dt {FRAME_DT}, "
      f"재질 {cfg.get('material')}", flush=True)

# --- Spring-Gaus 시뮬레이터 ---
sys.path.insert(0, a.springgaus)
from lib.models.spring_mass.Spring_Mass import Spring_Mass  # noqa: E402
from yacs.config import CfgNode as CN  # noqa: E402

# 값은 공식 config/mpm_synthetic/default.yaml 그대로. 중력은 씬 config 를 따른다
# (다른 베이스라인과 같은 조건이어야 한다).
scfg = CN()
scfg.K_NEIGHBORS = a.k_neighbors
scfg.K_BINDING = 16
scfg.N_STEP = a.n_step
scfg.INIT_VELOCITY = [0, 0, 0]
scfg.G = list(cfg.get("g", [0.0, 0.0, 0.0]))
scfg.PRETRAINED = None
scfg.DATA = CN()
sim = Spring_Mass(scfg, mat.clone())
sim.set_dt(dt=FRAME_DT)
print(f"[Spring-Gaus] 이웃 {a.k_neighbors}, 서브스텝/프레임 {a.n_step}", flush=True)

x = mat.clone()
v = torch.zeros_like(x)

if a.bench:
    for _ in range(3):
        out = sim(x, x, v, frame_id=1)
    torch.cuda.synchronize()
    t0 = time.time()
    n = 10
    for _ in range(n):
        out = sim(x, x, v, frame_id=1)
    torch.cuda.synchronize()
    ms = (time.time() - t0) / n * 1000
    print(f"[속도] {ms:.2f} ms/frame ({1000/ms:.1f} fps), 점 {N}, "
          f"이웃 {a.k_neighbors}, 서브스텝 {a.n_step}", flush=True)
    os.makedirs(a.out, exist_ok=True)
    json.dump({"ms_per_frame": ms, "fps": 1000 / ms, "n_points": N,
               "k_neighbors": a.k_neighbors, "n_step": a.n_step},
              open(os.path.join(a.out, "spring_gaus_bench.json"), "w"), indent=1)
    print("SG_BENCH_OK")
    sys.exit(0)

# --- 롤아웃 ---
traj, bad = [x.clone()], False
t0 = time.time()
for fr in range(1, a.frames):
    out = sim(x, x, v, frame_id=fr)
    x, v = out[0].detach(), out[1].detach()
    if not torch.isfinite(x).all():
        print(f"[롤아웃] 프레임 {fr} 비유한값 -- 중단", flush=True)
        bad = True
        break
    traj.append(x.clone())
    if fr % 50 == 0:
        print(f"  {fr}/{a.frames} ({time.time()-t0:.0f}s)", flush=True)
X = torch.stack(traj)
NF = X.shape[0]
d = float((X - X[0]).norm(dim=-1).max()) / float(sc.extent) * 100
# 영구 변형이 남았는지: 마지막 프레임이 처음에서 얼마나 떨어져 있나 (평균)
resid = float((X[-1] - X[0]).norm(dim=-1).mean()) / float(sc.extent) * 100
print(f"[롤아웃] {NF} 프레임, 최대 변위 {d:.2f}%, 잔류 변형 {resid:.3f}% of 물체"
      f"{' (발산)' if bad else ''}", flush=True)

# --- 렌더 ---
from scene.gaussian_model import GaussianModel  # noqa: E402
from gaussian_renderer import render as _render  # noqa: E402
from utils.graphics_utils import (focal2fov, getProjectionMatrix,  # noqa: E402
                                  getWorld2View2)


class MiniCam:
    def __init__(self, w, h, fy, fx, zn, zf, wvt, fp):
        self.image_width, self.image_height = w, h
        self.FoVy, self.FoVx, self.znear, self.zfar = fy, fx, zn, zf
        self.world_view_transform, self.full_proj_transform = wvt, fp
        self.camera_center = wvt.inverse()[3, :3]


def build_cam(R, T, fovx, fovy, w, h):
    wvt = torch.tensor(getWorld2View2(R, T)).transpose(0, 1).float().to(dev)
    pmx = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=fovx,
                              fovY=fovy).transpose(0, 1).to(dev)
    return MiniCam(w, h, fovy, fovx, 0.01, 100.0, wvt,
                   (wvt.unsqueeze(0).bmm(pmx.unsqueeze(0))).squeeze(0))


if a.cam_seq:
    cs = json.load(open(a.cam_seq))
    w = int(a.width); h = int(round(cs[0]["height"] * w / cs[0]["width"]))
    CAMS = [build_cam(np.array(c["R"], dtype=np.float64),
                      np.array(c["T"], dtype=np.float64),
                      c["FoVx"], c["FoVy"], w, h) for c in cs]
else:
    cams = json.load(open(a.cameras))
    c = cams[0]
    w0, h0 = int(c["width"]), int(c["height"])
    w = int(a.width); h = int(round(h0 * w / w0))
    R = np.array(c["rotation"], dtype=np.float64)
    T = -R.T @ np.array(c["position"], dtype=np.float64)
    CAMS = [build_cam(R, T, focal2fov(float(c["fx"]) * w / w0, w),
                      focal2fov(float(c["fy"]) * h / h0, h), w, h)]


class _P:
    debug = False
    compute_cov3D_python = False
    convert_SHs_python = False


pipe = _P()
BG = torch.tensor([1., 1., 1.], device=dev)
gs = GaussianModel(3, fea_dim=0)
gs.load_ply(a.ply)
N_ALL = gs._xyz.shape[0]
idx_all = torch.arange(N_ALL, device=dev)
if getattr(sc, "crop", None) is not None:
    idx_all = idx_all[sc.crop]
MAT = idx_all[KEEP]
ZR = torch.zeros(N_ALL, 4, device=dev); ZR[:, 0] = 1.0
ZS = torch.zeros(N_ALL, 3, device=dev)
G0 = sc.pos[KEEP]

import imageio  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

os.makedirs(a.out, exist_ok=True)
frames = []
for t in range(NF):
    dx = torch.zeros(N_ALL, 3, device=dev, dtype=sc.pos.dtype)
    dx[MAT] = sc.undo(X[t]) - sc.undo(G0)
    cam = CAMS[min(t, len(CAMS) - 1)]
    im = torch.clamp(_render(cam, gs, pipe, BG, dx, ZR, ZS,
                             d_rot_as_res=True)["render"], 0, 1)
    arr = (im.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
    pim = Image.fromarray(arr); dr = ImageDraw.Draw(pim)
    dr.rectangle([0, 0, pim.width, 18], fill=(0, 0, 0))
    dr.text((4, 4), f"Spring-Gaus ({cfg.get('material')} scene) f{t:03d}",
            fill=(255, 255, 255))
    frames.append(np.array(pim))

p_out = os.path.join(a.out, "spring_gaus.mp4")
imageio.mimsave(p_out, frames, fps=a.fps, quality=8)
json.dump({"frames": NF, "diverged": bool(bad), "max_disp_pct": d,
           "residual_pct": resid, "n_points": N,
           "k_neighbors": a.k_neighbors, "n_step": a.n_step,
           "scene_material": cfg.get("material")},
          open(os.path.join(a.out, "spring_gaus.json"), "w"), indent=1)
print(f"[저장] {p_out}", flush=True)
print("SG_OK")

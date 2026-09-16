"""기하 학습이 쓰는 MPM 궤적 몇 개를 그대로 렌더해 무엇을 학습하고 있는지 본다.

궤적은 임펄스 세기를 기준 힘의 [0.5, 0.5*impulse_range] 에서 뽑아 만든다.
세기가 클수록 크게 변형되는데, 그 분포의 어디쯤에서 실제로 **찢어지는지**는
숫자만으로는 안 보인다. 눈으로 확인하려고 만든 것이다.

찢어짐 여부를 같이 잰다: 초기 k-NN 이웃 중 시각 t 에 반경 밖으로 나간 비율.
이 값이 토폴로지 변화량이고, 고정 연결성 방법이 못 따라가는 바로 그 양이다.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--ply", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--dreamphysics", required=True)
ap.add_argument("--scgs", required=True)
ap.add_argument("--anchorflow", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--base_force", type=float, nargs=3, required=True)
ap.add_argument("--impulse_range", type=float, default=16.0)
ap.add_argument("--which", type=float, nargs="+", default=[0.5, 2.0, 4.0, 8.0],
                help="렌더할 임펄스 배수. 학습 분포는 [0.5, 0.5*range]")
ap.add_argument("--frames", type=int, default=60)
ap.add_argument("--dt_mult", type=int, default=40)
ap.add_argument("--width", type=int, default=540)
ap.add_argument("--fps", type=int, default=12)
ap.add_argument("--knn", type=int, default=16, help="이웃 변화율을 잴 때의 k")
ap.add_argument("--nbr_thresh", type=float, default=3.0,
                help="초기 이웃 거리의 몇 배를 넘으면 '떨어졌다'로 볼지")
a = ap.parse_args()

sys.path.insert(0, a.anchorflow)
sys.path.insert(0, a.scgs)
sys.path.insert(0, a.dreamphysics)
dev = "cuda"
torch.set_grad_enabled(False)
os.makedirs(a.out, exist_ok=True)

import warp as wp  # noqa: E402

wp.init()
from anchorflow import scene_setup  # noqa: E402
from anchorflow.mpm_teacher import MPMTeacher  # noqa: E402

sc = scene_setup.build(a.ply, a.config, 512, 8, device=dev,
                       frozen_weights=True, rot_fallback=True, eig_floor=0.02)
EXT = float(sc.extent)
FRAME_DT = float(sc.sub_dt) * a.dt_mult
print(f"[씬] 물질 {int(sc.keep.sum())}, 물체 {EXT:.4f}", flush=True)

T = MPMTeacher(sc, horizon=a.frames * FRAME_DT * 1.2)

# 이웃 변화율을 잴 기준: 초기 배치에서의 k-NN 과 그 거리
sub = torch.randperm(T.n, device=dev)[:4000]
X0 = T.pos_m[sub]
d0 = torch.cdist(X0, X0)
d0.fill_diagonal_(float("inf"))
nbr_d0, nbr_i0 = d0.topk(a.knn, largest=False)
print(f"[이웃] {sub.numel()} 점 표본, k={a.knn}, 초기 이웃 거리 중앙 "
      f"{float(nbr_d0.median()):.5f}", flush=True)

# --- 렌더 준비 ---
from scene.gaussian_model import GaussianModel  # noqa: E402
from gaussian_renderer import render as _render  # noqa: E402
from utils.graphics_utils import (focal2fov, getProjectionMatrix,  # noqa: E402
                                  getWorld2View2)
import imageio  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402


class MiniCam:
    def __init__(self, w, h, fy, fx, zn, zf, wvt, fp):
        self.image_width, self.image_height = w, h
        self.FoVy, self.FoVx, self.znear, self.zfar = fy, fx, zn, zf
        self.world_view_transform, self.full_proj_transform = wvt, fp
        self.camera_center = wvt.inverse()[3, :3]


class _P:
    debug = False
    compute_cov3D_python = False
    convert_SHs_python = False


cams = json.load(open(os.path.join(os.path.dirname(a.ply), "..", "..",
                                   "cameras.json"))) \
    if os.path.exists(os.path.join(os.path.dirname(a.ply), "..", "..",
                                   "cameras.json")) else None
if cams is None:
    raise SystemExit("cameras.json 을 못 찾았다")
# 물체가 크게, 온전히 담기는 시점을 고른다
xw = sc.undo(sc.pos[sc.keep]).double().cpu().numpy()
rng = np.random.RandomState(0)
xs = xw[rng.choice(xw.shape[0], min(4000, xw.shape[0]), replace=False)]
best, bi = -1, 0
for i, c in enumerate(cams):
    w0, h0 = int(c["width"]), int(c["height"])
    R = np.array(c["rotation"]); t_ = -R.T @ np.array(c["position"])
    cx = xs @ R + t_
    z = cx[:, 2]; fr = z > 1e-6
    if fr.sum() < 16:
        continue
    u = c["fx"] * cx[fr, 0] / z[fr] + w0 / 2
    v = c["fy"] * cx[fr, 1] / z[fr] + h0 / 2
    ins = (u >= 0) & (u < w0) & (v >= 0) & (v < h0)
    if ins.sum() < 16:
        continue
    sc_ = float(ins.mean()) * float(fr.mean()) * min(
        (np.ptp(u[ins]) / w0) * (np.ptp(v[ins]) / h0), 1.0)
    if sc_ > best:
        best, bi = sc_, i
c = cams[bi]
w0, h0 = int(c["width"]), int(c["height"])
w = int(a.width); h = int(round(h0 * w / w0))
R = np.array(c["rotation"], dtype=np.float64)
Tv = -R.T @ np.array(c["position"], dtype=np.float64)
wvt = torch.tensor(getWorld2View2(R, Tv)).transpose(0, 1).float().to(dev)
pmx = getProjectionMatrix(znear=0.01, zfar=100.0,
                          fovX=focal2fov(float(c["fx"]) * w / w0, w),
                          fovY=focal2fov(float(c["fy"]) * h / h0, h)
                          ).transpose(0, 1).to(dev)
cam = MiniCam(w, h, focal2fov(float(c["fy"]) * h / h0, h),
              focal2fov(float(c["fx"]) * w / w0, w), 0.01, 100.0, wvt,
              (wvt.unsqueeze(0).bmm(pmx.unsqueeze(0))).squeeze(0))
print(f"[카메라] 시점 {bi} (점수 {best:.3f}), {w}x{h}", flush=True)

pipe = _P()
BG = torch.tensor([1., 1., 1.], device=dev)
gs = GaussianModel(3, fea_dim=0)
gs.load_ply(a.ply)
N_ALL = gs._xyz.shape[0]
idx_all = torch.arange(N_ALL, device=dev)
if getattr(sc, "crop", None) is not None:
    idx_all = idx_all[sc.crop]
MAT = idx_all[sc.keep]
ZR = torch.zeros(N_ALL, 4, device=dev); ZR[:, 0] = 1.0
ZS = torch.zeros(N_ALL, 3, device=dev)
G0 = sc.pos[sc.keep]

base = torch.tensor(a.base_force, device=dev)
report = []
for mult in a.which:
    f = (base * mult).unsqueeze(0).expand(sc.pos.shape[0], 3).contiguous()
    dv = sc.impulse_dv(f)
    v0 = (T.w.unsqueeze(-1) * dv[T.idx]).sum(1).contiguous()
    T._set(T.pos_m.clone(), v0, T.eye.clone(), torch.zeros_like(T.eye))
    frames, torn, bad = [], [], False
    for t in range(a.frames):
        if t:
            for k in range(a.dt_mult):
                T.solver.p2g2p(None, float(sc.sub_dt), device=T.wp_dev)
                if (k + 1) % 4 == 0 and (not T._in_domain()
                                         or not T._vel_safe(4)):
                    bad = True
                    break
            if bad:
                break
        x = T.solver.export_particle_x_to_torch() if t else T.pos_m
        # 이웃 변화율
        xs_ = x[sub]
        dn = (xs_.unsqueeze(1) - xs_[nbr_i0]).norm(dim=-1)     # [S, k]
        torn.append(float((dn > a.nbr_thresh * nbr_d0).float().mean()))
        # 렌더
        dx = torch.zeros(N_ALL, 3, device=dev, dtype=sc.pos.dtype)
        dx[MAT] = sc.undo(x[sc.keep] if x.shape[0] != MAT.shape[0] else x) \
            - sc.undo(G0)
        im = torch.clamp(_render(cam, gs, pipe, BG, dx, ZR, ZS,
                                 d_rot_as_res=True)["render"], 0, 1)
        arr = (im.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
        pim = Image.fromarray(arr); dr = ImageDraw.Draw(pim)
        dr.rectangle([0, 0, pim.width, 18], fill=(0, 0, 0))
        dr.text((4, 4), f"impulse x{mult:g}  f{t:02d}  torn {100*torn[-1]:.1f}%",
                fill=(255, 255, 255))
        frames.append(np.array(pim))
    p = os.path.join(a.out, f"traj_x{mult:g}.mp4")
    imageio.mimsave(p, frames, fps=a.fps, quality=8)
    disp = 100 * float((x - T.pos_m).norm(dim=-1).max()) / EXT
    report.append(dict(mult=mult, frames=len(frames), diverged=bad,
                       torn_last=torn[-1], torn_max=max(torn),
                       max_disp_pct=disp))
    print(f"  x{mult:g}: {len(frames)} 프레임, 최대 변위 {disp:.2f}%, "
          f"이웃 이탈 {100*torn[-1]:.2f}% (최대 {100*max(torn):.2f}%)"
          f"{' [격자 이탈]' if bad else ''} -> {p}", flush=True)

json.dump(dict(base_force=a.base_force, knn=a.knn,
               nbr_thresh=a.nbr_thresh, rows=report),
          open(os.path.join(a.out, "fit_traj.json"), "w"), indent=1)
print(f"[저장] {a.out}", flush=True)
print("FITTRAJ_OK")

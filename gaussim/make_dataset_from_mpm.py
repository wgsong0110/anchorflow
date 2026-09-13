"""PhysGaussian MPM 소성 궤적을 GausSim 학습 데이터 형식으로 내보낸다.

GausSim 의 데이터셋은 공개되지 않았고(README 의 "Full dataset" 이 TODO),
배포된 데모에는 1 시퀀스 1 카메라 10 프레임뿐이라 그대로는 학습이 불가능하다.
그래서 같은 형식을 우리 소성 씬에서 만든다.

  transforms_train.json   Blender(NeRF) 형식. 파일명이 seq_XXXXX_XXXXX.png 여야
                          데이터셋이 시퀀스/카메라를 갈라낸다.
  images/                 시퀀스·카메라별 대표 프레임
  video_images/seq_.../   그 카메라에서 본 프레임 시퀀스
  point_cloud.ply         정지 3DGS
  pc_mask.pkl             움직이는 가우시안 마스크
  cln_pc_mask.pkl         렌더 대상 마스크
  pin_mask.json           고정 정점
  moving_part_points.ply  움직이는 부분의 점

한 시퀀스 = 한 초기 조건이고, 그 안의 여러 카메라가 같은 동역학을 다른 시점에서
본 것이다.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--ply", required=True)
ap.add_argument("--config", required=True, help="DreamPhysics 용 소성 config")
ap.add_argument("--cameras", required=True)
ap.add_argument("--dreamphysics", required=True)
ap.add_argument("--anchorflow", required=True)
ap.add_argument("--out", required=True, help="data_real/<scene> 경로")
ap.add_argument("--scene", default="plastic")
ap.add_argument("--n_seq", type=int, default=8)
ap.add_argument("--n_cam", type=int, default=4)
ap.add_argument("--frames", type=int, default=14)
ap.add_argument("--width", type=int, default=400)
ap.add_argument("--vel_scale", type=float, default=0.3)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

sys.path.insert(0, a.anchorflow)
sys.path.insert(0, a.dreamphysics)
import warp as wp

wp.init()
from anchorflow import scene_setup
from anchorflow.mpm_teacher import MPMTeacher

dev = "cuda"
torch.set_grad_enabled(False)
OUT = os.path.join(a.out, a.scene)
os.makedirs(os.path.join(OUT, "images"), exist_ok=True)
os.makedirs(os.path.join(OUT, "video_images"), exist_ok=True)

sc = scene_setup.build(a.ply, a.config, 512, 8, device=dev, frozen_weights=True,
                       rot_fallback=True, eig_floor=0.02)
FRAME_DT = float(sc.cfg.get("frame_dt", 0.04))
print("[씬] 재질 %s, 가우시안 %d, 물체 %.4f"
      % (sc.cfg.get("material"), int(sc.keep.sum()), float(sc.extent)), flush=True)

# --- 렌더러 (원본 gaussian-splatting 경로) ---
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render as _render
from utils.graphics_utils import focal2fov, getProjectionMatrix, getWorld2View2


class MiniCam:
    def __init__(self, w, h, fy, fx, wvt, fp):
        self.image_width, self.image_height = w, h
        self.FoVy, self.FoVx, self.znear, self.zfar = fy, fx, 0.01, 100.0
        self.world_view_transform, self.full_proj_transform = wvt, fp
        self.camera_center = wvt.inverse()[3, :3]


class _P:
    debug = False
    compute_cov3D_python = False
    convert_SHs_python = False


pipe = _P()
BG = torch.tensor([1., 1., 1.], device=dev)
gs = GaussianModel(3, fea_dim=0)
gs.load_ply(a.ply)
FIELDS = ("_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling",
          "_rotation")
# ply 전체를 그대로 둔다. crop -> keep 을 렌더러에 그대로 먹이면 배경 가우시안이
# 사라져 흰 공백 위의 물체만 남는다 -- PhysGaussian 공식 경로도 래스터화 직전에
# 비선택 가우시안을 다시 합친다. 움직이는 것은 물질 가우시안뿐이므로 전체 길이의
# 변위를 만들어 그 자리에만 채운다.
KEEP = sc.keep
N_ALL = gs._xyz.shape[0]
idx_all = torch.arange(N_ALL, device=dev)
if getattr(sc, "crop", None) is not None:
    idx_all = idx_all[sc.crop]
MAT = idx_all[KEEP]
ZR = torch.zeros(N_ALL, 4, device=dev); ZR[:, 0] = 1.0
ZS = torch.zeros(N_ALL, 3, device=dev)
N = int(MAT.shape[0])
G0 = sc.pos[KEEP]
print(f"[렌더] 가우시안 전체 {N_ALL}, 그중 물질 {N} (나머지는 정지 배경)",
      flush=True)

cams_all = json.load(open(a.cameras))
rng = np.random.RandomState(a.seed)
xw = sc.undo(G0).double().cpu().numpy()
xs = xw[rng.choice(xw.shape[0], min(4000, xw.shape[0]), replace=False)]
score = []
for i, c in enumerate(cams_all):
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
    score.append((float(ins.mean()) * float(fr.mean())
                  * min((np.ptp(u[ins]) / w0) * (np.ptp(v[ins]) / h0), 1.0), i))
score.sort(reverse=True)
cand = [i for _, i in score[: max(6 * a.n_cam, a.n_cam)]]
pos = np.array([cams_all[i]["position"] for i in cand])
sel = [0]
while len(sel) < min(a.n_cam, len(cand)):
    d = np.linalg.norm(pos[:, None] - pos[None, sel], axis=-1).min(1)
    d[sel] = -1
    sel.append(int(d.argmax()))
VIEWS = [cand[i] for i in sel]
print("[카메라] %s" % VIEWS, flush=True)

CAMS, FOVX = [], None
for i in VIEWS:
    c = cams_all[i]
    w0, h0 = int(c["width"]), int(c["height"])
    w = int(a.width); h = int(round(h0 * w / w0))
    fx, fy = float(c["fx"]) * w / w0, float(c["fy"]) * h / h0
    R = np.array(c["rotation"], dtype=np.float64)
    Tv = -R.T @ np.array(c["position"], dtype=np.float64)
    fovx, fovy = focal2fov(fx, w), focal2fov(fy, h)
    FOVX = fovx if FOVX is None else FOVX
    wvt = torch.tensor(getWorld2View2(R, Tv)).transpose(0, 1).float().to(dev)
    pmx = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=fovx,
                              fovY=fovy).transpose(0, 1).to(dev)
    # NeRF 형식은 camera-to-world 를 요구한다 (Blender 축: Y up, Z back)
    w2c = np.eye(4); w2c[:3, :3] = R.T; w2c[:3, 3] = Tv
    c2w = np.linalg.inv(w2c)
    c2w[:3, 1:3] *= -1
    CAMS.append({"cam": MiniCam(w, h, fovy, fovx, wvt,
                                (wvt.unsqueeze(0).bmm(pmx.unsqueeze(0))).squeeze(0)),
                 "c2w": c2w, "w": w, "h": h})


def draw(x, cam):
    dx = torch.zeros(N_ALL, 3, device=dev, dtype=sc.pos.dtype)
    dx[MAT] = sc.undo(x) - sc.undo(G0)
    im = torch.clamp(_render(cam, gs, pipe, BG, dx, ZR, ZS,
                             d_rot_as_res=True)["render"], 0, 1)
    return (im.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")


import imageio

T = MPMTeacher(sc, horizon=a.frames * FRAME_DT)
n_sub = max(1, int(round(FRAME_DT / float(sc.sub_dt))))
BASE = None
for bc in sc.cfg.get("boundary_conditions", []):
    if bc["type"] == "particle_impulse":
        BASE = torch.tensor(bc["force"], device=dev)

frames_json, n_ok = [], 0
for s in range(a.n_seq):
    g = torch.Generator(device=dev).manual_seed(a.seed * 1000 + s)
    if BASE is not None:
        f = (BASE * (0.5 + 1.5 * torch.rand(1, device=dev, generator=g))
             ).unsqueeze(0).expand(sc.pos.shape[0], 3).contiguous()
        dv = sc.impulse_dv(f)
        v0 = (T.w.unsqueeze(-1) * dv[T.idx]).sum(1).contiguous()
    else:
        d = torch.randn(3, device=dev, generator=g)
        v0 = (a.vel_scale * float(sc.extent) / (a.frames * FRAME_DT)
              * d / d.norm()).expand(T.n, 3).contiguous()
    T._set(T.pos_m.clone(), v0, T.eye.clone(), torch.zeros_like(T.eye))
    xs_t, bad = [T.pos_m.clone()], False
    for _ in range(a.frames - 1):
        for k in range(n_sub):
            T.solver.p2g2p(None, float(sc.sub_dt), device=T.wp_dev)
            if (k + 1) % 4 == 0 and (not T._in_domain() or not T._vel_safe(4)):
                bad = True
                break
        if bad:
            break
        xs_t.append(T.solver.export_particle_x_to_torch().clone())
    if bad or len(xs_t) < a.frames:
        print("  시퀀스 %02d: 격자 이탈 -- 버림" % s, flush=True)
        continue
    X = torch.stack(xs_t)
    for ci in range(len(CAMS)):
        name = "seq_%05d_%05d" % (s, ci)
        vd = os.path.join(OUT, "video_images", name)
        os.makedirs(vd, exist_ok=True)
        for t in range(a.frames):
            im = draw(X[t], CAMS[ci]["cam"])
            imageio.imwrite(os.path.join(vd, "%05d.jpg" % t), im, quality=92)
            if t == 0:
                imageio.imwrite(os.path.join(OUT, "images", name + ".jpg"), im,
                                quality=92)
        frames_json.append({"file_path": "images/%s.jpg" % name,
                            "transform_matrix": CAMS[ci]["c2w"].tolist()})
    n_ok += 1
    print("  시퀀스 %02d: %d 프레임 x %d 시점  최대 변위 %.2f%%"
          % (s, a.frames, len(CAMS),
             100 * float((X - X[0]).norm(dim=-1).max()) / float(sc.extent)),
          flush=True)

for phase in ("train", "test"):
    json.dump({"camera_angle_x": float(FOVX), "frames": frames_json},
              open(os.path.join(OUT, "transforms_%s.json" % phase), "w"))
gs.save_ply(os.path.join(OUT, "point_cloud.ply"))
# 마스크는 **저장한 ply 전체 길이**여야 한다. 배경을 살려두었으므로 N_ALL 이고,
# 움직이는 것은 물질 가우시안 자리뿐이다. 길이가 물질 수(N)면 로더가 전체 ply 를
# 그 마스크로 인덱싱하다 터진다.
mov = torch.zeros(N_ALL, dtype=torch.bool)
mov[MAT.cpu()] = True
# GausSim 은 이 pkl 을 torch 텐서로 읽는다 (torch.sum(pcmask) 를 그대로 부른다).
# numpy 로 쓰면 데이터셋 생성 단계에서 TypeError 가 난다.
pickle.dump(mov, open(os.path.join(OUT, "pc_mask.pkl"), "wb"))
pickle.dump(mov, open(os.path.join(OUT, "cln_pc_mask.pkl"), "wb"))
pin_g = np.zeros(N_ALL, dtype=bool)
json.dump([np.nonzero(pin_g)[0].tolist()], open(os.path.join(OUT, "pin_mask.json"), "w"))
json.dump({"train": [f["file_path"].split("/")[-1].replace(".jpg", "")
                     for f in frames_json],
           "test": [], "twin": [], "invalid": [], "force_invalid": [],
           "replace_seq": [], "replace_seq_avoid": [], "twin_replace": [],
           "replace": [], "replace_index": []},
          open(os.path.join(OUT, "meta.json"), "w"))
print("[완료] 시퀀스 %d, 프레임 항목 %d -> %s" % (n_ok, len(frames_json), OUT))
print("MAKE_DATASET_OK")

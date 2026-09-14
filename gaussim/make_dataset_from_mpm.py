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
ap.add_argument("--n_sim", type=int, default=8000,
                help="시뮬레이션할 물질 가우시안 수 (0 이면 전부). "
                     "GausSim 군집화가 n^2 메모리를 쓴다")
ap.add_argument("--n_pin", type=int, default=32,
                help="고정 영역에서 뽑을 앵커 수 (GausSim 최상위 계층용)")
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
os.makedirs(os.path.join(OUT, "images_mov"), exist_ok=True)
os.makedirs(os.path.join(OUT, "video_mov_images"), exist_ok=True)

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
KEEP = sc.keep.clone()
if a.n_sim > 0 and int(KEEP.sum()) > a.n_sim:
    # GausSim 의 계층 군집화는 complete-linkage 응집 군집이라 첫 단계에서 전체
    # 거리행렬(n^2)을 잡는다. 물질 가우시안 62k 면 31GB 로, 컨테이너 한도(42.5GB)를
    # 넘겨 프로세스가 아니라 **컨테이너가 통째로** 죽는다. 시뮬 대상을 고르게 솎는다.
    _i = torch.nonzero(KEEP).squeeze(-1)
    _g = torch.Generator(device="cpu").manual_seed(a.seed)
    _pick = _i[torch.randperm(_i.numel(), generator=_g)[:a.n_sim]]
    KEEP = torch.zeros_like(KEEP)
    KEEP[_pick] = True
    print(f"[시뮬] 물질 {int(sc.keep.sum())} -> {int(KEEP.sum())} 로 솎음 "
          f"(군집화 메모리 n^2)", flush=True)
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
    """전체 씬. 배경 가우시안은 정지한 채 같이 그려진다."""
    dx = torch.zeros(N_ALL, 3, device=dev, dtype=sc.pos.dtype)
    dx[MAT] = sc.undo(x) - sc.undo(G0)
    im = torch.clamp(_render(cam, gs, pipe, BG, dx, ZR, ZS,
                             d_rot_as_res=True)["render"], 0, 1)
    return (im.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")


# GausSim 의 config 는 render_mov_only=True 라 images_mov / video_mov_images 를
# 읽는다 -- 움직이는 부분만 남긴 렌더다. 전체 씬과 별개로 한 벌 더 만든다.
gs_mov = GaussianModel(3, fea_dim=0)
gs_mov.load_ply(a.ply)
for nm in FIELDS:
    setattr(gs_mov, nm, getattr(gs_mov, nm)[MAT])
ZR_M = torch.zeros(N, 4, device=dev); ZR_M[:, 0] = 1.0
ZS_M = torch.zeros(N, 3, device=dev)


BG_K = torch.zeros(3, device=dev)
BG_W = torch.ones(3, device=dev)


def draw_mov(x, cam, rgba=False):
    """움직이는 부분만. rgba 면 알파(물체 덮개)까지 낸다.

    래스터라이저는 배경 위에 합성한 결과만 준다. 같은 장면을 검정과 흰 배경에
    두 번 그리면 흰 쪽이 (1-a) 만큼 밝으므로 a = 1 - (흰 - 검정) 이고, 검정 쪽이
    곧 a*C 라 C = (검정)/a 로 되돌릴 수 있다. GausSim 의 로더가 다시 a 를 곱하므로
    여기서는 **곱하지 않은** C 와 a 를 내보내야 한다.
    """
    dx = sc.undo(x) - sc.undo(G0)
    ik = torch.clamp(_render(cam, gs_mov, pipe, BG_K, dx, ZR_M, ZS_M,
                             d_rot_as_res=True)["render"], 0, 1)
    if not rgba:
        return (ik.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
    iw = torch.clamp(_render(cam, gs_mov, pipe, BG_W, dx, ZR_M, ZS_M,
                             d_rot_as_res=True)["render"], 0, 1)
    alpha = (1.0 - (iw - ik).mean(0)).clamp(0, 1)
    rgb = (ik / alpha.clamp(min=1e-3).unsqueeze(0)).clamp(0, 1)
    out = torch.cat([rgb, alpha.unsqueeze(0)], 0)
    return (out.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")


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
        vdm = os.path.join(OUT, "video_mov_images", name)
        os.makedirs(vd, exist_ok=True)
        os.makedirs(vdm, exist_ok=True)
        for t in range(a.frames):
            im = draw(X[t], CAMS[ci]["cam"])
            imv = draw_mov(X[t], CAMS[ci]["cam"], rgba=True)
            # _mov 쪽은 png 로 읽는다 (로더의 suffix_replace=['.jpg','.png']).
            imageio.imwrite(os.path.join(vd, "%05d.jpg" % t), im, quality=92)
            imageio.imwrite(os.path.join(vdm, "%05d.png" % t), imv)
            if t == 0:
                imageio.imwrite(os.path.join(OUT, "images", name + ".jpg"), im,
                                quality=92)
                # 대표 프레임(images_mov)은 jpg 그대로다 -- 로더가 확장자를 png 로
                # 바꾸는 것은 video_mov_images 쪽뿐이다.
                imageio.imwrite(os.path.join(OUT, "images_mov", name + ".jpg"),
                                imv[..., :3], quality=92)
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
# pin 은 **움직이는 부분 안에서의 인덱스**다 (GausSim 이 torch.sum(pcmask) 길이의
# 마스크를 만들고 여기 담긴 번호로 True 를 세운다). 씬 config 가 속도를 0 으로
# 묶어두는 상자(enforce_particle_translation / cuboid)를 그대로 쓴다 -- 비워두면
# 계층 클러스터링이 빈 배열을 받아 터진다.
pin_local = torch.zeros(N, dtype=torch.bool)
mat_pos = sc.pos[KEEP].to(dev)
for bc in sc.cfg.get("boundary_conditions", []):
    if bc.get("type") not in ("cuboid", "enforce_particle_translation"):
        continue
    if float(np.linalg.norm(bc.get("velocity", [0, 0, 0]))) > 0:
        continue                       # 움직이는 구동기는 고정점이 아니다
    c = torch.tensor(bc["point"], device=dev, dtype=mat_pos.dtype)
    hs = torch.tensor(bc["size"], device=dev, dtype=mat_pos.dtype)
    pin_local |= (((mat_pos - c).abs() <= hs).all(-1)).cpu()
# GausSim 에서 pin 은 "고정된 점 전부"가 아니다. 최상위 계층에서 각 클러스터가
# **가장 가까운 pin 하나**에 붙고(n_clusters = max(index)+1), 모든 pin 이 적어도
# 하나의 클러스터에 뽑혀야 노드 수가 맞는다. 고정 영역 전체(수만 개)를 그대로 주면
# 대부분이 안 쓰여 노드 수 불일치로 터진다. 그래서 고정 영역 안에서 서로 멀리 떨어진
# 소수만 최원점 샘플링으로 고른다 -- 어디가 잡혀 있는지는 그대로 담긴다.
pin_idx_all = torch.nonzero(pin_local).squeeze(-1)
P = pin_idx_all.numel()
if P > a.n_pin:
    pts = mat_pos[pin_idx_all.to(dev)]
    sel = [0]
    d = (pts - pts[0]).norm(dim=-1)
    for _ in range(a.n_pin - 1):
        j = int(d.argmax())
        sel.append(j)
        d = torch.minimum(d, (pts - pts[j]).norm(dim=-1))
    pin_idx = pin_idx_all[torch.tensor(sorted(set(sel)))]
else:
    pin_idx = pin_idx_all
print(f"[pin] 고정 영역 {P} / {N} -> 앵커 {pin_idx.numel()} 개로 추림", flush=True)
json.dump([int(i) for i in pin_idx.tolist()],
          open(os.path.join(OUT, "pin_mask.json"), "w"))
json.dump({"train": [f["file_path"].split("/")[-1].replace(".jpg", "")
                     for f in frames_json],
           "test": [], "twin": [], "invalid": [], "force_invalid": [],
           "replace_seq": [], "replace_seq_avoid": [], "twin_replace": [],
           "replace": [], "replace_index": []},
          open(os.path.join(OUT, "meta.json"), "w"))
print("[완료] 시퀀스 %d, 프레임 항목 %d -> %s" % (n_ok, len(frames_json), OUT))
print("MAKE_DATASET_OK")

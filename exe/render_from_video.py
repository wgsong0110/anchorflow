"""영상 학습한 학생의 소성 씬 롤아웃을 MPM 정답과 나란히 렌더한다.

train_from_video.py 가 남긴 ckpt.pt(학생, 초기 속도, 앵커 기하)를 읽어 롤아웃하고,
같은 카메라에서 MPM 정답과 좌우로 붙여 영상을 만든다. 위치만 옮겨 그린다 --
정답 쪽 F 를 따로 내보내지 않으므로 한쪽만 공분산을 반영하면 비교가 불공정하다.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))

import imageio
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--ply", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--cameras", required=True)
ap.add_argument("--dreamphysics", required=True)
ap.add_argument("--ckpt", default=None,
                help="학생 ckpt. --gt_only 면 없어도 된다")
ap.add_argument("--gt_only", action="store_true",
                help="학생 없이 MPM 정답만 렌더한다 (ckpt 를 못 구할 때)")
ap.add_argument("--out", required=True)
ap.add_argument("--n_anchors", type=int, default=512)
ap.add_argument("--K", type=int, default=8)
ap.add_argument("--frames", type=int, default=40)
ap.add_argument("--width", type=int, default=480)
ap.add_argument("--fps", type=int, default=10)
ap.add_argument("--view", type=int, default=0, help="ckpt 에 저장된 시점 중 몇 번째")
ap.add_argument("--pick_view", action="store_true",
                help="ckpt 의 학습 시점 대신 커버리지 기준으로 고른다 -- "
                     "학습 시점은 근접 촬영이라 물체가 화면을 벗어난다")
a = ap.parse_args()

dev = "cuda"
torch.set_grad_enabled(False)
os.makedirs(a.out, exist_ok=True)
sys.path.insert(0, a.dreamphysics)
import warp as wp

wp.init()
from anchorflow import scene_setup
from anchorflow.mpm_teacher import MPMTeacher
from anchorflow.nextstate import NextStep, apply_step

sc = scene_setup.build(a.ply, a.config, a.n_anchors, a.K, device=dev,
                       frozen_weights=True, rot_fallback=True, eig_floor=0.02)
FRAME_DT = float(sc.cfg.get("frame_dt", 0.04))
if a.gt_only:
    st = None
    print(f"[gt_only] 학생 없이 MPM 정답만 그린다, 재질 {sc.cfg.get('material')}",
          flush=True)
else:
    st = torch.load(a.ckpt, map_location=dev, weights_only=False)
    print(f"[ckpt] iter {st['iter']}, 시점 {st['views']}, "
          f"재질 {sc.cfg.get('material')}", flush=True)

if st is not None:
    net = NextStep(hidden=128, depth=4, heads=4, use_accel=False,
                   scale=float(sc.extent),
                   vel_scale=float(sc.extent) / max(FRAME_DT, 1e-6),
                   zero_init=True).to(dev)
    net.load_state_dict(st["net"])
    net.eval()
    AC = st["ac"].to(dev)
    v0 = st["v0"].to(dev)

# --- MPM 정답 ---
T = MPMTeacher(sc, horizon=a.frames * FRAME_DT)
n_sub = max(1, int(round(FRAME_DT / float(sc.sub_dt))))
T._set(T.pos_m.clone(), torch.zeros(T.n, 3, device=dev), T.eye.clone(),
       torch.zeros_like(T.eye))
truth = [T.pos_m.clone()]
for _ in range(a.frames - 1):
    bad = False
    for k in range(n_sub):
        T.solver.p2g2p(None, float(sc.sub_dt), device=T.wp_dev)
        if (k + 1) % 4 == 0 and (not T._in_domain() or not T._vel_safe(4)):
            bad = True
            break
    if bad:
        break
    truth.append(T.solver.export_particle_x_to_torch().clone())
TR = torch.stack(truth)
NF = TR.shape[0]
print(f"[mpm] {NF} 프레임", flush=True)

# --- 학생 롤아웃 ---
PR = None
if st is not None:
    p, v, gp = AC.clone(), v0.clone(), sc.pos.clone()
    pred = [gp[sc.keep].clone()]
    for _ in range(NF - 1):
        q, d = apply_step(net, p, v, None, FRAME_DT, sc.fixed_mask)[-1]
        p, v = q, d / FRAME_DT
        gp = sc.skin(p, gp)
        pred.append(gp[sc.keep].clone())
    PR = torch.stack(pred)
    print(f"[학생] {PR.shape[0]} 프레임, 최대 변위 "
          f"{float((PR - PR[0]).norm(dim=-1).max()) / float(sc.extent) * 100:.2f}%"
          " of 물체", flush=True)

# --- 렌더 ---
sys.path.insert(0, os.path.join(os.path.dirname(sc.cfg_path)
                                if hasattr(sc, "cfg_path") else ".", ""))
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render as _render
from utils.graphics_utils import focal2fov, getProjectionMatrix, getWorld2View2


class MiniCam:
    def __init__(self, w, h, fy, fx, zn, zf, wvt, fp):
        self.image_width, self.image_height = w, h
        self.FoVy, self.FoVx, self.znear, self.zfar = fy, fx, zn, zf
        self.world_view_transform, self.full_proj_transform = wvt, fp
        self.camera_center = wvt.inverse()[3, :3]


cams = json.load(open(a.cameras))
if st is not None and not a.pick_view:
    view_idx = st["views"][a.view]
else:
    # ckpt 가 없으면 train_from_video.py 와 같은 기준으로 고른다:
    # 물체가 화면 안에 많이, 그리고 크게 담기는 시점. 아무 학습 시점이나 쓰면
    # 근접 촬영이 걸려 물체가 화면 밖으로 나간다.
    _xw = sc.undo(sc.pos[sc.keep]).double().cpu().numpy()
    _rng = np.random.RandomState(0)
    _xs = _xw[_rng.choice(_xw.shape[0], min(4000, _xw.shape[0]), replace=False)]
    _sc = []
    for _i, _c in enumerate(cams):
        _w0, _h0 = int(_c["width"]), int(_c["height"])
        _R = np.array(_c["rotation"]); _t = -_R.T @ np.array(_c["position"])
        _cx = _xs @ _R + _t
        _z = _cx[:, 2]; _fr = _z > 1e-6
        if _fr.sum() < 16:
            continue
        _u = _c["fx"] * _cx[_fr, 0] / _z[_fr] + _w0 / 2
        _v = _c["fy"] * _cx[_fr, 1] / _z[_fr] + _h0 / 2
        _in = (_u >= 0) & (_u < _w0) & (_v >= 0) & (_v < _h0)
        if _in.sum() < 16:
            continue
        _sc.append((float(_in.mean()) * float(_fr.mean())
                    * min((np.ptp(_u[_in]) / _w0) * (np.ptp(_v[_in]) / _h0), 1.0),
                    _i))
    _sc.sort(reverse=True)
    view_idx = _sc[min(a.view, len(_sc) - 1)][1]
    print(f"[카메라] ckpt 없음 -> 커버리지 상위 시점 {view_idx} 선택 "
          f"(점수 {_sc[min(a.view, len(_sc)-1)][0]:.3f})", flush=True)
c = cams[view_idx]
w0, h0 = int(c["width"]), int(c["height"])
w = int(a.width); h = int(round(h0 * w / w0))
fx, fy = float(c["fx"]) * w / w0, float(c["fy"]) * h / h0
R = np.array(c["rotation"], dtype=np.float64)
Tv = -R.T @ np.array(c["position"], dtype=np.float64)
fovx, fovy = focal2fov(fx, w), focal2fov(fy, h)
wvt = torch.tensor(getWorld2View2(R, Tv)).transpose(0, 1).float().to(dev)
pmx = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=fovx,
                          fovY=fovy).transpose(0, 1).to(dev)
cam = MiniCam(w, h, fovy, fovx, 0.01, 100.0, wvt,
              (wvt.unsqueeze(0).bmm(pmx.unsqueeze(0))).squeeze(0))


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
# 씬 셋업은 sim_area 로 잘라낸 뒤(crop) 그 안에서 물질을 골랐다(keep). 그 둘을
# 그대로 렌더러에 적용하면 배경 가우시안이 통째로 사라져 흰 공백만 남는다 --
# PhysGaussian 공식 경로도 래스터화 직전에 비선택 가우시안을 다시 합친다
# (gs_simulation.py 의 torch.cat([opacity_render, unselected_opacity])).
# 그래서 여기서는 ply 전체를 그대로 두고, 움직이는 것은 물질 가우시안뿐이도록
# 전체 길이의 변위 벡터를 만들어 그 자리에만 채운다. 배경은 정지한 채 그려진다.
N_ALL = gs._xyz.shape[0]
idx_all = torch.arange(N_ALL, device=dev)
if getattr(sc, "crop", None) is not None:
    idx_all = idx_all[sc.crop]
MAT = idx_all[sc.keep]                      # ply 전체 인덱스 중 물질인 것
ZR = torch.zeros(N_ALL, 4, device=dev); ZR[:, 0] = 1.0
ZS = torch.zeros(N_ALL, 3, device=dev)
N = int(MAT.shape[0])
G0 = sc.pos[sc.keep]
print(f"[렌더] 가우시안 전체 {N_ALL}, 그중 물질 {N} (나머지는 정지 배경)",
      flush=True)


def draw(x):
    dx = torch.zeros(N_ALL, 3, device=dev, dtype=sc.pos.dtype)
    dx[MAT] = sc.undo(x) - sc.undo(G0)
    im = torch.clamp(_render(cam, gs, pipe, BG, dx, ZR, ZS,
                             d_rot_as_res=True)["render"], 0, 1)
    return (im.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")


def label(img, text):
    from PIL import Image, ImageDraw
    im = Image.fromarray(img); dr = ImageDraw.Draw(im)
    dr.rectangle([0, 0, im.width, 18], fill=(0, 0, 0))
    dr.text((4, 4), text, fill=(255, 255, 255))
    return np.array(im)


frames = []
TRk = TR[:, sc.keep] if TR.shape[1] != N else TR
for t in range(NF):
    left = label(draw(TRk[t]), f"MPM ({sc.cfg.get('material')}) f{t:02d}")
    if PR is None:
        frames.append(left)
    else:
        right = label(draw(PR[t]), "student (video-trained)")
        frames.append(np.concatenate([left, right], axis=1))
p_out = os.path.join(a.out, "mpm_only.mp4" if PR is None else "rollout_vs_mpm.mp4")
imageio.mimsave(p_out, frames, fps=a.fps, quality=8)
print(f"[저장] {p_out}", flush=True)
print("RENDER_OK")

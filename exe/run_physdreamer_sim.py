"""PhysDreamer 의 시뮬레이터만 우리 소성 씬에 돌린다.

PhysDreamer 의 기여는 (a) 영상에서 물질장을 학습하는 것과 (b) 그 물질장을 쓰는
MPM 시뮬레이터다. 여기서 재는 것은 (b) 뿐이다 -- 물성은 다른 베이스라인과 **같은
값으로 고정**해서 넣고, 시뮬레이터가 소성 변형을 어떻게 다루는지만 본다.
물질장을 학습시키면 비교 대상이 물성 추정 성능이 되어버려 뜻이 달라진다.

그쪽 솔버(`physdreamer/warp_mpm`)는 재질 이름과 항복응력을 그대로 받으므로
(`material="metal"`, `yield_stress=...`), 우리 plane config 의 값을 그대로 넘긴다.
초기 조건도 다른 실행과 같은 임펄스를 쓴다.

렌더는 PhysGaussian 공식 궤도 카메라(dump_cams.py 산출)를 그대로 쓸 수 있어,
같은 시점에서 다른 방법들과 나란히 붙일 수 있다.
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
ap.add_argument("--config", required=True, help="소성 씬 config (plane_dp.json 등)")
ap.add_argument("--cameras", required=True)
ap.add_argument("--physdreamer", required=True, help="PhysDreamer 저장소 경로")
ap.add_argument("--scgs", required=True, help="렌더러(SC-GS) 경로")
ap.add_argument("--anchorflow", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--cam_seq", default=None, help="PhysGaussian 궤도 카메라 JSON")
ap.add_argument("--frames", type=int, default=40)
ap.add_argument("--width", type=int, default=540)
ap.add_argument("--n_grid", type=int, default=100)
ap.add_argument("--fps", type=int, default=10)
a = ap.parse_args()

sys.path.insert(0, a.anchorflow)
sys.path.insert(0, a.scgs)
sys.path.insert(0, a.physdreamer)

import warp as wp  # noqa: E402

wp.init()
from anchorflow import scene_setup  # noqa: E402
from physdreamer.warp_mpm.mpm_data_structure import (MPMModelStruct,  # noqa: E402
                                                     MPMStateStruct)
from physdreamer.warp_mpm.mpm_solver_diff import MPMWARPDiff  # noqa: E402

dev = "cuda"
torch.set_grad_enabled(False)
os.makedirs(a.out, exist_ok=True)

# 씬은 다른 실행과 같은 경로로 만든다 -- 같은 크롭, 같은 물질 선택, 같은 정규화.
sc = scene_setup.build(a.ply, a.config, 512, 8, device=dev,
                       frozen_weights=True, rot_fallback=True, eig_floor=0.02)
cfg = sc.cfg
FRAME_DT = float(cfg.get("frame_dt", 0.01))
SUB_DT = float(cfg.get("substep_dt", 1e-4))
n_sub = max(1, int(round(FRAME_DT / SUB_DT)))
mat_pos = sc.pos[sc.keep].contiguous()
N = mat_pos.shape[0]
vol = sc.volume[sc.keep].contiguous()
print(f"[씬] 물질 입자 {N}, frame_dt {FRAME_DT}, 서브스텝 {n_sub}, "
      f"재질 {cfg.get('material')}", flush=True)

# --- PhysDreamer 솔버 구성 (demo.py 와 같은 순서) ---
state = MPMStateStruct()
state.init(N, device=dev, requires_grad=False)
state.from_torch(mat_pos.clone(), vol.clone(), None, device=dev,
                 requires_grad=False, n_grid=a.n_grid, grid_lim=2.0)
model = MPMModelStruct()
model.init(N, device=dev, requires_grad=False)
model.init_other_params(n_grid=a.n_grid, grid_lim=2.0, device=dev)

# 물성은 **우리 config 값 그대로**. 학습하지 않는다.
mp = {"material": cfg.get("material", "metal"),
      "g": list(cfg.get("g", [0.0, 0.0, 0.0])),
      "density": float(cfg.get("density", 1000)),
      "grid_v_damping_scale": 1.1}
if "yield_stress" in cfg:
    mp["yield_stress"] = float(cfg["yield_stress"])
if "friction_angle" in cfg:
    mp["friction_angle"] = float(cfg["friction_angle"])
solver = MPMWARPDiff(N, n_grid=a.n_grid, grid_lim=2.0, device=dev)
solver.set_parameters_dict(model, state, mp)
solver.set_E_nu(model, float(cfg.get("E", 1e5)), float(cfg.get("nu", 0.3)),
                device=dev)
solver.prepare_mu_lam(model, state, device=dev)
print(f"[물성] {mp} | E {cfg.get('E')} nu {cfg.get('nu')}", flush=True)

# --- 초기 조건: 다른 실행과 같은 임펄스 ---
v0 = torch.zeros(N, 3, device=dev)
for bc in cfg.get("boundary_conditions", []):
    if bc.get("type") == "particle_impulse":
        f = torch.tensor(bc["force"], device=dev).unsqueeze(0).expand(
            sc.pos.shape[0], 3).contiguous()
        v0 = sc.impulse_dv(f)[sc.keep].contiguous()
        break
state.continue_from_torch(mat_pos.clone(), v0, None, device=dev,
                          requires_grad=False)
print(f"[초기] |v0| 최대 {float(v0.norm(dim=-1).max()):.4f}", flush=True)

# --- 롤아웃 ---
traj, bad = [mat_pos.clone()], False
for fr in range(a.frames - 1):
    for k in range(n_sub):
        solver.p2g2p(model, state, fr, SUB_DT, device=dev)
    x = state.particle_x.to(torch.device(dev)) if hasattr(state, "particle_x") else None
    x = wp.to_torch(state.particle_x).clone()
    if not torch.isfinite(x).all():
        print(f"[롤아웃] 프레임 {fr+1} 에서 비유한값 -- 여기까지만 쓴다", flush=True)
        bad = True
        break
    traj.append(x)
X = torch.stack(traj)
NF = X.shape[0]
d = float((X - X[0]).norm(dim=-1).max()) / float(sc.extent) * 100
print(f"[롤아웃] {NF} 프레임, 최대 변위 {d:.2f}% of 물체"
      f"{' (도중 발산)' if bad else ''}", flush=True)

# --- 렌더: 다른 방법들과 같은 카메라 ---
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


CAM_SEQ = None
if a.cam_seq:
    cs = json.load(open(a.cam_seq))
    w = int(a.width); h = int(round(cs[0]["height"] * w / cs[0]["width"]))
    CAM_SEQ = [build_cam(np.array(c["R"], dtype=np.float64),
                         np.array(c["T"], dtype=np.float64),
                         c["FoVx"], c["FoVy"], w, h) for c in cs]
    print(f"[카메라] PhysGaussian 궤도 {len(CAM_SEQ)} 프레임, {w}x{h}", flush=True)
else:
    cams = json.load(open(a.cameras))
    c = cams[0]
    w0, h0 = int(c["width"]), int(c["height"])
    w = int(a.width); h = int(round(h0 * w / w0))
    R = np.array(c["rotation"], dtype=np.float64)
    T = -R.T @ np.array(c["position"], dtype=np.float64)
    cam0 = build_cam(R, T, focal2fov(float(c["fx"]) * w / w0, w),
                     focal2fov(float(c["fy"]) * h / h0, h), w, h)
    CAM_SEQ = [cam0] * NF


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
# 배경 가우시안을 살려둔다 -- 공식 경로도 래스터화 직전에 다시 합친다.
N_ALL = gs._xyz.shape[0]
idx_all = torch.arange(N_ALL, device=dev)
if getattr(sc, "crop", None) is not None:
    idx_all = idx_all[sc.crop]
MAT = idx_all[sc.keep]
ZR = torch.zeros(N_ALL, 4, device=dev); ZR[:, 0] = 1.0
ZS = torch.zeros(N_ALL, 3, device=dev)
G0 = sc.pos[sc.keep]
print(f"[렌더] 전체 {N_ALL}, 물질 {int(MAT.shape[0])}", flush=True)

import imageio  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

frames = []
for t in range(NF):
    dx = torch.zeros(N_ALL, 3, device=dev, dtype=sc.pos.dtype)
    dx[MAT] = sc.undo(X[t]) - sc.undo(G0)
    cam = CAM_SEQ[min(t, len(CAM_SEQ) - 1)]
    im = torch.clamp(_render(cam, gs, pipe, BG, dx, ZR, ZS,
                             d_rot_as_res=True)["render"], 0, 1)
    arr = (im.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
    pim = Image.fromarray(arr); dr = ImageDraw.Draw(pim)
    dr.rectangle([0, 0, pim.width, 18], fill=(0, 0, 0))
    dr.text((4, 4), f"PhysDreamer sim ({cfg.get('material')}) f{t:02d}",
            fill=(255, 255, 255))
    frames.append(np.array(pim))

p_out = os.path.join(a.out, "physdreamer_sim.mp4")
imageio.mimsave(p_out, frames, fps=a.fps, quality=8)
json.dump({"frames": NF, "diverged": bool(bad), "max_disp_pct": d,
           "material": cfg.get("material"), "E": cfg.get("E"),
           "nu": cfg.get("nu"), "density": cfg.get("density")},
          open(os.path.join(a.out, "physdreamer_sim.json"), "w"), indent=1)
print(f"[저장] {p_out}", flush=True)
print("PD_SIM_OK")

"""PhysGaussian/GaussianFluent 의 **warp MPM 솔버만** 가지고 씬을 돌린다.

3DGS 는 쓰지 않는다. `mpm_solver_warp` 는 위치·부피만 받는 순수 입자 솔버이고,
가우시안 결합은 바깥 러너(`gs_simulation.py`)가 하는 일이다. 여기서는 기하만
h5 (GF 의 채우기를 거친 입자 구름) 에서 가져오고, 재질·경계·구동은 GF 의 config
어휘를 그대로 쓴다 -- 경계는 GF 의 `set_boundary_conditions` 를 그대로 부르므로
일곱 종류가 다 된다.

config 에 더한 것은 `clouds` 하나뿐이다. 구름을 여러 개 놓고 옮기고 노치를 낼 수
있어야 접합·인열 시험이 된다.

  {
    "clouds": [{"h5": "...", "translate": [0,-0.2,0]},
               {"h5": "...", "translate": [0, 0.2,0]}],
    "material": "watermelon", "E": 2e3, ... ,
    "boundary_conditions": [...]
  }

  python exe/run_warp_mpm.py --gf <GaussianFluent> --config <json> --out DIR
"""
import argparse
import json
import os
import sys
import time

import h5py
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--gf", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--frames", type=int, default=None)
ap.add_argument("--sort_every", type=int, default=1,
                help="몇 프레임마다 입자를 칸 순서로 다시 세울지 (0 이면 안 함)")
a = ap.parse_args()

sys.path.insert(0, a.gf)
os.chdir(a.gf)

import torch
import warp as wp
from mpm_solver_warp.mpm_solver_warp import MPM_Simulator_WARP
from mpm_solver_warp.engine_utils import save_data_at_frame
from utils.decode_param import set_boundary_conditions

wp.init()
cfg = json.load(open(a.config))
dev = "cuda:0"
n_grid = int(cfg.get("n_grid", 100))
grid_lim = float(cfg.get("grid_lim", 2.0))
dx = grid_lim / n_grid


def load_cloud(spec):
    with h5py.File(spec["h5"], "r") as h:
        x = np.array(h["x"])
    x = (x.T if x.shape[0] == 3 else x).astype(np.float64)
    x = x[np.isfinite(x).all(1)]
    if spec.get("stride", 1) > 1:
        x = x[::int(spec["stride"])]
    if "scale" in spec:
        c = 0.5 * (x.min(0) + x.max(0))
        x = (x - c) * float(spec["scale"]) + c
    if "notch" in spec:
        # 균열 자리를 정해 주는 쐐기. 파괴 시험에서 표준이다.
        nd = spec["notch"]
        ax, cut, dep, wid = int(nd["axis"]), float(nd["at"]), float(nd["depth"]), float(nd["width"])
        oth = [i for i in range(3) if i != ax]
        lo, hi = x[:, oth[1]].min(), x[:, oth[1]].max()
        keep = ~((np.abs(x[:, ax] - cut) < wid)
                 & (x[:, oth[1]] > hi - dep * (hi - lo)))
        x = x[keep]
    if "translate" in spec:
        x = x + np.asarray(spec["translate"], np.float64)
    return x


clouds = [load_cloud(s) for s in cfg["clouds"]]
sizes = [len(c) for c in clouds]
X = np.concatenate(clouds).astype(np.float32)
N = len(X)
# 구름마다 표시를 남긴다 -- 접합 시험에서 경계면 쌍과 내부 쌍을 갈라야 한다
GRP = np.concatenate([np.full(n, i, np.int32) for i, n in enumerate(sizes)])

# 입자 부피는 GF 와 같은 정의: 셀마다 세고 dx^3/개수
cell = np.clip(np.floor(X / dx).astype(np.int64), 0, n_grid - 1)
flat = (cell[:, 0] * n_grid + cell[:, 1]) * n_grid + cell[:, 2]
_u, _i, _c = np.unique(flat, return_inverse=True, return_counts=True)
VOL = ((dx ** 3) / _c[_i]).astype(np.float32)
if cfg["material"] == "sand":
    VOL[:] = VOL.mean()

print(f"[구름] {len(clouds)} 개 {sizes} -> 입자 {N}, "
      f"범위 {np.round(X.min(0),3)}~{np.round(X.max(0),3)}, "
      f"부피 중앙 {np.median(VOL):.3e}", flush=True)

solver = MPM_Simulator_WARP(10)
solver.load_initial_data_from_torch(
    torch.from_numpy(np.ascontiguousarray(X)).to(dev).contiguous(),
    torch.from_numpy(np.ascontiguousarray(VOL)).to(dev).contiguous(),
    n_grid=n_grid, grid_lim=grid_lim, device=dev)

mp = {k: cfg[k] for k in (
    "material", "E", "nu", "density", "friction_angle", "beta", "xi", "hardening",
    "yield_stress", "softening", "plastic_viscosity", "rpic_damping",
    "grid_v_damping_scale", "alpha_0", "g", "n_grid", "grid_lim") if k in cfg}
mp.setdefault("n_grid", n_grid); mp.setdefault("grid_lim", grid_lim)
solver.set_parameters_dict(mp, device=dev)
solver.finalize_mu_lam(device=dev)

tp = dict(substep_dt=float(cfg["substep_dt"]), frame_dt=float(cfg["frame_dt"]),
          frame_num=int(cfg.get("frame_num", 100)))
set_boundary_conditions(solver, cfg.get("boundary_conditions", []), tp)

v0 = np.asarray(cfg.get("init_velocity", [0.0, 0.0, 0.0]), np.float32)
solver.import_particle_v_from_torch(
    torch.from_numpy(np.tile(v0, (N, 1))).to(dev).contiguous(), device=dev)

dt = tp["substep_dt"]
if cfg.get("auto_dt", False):
    E, nu, rho = float(cfg["E"]), float(cfg["nu"]), float(cfg["density"])
    c = np.sqrt(E * (1 - nu) / ((1 + nu) * (1 - 2 * nu) * rho))
    dt = 0.6 * dx / c
nsub = max(1, int(tp["frame_dt"] / dt))
frames = a.frames if a.frames is not None else tp["frame_num"]
# gs_simulation.py:429 의 규칙. `p2g2p` 의 flip_pic 기본값이 True 라 이걸 안 넘기면
# 비율 0 을 줘도 **FLIP 경로가 비율 0 으로** 돌아 순수 PIC 이 된다 -- 회전 성분을
# 통째로 버려 모래도 찰흙도 뭉개진다. 0 이면 APIC 이어야 한다.
flip = float(cfg.get("flip_pic_ratio", 0.7))
use_flip = flip > 0.0
print(f"[설정] 재질 {cfg['material']}, 격자 {n_grid}, dt {dt:.3e} x {nsub}, "
      f"{frames} 프레임, {'FLIP '+str(flip) if use_flip else 'APIC'}, "
      f"v0 {v0}", flush=True)

out = os.path.abspath(a.out)
os.makedirs(out, exist_ok=True)
np.save(os.path.join(out, "group.npy"), GRP)
save_data_at_frame(solver, out, 0, save_to_ply=False, save_to_h5=True)
t0 = time.time()
for f in range(frames):
    if a.sort_every and f % a.sort_every == 0 and hasattr(solver, "af_sort_by_cell"):
        solver.af_sort_by_cell()
    for s in range(nsub):
        solver.p2g2p(s, dt, device=dev, flip_pic_ratio=flip, flip_pic=use_flip)
    save_data_at_frame(solver, out, f + 1, save_to_ply=False, save_to_h5=True)
    if (f + 1) % 10 == 0:
        print(f"  f{f+1:4d}  {time.time()-t0:.0f}s", flush=True)
print(f"[저장] {out}  {frames+1} 프레임  {time.time()-t0:.0f}s", flush=True)
print("WARP_MPM_DONE", flush=True)

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
ap.add_argument("--seed", type=int, default=0, help="제어점과 궤적의 시드")
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
    if "box" in spec:
        # 절차적 기둥/상자. 안식각은 **기둥 붕괴**로 재는 것이 표준이라
        # (늑대 모양이 무너진 더미는 원뿔이 아니어서 각을 못 읽는다) 필요하다.
        lo, hi = np.asarray(spec["box"][0], float), np.asarray(spec["box"][1], float)
        sp = float(spec.get("spacing", 0.01))
        g = np.stack(np.meshgrid(*[np.arange(lo[i] + sp / 2, hi[i], sp)
                                   for i in range(3)], indexing="ij"), -1)
        x = g.reshape(-1, 3)
        if spec.get("jitter", 0.0):
            r = np.random.default_rng(int(spec.get("seed", 0)))
            x = x + r.uniform(-1, 1, x.shape) * float(spec["jitter"]) * sp
        return x
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

# ------------------------------------------------------- 제어점 (Dirichlet)
# 표면 입자 몇 개를 골라 매 스텝 위치·속도를 궤적으로 **강제**한다. 나머지는
# 평소대로 푼다. GF 의 BC 는 구역마다 속도가 하나로 고정이라 제어점마다 다른
# 궤적을 줄 수 없어서, 솔버 배열을 직접 쓰는 쪽으로 한다.
CC = cfg.get("control", None)
ctrl_idx = None
if CC:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
    from control_traj import ControlTraj, pick_control_points
    seed = int(CC.get("seed", a.seed))
    ctrl_idx, surf = pick_control_points(X.astype(np.float64), CC.get("cell", None),
                                         int(CC.get("n_points", 4)), seed=seed)
    # 제어점 하나를 입자 한 개로 두면 물체 지름의 0.7% 라 아무 영향이 없다.
    # 반경 안의 입자를 **통째로** 같이 끌고 간다 (손가락/도구에 해당).
    _ext = float(np.linalg.norm(X.max(0) - X.min(0)))
    _rad = float(CC.get("radius", 0.0)) * _ext
    if _rad > 0:
        _mem, _off = [], []
        for c in ctrl_idx:
            d = np.linalg.norm(X - X[c], axis=1)
            m = np.flatnonzero(d <= _rad)
            _mem.append(m)
            _off.append((X[m] - X[c]).astype(np.float32))
        print(f"[제어점 반경] {CC.get('radius')} x 지름 = {_rad:.4f}, "
              f"잡은 입자 {[len(m) for m in _mem]}", flush=True)
    else:
        _mem = [np.array([c]) for c in ctrl_idx]
        _off = [np.zeros((1, 3), np.float32) for _ in ctrl_idx]
    # 잡은 입자가 격자 밖으로 나가지 않게 궤적을 안쪽으로 묶는다
    _pad = 4.0 * dx + _rad
    _bnd = (np.full(3, _pad), np.full(3, grid_lim - _pad))
    traj = ControlTraj(X.astype(np.float64), ctrl_idx,
                       dt=float(cfg["frame_dt"]),
                       steps=int(cfg.get("frame_num", 100)) + 1,
                       every_n=int(CC.get("every_n", 20)),
                       p_touch=float(CC.get("p_touch", 0.6)),
                       depth=float(CC.get("depth", 0.08)),
                       v_max=float(CC.get("v_max", 0.5)),
                       seed=seed, bounds=_bnd)
    print(f"[제어점] {len(ctrl_idx)} 개 (표면 {int(surf.sum())}/{len(X)}), "
          f"n={CC.get('every_n',20)} p={CC.get('p_touch',0.6)} "
          f"깊이={CC.get('depth',0.08)} v_max={CC.get('v_max',0.5)} 시드={seed}, "
          f"실제 최대속도 {traj.max_speed():.4f}", flush=True)

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
if ctrl_idx is not None:
    np.savez(os.path.join(out, "control.npz"), idx=ctrl_idx,
             pos=traj.P.astype(np.float32), vel=traj.V.astype(np.float32),
             x0=X[ctrl_idx].astype(np.float32),
             members=np.concatenate(_mem).astype(np.int64),
             member_ptr=np.cumsum([0] + [len(m) for m in _mem]).astype(np.int64),
             cfg=json.dumps(CC))
save_data_at_frame(solver, out, 0, save_to_ply=False, save_to_h5=True)
t0 = time.time()
# 정렬은 입자 순서를 바꾸므로 제어점을 쓸 때는 끈다 (색인이 어긋난다)
if ctrl_idx is not None:
    a.sort_every = 0
if ctrl_idx is not None:
    _cx = torch.from_numpy(np.ascontiguousarray(ctrl_idx)).to(dev)
    _mem_t = [torch.from_numpy(np.ascontiguousarray(m)).to(dev) for m in _mem]
    _off_t = [torch.from_numpy(o).to(dev) for o in _off]
else:
    _cx = None

for f in range(frames):
    if a.sort_every and f % a.sort_every == 0 and hasattr(solver, "af_sort_by_cell"):
        solver.af_sort_by_cell()
    if ctrl_idx is not None:
        # 프레임 안에서는 목표 위치로 **선형 보간**하며 서브스텝마다 다시 박는다
        p_now = torch.from_numpy(traj.pos(f).astype(np.float32)).to(dev)
        p_next = torch.from_numpy(traj.pos(f + 1).astype(np.float32)).to(dev)
        v_now = torch.from_numpy(traj.vel(f + 1).astype(np.float32)).to(dev)
    for s in range(nsub):
        if ctrl_idx is not None:
            w = (s + 1.0) / nsub
            cpos = p_now * (1.0 - w) + p_next * w
            tx = wp.to_torch(solver.mpm_state.particle_x)
            tv = wp.to_torch(solver.mpm_state.particle_v)
            for k in range(len(_mem_t)):      # 반경 안 입자를 통째로 옮긴다
                tx[_mem_t[k]] = cpos[k] + _off_t[k]
                tv[_mem_t[k]] = v_now[k]
        solver.p2g2p(s, dt, device=dev, flip_pic_ratio=flip, flip_pic=use_flip)
        if ctrl_idx is not None:     # g2p 가 옮겨 놓았으니 다시 박는다
            tx = wp.to_torch(solver.mpm_state.particle_x)
            tv = wp.to_torch(solver.mpm_state.particle_v)
            for k in range(len(_mem_t)):
                tx[_mem_t[k]] = cpos[k] + _off_t[k]
                tv[_mem_t[k]] = v_now[k]
    save_data_at_frame(solver, out, f + 1, save_to_ply=False, save_to_h5=True)
    if (f + 1) % 10 == 0:
        print(f"  f{f+1:4d}  {time.time()-t0:.0f}s", flush=True)
print(f"[저장] {out}  {frames+1} 프레임  {time.time()-t0:.0f}s", flush=True)
print("WARP_MPM_DONE", flush=True)

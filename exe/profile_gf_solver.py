"""GF 의 MPM 솔버만 떼어 **단계별로 시간을 잰다**. 가우시안 쪽은 건드리지 않는다.

솔버 안에 이미 `wp.ScopedTimer(..., dict=self.time_profile)` 가 박혀 있어
p2g / grid_update / apply_BC_on_grid / g2p / compute_stress 가 따로 잡힌다.
h5 한 장(위치·속도)과 config 만 있으면 3DGS 파이프라인 없이 돌릴 수 있다.

  python exe/profile_gf_solver.py --gf <GaussianFluent> --config <json> \
      --h5 <sim_0000000000.h5> --substeps 200
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
ap.add_argument("--h5", required=True)
ap.add_argument("--substeps", type=int, default=200)
ap.add_argument("--warmup", type=int, default=10)
ap.add_argument("--auto_dt", action="store_true")
ap.add_argument("--shuffle", action="store_true",
                help="입자 순서를 섞어서 잰다. p2g 가 **빨라지면** 느린 이유가 "
                     "대역폭이 아니라 같은 주소에 몰리는 원자적 덧셈이라는 뜻이다")
ap.add_argument("--sort", default="none", choices=("none", "cell", "morton"),
                help="입자를 격자 순서로 다시 줄 세운다. 섞으면 4.4 배 느려졌으므로 "
                     "지역성이 지배한다 -- 더 좋게 세우면 더 빨라질 수 있다")
ap.add_argument("--resort_every", type=int, default=0,
                help="몇 서브스텝마다 다시 줄 세울지. 0 이면 처음 한 번만")
ap.add_argument("--out", default=None)
a = ap.parse_args()

# gf 루트 **하나만** 넣는다. 안쪽 `mpm_solver_warp/` 를 같이 넣으면 그 안의
# 같은 이름 모듈(`mpm_solver_warp.py`)이 네임스페이스 패키지를 가려서
# "is not a package" 로 깨진다. 작업 디렉토리도 gf 로 옮긴다 -- 원본이 그렇게 돈다.
sys.path.insert(0, a.gf)
os.chdir(a.gf)
import torch
import warp as wp
from mpm_solver_warp.mpm_solver_warp import MPM_Simulator_WARP

wp.init()

cfg = json.load(open(a.config))
n_grid = int(cfg.get("n_grid", 100))
grid_lim = float(cfg.get("grid_lim", 2.0))
dx = grid_lim / n_grid
dev = "cuda:0"

with h5py.File(a.h5, "r") as h:
    X = np.array(h["x"]); X = (X.T if X.shape[0] == 3 else X).astype(np.float32)
    V = np.array(h["v"]); V = (V.T if V.shape[0] == 3 else V).astype(np.float32)

if a.shuffle:
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(X))
    X, V = X[perm], V[perm]
    print(f"[섞음] 입자 순서를 무작위로 바꿨다 ({len(X)} 개)", flush=True)

if a.sort != "none":
    _c = np.clip(np.floor(X / dx).astype(np.int64), 0, n_grid - 1)
    if a.sort == "cell":
        key = (_c[:, 0] * n_grid + _c[:, 1]) * n_grid + _c[:, 2]
    else:
        def _spread(v):           # 21 비트를 3 칸 간격으로 벌린다
            v = v & 0x1FFFFF
            v = (v | (v << 32)) & 0x1F00000000FFFF
            v = (v | (v << 16)) & 0x1F0000FF0000FF
            v = (v | (v << 8)) & 0x100F00F00F00F00F
            v = (v | (v << 4)) & 0x10C30C30C30C30C3
            v = (v | (v << 2)) & 0x1249249249249249
            return v
        key = (_spread(_c[:, 0]) | (_spread(_c[:, 1]) << 1)
               | (_spread(_c[:, 2]) << 2))
    ordr = np.argsort(key, kind="stable")
    X, V = X[ordr], V[ordr]
    print(f"[정렬] {a.sort} 순서로 다시 세웠다", flush=True)

# particle_filling.get_particle_volume 와 같은 정의
cell = np.clip(np.floor(X / dx).astype(np.int64), 0, n_grid - 1)
flat = (cell[:, 0] * n_grid + cell[:, 1]) * n_grid + cell[:, 2]
_u, _i, _c = np.unique(flat, return_inverse=True, return_counts=True)
vol = ((dx ** 3) / _c[_i]).astype(np.float32)
if cfg["material"] == "sand":
    vol[:] = vol.mean()

solver = MPM_Simulator_WARP(10)
# warp 로 감쌀 텐서는 연속이어야 한다 (`torch2warp_vec3` 가 단언한다)
_x = torch.from_numpy(np.ascontiguousarray(X)).to(dev).contiguous()
_v = torch.from_numpy(np.ascontiguousarray(V)).to(dev).contiguous()
solver.load_initial_data_from_torch(
    _x, torch.from_numpy(np.ascontiguousarray(vol)).to(dev).contiguous(),
    n_grid=n_grid, grid_lim=grid_lim, device=dev)
solver.import_particle_v_from_torch(_v, device=dev)

mp = {k: cfg[k] for k in ("material", "E", "nu", "density") if k in cfg}
for k in ("friction_angle", "beta", "xi", "hardening", "yield_stress", "softening",
          "plastic_viscosity", "rpic_damping", "grid_v_damping_scale", "alpha_0",
          "g", "n_grid", "grid_lim"):
    if k in cfg:
        mp[k] = cfg[k]
mp.setdefault("grid_lim", grid_lim); mp.setdefault("n_grid", n_grid)
solver.set_parameters_dict(mp, device=dev)
solver.finalize_mu_lam(device=dev)
for bc in cfg.get("boundary_conditions", []):
    if bc["type"] == "bounding_box":
        solver.add_bounding_box()
    elif bc["type"] == "surface_collider":
        solver.add_surface_collider(bc["point"], bc["normal"],
                                    bc.get("surface", "sticky"),
                                    bc.get("friction", 0.0),
                                    bc.get("start_time", 0.0),
                                    bc.get("end_time", 1e3))

dt = float(cfg["substep_dt"])
if a.auto_dt:
    E, nu, rho = float(cfg["E"]), float(cfg["nu"]), float(cfg["density"])
    c = np.sqrt(E * (1 - nu) / ((1 + nu) * (1 - 2 * nu) * rho))
    dt = 0.6 * dx / c
flip = float(cfg.get("flip_pic_ratio", 0.7))
print(f"[설정] 입자 {solver.n_particles}, 격자 {n_grid}^3 = "
      f"{n_grid**3/1e6:.1f}M 칸, dt {dt:.3e}, FLIP {flip}", flush=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gf_particle_sort import sort_by_cell
if a.resort_every:
    nm = sort_by_cell(solver, n_grid, grid_lim)
    print(f"[재정렬] 배열 {nm} 개를 옮겼다", flush=True)
for i in range(a.warmup):
    solver.p2g2p(i, dt, device=dev, flip_pic_ratio=flip)
wp.synchronize()
solver.time_profile = {}
t0 = time.time()
for i in range(a.substeps):
    if a.resort_every and i % a.resort_every == 0:
        sort_by_cell(solver, n_grid, grid_lim)
    solver.p2g2p(i, dt, device=dev, flip_pic_ratio=flip)
wp.synchronize()
el = time.time() - t0

tot = sum(sum(v) for v in solver.time_profile.values())
print(f"\n[합계] 서브스텝 {a.substeps} 개 {el*1000:.0f} ms "
      f"({el/a.substeps*1000:.3f} ms/서브스텝)", flush=True)
rows = sorted(((k, sum(v)) for k, v in solver.time_profile.items()),
              key=lambda r: -r[1])
for k, v in rows:
    print(f"  {k:28s} {v:8.1f} ms  {100*v/max(tot,1e-9):5.1f}%  "
          f"{v/a.substeps:.3f} ms/서브스텝")
print(f"  {'(타이머 밖)':28s} {el*1000-tot:8.1f} ms", flush=True)
if a.out:
    json.dump(dict(substeps=a.substeps, total_ms=el * 1000,
                   stages={k: v for k, v in rows}), open(a.out, "w"), indent=1)
    print(f"[저장] {a.out}")
print("PROFILE_DONE", flush=True)

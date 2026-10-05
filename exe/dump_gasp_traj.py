"""GASP(taichi_elements) 궤적을 h5 로 떨군다 -- 물리 잔차용.

잔차·증분 포텐셜을 재려면 x, v 말고 **그 솔버가 실제로 들고 있는 F** 가 필요하다.
`particle_info()` 는 F 를 안 주므로 커널로 직접 복사한다. snow 는 경화가 Jp 에
들어 있어 함께 떨군다.

그쪽 고정값은 그대로 둔다: E=1e6*size=2e6, nu=0.2, rho=1000, p_vol=dx^3
(**입자마다 균일** -- PG 처럼 셀 개수로 나누지 않는다).

  python exe/dump_gasp_traj.py --shape mic --material elastic --run run5
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import h5py
import numpy as np
import torch
from tqdm import tqdm

W = "/home/dkta/work"
sys.path.insert(0, f"{W}/taichi_elements")

ap = argparse.ArgumentParser()
ap.add_argument("--shape", required=True)
ap.add_argument("--material", default="elastic",
                choices=["elastic", "elastoplastic", "viscoplastic"])
ap.add_argument("--run", default="run5")
ap.add_argument("--frames", type=int, default=10)
ap.add_argument("--skip", type=int, default=24)
ap.add_argument("--floor", type=float, default=0.1)
ap.add_argument("--n_grid", type=int, default=100)
ap.add_argument("--s", type=int, default=0, help="0 이면 json 의 s_conv")
ap.add_argument("--out", default="")
a = ap.parse_args()

SJ = f"{W}/bench/{a.run}/gasp_{a.shape}_{a.material}.json"
d = json.load(open(SJ)) if os.path.exists(SJ) else {}
s = a.s or d.get("s_conv")
if not s:
    raise SystemExit(f"[건너뜀] 서브스텝을 모른다 ({SJ})")
s = int(s)

VP = f"{W}/gamesout/{a.shape}/pseudomesh_info/ours_30000/vertices.pt"
V = torch.load(VP, map_location="cpu").cpu().numpy().reshape(-1, 3)
X0 = np.load(f"{W}/anfill_{a.shape}.npy")
lo_t, hi_t = X0.min(0), X0.max(0)
lo_v, hi_v = V.min(0), V.max(0)
sc = float((hi_t - lo_t).max() / (hi_v - lo_v).max())
P0 = ((V - (lo_v + hi_v) / 2.0) * sc + (lo_t + hi_t) / 2.0).astype(np.float32)

OD = a.out or f"{W}/gaspsim/{a.shape}_{a.material}_s{s}/simulation_ply"
os.makedirs(OD, exist_ok=True)

import taichi as ti                                   # noqa: E402
from engine.mpm_solver import MPMSolver               # noqa: E402

ti.init(arch=ti.gpu, log_level=ti.ERROR,
        device_memory_fraction=float(os.environ.get("AF_TI_FRAC", 0.8)))
MATID = {"elastic": MPMSolver.material_elastic,
         "elastoplastic": MPMSolver.material_snow,
         "viscoplastic": MPMSolver.material_sand}

dx = 2.0 / a.n_grid
dt_scale = (1.0 / 60.0) / (2e-2 * dx / 2.0) / float(s)
mpm = MPMSolver(res=(a.n_grid,) * 3, size=2, dt_scale=dt_scale, E_scale=1.0,
                unbounded=False, support_plasticity=True)
mpm.set_gravity((0.0, 0.0, -9.8))
mpm.add_surface_collider(point=(0.0, 0.0, a.floor), normal=(0.0, 0.0, 1.0),
                         surface=MPMSolver.surface_sticky)
mpm.add_particles(particles=P0, material=MATID[a.material])
N = int(mpm.n_particles[None])

_F = ti.Vector.field(9, dtype=ti.f32, shape=N)
_Jp = ti.field(dtype=ti.f32, shape=N)


@ti.kernel
def grab():                 # 인자 없음: future annotations 가 주석을 문자열로 만든다
    for p in range(N):
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                _F[p][3 * i + j] = mpm.F[p][i, j]
        _Jp[p] = mpm.Jp[p]


print(f"[설정] gasp {a.shape} {a.material}  s={s}  입자 {N}  -> {OD}",
      flush=True)
nrun = a.frames + a.skip
for f in tqdm(range(nrun + 1), desc="프레임"):
    if f >= a.skip:
        pi = mpm.particle_info()
        grab()
        with h5py.File(f"{OD}/{f - a.skip:04d}.h5", "w") as hf:
            hf["x"] = pi["position"][:N].astype(np.float32)
            hf["v"] = pi["velocity"][:N].astype(np.float32)
            hf["f_tensor"] = _F.to_numpy()[:N].astype(np.float32)
            hf["jp"] = _Jp.to_numpy()[:N].astype(np.float32)
    if f == nrun:
        break
    mpm.step(1.0 / 60.0)

# 잔차 쪽이 그대로 읽을 cfg (그쪽 고정값)
MATNAME = {"elastic": "jelly", "elastoplastic": "ti_snow",
           "viscoplastic": "ti_sand"}
cfg = dict(material=MATNAME[a.material], E=2e6, nu=0.2, density=1000.0,
           n_grid=a.n_grid, grid_lim=2.0, frame_dt=1.0 / 60.0,
           g=[0.0, 0.0, -9.8], vol_mode="uniform",
           boundary_conditions=[dict(type="surface_collider",
                                     point=[0.0, 0.0, a.floor],
                                     normal=[0.0, 0.0, 1.0],
                                     surface="sticky", start_time=0.0,
                                     end_time=1e9)])
cp = os.path.join(os.path.dirname(OD), "cfg.json")
json.dump(cfg, open(cp, "w"), indent=1)
print(f"[저장] {OD} (프레임 {a.frames + 1}), cfg {cp}", flush=True)

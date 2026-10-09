"""학습 비교(3DGSim · NGFF · 우리)용 공통 궤적 하나를 PG 공식 MPM(jelly)으로 만든다.

입력: lego 0 프레임 입자(i-PG visco_drop, PG 공식 채우기) + 탄성 config (stab/lego 와 같은 값).
무작위 (시드 하나로 전부): 초기 회전(SO(3) 균일), 낙하 높이(바닥 위 여유 [h0, h1]), 초기 속도(크기 [v0, v1], 방향 구 균일, 강체 병진).
출력 DIR/sim_XXXXXXXXXX.h5 프레임마다 x (N,3) float32, F (N,9) float32 -- F 는 초기 회전 R 을 실은 F_sim·R
  (렌더러가 가우시안 공분산을 F C0 Fᵀ 로 쓰므로 회전이 모양에 반영된다), DIR/meta.json 에 무작위 값.

  cd i-physgaussian && python <anchorflow>/exe/lb_gen.py --h5 init.h5 --config cfg.json --seed 0 --out DIR
"""
import argparse
import json
import math
import os
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--h5", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--frames", type=int, default=80)
ap.add_argument("--h0", type=float, default=0.05, help="바닥 위 여유 최소 (시뮬 단위)")
ap.add_argument("--h1", type=float, default=0.30)
ap.add_argument("--v0", type=float, default=0.0, help="초기 속도 크기 최소")
ap.add_argument("--v1", type=float, default=1.5)
ap.add_argument("--pg", default="/home/dkta/work/i-physgaussian")
a = ap.parse_args()
sys.path.insert(0, a.pg); os.chdir(a.pg)
import h5py                                                      # noqa: E402
import numpy as np                                               # noqa: E402
import torch                                                     # noqa: E402
import warp as wp                                                # noqa: E402
from tqdm import tqdm                                            # noqa: E402
from mpm_solver_warp.mpm_solver_warp import MPM_Simulator_WARP   # noqa: E402

cfg = json.load(open(a.config))
n_grid, GL = int(cfg["n_grid"]), float(cfg["grid_lim"]); dx = GL / n_grid
frame_dt = float(cfg["frame_dt"]); nsub = int(round(frame_dt / float(cfg["substep_dt"]))); dt = frame_dt / nsub
G = [float(q) for q in cfg["g"]]
floor = [b for b in cfg["boundary_conditions"] if b["type"] == "surface_collider"][0]
ZF = float(floor["point"][2])
with h5py.File(a.h5) as h:
    x = np.array(h["x"]); x = (x.T if x.shape[0] == 3 else x).astype(np.float64)
N = x.shape[0]

rng = np.random.default_rng(a.seed)
q = rng.normal(size=4); q /= np.linalg.norm(q)                     # 균일 회전 (정규분포 사원수)
w_, i_, j_, k_ = q
R = np.array([[1 - 2 * (j_ * j_ + k_ * k_), 2 * (i_ * j_ - k_ * w_), 2 * (i_ * k_ + j_ * w_)],
              [2 * (i_ * j_ + k_ * w_), 1 - 2 * (i_ * i_ + k_ * k_), 2 * (j_ * k_ - i_ * w_)],
              [2 * (i_ * k_ - j_ * w_), 2 * (j_ * k_ + i_ * w_), 1 - 2 * (i_ * i_ + j_ * j_)]])
c = x.mean(0)
xr = (x - c) @ R.T
lift = float(rng.uniform(a.h0, a.h1))
xr += np.array([GL / 2, GL / 2, ZF + lift - xr[:, 2].min()])        # xy 가운데, 바닥 위 lift
d = rng.normal(size=3); d /= np.linalg.norm(d)
spd = float(rng.uniform(a.v0, a.v1))
v0 = d * spd
lo, hi = xr.min(0), xr.max(0)
margin = 3 * dx
print(f"[무작위] 시드 {a.seed}  높이 여유 {lift:.3f}  속도 {spd:.3f} 방향 {d.round(3)}  상자 {lo.round(3)} ~ {hi.round(3)}", flush=True)
if (lo < margin).any() or (hi > GL - margin).any():
    raise SystemExit(f"[거부] 회전 뒤 상자 밖 (여유 {margin:.3f})")
# 최고점 검산: 위로 쏘면 v_z²/2g 만큼 더 오른다
rise = max(0.0, v0[2]) ** 2 / (2 * abs(G[2]))
if hi[2] + rise > GL - margin:
    raise SystemExit(f"[거부] 최고점 {hi[2] + rise:.3f} 가 상자 위 {GL - margin:.3f} 를 넘는다")

X = torch.as_tensor(xr, dtype=torch.float32).cuda().contiguous()
cell = (X / dx).floor().long(); key = (cell[:, 0] * n_grid + cell[:, 1]) * n_grid + cell[:, 2]
_, inv, cnt = torch.unique(key, return_inverse=True, return_counts=True)
VOL = (dx ** 3 / cnt[inv].float()).contiguous()
wp.init()
sol = MPM_Simulator_WARP(10)
sol.load_initial_data_from_torch(X, VOL, None, n_grid=n_grid, grid_lim=GL)
sol.set_parameters_dict({"material": "jelly", "E": float(cfg["E"]), "nu": float(cfg["nu"]), "density": float(cfg["density"]),
                         "g": G, "n_grid": n_grid, "grid_lim": GL, "grid_v_damping_scale": 1.1})
sol.add_bounding_box()
sol.add_surface_collider(tuple(floor["point"]), tuple(floor["normal"]), floor["surface"], float(floor.get("friction", 0.0)))
sol.finalize_mu_lam()
sol.import_particle_v_from_torch(torch.as_tensor(np.tile(v0, (N, 1)), dtype=torch.float32).cuda().contiguous())
os.makedirs(a.out, exist_ok=True)
Rt = torch.as_tensor(R, dtype=torch.float32).cuda()


def dump(f, xt, Ft):
    with h5py.File(f"{a.out}/sim_{f:010d}.h5", "w") as h:
        h.create_dataset("x", data=xt.astype(np.float32))
        h.create_dataset("F", data=Ft.reshape(N, 9).astype(np.float32))


dump(0, xr, np.tile(R, (N, 1, 1)))
print(f"[pg] 입자 {N}, dx {dx:.4f}, 서브스텝 {dt:.3e} × {nsub}, {a.frames} 프레임", flush=True)
t0 = time.time(); step = 0
for f in tqdm(range(1, a.frames + 1), desc=f"lb {a.seed}"):
    for s in range(nsub):
        sol.p2g2p(step, dt); step += 1
    xt = sol.export_particle_x_to_torch()
    Ft = sol.export_particle_F_to_torch().reshape(N, 3, 3) @ Rt           # PG 는 (N, 9) 로 내보낸다
    if not torch.isfinite(xt).all():
        raise SystemExit(f"[발산] 프레임 {f}")
    dump(f, xt.cpu().numpy(), Ft.cpu().numpy())
json.dump(dict(seed=a.seed, R=R.tolist(), lift=lift, speed=spd, dir=d.tolist(), v0=v0.tolist(), frames=a.frames,
               frame_dt=frame_dt, nsub=nsub, N=N, config=os.path.abspath(a.config), h5=os.path.abspath(a.h5),
               range=dict(h=[a.h0, a.h1], v=[a.v0, a.v1]), sec=time.time() - t0), open(f"{a.out}/meta.json", "w"), indent=1)
print(f"[저장] {a.out}  {time.time() - t0:.0f}s", flush=True)

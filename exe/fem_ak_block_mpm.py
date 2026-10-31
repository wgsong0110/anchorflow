"""절 AK.24: 같은 블록 시험을 정답 MPM (PG 공식 warp 솔버, lego 설정: jelly, n_grid 100, 서브스텝 833/프레임, APIC) 으로. 초기 F = A 를 particle_F·particle_F_trial 둘 다에.
출력 <out>/x.pt [T, N, 3] (프레임마다). 사용: (i-physgaussian 에서) python <anchorflow>/exe/fem_ak_block_mpm.py <block.pt> <out> [프레임 80]"""
import os
import sys

import torch
import warp as wp

sys.path.insert(0, os.getcwd())
from mpm_solver_warp.mpm_solver_warp import MPM_Simulator_WARP   # noqa: E402

bp, out = sys.argv[1], sys.argv[2]
T = int(sys.argv[3]) if len(sys.argv) > 3 else 80
os.makedirs(out, exist_ok=True)
d = torch.load(bp, weights_only=False)
cfg = d["cfg"]
x0 = d["x"][0].cuda().contiguous()
N = x0.shape[0]
dx0 = float(cfg["grid_lim"]) / int(cfg["n_grid"])
cell = (x0 / dx0).floor().long()
key = (cell[:, 0] * 100 + cell[:, 1]) * 100 + cell[:, 2]
_, inv, cnt = torch.unique(key, return_inverse=True, return_counts=True)
VOL = (dx0 ** 3 / cnt[inv].float()).contiguous()
wp.init()
sol = MPM_Simulator_WARP(10)
sol.load_initial_data_from_torch(x0, VOL, None, n_grid=int(cfg["n_grid"]), grid_lim=float(cfg["grid_lim"]))
sol.set_parameters_dict({"material": "jelly", "E": float(cfg["E"]), "nu": float(cfg["nu"]), "density": float(cfg["density"]), "g": [0.0, 0.0, 0.0],
                         "n_grid": int(cfg["n_grid"]), "grid_lim": float(cfg["grid_lim"]), "grid_v_damping_scale": 1.1, "rpic_damping": 0.0})
sol.finalize_mu_lam()
Fm = d["F"][0].cuda().reshape(-1, 3, 3).contiguous()
sol.import_particle_F_from_torch(Fm)
sol.mpm_state.particle_F_trial = wp.clone(sol.mpm_state.particle_F)
nsub = int(round(float(cfg["frame_dt"]) / float(cfg["substep_dt"])))
dt = float(cfg["frame_dt"]) / nsub
XD = torch.empty(T, N, 3)
step = 0
for f in range(T):
    for s in range(nsub):
        sol.p2g2p(step, dt)
        step += 1
    XD[f] = sol.export_particle_x_to_torch().cpu()
    print(f"프레임 {f + 1}/{T}", flush=True)
torch.save(XD, os.path.join(out, "x.pt"))
print("[끝]", flush=True)

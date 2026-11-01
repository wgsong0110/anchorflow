"""절 AK.29: 떨어지는 블록을 정답 MPM 과 같은 조건으로 (gt_sens.py 기본 경로 = lb_gen.py: PG warp 솔버, jelly, APIC rpic 0, n_grid 100, 감쇠 1.1, bounding_box + sticky surface_collider,
입자 부피 dx_g³ / 칸 입자 수). 서브스텝만 1/60 s 를 832 등분 (정답 833 대비 0.12% 큼) 해 1/240 s 마다 (208 서브스텝) 기록 -- 프레임 1/60·1/120·1/240 판 정답을 한 번에.
출력 <out>/x240.pt [321, N, 3] (0 = 처음 상태). 사용: (i-physgaussian 에서) python <anchorflow>/exe/fem_ak_drop_mpm.py <init.pt> <out> [1/60 프레임 수 80]"""
import os
import sys

import torch
import warp as wp

sys.path.insert(0, os.getcwd())
from mpm_solver_warp.mpm_solver_warp import MPM_Simulator_WARP   # noqa: E402

ip, out = sys.argv[1], sys.argv[2]
T = int(sys.argv[3]) if len(sys.argv) > 3 else 80
os.makedirs(out, exist_ok=True)
d = torch.load(ip, weights_only=False)
cfg = d["cfg"]
x0 = d["x"][0].cuda().contiguous()
N = x0.shape[0]
ng = int(cfg["n_grid"]); GL = float(cfg["grid_lim"]); dx0 = GL / ng
cell = (x0 / dx0).floor().long()
_, inv, cnt = torch.unique((cell[:, 0] * ng + cell[:, 1]) * ng + cell[:, 2], return_inverse=True, return_counts=True)
VOL = (dx0 ** 3 / cnt[inv].float()).contiguous()
fl = [b for b in cfg["boundary_conditions"] if b["type"] == "surface_collider"][0]
wp.init()
sol = MPM_Simulator_WARP(10)
sol.load_initial_data_from_torch(x0, VOL, None, n_grid=ng, grid_lim=GL)
sol.set_parameters_dict({"material": "jelly", "E": float(cfg["E"]), "nu": float(cfg["nu"]), "density": float(cfg["density"]), "g": [float(q) for q in cfg["g"]],
                         "n_grid": ng, "grid_lim": GL, "grid_v_damping_scale": 1.1, "rpic_damping": 0.0})
sol.add_bounding_box()
sol.add_surface_collider(tuple(fl["point"]), tuple(fl["normal"]), fl["surface"], float(fl.get("friction", 0.0)))
sol.finalize_mu_lam()
sol.import_particle_v_from_torch(d["v"][0].cuda().contiguous())
dt = float(cfg["frame_dt"]) / 832
XD = torch.empty(4 * T + 1, N, 3)
XD[0] = x0.cpu()
step = 0
for f in range(1, 4 * T + 1):
    for s in range(208):
        sol.p2g2p(step, dt)
        step += 1
    XD[f] = sol.export_particle_x_to_torch().cpu()
    if f % 40 == 0:
        print(f"1/240 프레임 {f}/{4 * T} 최저 {float(XD[f][:, 2].min()):.4f} 중심 {float(XD[f][:, 2].mean()):.4f}", flush=True)
torch.save(XD, os.path.join(out, "x240.pt"))
print("[끝]", flush=True)

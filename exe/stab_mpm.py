"""서브스텝 안정성 실험: PG(공식 명시 MPM) / i-PG(암시 MPM, newton_gmres) 를 프레임당 N 서브스텝으로.

입력: 초기 상태 h5 (x, v) + config (탄성 jelly). 출력 npz: 프레임별 측정(stabstat) + 고정 부분표본 위치 + 시간.

  cd i-physgaussian && python <anchorflow>/exe/stab_mpm.py --method pg --h5 init.h5 --config cfg.json --substeps 64 --out r.npz
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lib"))
ap = argparse.ArgumentParser()
ap.add_argument("--method", choices=["pg", "ipg"], required=True)
ap.add_argument("--h5", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--substeps", type=int, required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--pg", default="/home/dkta/work/i-physgaussian")
a = ap.parse_args()
sys.path.insert(0, a.pg); os.chdir(a.pg)
import h5py                                                      # noqa: E402
import numpy as np                                               # noqa: E402
import torch                                                     # noqa: E402
import warp as wp                                                # noqa: E402
from tqdm import tqdm                                            # noqa: E402
from anchorflow import stabstat as ss                            # noqa: E402
from mpm_solver_warp.mpm_solver_warp import MPM_Simulator_WARP   # noqa: E402

cfg = json.load(open(a.config))
n_grid, GL = int(cfg["n_grid"]), float(cfg["grid_lim"]); dx = GL / n_grid
frame_dt, NF = float(cfg["frame_dt"]), int(cfg["frame_num"])
dt = frame_dt / a.substeps
G = [float(q) for q in cfg["g"]]
floor = [b for b in cfg["boundary_conditions"] if b["type"] == "surface_collider"][0]
with h5py.File(a.h5) as h:
    x = np.array(h["x"]); x = np.ascontiguousarray((x.T if x.shape[0] == 3 else x).astype(np.float32))
    v = np.array(h["v"]) if "v" in h else np.zeros_like(x.T)
    v = np.ascontiguousarray((v.T if v.shape[0] == 3 else v).astype(np.float32))
N = x.shape[0]
X = torch.as_tensor(x).cuda()
cell = (X / dx).floor().long(); key = (cell[:, 0] * n_grid + cell[:, 1]) * n_grid + cell[:, 2]
_, inv, cnt = torch.unique(key, return_inverse=True, return_counts=True)
VOL = (dx ** 3 / cnt[inv].float()).contiguous()
MASS = VOL.double().cpu().numpy() * float(cfg["density"])
SUB = ss.subset(N)
wp.init()
if a.method == "ipg":
    from implicit_mpm_solver import ImplicitMPMSolver
    sol = ImplicitMPMSolver(n_particles=10, n_grid=n_grid, grid_lim=GL)
else:
    sol = MPM_Simulator_WARP(10)
sol.load_initial_data_from_torch(X, VOL, None, n_grid=n_grid, grid_lim=GL)
sol.set_parameters_dict({"material": "jelly", "E": float(cfg["E"]), "nu": float(cfg["nu"]), "density": float(cfg["density"]),
                         "g": G, "n_grid": n_grid, "grid_lim": GL, "grid_v_damping_scale": 1.1})
sol.add_bounding_box()
sol.add_surface_collider(tuple(floor["point"]), tuple(floor["normal"]), floor["surface"], float(floor.get("friction", 0.0)))
sol.finalize_mu_lam()
sol.import_particle_v_from_torch(torch.as_tensor(v).cuda())
print(f"[{a.method}] 입자 {N}, dx {dx:.4f}, N {a.substeps} (dt {dt:.3e}), {NF} 프레임", flush=True)
rows = [ss.frame_stats(x, v, np.tile(np.eye(3), (N, 1, 1)), MASS, G, float(floor["point"][2]))]
XS = [x[SUB]]; T = []
step = 0
for f in tqdm(range(1, NF + 1), desc=f"{a.method} N{a.substeps}"):
    t0 = time.time()
    for s in range(a.substeps):
        if a.method == "ipg":
            sol.p2g2p_newton_gmres(step, dt)
        else:
            sol.p2g2p(step, dt)
        step += 1
    torch.cuda.synchronize(); T.append(time.time() - t0)
    xt = sol.export_particle_x_to_torch().cpu().numpy()
    vt = wp.to_torch(sol.mpm_state.particle_v).cpu().numpy()
    Ft = sol.export_particle_F_to_torch().cpu().numpy()
    rows.append(ss.frame_stats(xt, vt, Ft, MASS, G, float(floor["point"][2])))
    XS.append(xt[SUB])
    if rows[-1]["nan"] > 0.5:                                    # 대부분 발산했으면 더 돌 이유가 없다
        print(f"  발산 프레임 {f}", flush=True); break
np.savez_compressed(a.out, sub=SUB, x=np.stack(XS).astype(np.float32), t=np.array(T),
                    stats=np.array([[r.get(k, np.nan) for k in ("ke", "pe", "nan", "out", "vmax", "detneg", "detmin")] for r in rows]),
                    keys=np.array(["ke", "pe", "nan", "out", "vmax", "detneg", "detmin"]), substeps=a.substeps, dt=dt)
print(f"[저장] {a.out}  프레임당 {np.mean(T):.3f}s", flush=True)

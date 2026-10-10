"""치마(Actor01 옷)를 GaussianFluent 솔버(gf_mpm 이식본)로: mpma_pg 의 PG 입력을 그대로 써서 같은 조건, 같은 렌더러.

  --stage prep  pg_in.pt -> GF 입력 h5 (x, v, pin) + config (jelly, 같은 E·밀도·ν, 격자, 중력 -y, y=0.1 sticky 바닥, 25 fps)
  --stage conv  GF 출력 h5 (x, F) -> pg_traj.pt (x, cov = F C0 Fᵀ) -- mpma_pg --stage render 가 그대로 읽는다

GF 에는 천 모델도 입자 고정 기능도 없어 PG 비교와 같게: 등방 jelly, 고정 입자(몸·몸에 붙은 옷 면)는 매 서브스텝 처음 위치·속도 0 (gf_mpm 의 pin).
"""
import argparse
import glob
import json
import os

import h5py
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--stage", choices=["prep", "conv"], required=True)
ap.add_argument("--pg_in", required=True, help="mpma_pg --stage dump 결과 pg_in.pt")
ap.add_argument("--work", required=True)
ap.add_argument("--n_grid", type=int, default=250)
ap.add_argument("--frames", type=int, default=50)
ap.add_argument("--substeps", type=int, default=400)
a = ap.parse_args()
os.makedirs(a.work, exist_ok=True)
I = torch.load(a.pg_in)

if a.stage == "prep":
    x = I["x"].float().numpy(); v = I["v"].float().numpy(); pin = I["pin"].numpy().astype(np.int32)
    with h5py.File(f"{a.work}/init.h5", "w") as h:
        h.create_dataset("x", data=x); h.create_dataset("v", data=v); h.create_dataset("pin", data=pin)
    cfg = dict(material="jelly", E=float(I["E"]), nu=0.3, density=float(I["D"]), n_grid=a.n_grid, grid_lim=2.0,
               g=[0.0, -9.8, 0.0], frame_dt=1.0 / 25, frame_num=a.frames, substep_dt=(1.0 / 25) / a.substeps,
               grid_v_damping_scale=1.1, init_velocity=[0.0, 0.0, 0.0],
               boundary_conditions=[dict(type="surface_collider", point=[0.0, 0.1, 0.0], normal=[0.0, 1.0, 0.0],
                                         surface="sticky", friction=0.0, start_time=0.0, end_time=1000.0)])
    json.dump(cfg, open(f"{a.work}/config.json", "w"), indent=1)
    print(f"[prep] 입자 {len(x)} (고정 {int(pin.sum())}), E {cfg['E']:.4g} 밀도 {cfg['density']:.4g} 격자 {a.n_grid}", flush=True)
else:
    c6 = I["cov"].float()
    C0 = torch.zeros(c6.shape[0], 3, 3)
    C0[:, 0, 0], C0[:, 0, 1], C0[:, 0, 2] = c6[:, 0], c6[:, 1], c6[:, 2]
    C0[:, 1, 1], C0[:, 1, 2], C0[:, 2, 2] = c6[:, 3], c6[:, 4], c6[:, 5]
    C0[:, 1, 0], C0[:, 2, 0], C0[:, 2, 1] = c6[:, 1], c6[:, 2], c6[:, 4]
    XS, CS = [], []
    for fp in sorted(glob.glob(f"{a.work}/out/sim_*.h5")):
        with h5py.File(fp) as h:
            xx = np.array(h["x"]); xx = xx.T if xx.shape[0] == 3 else xx
            F = torch.as_tensor(np.array(h["F"])).float().reshape(-1, 3, 3)
        C = F @ C0 @ F.transpose(1, 2)
        XS.append(torch.as_tensor(xx).float())
        CS.append(torch.stack([C[:, 0, 0], C[:, 0, 1], C[:, 0, 2], C[:, 1, 1], C[:, 1, 2], C[:, 2, 2]], 1))
    torch.save(dict(x=torch.stack(XS), cov=torch.stack(CS)), f"{a.work}/pg_traj.pt")
    print(f"[conv] {len(XS)} 프레임 -> {a.work}/pg_traj.pt", flush=True)

"""공 궤적(.pt)에서 PG/i-PG 입력을 만든다 -- 입자 npy + 시나리오 npz + config.

채우기(`particle_filling`)는 **넣지 않는다**. 입자를 우리가 직접 주기 때문이다
(patch_ipg_particles.py 가 AF_PARTICLES_NPY 로 갈아끼운다).
"""
from __future__ import annotations
import argparse
import json
import os
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--traj", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--substep", type=int, default=0,
                help="0 이면 config 에 넣지 않는다 (PG 기본값을 쓴다)")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
try:
    d = torch.load(a.traj, map_location="cpu", weights_only=False)
except TypeError:
    d = torch.load(a.traj, map_location="cpu")

P = d["x"][0].float().numpy().astype(np.float32)
pp = os.path.join(a.out, "particles.npy")
np.save(pp, P)
print(f"입자 -> {pp}  {P.shape[0]} 개, "
      f"범위 {P.min(0).round(3).tolist()} ~ {P.max(0).round(3).tolist()}")

if "ctrl_id" in d:
    hid = d["ctrl_id"].reshape(-1).numpy().astype(np.int64)
    vel = d["ctrl_vel"].numpy().astype(np.float32)
    sp = os.path.join(a.out, "scen.npz")
    np.savez(sp, hid=hid, vel=vel)
    print(f"시나리오 -> {sp}  손잡이 {hid.tolist()}  vel {vel.shape}  "
          f"R {float(d['ctrl_R'].reshape(-1)[0])}")

c = {k: v for k, v in dict(d["cfg"]).items() if v is not None}
c.pop("particle_filling", None)              # 입자를 직접 준다
c["grid_lim"] = float(c.get("grid_lim", 2.0))
if a.substep:
    c["substep_dt"] = float(c["frame_dt"]) / a.substep
cp = os.path.join(a.out, "cfg.json")
json.dump(c, open(cp, "w"), indent=2)
print(f"config -> {cp}  n_grid {c['n_grid']} material {c.get('material')} "
      f"g {c['g']} E {c['E']}")

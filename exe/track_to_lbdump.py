"""rep_track2 결과(--save_traj --save_F, topo_flow --particles 목표) -> lb_roll_to_h5 가 읽는 덤프 (pred, F; 시뮬 좌표).

정규화 좌표 y 를 topo_flow 와 같은 식으로 되돌린다: x = (y - off)·s + lo, off = (1 - (X0.max - lo)/s)/2 (X0 = 원래 궤적 0 프레임).

  python exe/track_to_lbdump.py --res res.npz --flow flow.npz --traj lbdata/test/0000 --out dump.pt
"""
import argparse

import h5py
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--res", required=True)
ap.add_argument("--flow", required=True)
ap.add_argument("--traj", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
R = np.load(a.res); D = np.load(a.flow)
lo, s = D["lo"].astype(np.float64), float(D["s"])
with h5py.File(f"{a.traj}/sim_0000000000.h5") as h:
    X0 = np.array(h["x"]); X0 = X0.T if X0.shape[0] == 3 else X0
off = (1.0 - ((X0.max(0) - lo) / s)) / 2.0
Y = R["traj"].astype(np.float64)                      # [T, N, 3]
P = (Y - off) * s + lo
assert P.shape[1] == X0.shape[0], f"점 수가 다르다 {P.shape} vs {X0.shape}"
F = R["F"].astype(np.float32)
torch.save(dict(pred=torch.as_tensor(P, dtype=torch.float32), F=torch.as_tensor(F)), a.out)
print(f"[덤프] {a.out}  {P.shape[0]} 프레임, 0 프레임 대비 처음 예측 이동 최대 {np.abs(P[0] - X0).max():.3e}", flush=True)

"""rep_track2 --method fem 의 격자·보간이 명시적 FEM (fem_rt.Grid, fem_ex_sim 과 같은 격자) 과 같은지 확인.

정규화 좌표 P 의 가우시안을 시뮬 좌표로 옮겨 FEM 격자 (node_h = MPM dx·(2√2)^(1/3), grid_for_nodes h_fix) 를 깔고,
rep_track2 fem 과 같은 식 (시뮬 원점 기준 격자) 으로 정규화 좌표에서 깐 격자와 비교한다:
입자마다 든 사면체의 4 꼭짓점 위치 (시뮬 좌표) 와 barycentric 좌표가 같아야 한다.

  python exe/topo_fem_equiv.py --aux topo/aux_bread_v2.npz --mpm_dx 0.0133333
"""
import argparse
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from anchorflow import simplex as sx            # noqa: E402
from anchorflow import fem_rt as RTC            # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--aux", required=True)
ap.add_argument("--mpm_dx", type=float, required=True)
a = ap.parse_args()
dev = "cuda"
AX = np.load(a.aux, allow_pickle=True)
G = torch.as_tensor(AX["G"], dtype=torch.float64, device=dev)
S, LO, OFF = float(AX["s"]), np.asarray(AX["lo"], np.float64), np.asarray(AX["off"], np.float64)
node_h = a.mpm_dx * (2.0 * math.sqrt(2.0)) ** (1.0 / 3.0)
lo_t = torch.as_tensor(LO, device=dev); off_t = torch.as_tensor(OFF, device=dev)
Psim = (G - off_t) * S + lo_t
# 명시적 FEM 격자 (시뮬 좌표)
Gf = RTC.Grid(Psim, node_h)
Xf = Gf.npos[Gf.rows]                                                   # [N,4,3] 시뮬
# rep_track2 fem (정규화 좌표, 시뮬 원점 기준)
h = node_h / S
c0 = off_t - lo_t / S
lo, lat, nn = sx.grid_for_nodes(G - c0, 1.0, h_fix=h)
lo = lo + c0
idx, lam, _ = sx.locate(G, lo, lat, nn)
rows, uniq = sx.active_nodes(idx)
Xn = sx.node_pos(lo, lat, nn, uniq)
Xr = (Xn[rows] - off_t) * S + lo_t                                      # 시뮬로
dpos = float((Xr - Xf).abs().max())
dlam = float((lam - Gf.lam).abs().max())
print(f"[FEM 격자 대조] 가우시안 {G.shape[0]}  node_h {node_h:.6f} (시뮬)  활성 노드 rep {uniq.numel()} / FEM {Gf.M}  "
      f"사면체 꼭짓점 위치 최대 차 {dpos:.3e} (시뮬, node_h 의 {dpos / node_h:.1e})  barycentric 최대 차 {dlam:.3e}", flush=True)

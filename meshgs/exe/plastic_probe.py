"""고정 위상 사면체 메시가 소성 변형에서 어디서 무너지는지 진단한다.

발산이 (a) 요소 뒤집힘 때문인지 (b) 명시적 적분의 CFL 때문인지 가른다.
(a) 면 메시 표현의 한계이고 -- MPM 은 격자를 매 스텝 다시 쓰므로 이 문제가 없다 --
(b) 면 적분기를 고치면 되는 문제라 주장 근거가 못 된다.

그래서 스텝마다 뒤집힌 요소 비율, det F 최소, 최대 속도를 같이 찍는다.
"""
from __future__ import annotations

import argparse
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, "..", "lib"))

import numpy as np
import torch

from meshgs.fem import TetFEM
from meshgs.tetcage import build_tet_mesh, dilate_fill, occupancy

ap = argparse.ArgumentParser()
ap.add_argument("--res", type=int, default=32)
ap.add_argument("--E", type=float, default=1e5)
ap.add_argument("--nu", type=float, default=0.3)
ap.add_argument("--density", type=float, default=200.0)
ap.add_argument("--compress", type=float, default=0.20, help="높이 대비 압축량")
ap.add_argument("--dts", type=float, nargs="+", default=[2e-4, 5e-5, 1e-5])
ap.add_argument("--yields", type=float, nargs="+",
                default=[1e9, 1e4, 3e3, 1e3, 3e2, 1e2])
ap.add_argument("--damping", type=float, default=8.0)
a = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
g = torch.stack(torch.meshgrid(*[torch.linspace(-0.4, 0.4, 16, device=dev)] * 3,
                               indexing="ij"), -1).reshape(-1, 3)
occ, org, h = occupancy(g, res=a.res)
occ = dilate_fill(occ, 1)
V, T = build_tet_mesh(occ, org, h)
H = float(V[:, 2].max() - V[:, 2].min())
top = V[:, 2] > V[:, 2].max() - 1.5 * h
bot = V[:, 2] < V[:, 2].min() + 1.5 * h
c_wave = (a.E / a.density) ** 0.5
print(f"[설정] 정점 {V.shape[0]} 사면체 {T.shape[0]} h {h:.4f} 높이 {H:.3f}")
print(f"[설정] 파속 {c_wave:.1f} m/s -> 명시적 CFL 한계 dt < {h/c_wave:.2e}")
print(f"{'dt':>9} {'yield':>9} {'결과':>10} {'잔류%':>8} {'뒤집힘%':>9} "
      f"{'detF최소':>9} {'실패스텝':>8} {'첫뒤집힘':>9} {'직전뒤집힘%':>12} {'직전detF':>9}")

for dt in a.dts:
    PRESS = int(round(0.12 / dt))
    HOLD = PRESS // 3
    REL = PRESS * 3
    v_press = a.compress * H / (PRESS * dt)
    for ys in a.yields:
        kind = "none" if ys >= 1e8 else "von_mises"
        f = TetFEM(V, T, density=a.density, E=a.E, nu=a.nu, plastic=kind,
                   yield_stress=ys, damping=a.damping)
        Vc, vc = V.clone(), torch.zeros_like(V)
        fail, vmax = -1, 0.0
        first_inv, inv_prev, dmin_prev = -1, 0.0, 1.0
        for s in range(PRESS + HOLD + REL):
            hold = s < PRESS + HOLD
            Vc, vc = f.step(Vc, vc, dt, fixed=(bot | top) if hold else bot)
            if s < PRESS:
                Vc[top, 2] -= v_press * dt
            vmax = max(vmax, float(vc.norm(dim=-1).max())
                       if torch.isfinite(vc).all() else float("inf"))
            if not torch.isfinite(Vc).all():
                fail = s
                break
            iv, dmin, _ = f.quality(Vc)       # 매 스텝 추적
            if first_inv < 0 and iv > 0:
                first_inv = s
            inv_prev, dmin_prev = iv, dmin
        ok = fail < 0
        if ok:
            r = float((Vc - V).norm(dim=-1).mean()) / H
            iv, dmin, _ = f.quality(Vc)
            print(f"{dt:9.1e} {ys:9.1e} {'완주':>10} {100*r:8.2f} {100*iv:9.2f} "
                  f"{dmin:9.3f} {'-':>8} "
                  f"{(first_inv if first_inv >= 0 else -1):9d} {'-':>12} {'-':>9}")
        else:
            print(f"{dt:9.1e} {ys:9.1e} {'발산':>10} {'-':>8} {'-':>9} "
                  f"{'-':>9} {fail:8d} "
                  f"{(first_inv if first_inv >= 0 else -1):9d} "
                  f"{100*inv_prev:12.3f} {dmin_prev:9.3f}")

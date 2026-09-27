"""격자 왕복 왜곡이 **해상도** 탓인지 본다.

P2G 는 질량가중 평균이고 G2P 는 B-spline 도함수라 공식은 MPM 과 같다. 문제는
셀당 입자 수다: 100^3 격자(dx=0.02)에 물체가 80 셀쯤 걸치면 점유 셀이 수천 개라
입자 8000 개면 셀당 2 개도 안 된다. PG 는 243621 개(셀당 ~50)를 쓴다.

입자 수와 격자 해상도를 바꿔가며 |(I+G) - J| / |J-I| 를 잰다 (J 는 이웃
최소제곱으로 뽑은 실제 야코비안).
"""
import argparse
import os

import torch

from anchorflow import phys_resid

ap = argparse.ArgumentParser()
ap.add_argument("--traj", required=True)
ap.add_argument("--t0", type=int, default=10)
ap.add_argument("--n_pts", type=int, nargs="+", default=[8000, 20000])
ap.add_argument("--n_grid", type=int, nargs="+", default=[100, 50, 32, 20])
ap.add_argument("--k", type=int, default=16)
a = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
d = torch.load(a.traj, map_location="cpu", weights_only=False)
cfg = d["cfg"]
gl = float(cfg.get("grid_lim", 2.0))
X = d["x"].float()
t0 = a.t0
I3 = torch.eye(3, device=dev)


def jac_ls(xa, xb, k=16, chunk=2048):
    N = xa.shape[0]
    idx = torch.empty(N, k, dtype=torch.long, device=dev)
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        idx[s:e] = torch.cdist(xa[s:e], xa).topk(k + 1, largest=False).indices[:, 1:]
    d0 = xa[idx] - xa.unsqueeze(1)
    d1 = xb[idx] - xb.unsqueeze(1)
    w = 1.0 / (d0.norm(dim=-1, keepdim=True) ** 2 + 1e-12)
    w = w / w.sum(1, keepdim=True)
    A = torch.einsum("nkc,nki,nkj->nij", w, d1, d0)
    B = torch.einsum("nkc,nki,nkj->nij", w, d0, d0) + 1e-10 * I3
    return A @ torch.linalg.inv(B)


print(f"[궤적] {os.path.basename(a.traj)} t0={t0} 전체 입자 {X.shape[1]}")
for NP in a.n_pts:
    g0 = torch.Generator().manual_seed(0)
    sel = torch.randperm(X.shape[1], generator=g0)[:NP].sort().values
    x0 = X[t0, sel].to(dev)
    x1 = X[t0 + 1, sel].to(dev)
    du = x1 - x0
    mass = torch.full((sel.numel(),), float(cfg["density"]), device=dev)
    J = jac_ls(x0, x1, a.k)
    jn = (J - I3).reshape(-1, 9).norm(dim=-1)
    for NG in a.n_grid:
        dx = gl / NG
        vi = (x0 / dx).long().clamp(0, NG - 1)
        fl = (vi[:, 0] * NG + vi[:, 1]) * NG + vi[:, 2]
        occ = int(torch.unique(fl).numel())
        m_I, du_I, _v, info, _ = phys_resid.p2g_increment(
            x0, du, torch.zeros_like(x0), mass, NG, gl)
        gu = phys_resid.g2p_grad(x0, du_I, info, NG)
        dist = ((gu + I3 - J).reshape(-1, 9).norm(dim=-1) / jn.clamp_min(1e-9))
        print(f"  입자 {NP:6d} 격자 {NG:4d}^3 (dx {dx:.4f})  점유셀 {occ:6d} "
              f"셀당 {NP / max(occ, 1):5.2f}개  왜곡 중앙 {float(dist.median()):.4f}")

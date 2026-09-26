"""궤적의 F 가 PG 의 F 와 맞는지, 그리고 그것이 목적함수를 어긋나게 하는지 본다.

확인 순서
  1) 궤적의 F 가 위치와 **정합**한가: 프레임 사이 야코비안 J 를 위치에서 뽑아
     F^{n+1} ~= J F^n 이 성립하는지 (성립 안 하면 F 가 다른 경로로 만들어진 것)
  2) F 를 아예 위치에서 다시 만들어 넣으면 교사오차가 줄어드는가
  3) 격자 왕복(P2G -> G2P)이 변형기울기를 얼마나 왜곡하는가
"""
import argparse
import os

import torch

from anchorflow import phys_resid

ap = argparse.ArgumentParser()
ap.add_argument("--traj", required=True)
ap.add_argument("--t0", type=int, nargs="+", default=[10])
ap.add_argument("--n_pts", type=int, default=8000)
ap.add_argument("--steps", type=int, default=400)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--k", type=int, default=16, help="야코비안 최소제곱 이웃 수")
a = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
d = torch.load(a.traj, map_location="cpu", weights_only=False)
cfg = d["cfg"]
h = float(cfg["frame_dt"])
ng, gl = int(cfg["n_grid"]), float(cfg.get("grid_lim", 2.0))
gv = torch.tensor(cfg["g"], device=dev)
X = d["x"].float()
F_all = d.get("F")
g0 = torch.Generator().manual_seed(0)
sel = torch.randperm(X.shape[1], generator=g0)[:a.n_pts].sort().values
X = X[:, sel].to(dev)
mass = torch.full((sel.numel(),), float(cfg["density"]), device=dev)
vol = mass / float(cfg["density"])
I3 = torch.eye(3, device=dev)


def jac_ls(xa, xb, k=16, chunk=2048):
    """국소 최소제곱으로 한 스텝 사상의 야코비안 [N,3,3]."""
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


for t0 in a.t0:
    x0, tgt = X[t0], X[t0 + 1]
    v0 = (X[t0] - X[t0 - 1]) / h
    ext = float((x0.max(0).values - x0.min(0).values).norm())
    stay = float((x0 - tgt).norm(dim=-1).mean()) / ext * 100
    F_traj = (F_all[t0, sel].float().to(dev) if F_all is not None else None)
    F_next = (F_all[t0 + 1, sel].float().to(dev) if F_all is not None else None)
    J = jac_ls(x0, tgt, a.k)
    print(f"[t0={t0}] 정지오차 {stay:.3f}%  |J-I| 중앙 "
          f"{float((J - I3).reshape(-1, 9).norm(dim=-1).median()):.5f}")
    if F_traj is not None and F_next is not None:
        pred = J @ F_traj
        rel = ((pred - F_next).reshape(-1, 9).norm(dim=-1)
               / F_next.reshape(-1, 9).norm(dim=-1).clamp_min(1e-9))
        print(f"   F 정합성: |J F^n - F^{{n+1}}| / |F^{{n+1}}| 중앙 "
              f"{float(rel.median()):.5f} 상위10% {float(rel.quantile(0.9)):.5f}")
    # 격자 왕복이 변형기울기를 얼마나 바꾸나
    du_t = tgt - x0
    m_I, du_I, _v, info, _ = phys_resid.p2g_increment(
        x0, du_t, torch.zeros_like(x0), mass, ng, gl)
    gu = phys_resid.g2p_grad(x0, du_I, info, ng)
    dJ = ((gu + I3 - J).reshape(-1, 9).norm(dim=-1)
          / (J - I3).reshape(-1, 9).norm(dim=-1).clamp_min(1e-9))
    print(f"   격자 왕복 왜곡: |(I+G)-J| / |J-I| 중앙 {float(dJ.median()):.4f}")

    def best_du(F):
        nrm = float(mass.sum()) * ext ** 2 / h ** 2
        du = torch.zeros_like(x0).requires_grad_(True)
        opt = torch.optim.Adam([du], lr=a.lr * ext)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
        best, bdu = float("inf"), du.detach().clone()
        for _ in range(a.steps):
            opt.zero_grad()
            E, _, _, _ = phys_resid.grid_ip_energy(
                x0, du, v0, F, mass, vol, cfg, h, ng, gl, g=gv, norm=nrm)
            E.backward(); opt.step(); sch.step()
            if float(E) < best:
                best, bdu = float(E), du.detach().clone()
        return float((x0 + bdu - tgt).norm(dim=-1).mean()) / ext * 100

    if F_traj is not None:
        print(f"   최적(궤적 F)   교사오차 {best_du(F_traj):6.3f}%")
    print(f"   최적(F=I)      교사오차 "
          f"{best_du(I3.expand(sel.numel(), 3, 3).contiguous()):6.3f}%")

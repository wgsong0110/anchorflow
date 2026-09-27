"""격자에서 뽑은 ∇Δu 가 실제 야코비안과 상관이 있는지 직접 확인한다.

왜곡비가 해상도와 무관하게 1 로 수렴하는 것은 두 양이 무상관이라는 뜻이다.
크기·상관·부호를 따로 보고, 균일 변형(정확히 아는 답)으로 검산한다.
"""
import argparse
import os

import torch

from anchorflow import phys_resid

ap = argparse.ArgumentParser()
ap.add_argument("--traj", required=True)
ap.add_argument("--t0", type=int, default=10)
ap.add_argument("--n_pts", type=int, default=20000)
ap.add_argument("--n_grid", type=int, default=100)
a = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
d = torch.load(a.traj, map_location="cpu", weights_only=False)
cfg = d["cfg"]
gl = float(cfg.get("grid_lim", 2.0))
NG = a.n_grid
X = d["x"].float()
g0 = torch.Generator().manual_seed(0)
sel = torch.randperm(X.shape[1], generator=g0)[:a.n_pts].sort().values
x0 = X[a.t0, sel].to(dev)
mass = torch.full((sel.numel(),), float(cfg["density"]), device=dev)
I3 = torch.eye(3, device=dev)


def grid_grad(x, du):
    m_I, du_I, _v, info, _ = phys_resid.p2g_increment(
        x, du, torch.zeros_like(x), mass, NG, gl)
    return phys_resid.g2p_grad(x, du_I, info, NG)


print(f"[검산] 균일 변형: du = A x, ∇Δu 는 정확히 A 여야 한다")
for tag, A in (("등방 수축 0.01", -0.01 * I3),
               ("x 방향 신장 0.02",
                torch.tensor([[0.02, 0, 0], [0, 0, 0], [0, 0, 0.0]], device=dev)),
               ("전단 0.02",
                torch.tensor([[0, 0.02, 0], [0, 0, 0], [0, 0, 0.0]], device=dev))):
    du = (A @ (x0 - x0.mean(0)).unsqueeze(-1)).squeeze(-1)
    gu = grid_grad(x0, du)
    err = (gu - A).reshape(-1, 9).norm(dim=-1) / A.reshape(-1).norm().clamp_min(1e-12)
    print(f"  {tag:16s} |∇Δu-A|/|A| 중앙 {float(err.median()):.4f} "
          f"|∇Δu| 중앙 {float(gu.reshape(-1, 9).norm(dim=-1).median()):.5f} "
          f"|A| {float(A.reshape(-1).norm()):.5f}")

# 실제 프레임 변위
x1 = X[a.t0 + 1, sel].to(dev)
du = x1 - x0
gu = grid_grad(x0, du)
print(f"\n[실측] 프레임 변위: |∇Δu_격자| 중앙 "
      f"{float(gu.reshape(-1, 9).norm(dim=-1).median()):.5f}")
# 성분별 상관
kk = 16
idx = torch.cdist(x0, x0).topk(kk + 1, largest=False).indices[:, 1:]
d0 = x0[idx] - x0.unsqueeze(1)
d1 = x1[idx] - x1.unsqueeze(1)
w = 1.0 / (d0.norm(dim=-1, keepdim=True) ** 2 + 1e-12)
w = w / w.sum(1, keepdim=True)
Am = torch.einsum("nkc,nki,nkj->nij", w, d1, d0)
Bm = torch.einsum("nkc,nki,nkj->nij", w, d0, d0) + 1e-10 * I3
J = Am @ torch.linalg.inv(Bm)
Gt = J - I3
print(f"[실측] |J-I| 중앙 {float(Gt.reshape(-1, 9).norm(dim=-1).median()):.5f}")
gf = gu.reshape(-1, 9)
tf = Gt.reshape(-1, 9)
for c in range(9):
    u, vv = gf[:, c], tf[:, c]
    cc = float(((u - u.mean()) * (vv - vv.mean())).mean()
               / (u.std().clamp_min(1e-20) * vv.std().clamp_min(1e-20)))
    print(f"   성분 {c}: 상관 {cc:+.4f}  격자 표준편차 {float(u.std()):.5f} "
          f"실제 {float(vv.std()):.5f}")

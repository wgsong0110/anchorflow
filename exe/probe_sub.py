"""서브스텝 분할 목적함수가 교사를 가리키는지 확인한다 (K 를 키워가며)."""
import argparse
import os

import torch

from anchorflow import phys_resid

ap = argparse.ArgumentParser()
ap.add_argument("--traj", required=True)
ap.add_argument("--t0", type=int, nargs="+", default=[3, 10, 20])
ap.add_argument("--n_pts", type=int, default=8000)
ap.add_argument("--K", type=int, nargs="+", default=[1, 4, 16, 64])
ap.add_argument("--steps", type=int, default=400)
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
print(f"[궤적] {os.path.basename(a.traj)}")
for t0 in a.t0:
    x = X[t0]
    du_t = X[t0 + 1] - x
    v = (X[t0] - X[t0 - 1]) / h
    F = (F_all[t0, sel].float().to(dev) if F_all is not None
         else torch.eye(3, device=dev).expand(sel.numel(), 3, 3).contiguous())
    ext = float((x.max(0).values - x.min(0).values).norm())
    e_stay = float((torch.zeros_like(x) - du_t).norm(dim=-1).mean()) / ext * 100
    print(f"[t0={t0}] 정지의 교사오차 {e_stay:.3f}%")
    for K in a.K:
        nrm = float(mass.sum()) * ext ** 2 / (h / K) ** 2
        du = torch.zeros_like(x).requires_grad_(True)
        opt = torch.optim.Adam([du], lr=3e-3 * ext)
        for _ in range(a.steps):
            opt.zero_grad()
            E, _, _, _ = phys_resid.grid_ip_sub(
                x, du, v, F, mass, vol, cfg, h, ng, gl, g=gv, norm=nrm, K=K)
            E.backward()
            opt.step()
        err = float((du.detach() - du_t).norm(dim=-1).mean()) / ext * 100
        with torch.no_grad():
            Et, _, _, _ = phys_resid.grid_ip_sub(
                x, du_t, v, F, mass, vol, cfg, h, ng, gl, g=gv, norm=nrm, K=K)
            Eo, _, _, _ = phys_resid.grid_ip_sub(
                x, du.detach(), v, F, mass, vol, cfg, h, ng, gl, g=gv, norm=nrm, K=K)
        print(f"   K={K:3d}  최적의 교사오차 {err:6.3f}%  "
              f"(정지 대비 {err / max(e_stay, 1e-9):.3f})  "
              f"E(최적)-E(교사) {float(Eo - Et):+.3e}")

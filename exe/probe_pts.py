"""탄성항을 입자 이웃 기울기로 바꾼 목적함수가 교사를 가리키는지 확인한다."""
import argparse
import os

import torch

from anchorflow import phys_resid

ap = argparse.ArgumentParser()
ap.add_argument("--traj", required=True)
ap.add_argument("--t0", type=int, nargs="+", default=[3, 10, 20])
ap.add_argument("--n_pts", type=int, default=8000)
ap.add_argument("--steps", type=int, default=400)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--k", type=int, default=16)
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
    x0, tgt = X[t0], X[t0 + 1]
    v0 = (X[t0] - X[t0 - 1]) / h
    F0 = (F_all[t0, sel].float().to(dev) if F_all is not None
          else torch.eye(3, device=dev).expand(sel.numel(), 3, 3).contiguous())
    ext = float((x0.max(0).values - x0.min(0).values).norm())
    stay = float((x0 - tgt).norm(dim=-1).mean()) / ext * 100
    nrm = float(mass.sum()) * ext ** 2 / h ** 2
    nb = phys_resid.jac_neighbors(x0, a.k)
    print(f"[t0={t0}] 정지오차 {stay:.3f}%")
    for name, fn in (("격자", phys_resid.grid_ip_energy),
                     ("입자", phys_resid.grid_ip_pts)):
        du = torch.zeros_like(x0).requires_grad_(True)
        opt = torch.optim.Adam([du], lr=a.lr * ext)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
        best, bdu = float("inf"), du.detach().clone()
        for _ in range(a.steps):
            opt.zero_grad()
            kw = dict(g=gv, norm=nrm)
            if name == "입자":
                kw["nb"] = nb
            E, _, _, _ = fn(x0, du, v0, F0, mass, vol, cfg, h, ng, gl, **kw)
            E.backward(); opt.step(); sch.step()
            if float(E) < best:
                best, bdu = float(E), du.detach().clone()
        err = float((x0 + bdu - tgt).norm(dim=-1).mean()) / ext * 100
        with torch.no_grad():
            kw = dict(g=gv, norm=nrm)
            if name == "입자":
                kw["nb"] = nb
            Et, _, _, it_ = fn(x0, tgt - x0, v0, F0, mass, vol, cfg, h, ng, gl, **kw)
            Eo, _, _, io_ = fn(x0, bdu, v0, F0, mass, vol, cfg, h, ng, gl, **kw)
        print(f"   {name} 탄성: 최적 교사오차 {err:6.3f}% (정지 대비 "
              f"{err / max(stay, 1e-9):.3f})  E(최적)-E(교사) {float(Eo - Et):+.3e}"
              f"  탄성 교사 {it_[1]:.3e} 최적 {io_[1]:.3e}")

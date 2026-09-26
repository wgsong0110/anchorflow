"""격자 물리손실이 손잡이가 크게 끌 때 왜 교사와 어긋나는지 항별로 분해한다.

비교 대상 변위 셋: 교사의 다음 프레임, 정지(0), 그리고 그 목적함수의 최적해.
각각에 대해 E 와 항별(관성/탄성/중력/접촉), 잔차, 교사 대비 위치오차를 낸다.
E(최적) < E(교사) 면 목적함수 자체가 틀린 답을 더 좋아하는 것이고, 어느 항이
그렇게 만드는지는 항별 차이로 드러난다.

  python exe/probe_terms.py --traj traj_h2/mic_clayC_t_s400706.pt --t0 3 10 20
"""
import argparse
import os

import torch

from anchorflow import phys_resid

ap = argparse.ArgumentParser()
ap.add_argument("--traj", required=True)
ap.add_argument("--t0", type=int, nargs="+", default=[3, 10, 20])
ap.add_argument("--n_pts", type=int, default=8000)
ap.add_argument("--steps", type=int, default=400)
ap.add_argument("--lr", type=float, default=3e-3)
ap.add_argument("--no_free", type=int, default=0,
                help="1 이면 손잡이 Dirichlet 제외를 끄고 잰다 (비교용)")
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


def free_of(t0):
    """손잡이 반경 안 입자를 구속으로 본다 (PG 규약). 없으면 전부 자유."""
    cid, cp, cr = d.get("ctrl_id"), d.get("ctrl_pos"), d.get("ctrl_R")
    if cid is None or a.no_free:
        return None
    ti = min(t0, cp.shape[0] - 1)
    R = float(cr[min(t0, cr.numel() - 1)])
    c = cp[ti].float().to(dev)
    dd = (X[t0].unsqueeze(1) - c.unsqueeze(0)).norm(dim=-1)
    return (dd.min(1).values > R)


print(f"[궤적] {os.path.basename(a.traj)}  손잡이 제외 {'끔' if a.no_free else '켬'}")
for t0 in a.t0:
    x = X[t0]
    v = (X[t0] - X[t0 - 1]) / h
    F = (F_all[t0, sel].float().to(dev) if F_all is not None
         else torch.eye(3, device=dev).expand(sel.numel(), 3, 3).contiguous())
    ext = float((x.max(0).values - x.min(0).values).norm())
    nrm = float(mass.sum()) * ext ** 2 / h ** 2
    fm = free_of(t0)
    n_free = int(fm.sum()) if fm is not None else sel.numel()

    def ev(du, tag):
        du = du.detach().clone().requires_grad_(True)
        E, _, _, info = phys_resid.grid_ip_energy(
            x, du, v, F, mass, vol, cfg, h, ng, gl, g=gv, norm=nrm, free=fm)
        gx, = torch.autograd.grad(E * nrm, du, retain_graph=False)
        r = float((gx * h * h / mass.unsqueeze(-1) / ext).norm(dim=-1).mean())
        err = float((x + du.detach() - X[t0 + 1]).norm(dim=-1).mean()) / ext * 100
        print(f"  {tag:10s} E {float(E):+.6e} | 관성 {info[0]:+.3e} 탄성 "
              f"{info[1]:+.3e} 중력 {info[2]:+.3e} 접촉 {info[3]:+.3e} "
              f"| 잔차 {r:.4e} | 교사오차 {err:6.3f}%")
        return float(E)

    du_t = X[t0 + 1] - x
    du_0 = torch.zeros_like(x)
    print(f"[t0={t0}] 자유입자 {n_free}/{sel.numel()}  물체 {ext:.4f}")
    E_t = ev(du_t, "교사")
    E_0 = ev(du_0, "정지")
    # 최적해
    du = torch.zeros_like(x).requires_grad_(True)
    opt = torch.optim.Adam([du], lr=a.lr * ext)
    for _ in range(a.steps):
        opt.zero_grad()
        E, _, _, _ = phys_resid.grid_ip_energy(
            x, du, v, F, mass, vol, cfg, h, ng, gl, g=gv, norm=nrm, free=fm)
        E.backward()
        opt.step()
    E_o = ev(du.detach(), "최적")
    print(f"  -> E(최적) {'<' if E_o < E_t else '>='} E(교사)   "
          f"차이 {E_o - E_t:+.4e}   (정지 대비 {E_0 - E_t:+.4e})")

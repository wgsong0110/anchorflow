"""서브스텝을 **순차로 따로** 풀어 교사에 수렴하는지 본다.

grid_ip_sub 은 프레임 변위를 직선으로 K 등분해 **한 번에** 최적화하므로 경로가
휘지 못한다. PG 는 서브스텝마다 상태를 갱신하며 나아가므로, 올바른 대조는 매
서브스텝의 증분 포텐셜을 따로 최소화하고 상태를 전진시키는 것이다.

K 를 키우며 최종 위치가 교사의 다음 프레임에 가까워지면, 목적함수는 맞고 문제는
'프레임 크기 한 스텝' 이라는 것이 된다. 가까워지지 않으면 목적함수 자체가 PG 를
가리키지 않는다.

  python exe/probe_seq.py --traj traj_h2/mic_clayC_t_s400706.pt --t0 10 --K 1 4 16 64
"""
import argparse
import os

import torch

from anchorflow import phys_resid

ap = argparse.ArgumentParser()
ap.add_argument("--traj", required=True)
ap.add_argument("--t0", type=int, nargs="+", default=[10])
ap.add_argument("--n_pts", type=int, default=8000)
ap.add_argument("--K", type=int, nargs="+", default=[1, 4, 16, 64])
ap.add_argument("--steps", type=int, default=300)
ap.add_argument("--lr", type=float, default=1e-3)
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
    x0 = X[t0]
    tgt = X[t0 + 1]
    v0 = (X[t0] - X[t0 - 1]) / h
    F0 = (F_all[t0, sel].float().to(dev) if F_all is not None
          else torch.eye(3, device=dev).expand(sel.numel(), 3, 3).contiguous())
    ext = float((x0.max(0).values - x0.min(0).values).norm())
    stay = float((x0 - tgt).norm(dim=-1).mean()) / ext * 100
    print(f"[t0={t0}] 정지의 교사오차 {stay:.3f}%  물체 {ext:.4f}")
    for K in a.K:
        hs = h / K
        x, v, F = x0, v0, F0
        for _k in range(K):
            nrm = float(mass.sum()) * ext ** 2 / hs ** 2
            du = torch.zeros_like(x).requires_grad_(True)
            opt = torch.optim.Adam([du], lr=a.lr * ext / K ** 0.5)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
            best, bdu = float("inf"), du.detach().clone()
            for _ in range(a.steps):
                opt.zero_grad()
                E, _, _, _ = phys_resid.grid_ip_energy(
                    x, du, v, F, mass, vol, cfg, hs, ng, gl, g=gv, norm=nrm)
                E.backward()
                opt.step()
                sch.step()
                if float(E) < best:
                    best, bdu = float(E), du.detach().clone()
            with torch.no_grad():
                _E, dlog, F_tr, _ = phys_resid.grid_ip_energy(
                    x, bdu, v, F, mass, vol, cfg, hs, ng, gl, g=gv, norm=nrm)
                F = phys_resid.plastic_step(F_tr, dlog)
                v = bdu / hs
                x = x + bdu
        err = float((x - tgt).norm(dim=-1).mean()) / ext * 100
        print(f"   K={K:3d} (h={hs:.3e})  최종 교사오차 {err:6.3f}%  "
              f"정지 대비 {err / max(stay, 1e-9):.3f}")

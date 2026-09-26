"""관성항과 탄성항의 저울이 프레임 간격 h 때문에 어긋나는지 확인한다.

증분 포텐셜은 E = sum m/(2h^2)|x - xtil|^2 + sum V Psi(F(x)) - sum m g.x 인데,
관성 계수가 1/h^2 이라 h 를 PG 의 서브스텝(5e-5)에서 프레임(1/60)으로 키우면
관성항이 (5e-5/1.67e-2)^2 = 9e-6 배로 줄어 탄성항에 압도된다. 그러면 교사의
한 프레임 운동은 이 목적함수의 정류점이 아니게 된다.

h 를 바꿔가며 교사/정지/최적의 E 와 교사오차를 재서 이것을 확인한다. 각 h 에서
관성항은 그 h 로 계산하고 변위는 **프레임 변위를 h/frame_dt 비율로 축소**해
같은 물리 구간을 보게 맞춘다.
"""
import argparse
import os

import torch

from anchorflow import phys_resid

ap = argparse.ArgumentParser()
ap.add_argument("--traj", required=True)
ap.add_argument("--t0", type=int, default=10)
ap.add_argument("--n_pts", type=int, default=8000)
ap.add_argument("--sub", type=int, nargs="+", default=[1, 4, 16, 64, 333],
                help="프레임을 이만큼으로 쪼갠 h 로 잰다")
ap.add_argument("--steps", type=int, default=400)
a = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
d = torch.load(a.traj, map_location="cpu", weights_only=False)
cfg = dict(d["cfg"])
fdt = float(cfg["frame_dt"])
ng, gl = int(cfg["n_grid"]), float(cfg.get("grid_lim", 2.0))
gv = torch.tensor(cfg["g"], device=dev)
X = d["x"].float()
F_all = d.get("F")
g0 = torch.Generator().manual_seed(0)
sel = torch.randperm(X.shape[1], generator=g0)[:a.n_pts].sort().values
X = X[:, sel].to(dev)
mass = torch.full((sel.numel(),), float(cfg["density"]), device=dev)
vol = mass / float(cfg["density"])
t0 = a.t0
x0 = X[t0]
ext = float((x0.max(0).values - x0.min(0).values).norm())
F0 = (F_all[t0, sel].float().to(dev) if F_all is not None
      else torch.eye(3, device=dev).expand(sel.numel(), 3, 3).contiguous())
du_frame = X[t0 + 1] - x0
v0 = (X[t0] - X[t0 - 1]) / fdt
print(f"[궤적] {os.path.basename(a.traj)} t0={t0} 물체 {ext:.4f} "
      f"프레임변위 중앙 {float(du_frame.norm(dim=-1).median()):.5f}")

for S in a.sub:
    h = fdt / S
    du_t = du_frame / S                      # 같은 물리 구간의 1/S
    nrm = float(mass.sum()) * ext ** 2 / h ** 2

    def ev(du, tag, show=True):
        du = du.detach().clone().requires_grad_(True)
        E, _, _, info = phys_resid.grid_ip_energy(
            x0, du, v0, F0, mass, vol, cfg, h, ng, gl, g=gv, norm=nrm)
        err = float((du.detach() - du_t).norm(dim=-1).mean()) / ext * 100
        if show:
            print(f"    {tag:6s} E {float(E):+.5e} | 관성 {info[0]:+.3e} "
                  f"탄성 {info[1]:+.3e} 중력 {info[2]:+.3e} | 교사변위오차 {err:6.3f}%")
        return float(E)

    print(f"  h = frame/{S} = {h:.3e}")
    E_t = ev(du_t, "교사")
    E_0 = ev(torch.zeros_like(x0), "정지")
    du = torch.zeros_like(x0).requires_grad_(True)
    opt = torch.optim.Adam([du], lr=3e-3 * ext / max(S ** 0.5, 1.0))
    for _ in range(a.steps):
        opt.zero_grad()
        E, _, _, _ = phys_resid.grid_ip_energy(
            x0, du, v0, F0, mass, vol, cfg, h, ng, gl, g=gv, norm=nrm)
        E.backward()
        opt.step()
    E_o = ev(du.detach(), "최적")
    print(f"    -> E(최적)-E(교사) {E_o - E_t:+.4e}  "
          f"{'교사가 정류점 아님' if E_o < E_t - 1e-12 else '교사가 최소에 가깝다'}")

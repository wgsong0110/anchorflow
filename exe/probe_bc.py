"""교사 프레임에서 바닥 경계조건이 있고 없고를 비교한다.

PG 의 설정에는 z=0.48 sticky 평면과 경계 상자가 들어 있는데, 격자 목적함수가
이것을 안 쓰면 중력을 막는 항이 없어 바닥 아래로 흐르는 것이 이득이 된다.
교사(PG)의 다음 프레임이 얼마나 정류점에 가까운지로 확인한다 -- 바닥을 넣으면
잔차가 낮아져야 맞다.
"""
import argparse
import json
import os

import torch

from anchorflow import phys_resid

ap = argparse.ArgumentParser()
ap.add_argument("--traj", required=True)
ap.add_argument("--cfg", required=True)
ap.add_argument("--t0", type=int, default=3)
ap.add_argument("--fall", type=float, default=0.0,
                help="바닥을 향해 이만큼(물체 크기 비율) 내려가는 변위를 시험한다")
ap.add_argument("--n_pts", type=int, default=8000)
a = ap.parse_args()

dev = "cuda"
d = torch.load(a.traj, map_location="cpu", weights_only=False)
cfg = json.load(open(a.cfg))
x_all = d["x"].float()
g = torch.Generator().manual_seed(0)
sel = torch.randperm(x_all.shape[1], generator=g)[:a.n_pts].sort().values
x_all = x_all[:, sel].to(dev)
F_all = d.get("F")
F = (F_all[a.t0, sel].float().to(dev) if F_all is not None
     else torch.eye(3, device=dev).expand(sel.numel(), 3, 3).contiguous())

h = float(cfg["frame_dt"])
n_grid, gl = int(cfg["n_grid"]), float(cfg.get("grid_lim", 2.0))
x = x_all[a.t0]
v = (x_all[a.t0] - x_all[a.t0 - 1]) / h
mass = torch.full((x.shape[0],), float(cfg["density"]), device=dev)
ext = float((x.max(0).values - x.min(0).values).norm())
vol = mass / float(cfg["density"])
gv = torch.tensor(cfg["g"], device=dev)
nrm = float(mass.sum()) * ext ** 2 / h ** 2

print(f"[기하] 물체 z {float(x[:,2].min()):.4f} ~ {float(x[:,2].max()):.4f}, "
      f"dx={gl / n_grid:.4f}, 바닥면 z=0.48 은 노드 {0.48 / (gl / n_grid):.1f}")
_m, _du, _v, _info, _ = phys_resid.p2g_increment(
    x, torch.zeros_like(x), v, mass, n_grid, gl)
os.environ["AF_NO_BC"] = ""
_bc = phys_resid.bc_node_mask(_info[3], n_grid, _info[4], cfg, x.dtype, dev)
print(f"[경계] 점유 노드 {_info[3].numel()} 개 중 규정 노드 {int(_bc.sum())} 개")

for off, tag in ((1, "교사 다음 프레임"), (0, "정지(아무것도 안 함)")):
    du0 = (x_all[a.t0 + off] - x) if off else torch.zeros_like(x)
    if a.fall > 0:
        du0 = du0 - torch.tensor([0.0, 0.0, a.fall * ext], device=dev)
    for no_bc in (1, 0):
        os.environ["AF_NO_BC"] = "1" if no_bc else ""
        du = du0.clone().requires_grad_(True)
        E, _, _, _ = phys_resid.grid_ip_energy(
            x, du, v, F, mass, vol, cfg, h, n_grid, gl, g=gv, norm=nrm)
        gx, = torch.autograd.grad(E * nrm, du)
        r = (gx * h ** 2 / mass.unsqueeze(-1) / ext).norm(dim=-1).mean()
        print(f"{tag:16s} 바닥 {'없음' if no_bc else '있음'}: "
              f"E={float(E):.6e}  잔차={float(r):.6e}")

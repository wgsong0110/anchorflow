"""절 AE (1): 입자 부피 V_p 확인 -- 우리 V_p (deform_model.traj_mass / ρ) 와 정답 MPM 이 쓴 V_p (exe/lb_gen.py, PhysGaussian get_particle_volume:
MPM 격자 칸 dx³ / 칸 안 입자 수, 0 프레임 위치) 가 같은지, 그리고 입자 점군이 실제로 덮는 부피와 Σ V_p 비교.
덮는 부피 근사: (a) MPM 칸 점유 (= Σ V_p, 정의상) (b) 더 잘게 (dx/2, /4, /8) 점유 칸 부피 (c) 닫힌 고체 근사 -- 해상도 r 로 점유 → k 칸 팽창 → 바깥 채움 (flood fill)
→ 속 채움 → k 칸 침식 (알파 셰이프 근사, 반경 ~ k r) (d) 최근접 이웃 간격 d_p 로 Σ d_p³.
사용: python exe/fem_ae_vp.py --data <.pt> [--gt_h5 <0 프레임 h5>] --out <json>"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from scipy import ndimage

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from anchorflow import deform_model as DM  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--region_density", action="store_true")
ap.add_argument("--gt_h5", default="")
ap.add_argument("--out", required=True)
a = ap.parse_args()
d = torch.load(a.data, map_location="cpu", weights_only=False)
cfg = d["cfg"]
DM.REGION_DENSITY[0] = a.region_density
m = DM.traj_mass(d, "cpu").double()
rho = d["_rho"].double() if a.region_density else torch.full_like(m, float(cfg["density"]))
vol = m / rho                                                               # fem_ex_sim.py 와 같은 V_p
x0 = d["x"][0].double()
N = x0.shape[0]
ng, GL = int(cfg.get("n_grid", 100)), float(cfg.get("grid_lim", 2.0))
dx = GL / ng
R = dict(data=a.data, N=N, n_grid=ng, grid_lim=GL, dx=dx, V_sum=float(vol.sum()), M_sum=float(m.sum()))

# 정답 생성기 식 그대로 (lb_gen.py: torch.unique 키 / PG: ti.floor(p/dx) 칸 개수) -- float32 위치에서
X32 = x0.float()
cell = (X32 / dx).floor().long()
key = (cell[:, 0] * ng + cell[:, 1]) * ng + cell[:, 2]
_, inv, cnt = torch.unique(key, return_inverse=True, return_counts=True)
VOLgt = (dx ** 3 / cnt[inv].float()).double()
R["V_gt_formula_sum"] = float(VOLgt.sum())
R["V_vs_gt_formula_maxrel"] = float(((vol - VOLgt).abs() / VOLgt).max())
R["occ_cells_mpm"] = int(cnt.numel())
R["parts_per_cell_q"] = [float(q) for q in np.quantile(cnt.numpy(), [0.1, 0.5, 0.9])]
if a.gt_h5:
    import h5py
    with h5py.File(a.gt_h5) as h:
        xg = np.array(h["x"])
    xg = xg.T if xg.shape[0] == 3 else xg
    R["gt_h5"] = a.gt_h5
    R["gt_h5_N"] = int(xg.shape[0])
    if xg.shape[0] == N:
        R["gt_h5_x0_maxdiff"] = float(np.abs(xg.astype(np.float64) - x0.numpy()).max())

# (b) 더 잘게 점유
lo = x0.min(0).values.numpy() - 1e-9
occ = {}
for k in (1, 2, 4, 8):
    r = dx / k
    c = np.floor((x0.numpy() - lo) / r).astype(np.int64)
    u = np.unique(c[:, 0] * 10**12 + c[:, 1] * 10**6 + c[:, 2])
    occ[f"dx/{k}"] = float(len(u) * r ** 3)
R["V_occupied"] = occ

# (c) 닫힌 고체 근사
sol = {}
for k_div, kk in ((4, 2), (4, 4), (8, 4)):
    r = dx / k_div
    c = np.floor((x0.numpy() - lo) / r).astype(np.int64) + kk + 2
    shp = tuple(c.max(0) + kk + 3)
    g = np.zeros(shp, bool)
    g[c[:, 0], c[:, 1], c[:, 2]] = True
    st = ndimage.generate_binary_structure(3, 1)
    gd = ndimage.binary_dilation(g, st, iterations=kk)
    gf = ndimage.binary_fill_holes(gd)
    ge = ndimage.binary_erosion(gf, st, iterations=kk, border_value=0) | g
    sol[f"r=dx/{k_div},k={kk} (반경 ~{kk * r:.4f} m)"] = float(ge.sum() * r ** 3)
R["V_solid_close"] = sol

# (d) 최근접 이웃 간격
xs = x0.float().cuda() if torch.cuda.is_available() else x0.float()
nn = torch.empty(N, device=xs.device)
for i in range(0, N, 8192):
    dd = torch.cdist(xs[i:i + 8192], xs)
    dd[torch.arange(dd.shape[0]), torch.arange(i, i + dd.shape[0])] = float("inf")
    nn[i:i + 8192] = dd.min(1).values
nn = nn.double().cpu()
R["nn_q"] = [float(q) for q in np.quantile(nn.numpy(), [0.1, 0.5, 0.9])]
R["V_nn3_sum"] = float((nn ** 3).sum())
R["V_p_q"] = [float(q) for q in np.quantile(vol.numpy(), [0.1, 0.5, 0.9])]
R["Vp_over_nn3_q"] = [float(q) for q in np.quantile((vol / nn ** 3).numpy(), [0.1, 0.5, 0.9])]
json.dump(R, open(a.out, "w"), indent=1, ensure_ascii=False)
print(json.dumps(R, indent=1, ensure_ascii=False))

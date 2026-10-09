"""찻주전자 충돌 장면 초기 상태: 학습한 찻주전자 3DGS(fgs_assets) + PG 공식 채우기, 한 벌 또는 두 벌.

  - 크기: 실제 크기(0.27 m) × --scale
  - 물성: Fracture-GS 표 6 의 Table top (E 1.5e4, ν 0.39, 밀도 1, NACC β 0.5, ξ 1, M 2.36). 초기 logJp 는 --alpha0
  - --n_obj 1 --rest : 바닥 위에 가만히 놓기 (자중으로 무너지는지 보는 정적 시험)
  - --n_obj 2       : x 로 --gap 떨어뜨려 ±--v 로 정면 충돌 (공중, 중력 켬)
렌더용 gaussians.pt 는 fgs_render.py 형식 (gi, c6, shs, op).

  cd i-physgaussian && python <anchorflow>/exe/fgs_pair_init.py --assets /home/dkta/work/fgs_assets --out DIR
"""
import argparse
import json
import math
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--pg", default="/home/dkta/work/i-physgaussian")
ap.add_argument("--assets", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--scale", type=float, default=3.0)
ap.add_argument("--n_obj", type=int, default=2)
ap.add_argument("--rest", action="store_true")
ap.add_argument("--gap", type=float, default=0.4)
ap.add_argument("--v", type=float, default=5.0)
ap.add_argument("--alpha0", type=float, default=math.log(0.99))
ap.add_argument("--E", type=float, default=1.5e4)
ap.add_argument("--nu", type=float, default=0.39)
ap.add_argument("--rho", type=float, default=1.0)
ap.add_argument("--beta", type=float, default=0.5)
ap.add_argument("--xi", type=float, default=1.0)
ap.add_argument("--material", default="watermelon", help="watermelon = NACC (GF/CD-MPM), jelly = 탄성만")
ap.add_argument("--floor_z", type=float, default=0.1)
ap.add_argument("--n_grid", type=int, default=200)
ap.add_argument("--frames", type=int, default=60)
ap.add_argument("--frame_dt", type=float, default=1.0 / 60.0)
a = ap.parse_args()
sys.path.append(a.pg); sys.path.append(os.path.join(a.pg, "gaussian-splatting"))
os.chdir(a.pg)

import h5py                                                      # noqa: E402
import numpy as np                                               # noqa: E402
import taichi as ti                                              # noqa: E402
import torch                                                     # noqa: E402
from scene.gaussian_model import GaussianModel                   # noqa: E402
from particle_filling.filling import fill_particles              # noqa: E402

ti.init(arch=ti.cuda, device_memory_GB=4.0)
dev = torch.device("cuda")
GL = 2.0
meta = json.load(open(f"{a.assets}/teapot_ns/meta.json"))
gs = GaussianModel(3)
gs.load_ply(f"{a.assets}/teapot_gs/point_cloud/iteration_30000/point_cloud.ply")
keep = gs.get_opacity.detach()[:, 0] > 0.02
S = a.scale / meta["scale"]
P = gs.get_xyz.detach()[keep] * S; C6 = gs.get_covariance()[keep].detach() * S * S
SH = gs.get_features[keep].detach(); OP = gs.get_opacity[keep].detach()
P = P - 0.5 * (P.min(0).values + P.max(0).values)               # 중심 원점
w, hgt = float(P[:, 0].max() - P[:, 0].min()), float(P[:, 2].max() - P[:, 2].min())
if a.rest or a.n_obj == 1:
    CEN = [np.array([1.0, 1.0, a.floor_z + 0.5 * hgt + 0.005])]
else:
    zc = a.floor_z + 0.5 * hgt + 0.3
    d = 0.5 * w + 0.5 * a.gap
    CEN = [np.array([1.0 - d, 1.0, zc]), np.array([1.0 + d, 1.0, zc])]
FP = dict(grid_n=a.n_grid, max_samples=2000000, grid_dx=GL / a.n_grid, density_thres=100.0, search_thres=1.0,
          max_particles_per_cell=1, search_exclude_dir=2, ray_cast_dir=4, boundary=None, smooth=True)
Q = P + torch.tensor(CEN[0], device=dev, dtype=P.dtype)
xf = fill_particles(pos=Q.float().contiguous(), opacity=OP.float().contiguous(), cov=C6.float().contiguous(), **FP)
xf = xf.to(dev).float() - torch.tensor(CEN[0], device=dev).float()     # 한 벌 (중심 원점)
ng = P.shape[0]; n1 = xf.shape[0]
print(f"[찻주전자] 크기 {np.round((P.max(0).values - P.min(0).values).cpu().numpy(), 3)} m, 가우시안 {ng} + 채움 {n1 - ng}",
      flush=True)
XS, VS, OB = [], [], []
for o, c in enumerate(CEN):
    XS.append(xf.cpu().numpy() + c)
    v = np.zeros((n1, 3))
    if not a.rest and a.n_obj == 2:
        v[:, 0] = a.v if o == 0 else -a.v
    VS.append(v); OB.append(np.full(n1, o, np.int32))
X, V, OBJ = np.concatenate(XS), np.concatenate(VS), np.concatenate(OB)
assert X.min() > 0.05 and X.max() < GL - 0.05, (X.min(0), X.max(0))
os.makedirs(a.out, exist_ok=True)
with h5py.File(f"{a.out}/init.h5", "w") as h:
    h.create_dataset("x", data=X.T.astype(np.float32)); h.create_dataset("v", data=V.T.astype(np.float32))
    h.create_dataset("obj", data=OBJ)
M = 2.36; sphi = 3 * M / (6 + M)
cfg = dict(material=a.material, E=a.E, nu=a.nu, density=a.rho, alpha_0=a.alpha0, beta=a.beta, xi=a.xi, hardening=1.0,
           friction_angle=math.degrees(math.asin(sphi)), n_grid=a.n_grid, grid_lim=GL, flip_pic_ratio=0.7,
           substep_dt=1e-5, frame_dt=a.frame_dt, frame_num=a.frames, g=[0.0, 0.0, -9.8], init_velocity=[0.0, 0.0, 0.0],
           boundary_conditions=[{"type": "bounding_box"},
                                {"type": "surface_collider", "point": [1.0, 1.0, a.floor_z], "normal": [0, 0, 1],
                                 "surface": "sticky", "friction": 0.0, "start_time": 0, "end_time": 1000.0}],
           repflow_note=dict(scene="찻주전자 (Table top 물성)", scale=a.scale, n_obj=a.n_obj, rest=a.rest, gap=a.gap, v=a.v))
json.dump(cfg, open(f"{a.out}/config.json", "w"), indent=1)
GI = np.concatenate([np.arange(ng) + o * n1 for o in range(len(CEN))])
K = len(CEN)
torch.save(dict(gi=torch.as_tensor(GI), c6=C6.repeat(K, 1).cpu(), shs=SH.repeat(K, 1, 1).cpu(), op=OP.repeat(K, 1).cpu()),
           f"{a.out}/gaussians.pt")
print(f"[저장] {a.out}: 물체 {K}, 입자 {X.shape[0]}, α0 {a.alpha0:.4f}, 재질 {a.material}", flush=True)

"""Fracture-GS Teapot & Table 대체 장면의 시뮬 초기 상태: 배치 + PG 공식 내부 채우기 + 입자별 물성.

fgs_scene.py 와 같은 배치(실제 크기, m, z 위)를 시뮬 상자 [0, 2]^3 안으로 옮긴다:
탁자 중심 xy = (1, 1), 바닥(탁자 다리 밑) z = --floor_z. 찻주전자는 상판 중앙 위 --drop_h, 정지 상태에서 낙하.
채우기: PhysGaussian fill_particles (wolf config 값: 채우기 격자 100, 밀도 100, 탐색 1.0, 제외 2, 광선 4, smooth),
물체마다 따로. 반환은 가우시안 먼저, 채운 입자 나중 (PG 와 같다).
물성 (Fracture-GS 표 6): 물체 0 찻주전자 = Teapot, 물체 1 탁자 = 상판 Table top / 다리 Table leg (z 로 가른다).
  NACC α 는 입자별 초기 logJp = ln α 로, M 2.36 은 셋 다 같다. β·ξ 는 솔버 전역값이라 찻주전자·상판의 (0.5, 1) 을
  쓴다 -- 다리의 (2, 3) 은 못 넣는다 (다리는 E 1e8 로 사실상 강체라 항복하지 않는다).

  cd i-physgaussian && python <anchorflow>/exe/fgs_init.py --assets /home/dkta/work/fgs_assets --out DIR
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
ap.add_argument("--teapot_scale", type=float, default=1.0)
ap.add_argument("--drop_h", type=float, default=0.3)
ap.add_argument("--v0", type=float, default=0.0, help="찻주전자 초기 하강 속도 (m/s)")
ap.add_argument("--floor_z", type=float, default=0.1)
ap.add_argument("--n_grid", type=int, default=200)
ap.add_argument("--frames", type=int, default=60)
ap.add_argument("--frame_dt", type=float, default=1.0 / 60.0)
ap.add_argument("--opacity", type=float, default=0.02)
a = ap.parse_args()
sys.path.append(a.pg)
sys.path.append(os.path.join(a.pg, "gaussian-splatting"))
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


def load(name):
    meta = json.load(open(f"{a.assets}/{name}_ns/meta.json"))
    gs = GaussianModel(3)
    gs.load_ply(f"{a.assets}/{name}_gs/point_cloud/iteration_30000/point_cloud.ply")
    keep = gs.get_opacity.detach()[:, 0] > a.opacity
    return dict(x=gs.get_xyz.detach()[keep] / meta["scale"], c6=gs.get_covariance()[keep].detach() / meta["scale"] ** 2,
                shs=gs.get_features[keep].detach(), op=gs.get_opacity[keep].detach())


T, P = load("table"), load("teapot")
P["x"] = P["x"] * a.teapot_scale; P["c6"] = P["c6"] * a.teapot_scale ** 2
tc = 0.5 * (T["x"].min(0).values + T["x"].max(0).values)
T["x"] = T["x"] - torch.tensor([float(tc[0]) - 1.0, float(tc[1]) - 1.0, float(T["x"][:, 2].min()) - a.floor_z], device=dev)
ztop = float(torch.quantile(T["x"][:, 2], 0.995))
pc = 0.5 * (P["x"].min(0).values + P["x"].max(0).values)
P["x"] = P["x"] - pc + torch.tensor([1.0, 1.0, ztop + a.drop_h + float(pc[2] - P["x"][:, 2].min())], device=dev)
assert float(P["x"][:, 2].max()) < GL - 0.05, "상자 위로 넘친다"

FP = dict(grid_n=100, max_samples=2000000, grid_dx=GL / 100, density_thres=100.0, search_thres=1.0,
          max_particles_per_cell=1, search_exclude_dir=2, ray_cast_dir=4, boundary=None, smooth=True)
XS, OB, NG = [], [], []
for o, Q in enumerate((P, T)):
    xf = fill_particles(pos=Q["x"].float().contiguous(), opacity=Q["op"].float().contiguous(),
                        cov=Q["c6"].float().contiguous(), **FP)
    xf = xf.to(dev).float()
    print(f"[채우기] 물체 {o}: 가우시안 {Q['x'].shape[0]} + 채움 {xf.shape[0] - Q['x'].shape[0]}", flush=True)
    XS.append(xf); OB.append(torch.full((xf.shape[0],), o, dtype=torch.int32)); NG.append(Q["x"].shape[0])
X = torch.cat(XS).cpu().numpy().astype(np.float64); OBJ = torch.cat(OB).numpy()
N = X.shape[0]
# 탁자 상판 / 다리: z 단면의 xy 상자 넓이가 최대의 절반 아래로 떨어지는 곳이 상판 밑면
xt = XS[1].cpu().numpy(); zs = np.linspace(xt[:, 2].min(), xt[:, 2].max(), 200)
area = []
for z0, z1 in zip(zs[:-1], zs[1:]):
    s = xt[(xt[:, 2] >= z0) & (xt[:, 2] < z1)]
    area.append(np.prod(s[:, :2].max(0) - s[:, :2].min(0)) if len(s) > 10 else 0.0)
area = np.array(area); amax = area.max()
ztb = zs[:-1][np.nonzero(area > 0.5 * amax)[0].min()]           # 상판 밑면 (아래에서부터 처음 넓어지는 곳)
# 다리 기둥만 넓은 경우를 막는다: 상판 밑면은 상판 윗면에서 아래로 15 cm 안
ztb = max(ztb, ztop - 0.15)
TOP = (OBJ == 1) & (X[:, 2] >= ztb); LEG = (OBJ == 1) & (X[:, 2] < ztb); POT = OBJ == 0
print(f"[물성 구역] 찻주전자 {POT.sum()}  상판 {TOP.sum()} (z >= {ztb:.3f}, 윗면 {ztop:.3f})  다리 {LEG.sum()}", flush=True)
E = np.where(POT, 5e5, np.where(TOP, 1.5e4, 1e8))
NU = np.where(POT, 0.46, 0.39)
RHO = np.where(POT, 5.0, np.where(TOP, 1.0, 1000.0))
AL0 = np.where(POT, math.log(0.98), np.where(TOP, math.log(0.99), math.log(0.94)))
V = np.zeros_like(X); V[POT, 2] = -a.v0
os.makedirs(a.out, exist_ok=True)
with h5py.File(f"{a.out}/init.h5", "w") as h:
    h.create_dataset("x", data=X.T.astype(np.float32)); h.create_dataset("v", data=V.T.astype(np.float32))
    h.create_dataset("obj", data=OBJ)
    for k, v in (("E", E), ("nu", NU), ("density", RHO), ("alpha0", AL0)):
        h.create_dataset(k, data=v.astype(np.float64))
M = 2.36; sphi = 3 * M / (6 + M)
cfg = dict(material="watermelon", E=5e5, nu=0.46, density=5.0, alpha_0=math.log(0.98), beta=0.5, xi=1.0,
           hardening=1.0, friction_angle=math.degrees(math.asin(sphi)), n_grid=a.n_grid, grid_lim=GL,
           flip_pic_ratio=0.7, substep_dt=1e-5, frame_dt=a.frame_dt, frame_num=a.frames,
           g=[0.0, 0.0, -9.8], init_velocity=[0.0, 0.0, 0.0],
           boundary_conditions=[{"type": "bounding_box"},
                                {"type": "surface_collider", "point": [1.0, 1.0, a.floor_z], "normal": [0, 0, 1],
                                 "surface": "sticky", "friction": 0.0, "start_time": 0, "end_time": 1000.0}],
           repflow_note=dict(scene="Fracture-GS Teapot & Table 대체 (Poly Haven CC0)", drop_h=a.drop_h, v0=a.v0,
                             teapot_scale=a.teapot_scale, table_top_z=ztop, table_top_bottom_z=float(ztb),
                             per_particle="init.h5 의 E, nu, density, alpha0", fill=FP))
json.dump(cfg, open(f"{a.out}/config.json", "w"), indent=1)
# 렌더용 가우시안 (입자 번호 = 각 물체 채우기 결과의 앞부분)
off = np.cumsum([0] + [x.shape[0] for x in XS])[:-1]
GI = np.concatenate([np.arange(n) + o_ for n, o_ in zip(NG, off)])
torch.save(dict(gi=torch.as_tensor(GI), c6=torch.cat([P["c6"], T["c6"]]).cpu(),
                shs=torch.cat([P["shs"], T["shs"]]).cpu(), op=torch.cat([P["op"], T["op"]]).cpu()),
           f"{a.out}/gaussians.pt")
print(f"[저장] {a.out}: 입자 {N}, 가우시안 {GI.size}", flush=True)

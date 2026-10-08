"""PhysTwin 공식 코드(Jianghanxiao/PhysTwin)의 실시간 루프 비용을 wolf 에서 잰다 (렌더링 제외).

공식 interactive_playground 의 프레임 = (1) 시뮬레이터 forward_graph 한 번 (스프링-질점 Warp, 667 서브스텝)
+ (2) 가우시안 이동 보간 (calc_weights_vals_from_indices + interpolate_motions_speedup, 이웃 16 은 첫 프레임에 한 번).
공개된 케이스 데이터(학습된 강성 등)가 없어, 설정은 공식 configs/real.yaml 기본값으로 둔다:
dt 5e-5, 667 서브스텝(30 FPS), init_spring_Y 3e4, dashpot 100, drag 3, 스프링은 반지름 0.02 m 안 최대 30 이웃
(trainer_warp._init_start 와 같은 규칙). 질점은 가우시안 중심에서 공식 data_process_sample 처럼 뽑는다
(11024 점 → 복셀 0.005 m, 물체를 0.25 m 크기로).

  python exe/fps_phystwin.py --repo /home/dkta/work/PhysTwin --pts repflow/aux_wolf_pts.npy
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True)
ap.add_argument("--pts", required=True)
ap.add_argument("--frames", type=int, default=30)
ap.add_argument("--warm", type=int, default=5)
a = ap.parse_args()
sys.path.insert(0, a.repo)
os.chdir(a.repo)
import types                                                         # noqa: E402
import importlib                                                     # noqa: E402
import warp as wp                                                    # noqa: E402
from scipy.spatial import cKDTree                                    # noqa: E402
# qqtt/__init__.py 는 학습기(open3d·gsplat 등)까지 끌어온다 -> 시뮬레이터에 필요한 모듈만 직접 올린다
for _name, _sub in (("qqtt", "qqtt"), ("qqtt.utils", "qqtt/utils"), ("qqtt.model", "qqtt/model"),
                    ("qqtt.model.diff_simulator", "qqtt/model/diff_simulator")):
    _m = types.ModuleType(_name); _m.__path__ = [os.path.join(a.repo, _sub)]; sys.modules[_name] = _m
cfg = importlib.import_module("qqtt.utils.config").cfg
sys.modules["qqtt.utils"].cfg = cfg
sys.modules["qqtt.utils"].logger = importlib.import_module("qqtt.utils.logger").logger
cfg.load_from_yaml(os.path.join(a.repo, "configs", "real.yaml"))
cfg.device = "cuda"
cfg.use_graph = True
from qqtt.model.diff_simulator.spring_mass_warp import SpringMassSystemWarp      # noqa: E402
from gaussian_splatting.dynamic_utils import (interpolate_motions_speedup, knn_weights_sparse,  # noqa: E402
                                              calc_weights_vals_from_indices, get_topk_indices)
wp.init()
dev = "cuda"
G = torch.as_tensor(np.load(a.pts), dtype=torch.float32)
ext = float((G.max(0).values - G.min(0).values).max())
G = (G - G.mean(0)) / ext * 0.25                                    # 물체 0.25 m (PhysTwin 실측 물체 크기)
g = torch.Generator().manual_seed(0)
P = G[torch.randperm(G.shape[0], generator=g)[:11024]]
key = torch.floor((P - P.min(0).values) / 0.005).long()
_, inv = torch.unique(key, dim=0, return_inverse=True)
B = torch.zeros(int(inv.max()) + 1, 3).index_reduce_(0, inv, P, "mean", include_self=False)
n = B.shape[0]
tree = cKDTree(B.numpy())
springs = set()
for i in range(n):                                                    # search_hybrid_vector_3d(r, max_nn)
    d, j = tree.query(B[i].numpy(), k=cfg.object_max_neighbours, distance_upper_bound=cfg.object_radius)
    for jj, dd in zip(j, d):
        if np.isfinite(dd) and jj != i:
            springs.add((min(i, jj), max(i, jj)))
S = torch.as_tensor(sorted(springs), dtype=torch.int32)
rest = (B[S[:, 0].long()] - B[S[:, 1].long()]).norm(dim=1)
print(f"[PhysTwin] 질점 {n}  스프링 {S.shape[0]}  가우시안 {G.shape[0]}  dt {cfg.dt}  서브스텝 {cfg.num_substeps}",
      flush=True)
gt = B[None].repeat(2, 1, 1).to(dev)
sim = SpringMassSystemWarp(
    B.to(dev), S.to(dev), rest.to(dev), torch.ones(n, device=dev), dt=cfg.dt, num_substeps=cfg.num_substeps,
    spring_Y=cfg.init_spring_Y, collide_elas=cfg.collide_elas, collide_fric=cfg.collide_fric,
    dashpot_damping=cfg.dashpot_damping, drag_damping=cfg.drag_damping,
    collide_object_elas=cfg.collide_object_elas, collide_object_fric=cfg.collide_object_fric,
    collision_dist=cfg.collision_dist, num_object_points=n, num_surface_points=n, num_original_points=n,
    controller_points=None, reverse_z=cfg.reverse_z, spring_Y_min=cfg.spring_Y_min,
    spring_Y_max=cfg.spring_Y_max, gt_object_points=gt,
    gt_object_visibilities=torch.ones(2, n, device=dev), gt_object_motions_valid=torch.ones(2, n, device=dev),
    self_collision=False, disable_backward=True)
xyz = G.to(dev)
quat = torch.zeros(G.shape[0], 4, device=dev); quat[:, 0] = 1
prev = wp.to_torch(sim.wp_states[0].wp_x, requires_grad=False).clone()
rel = get_topk_indices(prev, K=16)
_, widx = knn_weights_sparse(prev, xyz, K=16)
ts, tsim, tint = [], [], []
for f in range(a.warm + a.frames):
    torch.cuda.synchronize(); t0 = time.time()
    wp.capture_launch(sim.forward_graph)
    x = wp.to_torch(sim.wp_states[-1].wp_x, requires_grad=False)
    sim.set_init_state(sim.wp_states[-1].wp_x, sim.wp_states[-1].wp_v)
    torch.cuda.synchronize(); t1 = time.time()
    w = calc_weights_vals_from_indices(prev, xyz, widx)
    xyz, quat, _ = interpolate_motions_speedup(bones=prev, motions=x - prev, relations=rel, weights=w,
                                               weights_indices=widx, xyz=xyz, quat=quat)
    prev = x.clone()
    torch.cuda.synchronize(); t2 = time.time()
    if f >= a.warm:
        tsim.append(t1 - t0); tint.append(t2 - t1); ts.append(t2 - t0)
ms = lambda v: 1000 * float(np.median(v))
print(f"[FPS PhysTwin] 시뮬 {ms(tsim):.2f} ms + 보간 {ms(tint):.2f} ms = {ms(ts):.2f} ms/프레임 "
      f"-> {1000 / ms(ts):.1f} FPS", flush=True)

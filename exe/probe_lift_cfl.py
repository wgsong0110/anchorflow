"""기하가 험한가: MPM 상태를 앵커로 접었다 편 입자 속도가 격자를 견디는가.

DAgger 는 첫 호출에서 학생을 굴리기 전에 **진짜 MPM 궤적의 프레임 1** 을 앵커로
접어 그대로 lift 한다(train_nextstate.collect_dagger 참조). 즉 stu_dyn 이 죽은
상태에는 학생의 오차가 섞여 있지 않다 -- 기하만의 문제다.

그래서 여기서는 학생 없이, 캐시된 MPM 궤적을 각 기하로 접었다 펴서 입자 속도와
CFL 수(한 substep 이동거리 / 격자 간격)를 잰다. CFL 이 1 을 넘으면 입자가 한
substep 에 격자 한 칸을 건너뛰고, warp 는 격자 쓰기에 경계 검사가 없다.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
import torch

from anchorflow import scene_setup

ap = argparse.ArgumentParser()
ap.add_argument("--ply", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--dreamphysics", default="/workspace/DreamPhysics")
ap.add_argument("--traj_cache", required=True)
ap.add_argument("--ckpt", nargs="+", required=True, help="이름:경로")
ap.add_argument("--n_traj", type=int, default=12)
ap.add_argument("--frames", type=int, default=6, help="궤적 앞쪽 몇 프레임을 볼지")
ap.add_argument("--anchors", type=int, default=512)
ap.add_argument("--knn", type=int, default=8)
ap.add_argument("--sh_degree", type=int, default=3)
ap.add_argument("--grid_lim", type=float, default=2.0)
ap.add_argument("--n_grid", type=int, default=100)
args = ap.parse_args()

sys.path.insert(0, args.dreamphysics)
import warp as wp

dev = "cuda"
torch.set_grad_enabled(False)
wp.init()
from anchorflow.anchor_sparse import Traj, load_fitted

sys.modules["__main__"].Traj = Traj

sc = scene_setup.build(args.ply, args.config, args.anchors, args.knn, device=dev,
                       frozen_weights=True, rot_fallback=True, eig_floor=0.02,
                       sh_degree=args.sh_degree)
DX = args.grid_lim / args.n_grid
SUB = sc.sub_dt
DT_C = 40 * SUB
print(f"[setup] dx {DX:.5g}  sub_dt {SUB:.5g}  dt_c {DT_C:.5g}", flush=True)

FIT = torch.load(args.traj_cache, map_location=dev, weights_only=False)["fit"]
NT = min(args.n_traj, len(FIT))
print(f"[data] 궤적 {NT} x 앞 {args.frames} 프레임", flush=True)

for spec in args.ckpt:
    name, path = spec.split(":", 1)
    if not os.path.exists(path):
        print(f"  [건너뜀] {name}: {path} 없음", flush=True)
        continue
    fit = load_fitted(sc, path, dev)[0].fit
    cache = fit.prepare()
    fac = fit.ls_factor(cache)
    w, rc, q, Binv, blocked, amass = cache
    # 기하 자체의 건강 지표
    trB = Binv.diagonal(dim1=-2, dim2=-1).sum(-1)          # 큰 값 = B 가 특이함
    gpa = torch.zeros(fit.M, device=dev).index_add_(
        0, fit.pair_a, torch.ones_like(w))                  # 앵커가 쥔 짝 수
    worst_cfl = worst_v = worst_C = worst_ccfl = 0.0
    worst_det = 1e9
    blow = 0
    for i in range(NT):
        for t in range(1, 1 + args.frames):
            if t >= FIT[i][0].shape[0]:
                break
            x0 = FIT[i][0][t].to(dev, torch.float32)
            xm = FIT[i][0][t - 1].to(dev, torch.float32)
            p = fit.project_ls(x0, cache, fac)
            pm = fit.project_ls(xm, cache, fac)
            # DAgger 와 같은 속도: 앵커 위치의 유한차분이지 MPM 속도의 투영이 아니다
            v = (p - pm) / DT_C
            xh, vl, Fl, Cl = fit.lift(p, v, cache)
            sp = vl.norm(dim=-1).max().item()
            cfl = sp * SUB / DX
            worst_v = max(worst_v, sp)
            worst_cfl = max(worst_cfl, cfl)
            Fm = Fl.view(-1, 3, 3)
            worst_C = max(worst_C, Cl.abs().max().item())
            worst_det = min(worst_det, torch.linalg.det(Fm).min().item())
            # C 는 한 substep 동안 이웃 격자로 옮기는 속도 기울기다
            worst_ccfl = max(worst_ccfl, Cl.abs().max().item() * SUB)
            if cfl > 1.0:
                blow += 1
    print(f"  {name:<10} 앵커 {fit.M:>5}  최대 |v| {worst_v:>10.2f}  최대 CFL {worst_cfl:>9.3f}"
          f"  CFL>1 프레임 {blow}/{NT * args.frames}", flush=True)
    print(f"             최대 |C| {worst_C:.4g}  |C|·sub_dt {worst_ccfl:.4g}  "
          f"최소 det(F) {worst_det:.4g}", flush=True)
    print(f"             trace(Binv) 최대 {trB.max():.3g}, "
          f"앵커가 쥔 짝 최소 {int(gpa.min())}, 앵커질량 최소 {amass.min():.3g}",
          flush=True)
    fit = None
    torch.cuda.empty_cache()

# 비교 기준: MPM 자신의 입자 속도
mx = 0.0
for i in range(NT):
    for t in range(1, 1 + args.frames):
        if t >= FIT[i][1].shape[0]:
            break
        mx = max(mx, FIT[i][1][t].to(dev, torch.float32).norm(dim=-1).max().item())
print(f"\n  [기준] MPM 자신의 최대 입자 |v| {mx:.2f}  (CFL {mx * SUB / DX:.3f})")
print("CFL_DONE")

"""회전 목표가 정말 큰가, 아니면 우리가 큰 값을 만들고 있나.

|u_g| 이 1.61 rad(92도) 로 나왔다. 회전벡터를 선형 블렌딩하는 것은 회전이 작을 때만
타당하고, pi 근처에서는 +-pi 로 뒤집혀 평균이 무의미해진다. 그래서 세 가지를 나눠 본다.

  1. MPM 의 F 자체 -- det, 특이값, 항등원에서 얼마나 떨어져 있나
  2. 극분해가 주는 목표 |u_g*| 의 분포 (프레임별로도)
  3. 디코더가 내는 |u_g| 와 그 목표의 차이
"""
from __future__ import annotations

import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
import torch
from anchorflow import scene_setup

ap = argparse.ArgumentParser()
ap.add_argument("--ply", required=True); ap.add_argument("--config", required=True)
ap.add_argument("--dreamphysics", default="/workspace/DreamPhysics")
ap.add_argument("--fit", required=True); ap.add_argument("--traj_cache", required=True)
ap.add_argument("--frames", nargs="+", type=int, default=[0, 1, 5, 10, 20, 30])
ap.add_argument("--n_traj", type=int, default=4)
args = ap.parse_args()

sys.path.insert(0, args.dreamphysics)
import warp as wp
dev = "cuda"; torch.set_grad_enabled(False); wp.init()
from anchorflow.anchor_sparse import Traj, load_fitted
from anchorflow.frame_encode import FrameState, polar_target, expmap, logmap
from anchorflow.anchor_fit import closest_rotation
sys.modules["__main__"].Traj = Traj

sc = scene_setup.build(args.ply, args.config, 512, 8, device=dev,
                       frozen_weights=True, rot_fallback=True, eig_floor=0.02)
fit = load_fitted(sc, args.fit, dev)[0].fit
cache = fit.prepare(); w = cache[0]; Yg = fit.Xc - cache[1]
MASS = sc.volume[sc.keep].clone()
FS = FrameState(fit.pair_g, fit.pair_a, fit.N, fit.M, MASS)
FIT = torch.load(args.traj_cache, map_location=dev, weights_only=False)["fit"]
print(f"[setup] 앵커 {fit.M}, 가우시안 {fit.N}\n", flush=True)

print(f"{'프레임':>6}{'det 중앙':>10}{'det 최소':>10}{'|F-I| 평균':>12}"
      f"{'|u*| 평균':>11}{'|u*| 중앙':>11}{'|u*|>2 비율':>12}{'|u_g| 복호':>11}{'복호 오차':>10}")
for t in args.frames:
    da = dm = fi = ua = um = hi = ug_ = du = 0.0
    for i in range(args.n_traj):
        if t >= FIT[i][2].shape[0]: continue
        F0 = FIT[i][2][t].to(dev, torch.float32).view(-1, 3, 3)
        det = torch.linalg.det(F0)
        eye = torch.eye(3, device=dev).expand_as(F0)
        ut, st = polar_target(F0)
        n = ut.norm(dim=-1)
        p, u, s = FS.encode_closed(FIT[i][0][t].to(dev, torch.float32), F0, w, Yg,
                                    fixed=fit.fixed, p_fix=fit.pos, targets=(ut, st))
        ug, sg = FS.blend(u, s, w)
        da += float(det.median()); dm += float(det.min())
        fi += float((F0 - eye).norm(dim=(-1, -2)).mean())
        ua += float(n.mean()); um += float(n.median()); hi += float((n > 2.0).float().mean())
        ug_ += float(ug.norm(dim=-1).mean()); du += float((ug - ut).norm(dim=-1).mean())
    k = args.n_traj
    print(f"{t:>6}{da/k:>10.4f}{dm/k:>10.4f}{fi/k:>12.4f}{ua/k:>11.4f}{um/k:>11.4f}"
          f"{100*hi/k:>11.1f}%{ug_/k:>11.4f}{du/k:>10.4f}", flush=True)
print("\ndet 이 1 근처면 부피가 보존된 것. |u*| 는 극분해가 주는 가우시안별 회전 크기(rad).")
print("|u*|>2 비율이 크면 회전벡터 선형 블렌딩의 전제가 깨진 것이다(pi 근처 뒤집힘).")
print("ROT_DONE")

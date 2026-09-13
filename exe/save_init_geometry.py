"""학습 안 한 기하를 체크포인트로 저장한다 -- 학생 대조군의 출발점.

기하 학습이 학생의 롤아웃 오차를 실제로 얼마나 줄이는지는 재본 적이 없다.
같은 레시피의 학생을 초기 기하 위에 올려야 그 답이 나온다.
"""
from __future__ import annotations

import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
import torch
from anchorflow import scene_setup

ap = argparse.ArgumentParser()
ap.add_argument("--ply", required=True); ap.add_argument("--config", required=True)
ap.add_argument("--dreamphysics", default="/workspace/DreamPhysics")
ap.add_argument("--out", required=True)
ap.add_argument("--anchors", type=int, default=512)
args = ap.parse_args()
sys.path.insert(0, args.dreamphysics)
import warp as wp
dev = "cuda"; torch.set_grad_enabled(False); wp.init()
from anchorflow.anchor_sparse import AnchorSparse

sc = scene_setup.build(args.ply, args.config, args.anchors, 8, device=dev,
                       frozen_weights=True, rot_fallback=True, eig_floor=0.02)
fit = AnchorSparse(sc, c=0.25, eig_floor=0.02).to(dev)
fit.init_from_geometry()
torch.save({"pos": fit.pos.detach().cpu(), "quat": fit.quat.detach().cpu(),
            "log_s": fit.log_s.detach().cpu(), "log_k": fit.log_k.detach().cpu(),
            "log_amp": fit.log_amp.detach().cpu(),
            "c": 0.25, "eig_floor": 0.02, "iter": 0,
            "runtime": {"mass_floor": 0.0, "mass_freeze": 0.02, "finv_ridge": 1e-6,
                         "cfl_agg": "mean", "astress_eig": 0.0, "astress_jmax": 0.0,
                         "edge_only": False},
            "args": {"note": "init_from_geometry, 학습 없음"}}, args.out)
print(f"저장 {args.out}: 앵커 {fit.M}, 짝 {fit.pair_g.shape[0]}")
print("INIT_SAVED")

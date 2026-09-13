"""logm 이 실제로 맞는지부터 확인한다 -- exp(log F) 가 F 로 돌아오는가.

이 검증을 안 하고 결과를 보고한 것이 앞선 실패의 핵심이었다. 합성 행렬과 MPM 의
실제 F 양쪽에서 재고, 고윳값이 음수인 표본이 실제로 있는지도 함께 센다
(2차 판에서 확인 없이 그렇다고 단정하고 방법을 바꿨다).
"""
from __future__ import annotations

import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
import torch
from anchorflow import scene_setup

ap = argparse.ArgumentParser()
ap.add_argument("--ply", required=True); ap.add_argument("--config", required=True)
ap.add_argument("--dreamphysics", default="/workspace/DreamPhysics")
ap.add_argument("--traj_cache", required=True)
ap.add_argument("--frames", nargs="+", type=int, default=[2, 9, 10, 18, 30])
ap.add_argument("--trajs", nargs="+", type=int, default=[0, 8, 26, 39, 72],
                help="대표성 있게 -- 앞의 몇 개만 보면 검증이 아니다")
args = ap.parse_args()

sys.path.insert(0, args.dreamphysics)
import warp as wp
dev = "cuda"; torch.set_grad_enabled(False); wp.init()
from anchorflow.anchor_sparse import Traj
from anchorflow.frame_logeuc import logm3, expm3, roundtrip_error
sys.modules["__main__"].Traj = Traj

print("=== 1. 합성 행렬 (정답을 아는 경우) ===", flush=True)
torch.manual_seed(0)
n = 20000
X = torch.randn(n, 3, 3, device=dev) * 0.25
F = expm3(X)
Xb = logm3(F)
e = (Xb - X).norm(dim=(-1, -2)) / X.norm(dim=(-1, -2)).clamp(min=1e-12)
print(f"  log(exp(X)) 복원 상대오차: 평균 {float(e.mean()):.3e}  최대 {float(e.max()):.3e}")
rel, bad = roundtrip_error(F)
print(f"  exp(log(F)) 왕복 상대오차: 평균 {float(rel.mean()):.3e}  최대 {float(rel.max()):.3e}"
      f"  비유한 {int(bad.sum())}", flush=True)

print("\n=== 2. MPM 의 실제 F ===", flush=True)
sc = scene_setup.build(args.ply, args.config, 512, 8, device=dev,
                       frozen_weights=True, rot_fallback=True, eig_floor=0.02)
FIT = torch.load(args.traj_cache, map_location=dev, weights_only=False)["fit"]
print(f"{'프레임':>7}{'왕복 평균':>12}{'왕복 최대':>12}{'비유한':>9}"
      f"{'det 최소':>10}{'음수 고윳값 표본':>16}{'복소 고윳값':>12}")
for t in args.frames:
    rs, rx, nb, dmin, neg, cpx, tot = 0.0, 0.0, 0, 9e9, 0, 0, 0
    for i in args.trajs:
        if i >= len(FIT) or t >= FIT[i][2].shape[0]: continue
        F0 = FIT[i][2][t].to(dev, torch.float32).view(-1, 3, 3)
        rel, bad = roundtrip_error(F0)
        ok = torch.isfinite(rel)
        rs += float(rel[ok].mean()); rx = max(rx, float(rel[ok].max()))
        nb += int(bad.sum())
        dmin = min(dmin, float(torch.linalg.det(F0).min()))
        ev = torch.linalg.eigvals(F0)
        im = ev.imag.abs().max(-1).values
        realneg = ((im < 1e-6) & (ev.real.min(-1).values < 0)).float()
        neg += int(realneg.sum()); cpx += int((im > 1e-6).float().sum())
        tot += F0.shape[0]
    k = len(args.trajs)
    print(f"{t:>7}{rs/k:>12.3e}{rx:>12.3e}{nb:>9}{dmin:>10.4f}"
          f"{neg:>10}/{tot:<6}{cpx:>12}", flush=True)
print("\n음수 고윳값 표본이 0 이면 실수 로그가 모든 F 에 존재한다 --")
print("2차 판에서 극분해로 우회한 근거가 틀렸다는 뜻이다.")
print("LOGM_DONE")

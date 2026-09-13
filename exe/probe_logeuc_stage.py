"""로그-유클리드 인코더가 어느 단계에서 깨지는지 하나씩 짚는다.

logm 자체는 왕복 8e-7 로 검증됐는데 인코더 전체는 nan 이다. 후보는 셋:
  (a) 목표 X* 가 특정 창에서 비정상
  (b) 최소제곱이 증폭해 앵커값 X_a 가 폭주  (역블러링의 성질)
  (c) 블렌드된 X_g 가 커서 exp 가 오버플로  (float32 는 e^88 에서 넘친다)
각 단계의 크기와 비유한 개수를 찍어 어디인지 확정한다.
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
ap.add_argument("--n_win", type=int, default=6); ap.add_argument("--frames", type=int, default=30)
args = ap.parse_args()

sys.path.insert(0, args.dreamphysics)
import warp as wp
dev = "cuda"; torch.set_grad_enabled(False); wp.init()
from anchorflow.anchor_sparse import Traj, load_fitted
from anchorflow.frame_encode import FrameState
from anchorflow.frame_logeuc import LogEucState, logm_target, expm3
sys.modules["__main__"].Traj = Traj

sc = scene_setup.build(args.ply, args.config, 512, 8, device=dev,
                       frozen_weights=True, rot_fallback=True, eig_floor=0.02)
fit = load_fitted(sc, args.fit, dev)[0].fit
cache = fit.prepare(); w = cache[0]
MASS = sc.volume[sc.keep].clone()
FS = FrameState(fit.pair_g, fit.pair_a, fit.N, fit.M, MASS)
LE = LogEucState(FS)
L = FS.gram(w)
FIT = torch.load(args.traj_cache, map_location=dev, weights_only=False)["fit"]
g = torch.Generator(device="cpu"); g.manual_seed(20260908)
WIN = [(int(torch.randint(len(FIT), (1,), generator=g).item()),
        int(torch.randint(args.frames - 1, (1,), generator=g).item()))
       for _ in range(args.n_win)]


def st(name, T):
    f = torch.isfinite(T)
    v = T[f]
    return (f"{name:<14} 비유한 {int((~f).sum()):>8}  |.| 평균 "
            f"{float(v.abs().mean()) if v.numel() else float('nan'):>10.4f}  최대 "
            f"{float(v.abs().max()) if v.numel() else float('nan'):>12.4f}")


print(f"{'창':>4}  단계별", flush=True)
for k, (i, t) in enumerate(WIN):
    F0 = FIT[i][2][t].to(dev, torch.float32).view(-1, 3, 3)
    print(f"[창 {k}] 궤적 {i} 프레임 {t}")
    print("   " + st("0. F^MPM", F0))
    Xt = logm_target(F0)
    print("   " + st("1. 목표 X*", Xt))
    Xa = FS._wls(Xt.reshape(-1, 9), w, L)
    print("   " + st("2. 앵커 X_a", Xa))
    Xg = LE.blend(Xa, w)
    print("   " + st("3. 블렌드 X_g", Xg))
    Fh = expm3(Xg)
    print("   " + st("4. exp(X_g)", Fh))
    ok = torch.isfinite(Fh).all(dim=(-1, -2))
    if ok.any():
        e = ((MASS[ok] * (Fh[ok] - F0[ok]).pow(2).sum((-1, -2))).sum()
             / MASS[ok].sum()).sqrt()
        print(f"   유한 표본 {int(ok.sum())}/{ok.numel()}  그 위의 F 잔차 {float(e):.4f}")
    print(flush=True)
print("STAGE_DONE")

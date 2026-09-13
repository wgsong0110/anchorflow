"""부피 붕괴가 실제로 렌더링 문제인가, 그리고 cov3D_precomp 경로의 비용.

행렬식은 이 문제의 척도가 아니다 -- det 0.014 가 (0.24,0.24,0.24) 일 수도
(1,1,0.014) 일 수도 있고, 앞은 그냥 작아진 타원체이고 뒤만 납작한 조각이다.
봐야 할 것은 **최소 특이값** sigma_min 이다.

부호는 렌더링에서 문제가 아니다 -- Sigma' = F Sigma0 F^T 는 F 가 무엇이든 양반정부호이고
det Sigma' = (det F)^2 det Sigma0 이라 부호가 제곱으로 사라진다. 타원체는 거울 대칭이다.
다만 MPM 으로 되돌릴 때는 걸린다(교사가 det < 0.05 를 거부한다).
"""
from __future__ import annotations

import argparse, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
import torch
from anchorflow import scene_setup

ap = argparse.ArgumentParser()
ap.add_argument("--ply", required=True); ap.add_argument("--config", required=True)
ap.add_argument("--dreamphysics", default="/workspace/DreamPhysics")
ap.add_argument("--fit", required=True); ap.add_argument("--traj_cache", required=True)
ap.add_argument("--n_win", type=int, default=6); ap.add_argument("--frames", type=int, default=30)
ap.add_argument("--width", type=int, default=800); ap.add_argument("--height", type=int, default=800)
args = ap.parse_args()

sys.path.insert(0, args.dreamphysics)
import warp as wp
dev = "cuda"; torch.set_grad_enabled(False); wp.init()
from anchorflow.anchor_sparse import Traj, load_fitted
from anchorflow.frame_encode import FrameState, polar_target
from anchorflow.frame_logeuc import LogEucState
from anchorflow.anchor_fit import closest_rotation
sys.modules["__main__"].Traj = Traj

sc = scene_setup.build(args.ply, args.config, 512, 8, device=dev,
                       frozen_weights=True, rot_fallback=True, eig_floor=0.02)
fit = load_fitted(sc, args.fit, dev)[0].fit
cache = fit.prepare(); w = cache[0]; Yg = fit.Xc - cache[1]
MASS = sc.volume[sc.keep].clone()
FS = FrameState(fit.pair_g, fit.pair_a, fit.N, fit.M, MASS)
LE = LogEucState(FS)
L = FS.gram(w)
FIT = torch.load(args.traj_cache, map_location=dev, weights_only=False)["fit"]
g = torch.Generator(device="cpu"); g.manual_seed(20260908)
WIN = [(int(torch.randint(len(FIT), (1,), generator=g).item()),
        int(torch.randint(args.frames - 1, (1,), generator=g).item()))
       for _ in range(args.n_win)]
print(f"[setup] 앵커 {fit.M}, 가우시안 {fit.N}, 창 {len(WIN)}", flush=True)

Q = (0.001, 0.01, 0.1, 0.5, 0.9)
QL = ("p0.1", "p1", "p10", "p50", "p90")
def quant(v):
    s = v.reshape(-1); s = s[torch.isfinite(s)]
    s = s[torch.randperm(s.numel(), device=s.device)[:200000]]
    return [float(x) for x in torch.quantile(s, torch.tensor(Q, device=s.device))]

R = {}
for name in ("MPM 원본", "회전+신축", "자유 F", "로그-유클리드"):
    sg, dt, neg, flat, tot = [], [], 0, 0, 0
    for i, t in WIN:
        F0 = FIT[i][2][t].to(dev, torch.float32).view(-1, 3, 3)
        x0 = FIT[i][0][t].to(dev, torch.float32)
        if name == "MPM 원본":
            Fh = F0
        elif name == "회전+신축":
            _, u, s_ = FS.encode_closed(x0, F0, w, Yg, fixed=fit.fixed, p_fix=fit.pos)
            Fh = FS.decode(u, s_, w)
        elif name == "자유 F":
            Fa = FS._wls(F0.reshape(-1, 9), w, L)
            Fh = FS.decode_F(Fa, w)
        else:
            Fh = LE.decode(LE.encode(F0, w, L=L), w)
        sv = torch.linalg.svdvals(Fh)
        d = torch.linalg.det(Fh)
        sg.append(sv[:, -1]); dt.append(d)
        neg += int((d < 0).sum()); flat += int((sv[:, -1] < 0.2).sum()); tot += d.numel()
    R[name] = dict(sig=quant(torch.cat(sg)), det=quant(torch.cat(dt)),
                   neg=100 * neg / tot, flat=100 * flat / tot)
    print(f"  {name}: 최소특이값 p1 {R[name]['sig'][1]:.4f}, det<0 {R[name]['neg']:.3f}%, "
          f"sig<0.2 {R[name]['flat']:.3f}%", flush=True)

print(f"\n== 최소 특이값 sigma_min (한 축이 얼마나 눌렸는가 -- 이게 진짜 척도) ==")
print(f"{'':<14}" + "".join(f"{a:>10}" for a in QL) + f"{'sig<0.2':>10}{'det<0':>9}")
for k, v in R.items():
    print(f"{k:<14}" + "".join(f"{a:>10.4f}" for a in v["sig"])
          + f"{v['flat']:>9.3f}%{v['neg']:>8.3f}%")
print(f"\n== 참고: 행렬식 ==")
print(f"{'':<14}" + "".join(f"{a:>10}" for a in QL))
for k, v in R.items():
    print(f"{k:<14}" + "".join(f"{a:>10.4f}" for a in v["det"]))

# ---- cov3D_precomp 경로 비용 ---------------------------------------------
print(f"\n== 렌더 준비 비용 (프레임당) ==", flush=True)
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
import math
NG = sc.pos.shape[0]
S0 = torch.eye(3, device=dev).expand(fit.N, 3, 3) * (0.005 ** 2)   # 정준 공분산 대역
Fx = torch.eye(3, device=dev).expand(fit.N, 3, 3).contiguous()

def bench(fn, n=40):
    for _ in range(10): fn()
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.time() - t0) / n * 1e3

def cov_path():
    Sg = Fx @ S0 @ Fx.transpose(-1, -2)
    return torch.stack([Sg[:, 0, 0], Sg[:, 0, 1], Sg[:, 0, 2],
                        Sg[:, 1, 1], Sg[:, 1, 2], Sg[:, 2, 2]], -1)
def polar_path():
    closest_rotation(Fx, 8, 1e-6)

print(f"  cov3D_precomp (F S0 F^T, 상삼각 6): {bench(cov_path):.2f} ms")
print(f"  극분해 (회전+스케일 뽑기)            : {bench(polar_path):.2f} ms")
print("\nSIGMA_DONE")

"""앵커가 F 를 **상태로** 들고 있으면 MPM 의 F 를 얼마나 담을 수 있나.

지금 앵커 상태는 위치와 속도뿐이다. 회전·신축(log_s, quat)은 정지 상태 파라미터라
피팅으로 정해지고 시뮬레이션 중에는 변하지 않는다. 그래서 변형 기울기 F 는 앵커가
들고 있는 양이 아니라 **이웃 앵커들의 상대 위치에서 형상 매칭으로 유도**된다:

    F_i = (sum_a w_ia (p_a - cc_i) ⊗ q_i) Binv_i

512 개 앵커의 배치에서 171,553 개 입자 각각의 F 를 뽑는 것이라, 실측 원시 오차가
1.517 이었다 -- 항등원 주변의 무차원 양에서 이 값이면 사실상 재현하지 못한다.

대안은 앵커마다 자기 변형을 상태로 들고, 입자의 F 를 그 혼합으로 만드는 것이다:

    F_i = sum_a w_ia Fa_a          (Fa: 앵커별 3x3, 회전과 신축을 모두 담는다)

이때 최선의 Fa 는 최소제곱으로 정해진다: Fa* = (W^T W)^-1 W^T F_mpm, W[i,a] = w_ia.
그 Fa* 로 되돌린 F 의 오차가 **이 상태 설계의 표현 하한**이다. 지금 값(1.517)과
견주면, F 오차가 상태 설계 탓인지 512 라는 개수의 한계인지 갈린다.

위치 인코더가 쓰는 C 와는 다른 연산자다 -- C 는 x = Cp + b 의 것이고, 여기서는
가중 혼합 W 자체다.
"""
from __future__ import annotations

import argparse
import os
import sys

_lib = os.path.join(os.path.dirname(__file__), "..", "lib")
sys.path.insert(0, _lib)

import torch

from anchorflow import scene_setup

ap = argparse.ArgumentParser()
ap.add_argument("--ply", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--dreamphysics", default="/workspace/DreamPhysics")
ap.add_argument("--traj_cache", required=True)
ap.add_argument("--n_win", type=int, default=8)
ap.add_argument("--frames", type=int, default=30)
ap.add_argument("--ridge", type=float, default=1e-6)
args = ap.parse_args()

sys.path.insert(0, args.dreamphysics)
import warp as wp

dev = "cuda"
torch.set_grad_enabled(False)
wp.init()
from anchorflow.anchor_sparse import AnchorSparse, Traj
sys.modules["__main__"].Traj = Traj

sc = scene_setup.build(args.ply, args.config, 512, 8, device=dev,
                       frozen_weights=True, rot_fallback=True, eig_floor=0.02)
fit = AnchorSparse(sc, c=0.25, eig_floor=0.02).to(dev)
fit.init_from_geometry()
cache = fit.prepare()
fac = fit.ls_factor(cache)
MASS = sc.volume[sc.keep].clone()
w = cache[0]
print(f"[setup] 앵커 {fit.M}, 물질 가우시안 {fit.N}, 짝 {fit.pair_g.shape[0]}", flush=True)

# 가중 혼합 연산자 W 와 그 최소제곱 역
flat = fit.pair_g * fit.M + fit.pair_a
W = torch.zeros(fit.N * fit.M, device=dev).index_add_(0, flat, w).view(fit.N, fit.M)
Gw = W.t() @ W
Gw = Gw + args.ridge * Gw.diagonal().mean().clamp(min=1e-20) * torch.eye(fit.M, device=dev)
Lw = torch.linalg.cholesky(Gw)
print(f"[setup] W [{fit.N}, {fit.M}] 조밀, W^T W 조건수 "
      f"{float(torch.linalg.cond(Gw)):.3e}", flush=True)

blob = torch.load(args.traj_cache, map_location=dev, weights_only=False)
FIT = blob["fit"]

def mw(e2):
    return float(((MASS * e2).sum() / MASS.sum()).sqrt())

g = torch.Generator(device="cpu"); g.manual_seed(20260905)
hi = max(1, args.frames - 12)
cur, best, ident = [], [], []
for _ in range(args.n_win):
    i = int(torch.randint(len(FIT), (1,), generator=g).item())
    t = int(torch.randint(hi, (1,), generator=g).item())
    ent = FIT[i]
    x0 = ent[0][t]
    F0 = ent[2][t].reshape(-1, 9).float()
    # 1) 지금 방식: 위치에서 형상 매칭으로 유도
    p = fit.project_ls(x0, cache, fac)
    _, _, Fl, _ = fit.lift(p, torch.zeros_like(p), cache)
    cur.append(mw((Fl - F0).pow(2).sum(-1)))
    # 2) F 를 앵커 상태로: 최소제곱으로 가장 좋은 Fa 를 구해 되돌린다
    Fa = torch.cholesky_solve(W.t() @ F0, Lw)          # [M,9]
    F_hat = W @ Fa                                      # [N,9]
    best.append(mw((F_hat - F0).pow(2).sum(-1)))
    # 3) 기준선: 아예 항등원으로 둘 때
    I9 = torch.eye(3, device=dev).reshape(1, 9).expand_as(F0)
    ident.append(mw((I9 - F0).pow(2).sum(-1)))

n = len(cur)
a, b, c = sum(cur)/n, sum(best)/n, sum(ident)/n
print(f"\n{'F 를 만드는 방식':>34} {'원시 mwRMS':>13} {'항등원 대비':>12}")
print(f"{'지금: 위치에서 형상 매칭 (상태=p, v)':>34} {a:12.4f} {100*a/c:11.1f}%")
print(f"{'앵커가 F 를 상태로 (최소제곱 상한)':>34} {b:12.4f} {100*b/c:11.1f}%")
print(f"{'아무것도 안 함 (F = I)':>34} {c:12.4f} {100.0:11.1f}%")
print(f"\n앵커 상태에 F 를 넣으면 오차가 {a/max(b,1e-12):.1f} 배 줄어든다.")
print("\nF_STATE_DONE")

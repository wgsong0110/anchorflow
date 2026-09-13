"""기하 피팅의 기울기가 목적함수의 하강 방향인가?

고정 창 목적함수가 40 iteration 동안 0.0232 -> 0.0223 로 사실상 평평했는데, 파라미터는
앵커 간격의 0.166 배까지 단조로 움직였고 기울기 방향도 일관됐다(직전과의 코사인
0.23~0.38). "움직이는데 안 내려간다" 는 기울기가 이 목적함수의 하강 방향이 아닐 때
나오는 그림이다.

구조적으로 그럴 이유가 있다. 창이 출발하는 초기 앵커 상태는

    p = project_ls(x0)        # anchor_sparse.py:1202, @torch.no_grad()
    v = project_v_ls(v0)      # 1214, 마찬가지

로 만들어지고 이 셋(ls_factor 포함)은 전부 no_grad 다. 앵커를 움직이면 (1) 상태가
접히는 방식과 (2) 거기서 굴러가고 펴지는 방식이 함께 바뀌는데, 역전파는 (2) 만 본다.
즉 부분 도함수로 내려가려 한다.

두 가지로 판정한다.

  선 탐색     g 를 따라 여러 보폭으로 옮겨 고정 창 목적함수를 잰다. 어떤 보폭에서도
              내려가지 않으면 g 는 하강 방향이 아니다.
  유한차분    파라미터를 직접 흔들어 목적함수 변화를 재고 g 와 비교한다. 부호나 크기가
              어긋나면 기울기가 불완전하다는 직접 증거다.
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
ap.add_argument("--n_win", type=int, default=6, help="고정 창 개수")
ap.add_argument("--unroll", type=int, default=12)
ap.add_argument("--dt_mult", type=int, default=40)
ap.add_argument("--frames", type=int, default=30)
ap.add_argument("--grid_lim", type=float, default=2.0)
ap.add_argument("--fd_n", type=int, default=6, help="유한차분으로 볼 좌표 수")
ap.add_argument("--fd_eps", type=float, default=1e-3)
args = ap.parse_args()

sys.path.insert(0, args.dreamphysics)
import warp as wp

dev = "cuda"
wp.init()
from anchorflow.anchor_sparse import AnchorSparse, Traj
sys.modules["__main__"].Traj = Traj

sc = scene_setup.build(args.ply, args.config, 512, 8, device=dev,
                       frozen_weights=True, rot_fallback=True, eig_floor=0.02)
fit = AnchorSparse(sc, c=0.25, eig_floor=0.02).to(dev)
fit.init_from_geometry()
blob = torch.load(args.traj_cache, map_location=dev, weights_only=False)
FIT = blob["fit"]
MASS = sc.volume[sc.keep].clone()
print(f"[setup] 앵커 {fit.M}, 궤적 {len(FIT)}, 창 {args.n_win}개 x {args.unroll}프레임",
      flush=True)

g_ = torch.Generator(device="cpu"); g_.manual_seed(20260905)
hi = max(1, args.frames - args.unroll)
WIN = [(int(torch.randint(len(FIT), (1,), generator=g_).item()),
        int(torch.randint(hi, (1,), generator=g_).item())) for _ in range(args.n_win)]


def objective(create_graph=False):
    """고정 창의 위치 항. 학습 루프의 loss 와 같은 정의(mwrmsd), 표본만 고정."""
    cache = fit.prepare()
    fac = fit.ls_factor(cache)
    tot, n = 0.0, 0
    for i, t in WIN:
        X, V = FIT[i][0], FIT[i][1]
        fit.reset_carried()
        p = fit.project_ls(X[t], cache, fac)
        v = fit.project_v_ls(V[t], cache, fac)
        loss = 0.0
        m = min(args.unroll, X.shape[0] - 1 - t)
        for j in range(m):
            p, v, _ = fit.rollout(p, v, args.dt_mult, cache)
            got = fit.gaussian_pos(p, cache)
            e2 = (got - X[t + j + 1]).pow(2).sum(-1)
            loss = loss + ((MASS * e2).sum() / MASS.sum()).sqrt() / args.grid_lim
        if torch.isfinite(loss):
            tot = tot + loss / m; n += 1
    return (tot / n) if n else None


PAR = ["pos", "log_s", "quat", "log_amp", "log_k"]
L0 = objective()
print(f"\n[기준] 고정 창 목적함수 {float(L0):.6f}", flush=True)
fit.zero_grad(set_to_none=True)
L0.backward()
G = {n: getattr(fit, n).grad.detach().clone() for n in PAR
     if getattr(fit, n).grad is not None}
gn = {n: float(g.norm()) for n, g in G.items()}
print("[기울기] " + ", ".join(f"|g_{n}| {v:.3e}" for n, v in gn.items()), flush=True)

base = {n: getattr(fit, n).detach().clone() for n in PAR}
gsq = sum(float((G[n] ** 2).sum()) for n in G)
print(f"[기울기] ‖g‖² = {gsq:.4e}  -- 1차 예측 감소량 = α·‖g‖²\n", flush=True)

print(f"{'보폭 α':>10} {'목적함수':>12} {'변화':>12} {'1차 예측':>12} {'비율':>8}")
for a in (1e-5, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1):
    with torch.no_grad():
        for n in PAR:
            getattr(fit, n).copy_(base[n] - a * G[n] if n in G else base[n])
    with torch.no_grad():
        La = objective()
    d = float(La) - float(L0) if La is not None else float("nan")
    pred = -a * gsq
    print(f"{a:10.1e} {float(La):11.6f} {d:+11.6f} {pred:+11.6f} "
          f"{(d/pred if pred else float('nan')):7.2f}", flush=True)
with torch.no_grad():
    for n in PAR:
        getattr(fit, n).copy_(base[n])

print(f"\n[유한차분] 좌표 {args.fd_n}개, eps={args.fd_eps:g}")
print(f"{'파라미터':>10} {'해석적 g':>13} {'유한차분':>13} {'비율':>8}")
torch.manual_seed(0)
for n in ("pos", "log_s"):
    if n not in G: continue
    flat = getattr(fit, n).detach().reshape(-1)
    idx = torch.randperm(flat.numel())[: args.fd_n]
    for k in idx.tolist():
        with torch.no_grad():
            getattr(fit, n).reshape(-1)[k] = base[n].reshape(-1)[k] + args.fd_eps
            Lp = objective()
            getattr(fit, n).reshape(-1)[k] = base[n].reshape(-1)[k] - args.fd_eps
            Lm = objective()
            getattr(fit, n).reshape(-1)[k] = base[n].reshape(-1)[k]
        fd = (float(Lp) - float(Lm)) / (2 * args.fd_eps)
        an = float(G[n].reshape(-1)[k])
        print(f"{n+'['+str(k)+']':>10} {an:12.5e} {fd:12.5e} "
              f"{(an/fd if abs(fd) > 1e-12 else float('nan')):7.2f}", flush=True)
print("\nGRAD_PROBE_DONE")

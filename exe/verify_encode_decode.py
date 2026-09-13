"""앵커 <-> MPM 두 방향이 서로의 역인가.

  decode   앵커 상태 -> 가우시안/입자 위치   (gaussian_pos, lift)
  encode   입자 -> 앵커 상태                 (project_ls 최소제곱 역, project 가중평균)

검사 넷. 앞의 셋은 통과해야 하는 항등식이고, 마지막은 두 인코더를 같은 자로 견준다.

  1 정지    project(Xc) 는 canonical 앵커를 그대로 돌려줘야 한다. 이걸 못 하면
            나머지가 아무리 좋아도 틀린 것이다(probe_encoder 의 지적).
  2 왕복    표현 가능한 앵커 상태 p 를 decode 했다가 encode 하면 p 가 나와야 한다.
            decode 의 상 위에 있는 점이므로 정확한 역이라면 오차 0 이다.
  3 선형부  decode 는 cc(p) + F(p)(X-rc) 이고 F 는 극분해를 거쳐 p 에 비선형이다.
            project_ls 는 선형부 C 로 세운 최소제곱이므로, 비선형 항이 얼마나
            남는지가 곧 이 역의 한계다. F 를 고정하고 재면 그 몫이 분리된다.
  4 속도    project_v_ls 가 속도 decode 의 역인가. 위치와 달리 상수항이 없다.
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
ap.add_argument("--fit", default=None, help="피팅된 앵커 집합. 없으면 샘플링 집합")
ap.add_argument("--n_case", type=int, default=4)
ap.add_argument("--dt_mult", type=int, default=40)
ap.add_argument("--frames", type=int, default=12)
args = ap.parse_args()

sys.path.insert(0, args.dreamphysics)
import warp as wp

dev = "cuda"
torch.set_grad_enabled(False)
wp.init()
from anchorflow.anchor_sparse import AnchorSparse, load_fitted

sc0 = scene_setup.build(args.ply, args.config, 512, 8, device=dev,
                        frozen_weights=True, rot_fallback=True, eig_floor=0.02)
if args.fit:
    scw, _ = load_fitted(sc0, args.fit, dev)
    fit, cache = scw.fit, scw._cache
else:
    fit = AnchorSparse(sc0, c=0.25, eig_floor=0.02).to(dev)
    fit.init_from_geometry()
    cache = fit.prepare()
fac = fit.ls_factor(cache)
AC = fit.pos.clone()
free = ~fit.fixed
print(f"[setup] 앵커 {fit.M} (자유 {int(free.sum())}), 물질 가우시안 {fit.N}, "
      f"기하 {'피팅됨' if args.fit else '샘플링(피팅 전)'}", flush=True)
h = sc0.sim.radius


def rel_p(a, b):
    """앵커 오차를 앵커 간격으로 잰다."""
    return float((a - b)[free].norm(dim=-1).mean() / h)


print("\n=== 1. 정지 상태 항등식 ===")
x_rest = fit.gaussian_pos(AC, cache)
for name, enc in (("project_ls", lambda x: fit.project_ls(x, cache, fac)),
                  ("project(가중평균)", lambda x: fit.project(x, cache))):
    p_hat = enc(x_rest)
    print(f"  {name:18} ‖p̂ − AC‖/h = {rel_p(p_hat, AC):.3e}")

print("\n=== 2. 표현 가능한 상태의 왕복 ===")
print(f"  {'섭동':>10} {'project_ls':>14} {'project(평균)':>16}")
g = torch.Generator(device=dev); g.manual_seed(3)
for scale in (0.05, 0.2, 0.5, 1.0):
    d = torch.randn(fit.M, 3, device=dev, generator=g) * (scale * h)
    d[fit.fixed] = 0
    p = AC + d
    x = fit.gaussian_pos(p, cache)
    a = rel_p(fit.project_ls(x, cache, fac), p)
    b = rel_p(fit.project(x, cache), p)
    print(f"  {scale:9.2f}h {a:13.3e} {b:15.3e}")

print("\n=== 3. 시뮬레이터가 실제로 도달한 상태의 왕복 ===")
BASE = torch.tensor([-0.477, 0.0, 0.0], device=dev)
print(f"  {'프레임':>7} {'변위/h':>9} {'project_ls':>13} {'project(평균)':>16}")
for c in range(args.n_case):
    g.manual_seed(11 + c)
    uk = torch.rand(1, device=dev, generator=g).item()
    kk = max(1, int(round(32 ** uk)))
    ur = torch.rand(1, device=dev, generator=g).item()
    lo, hi = h * 0.125, sc0.extent
    rad = lo * ((hi / lo) ** ur)
    f = sc0.random_multi_poke(g, kk, rad, float(BASE.norm()))
    # 임펄스가 만드는 입자 속도장을 앵커로 접어 출발한다 -- 평가와 같은 방식
    dv = fit.impulse_dv(f, cache)
    vg = torch.zeros(fit.N, 3, device=dev).index_add_(
        0, fit.pair_g, cache[0].unsqueeze(-1) * dv[fit.pair_a])
    p, v = AC.clone(), fit.project_v_ls(vg, cache, fac)
    ok = True
    for t in range(args.frames):
        p, v, _ = fit.rollout(p, v, args.dt_mult, cache)
        if not torch.isfinite(p).all(): ok = False; break
    if not ok:
        print(f"  case{c}: 발산"); continue
    x = fit.gaussian_pos(p, cache)
    print(f"  {args.frames:7d} {float((p-AC)[free].norm(dim=-1).mean()/h):8.3f} "
          f"{rel_p(fit.project_ls(x, cache, fac), p):12.3e} "
          f"{rel_p(fit.project(x, cache), p):15.3e}")

print("\n=== 4. 속도 왕복 ===")
# 위치 decode 는 x = C·p + b 로 p 에 선형이고(cc 도 F 도 선형), 속도는 그 도함수라
# vx = C·v 다 -- lift 의 vc + Fdot(X-rc) 가 바로 그것이다. 앞선 검사는 vc 만 써서
# Fdot 항을 빠뜨렸고, 부분 decode 에 전체 역을 적용해 상대오차 1.0 이 나왔다.
print(f"  {'크기':>10} {'project_v_ls':>15} {'(vc 만, 잘못된 decode)':>24}")
w0, rc, q, Binv, blocked, _ = cache
for scale in (0.1, 1.0):
    vv = torch.randn(fit.M, 3, device=dev, generator=g) * scale
    vv[fit.fixed] = 0
    _, vx, _, _ = fit.lift(AC, vv, cache)          # 올바른 속도 decode
    a = float((fit.project_v_ls(vx, cache, fac) - vv)[free].norm(dim=-1).mean()
              / max(scale, 1e-9))
    vg = torch.zeros(fit.N, 3, device=dev).index_add_(
        0, fit.pair_g, w0.unsqueeze(-1) * vv[fit.pair_a])
    b_ = float((fit.project_v_ls(vg, cache, fac) - vv)[free].norm(dim=-1).mean()
               / max(scale, 1e-9))
    print(f"  {scale:9.2f} {a:14.3e} {b_:23.3e}")
print("\nVERIFY_DONE")

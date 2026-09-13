"""MPM 의 서브스텝 40 이 충분한지 잰다.

두 가지를 본다.

  1. CFL 수 -- 한 서브스텝에 입자가 격자 한 칸의 몇 배를 움직이는가.
     dx = grid_lim / n_grid, CFL = max|v| * sub_dt / dx. 통상 0.3~0.5 아래로 둔다.
  2. 수렴 -- 같은 초기조건을 40 / 80 / 160 서브스텝으로 굴려 최종 상태를 견준다.
     40 이 충분하면 두 배로 쪼개도 결과가 거의 안 변해야 한다.

40 을 두 번 굴린 차이(재현성 바닥)를 함께 내야 판단이 선다 -- warp 의 원자적 덧셈이
비결정적이라 같은 코드로도 갈라지고, 그 아래 차이는 읽을 수 없다.
"""
from __future__ import annotations

import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
import torch
from anchorflow import scene_setup

ap = argparse.ArgumentParser()
ap.add_argument("--ply", required=True); ap.add_argument("--config", required=True)
ap.add_argument("--dreamphysics", default="/workspace/DreamPhysics")
ap.add_argument("--n_imp", type=int, default=6)
ap.add_argument("--frames", type=int, default=30)
ap.add_argument("--dt_mult", type=int, default=40)
ap.add_argument("--mults", default="1,2,4", help="서브스텝 배수")
ap.add_argument("--mp_kmax", type=int, default=32); ap.add_argument("--mp_rmin", type=float, default=0.125)
ap.add_argument("--impulse_range", type=float, default=16.0)
ap.add_argument("--n_grid", type=int, default=100); ap.add_argument("--grid_lim", type=float, default=2.0)
ap.add_argument("--seed", type=int, default=4242)
args = ap.parse_args()

sys.path.insert(0, args.dreamphysics)
import warp as wp
dev = "cuda"; torch.set_grad_enabled(False); wp.init()
from anchorflow.anchor_sparse import AnchorSparse
from anchorflow.mpm_teacher import MPMTeacher

sc = scene_setup.build(args.ply, args.config, 512, 8, device=dev,
                       frozen_weights=True, rot_fallback=True, eig_floor=0.02)
fit = AnchorSparse(sc, c=0.25, eig_floor=0.02).to(dev); fit.init_from_geometry()
cache = fit.prepare()
T = MPMTeacher(sc)
DX = args.grid_lim / args.n_grid
MASS = sc.volume[sc.keep].clone()
print(f"[setup] 격자 칸 dx = {DX:.5f}, 기준 sub_dt = {sc.sub_dt:.3g}, "
      f"CFL=1 이 되는 속도 = {DX/sc.sub_dt:.1f}", flush=True)

base = None
for bc in sc.cfg.get("boundary_conditions", []):
    if "force" in bc:
        base = torch.tensor(bc["force"], device=dev); break
if base is None:
    base = torch.tensor([0.0, 0.0, 1.0], device=dev)


def draw(g):
    uk = torch.rand(1, device=dev, generator=g).item()
    kk = max(1, int(round(args.mp_kmax ** uk)))
    ur = torch.rand(1, device=dev, generator=g).item()
    lo, hi = sc.sim.radius * args.mp_rmin, sc.extent
    rad = lo * ((hi / lo) ** ur)
    us = torch.rand(1, device=dev, generator=g).item()
    mag = base.norm().item() * 0.5 * (args.impulse_range ** us)
    return sc.random_multi_poke(g, kk, rad, mag), kk, rad


@torch.no_grad()
def run(force, mult):
    """서브스텝을 mult 배로 쪼개 같은 물리 시간을 굴린다. 최종 위치와 최대 CFL."""
    dv = fit.impulse_dv(force, cache)
    v0 = torch.zeros(fit.N, 3, device=dev).index_add_(
        0, fit.pair_g, cache[0].unsqueeze(-1) * dv[fit.pair_a])
    T._set(T.pos_m.clone(), v0.contiguous(), T.eye.clone(), torch.zeros_like(T.eye))
    sub = sc.sub_dt / mult
    n = args.dt_mult * mult
    vmax = float(v0.norm(dim=-1).max())
    for _ in range(args.frames):
        for k in range(n):
            T.solver.p2g2p(None, sub, device=T.wp_dev)
            if (k + 1) % 8 == 0:
                if not T._in_domain():
                    return None, None
                vmax = max(vmax, float(T.solver.export_particle_v_to_torch().norm(dim=-1).max()))
    return T.solver.export_particle_x_to_torch().clone(), vmax * sub / DX


def rel(a, b, x0):
    """질량가중 RMS 차이를, MPM 이 그 궤적에서 실제로 움직인 거리로 나눈 값"""
    d = ((MASS * (a - b).pow(2).sum(-1)).sum() / MASS.sum()).sqrt()
    span = (b - x0).norm(dim=-1).max().clamp(min=1e-12)
    return float(d / span) * 100


MU = [int(x) for x in args.mults.split(",")]
g = torch.Generator(device=dev); g.manual_seed(args.seed)
X0 = T.pos_m.clone()
print(f"\n{'#':>3}{'K':>4}{'r/spacing':>11}{'CFL@40':>9}"
      + "".join(f"{'40 vs ' + str(m) + 'x':>12}" for m in MU[1:])
      + f"{'40 재현성':>11}")
rows = []
for i in range(args.n_imp):
    f, kk, rad = draw(g)
    xs, cfl = {}, None
    ok = True
    for m in MU:
        x, c = run(f, m)
        if x is None: ok = False; break
        xs[m] = x
        if m == 1: cfl = c
    if not ok:
        print(f"{i:>3}{kk:>4}{rad/sc.sim.radius:>11.2f}   (도메인 이탈)", flush=True)
        continue
    x40b, _ = run(f, 1)                       # 같은 설정 두 번째 -- 재현성 바닥
    ds = [rel(xs[1], xs[m], X0) for m in MU[1:]]
    dr = rel(xs[1], x40b, X0)
    rows.append((cfl, ds, dr))
    print(f"{i:>3}{kk:>4}{rad/sc.sim.radius:>11.2f}{cfl:>9.3f}"
          + "".join(f"{d:>12.2f}" for d in ds) + f"{dr:>11.2f}", flush=True)

if rows:
    n = len(rows)
    print(f"\n평균  CFL {sum(r[0] for r in rows)/n:.3f}  최대 {max(r[0] for r in rows):.3f}")
    for j, m in enumerate(MU[1:]):
        print(f"  40 대 {m}x 배 세분: 평균 {sum(r[1][j] for r in rows)/n:.2f}%  "
              f"최대 {max(r[1][j] for r in rows):.2f}%")
    print(f"  40 재현성 바닥      : 평균 {sum(r[2] for r in rows)/n:.2f}%  "
          f"최대 {max(r[2] for r in rows):.2f}%")
print("\nCFL_DONE")

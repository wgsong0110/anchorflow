"""여러 씬에서 표현 하한과 MPM 재현성을 잰다 -- 학습 없이.

학생은 특정 앵커 집합 전용 스테퍼라 씬을 넘나들 수 없다. 하지만 두 가지는 학습 없이
잴 수 있고, ficus 에서 얻은 결론이 그 씬만의 것인지 가른다.

  표현 하한   MPM 궤적을 프레임마다 독립으로 앵커에 접었다 편 잔차. 512 개 앵커가
              그 씬의 변형을 담을 수 있는지. ficus 에서는 0.29%(고정 스케일) 로
              학생 오차 1.6~1.8% 의 1/6 이었다 -- 병목이 표현이 아니라는 근거.
  재현성      같은 코드로 같은 초기 조건을 두 번 굴린 차이. warp 의 원자적 덧셈이
              순서 비결정적이라 2400 서브스텝에 걸쳐 자란다. ficus 에서 2.49%
              (자체 변위 기준) 였고, 그 아래 차이는 원리적으로 읽을 수 없다.

임펄스는 학습·평가와 같은 (K, r) 계열에서 뽑는다. 앵커는 피팅하지 않은 샘플링 집합
이므로, 여기서 나오는 하한은 "피팅 전 512 앵커" 의 것이다.
"""
from __future__ import annotations

import argparse
import json
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
ap.add_argument("--name", default="scene")
ap.add_argument("--n_imp", type=int, default=6)
ap.add_argument("--frames", type=int, default=60)
ap.add_argument("--dt_mult", type=int, default=40)
ap.add_argument("--seed", type=int, default=777)
ap.add_argument("--grid_lim", type=float, default=2.0)
ap.add_argument("--mp_kmax", type=int, default=32)
ap.add_argument("--mp_rmin", type=float, default=0.125)
ap.add_argument("--out", default=None)
args = ap.parse_args()

sys.path.insert(0, args.dreamphysics)
import warp as wp

dev = "cuda"
torch.set_grad_enabled(False)
wp.init()
from anchorflow.anchor_sparse import AnchorSparse
from anchorflow.mpm_teacher import MPMTeacher

sc = scene_setup.build(args.ply, args.config, 512, 8, device=dev,
                       frozen_weights=True, rot_fallback=True, eig_floor=0.02)
fit = AnchorSparse(sc, c=0.25, eig_floor=0.02).to(dev)
cache = fit.prepare()
fac = fit.ls_factor(cache)
T = MPMTeacher(sc)
X0 = T.pos_m.clone()
base = sc.cfg.get("particle_impulse") or [-0.477, 0.0, 0.0]
if isinstance(base, dict):
    base = base.get("force", [-0.477, 0.0, 0.0])
BASE = torch.tensor([float(x) for x in (base[:3] if isinstance(base, (list, tuple))
                                        else [-0.477, 0, 0])], device=dev)
if float(BASE.norm()) < 1e-9:
    BASE = torch.tensor([-0.477, 0.0, 0.0], device=dev)
print(f"[{args.name}] 입자 {T.n}, 앵커 {fit.M}, 물체 크기 {sc.extent:.4f}, "
      f"앵커 간격 {sc.sim.radius:.4f}, 기준 힘 {float(BASE.norm()):.4f}", flush=True)


def mpm(v0):
    T._set(X0.clone(), v0, T.eye.clone(), torch.zeros_like(T.eye))
    xs = [X0.clone()]
    for _ in range(args.frames):
        for k in range(args.dt_mult):
            T.solver.p2g2p(None, sc.sub_dt, device=T.wp_dev)
            if (k + 1) % 4 == 0 and not T._in_domain():
                return None
        xs.append(T.solver.export_particle_x_to_torch().clone())
    return torch.stack(xs)


rows = []
g = torch.Generator(device=dev)
for i in range(args.n_imp):
    g.manual_seed(args.seed + 1000 * i)
    uk = torch.rand(1, device=dev, generator=g).item()
    kk = max(1, int(round(args.mp_kmax ** uk)))
    ur = torch.rand(1, device=dev, generator=g).item()
    lo, hi = sc.sim.radius * args.mp_rmin, sc.extent
    rad = lo * ((hi / lo) ** ur)
    f = sc.random_multi_poke(g, kk, rad, float(BASE.norm()))
    dv = sc.impulse_dv(f)
    v0 = (T.w.unsqueeze(-1) * dv[T.idx]).sum(1).contiguous()
    ref = mpm(v0)
    if ref is None:
        print(f"  imp{i}: K={kk} r={rad/sc.sim.radius:.2f}x  MPM 이탈", flush=True)
        continue
    span = float((ref - ref[0]).norm(dim=-1).max())
    fl = torch.stack([fit.gaussian_pos(fit.project_ls(ref[t], cache, fac), cache)
                      for t in range(ref.shape[0])])
    d_fl = float((fl - ref).norm(dim=-1).mean())
    ref2 = mpm(v0)
    d_rp = float("nan") if ref2 is None else float((ref2 - ref).norm(dim=-1).mean())
    rows.append((kk, rad / sc.sim.radius, span, d_fl, d_rp))
    print(f"  imp{i}: K={kk:2d} r={rad/sc.sim.radius:6.2f}x  변위 {span:.4f}  "
          f"하한 {100*d_fl/args.grid_lim:.3f}%(고정)/{100*d_fl/max(span,1e-12):.2f}%(자체)  "
          f"재현 {100*d_rp/args.grid_lim:.3f}%(고정)/{100*d_rp/max(span,1e-12):.2f}%(자체)",
          flush=True)
    del ref, ref2, fl

if rows:
    n = len(rows)
    mf = sum(100 * r[3] / args.grid_lim for r in rows) / n
    ms = sum(100 * r[3] / max(r[2], 1e-12) for r in rows) / n
    rf = sum(100 * r[4] / args.grid_lim for r in rows if r[4] == r[4]) / n
    rs = sum(100 * r[4] / max(r[2], 1e-12) for r in rows if r[4] == r[4]) / n
    print(f"\n[{args.name}] 유효 {n}/{args.n_imp}  |  표현 하한 {mf:.3f}%(고정) "
          f"{ms:.2f}%(자체)  |  재현성 {rf:.3f}%(고정) {rs:.2f}%(자체)", flush=True)
    if args.out:
        json.dump({"name": args.name, "rows": rows, "floor_fixed": mf,
                   "floor_span": ms, "repro_fixed": rf, "repro_span": rs,
                   "extent": float(sc.extent), "spacing": float(sc.sim.radius),
                   "n_particles": int(T.n), "M": int(fit.M)}, open(args.out, "w"), indent=1)
print("SCENE_FLOOR_DONE")

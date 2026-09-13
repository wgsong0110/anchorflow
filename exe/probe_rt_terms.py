"""왕복 손실 세 항(x, v, F)의 크기를 따로 재서 견준다.

각 항은 "그 오차가 유발하는 위치 오차 / grid_lim" 으로 환산된다:

    x  ->  mwRMS(dx)                 / grid_lim
    v  ->  mwRMS(dv) · dt_c          / grid_lim
    F  ->  mwRMS(dF) · r_ref         / grid_lim

원시 크기와 환산 후 크기, 그리고 합에서 차지하는 비중을 함께 낸다. 한 항이 다른 항을
압도하면 그 항만 최적화되므로, 가중(--rt_w)을 정하려면 이 표가 먼저 있어야 한다.
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
ap.add_argument("--fit", default=None)
ap.add_argument("--n_win", type=int, default=8)
ap.add_argument("--dt_mult", type=int, default=40)
ap.add_argument("--frames", type=int, default=30)
ap.add_argument("--grid_lim", type=float, default=2.0)
args = ap.parse_args()

sys.path.insert(0, args.dreamphysics)
import warp as wp

dev = "cuda"
torch.set_grad_enabled(False)
wp.init()
from anchorflow.anchor_sparse import AnchorSparse, Traj, load_fitted
sys.modules["__main__"].Traj = Traj

sc = scene_setup.build(args.ply, args.config, 512, 8, device=dev,
                       frozen_weights=True, rot_fallback=True, eig_floor=0.02)
if args.fit:
    scw, _ = load_fitted(sc, args.fit, dev)
    fit, cache = scw.fit, scw._cache
else:
    fit = AnchorSparse(sc, c=0.25, eig_floor=0.02).to(dev)
    fit.init_from_geometry()
    cache = fit.prepare()
fac = fit.ls_factor(cache)
MASS = sc.volume[sc.keep].clone()
DT_C = args.dt_mult * sc.sub_dt
R_REF = float((fit.Xc - cache[1]).norm(dim=-1).pow(2).mean().sqrt())
print(f"[setup] 앵커 {fit.M}, dt_c {DT_C:.5g}, r_ref {R_REF:.5g}, "
      f"grid_lim {args.grid_lim:g}, 기하 {'피팅' if args.fit else '샘플링'}", flush=True)

blob = torch.load(args.traj_cache, map_location=dev, weights_only=False)
FIT = blob["fit"]
has_F = len(FIT[0]) == 4
print(f"[data] 궤적 {len(FIT)}, F 저장됨 {has_F}", flush=True)
if not has_F:
    raise SystemExit("F 가 없는 캐시다 -- --n_fine 으로 저장해야 한다")

def mw(e2):
    return float(((MASS * e2).sum() / MASS.sum()).sqrt())

g = torch.Generator(device="cpu"); g.manual_seed(20260905)
hi = max(1, args.frames - 12)
rows = []
for _ in range(args.n_win):
    i = int(torch.randint(len(FIT), (1,), generator=g).item())
    t = int(torch.randint(hi, (1,), generator=g).item())
    ent = FIT[i]
    x0, v0 = ent[0][t], ent[1][t]
    F0 = ent[2][t].reshape(-1, 9).float()
    p = fit.project_ls(x0, cache, fac)
    v = fit.project_v_ls(v0, cache, fac)
    xl, vl, Fl, _ = fit.lift(p, v, cache)
    rows.append((mw((xl - x0).pow(2).sum(-1)),
                 mw((vl - v0).pow(2).sum(-1)),
                 mw((Fl - F0).pow(2).sum(-1)),
                 float(x0.norm(dim=-1).mean()), float(v0.norm(dim=-1).mean())))

n = len(rows)
rx = sum(r[0] for r in rows) / n
rv = sum(r[1] for r in rows) / n
rF = sum(r[2] for r in rows) / n
cx = rx / args.grid_lim
cv = rv * DT_C / args.grid_lim
cF = rF * R_REF / args.grid_lim
tot = cx + cv + cF
print(f"\n{'항':>4} {'원시 mwRMS':>14} {'단위':>10} {'환산 계수':>14} "
      f"{'손실 기여':>14} {'비중':>8}")
print(f"{'x':>4} {rx:13.5e} {'길이':>10} {1/args.grid_lim:13.5e} {cx:13.5e} "
      f"{100*cx/tot:7.1f}%")
print(f"{'v':>4} {rv:13.5e} {'길이/시간':>10} {DT_C/args.grid_lim:13.5e} {cv:13.5e} "
      f"{100*cv/tot:7.1f}%")
print(f"{'F':>4} {rF:13.5e} {'무차원':>10} {R_REF/args.grid_lim:13.5e} {cF:13.5e} "
      f"{100*cF/tot:7.1f}%")
print(f"{'합':>4} {'':>13} {'':>10} {'':>13} {tot:13.5e} {100.0:7.1f}%")
print(f"\n참고: MPM 자체 크기 -- |x| 평균 {sum(r[3] for r in rows)/n:.4f}, "
      f"|v| 평균 {sum(r[4] for r in rows)/n:.4f}")
print("\nRT_TERMS_DONE")

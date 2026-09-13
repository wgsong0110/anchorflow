"""정답을 한 번 만들어 저장하고, A~D 를 같은 자로 채점한다.

i-PhysGaussian 과 우리는 mpm_solver_warp 사본이 다르고, 그 둘은 물리적으로 동등하지
않다 -- DreamPhysics 쪽 커널은 mu/lam 을 만들 때 E 에 1e7 을 곱하고 i-PG 쪽은 곱하지
않으며, 그 밖에도 차이가 있어 같은 임펄스가 한쪽에서만 도메인을 벗어난다. 그래서 한
프로세스에서 두 사본을 섞지 않는다. 여기서 **정답을 확정해 디스크에 남기고**,
지표를 둘 다 낸다.

  고정      mean_t mean_i ||pred - ref|| / grid_lim      <- 기본. i-PhysGaussian 이 쓰는
            정규화이고, 논문이 이유를 밝혀 두었다: 거의 정지한 구간에서 나눗셈이
            불안정해지는 것을 피하려고.
  자체변위  ... / (그 궤적의 MPM 자체 최대 변위)          <- 이전 격자가 쓰던 것

자체 변위로 나누면 거의 안 움직인 궤적에서 값이 폭발한다. 실측: 같은 12 칸 안에서
MPM 최대 변위가 0.021 ~ 1.004 로 48 배 벌어지고(전체 격자로는 150 배), log 변위와
log 오차의 상관이 네 모델 모두 -0.88 ~ -0.93 이었다. K=1 과 작은 반경이 취약해
보이던 것의 상당 부분이 물리가 아니라 이 분모였다. K=1, r=0.75x 에서는 MPM 최대
변위가 0.0006 이라 오차가 458% 로 찍혔다.

i-PG 는 별도 프로세스에서 그 정답을 불러와 같은 지표로 채점한다(exe/eval_ipg_rows.py).

임펄스 시드는 (K, r) 격자의 인덱스로 매기므로, 같은 --ks/--rs 를 주면 A~D 격자와
완전히 같은 임펄스가 재현된다.
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
ap.add_argument("--pair", action="append", default=[], help="이름:기하.pt:학생.pt")
ap.add_argument("--dump", required=True, help="정답을 남길 디렉토리")
ap.add_argument("--cells", required=True, help="K:R,K:R,...")
ap.add_argument("--draws", type=int, default=2)
ap.add_argument("--frames", type=int, default=60)
ap.add_argument("--dt_mult", type=int, default=40)
ap.add_argument("--seed", type=int, default=777)
ap.add_argument("--mag", type=float, default=1.0)
ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
ap.add_argument("--rs", type=float, nargs="+",
                 default=[0.125, 0.25, 0.5, 0.75, 1.0, 2.0, 5.0, 10.0])
ap.add_argument("--grid_lim", type=float, default=2.0)
ap.add_argument("--out", default=None)
args = ap.parse_args()

sys.path.insert(0, args.dreamphysics)
import warp as wp

dev = "cuda"
torch.set_grad_enabled(False)
wp.init()
from anchorflow.anchor_sparse import load_fitted
from anchorflow.mpm_teacher import MPMTeacher
from anchorflow.nextstate import apply_step, net_from_ckpt

os.makedirs(args.dump, exist_ok=True)
sc0 = scene_setup.build(args.ply, args.config, 512, 8, device=dev,
                        frozen_weights=True, rot_fallback=True, eig_floor=0.02)
T = MPMTeacher(sc0)
X0 = T.pos_m.clone()
BASE = torch.tensor([-0.477, 0.0, 0.0], device=dev)
spacing = sc0.sim.radius
dt_c = args.dt_mult * sc0.sub_dt

GEOM, RUNS = {}, []
for spec in args.pair:
    name, fitpt, ckpt = spec.split(":")
    if fitpt not in GEOM:
        sc, _ = load_fitted(sc0, fitpt, dev)
        GEOM[fitpt] = (sc, sc.fit, sc._cache)
    net = net_from_ckpt(torch.load(ckpt, map_location=dev, weights_only=False), dev).eval()
    RUNS.append((name, fitpt, net))
ROWS = [f"floor({os.path.basename(f).replace('.pt','')})" for f in GEOM] + [n for n, _, _ in RUNS]
want = set(args.cells.split(","))
print(f"[setup] 칸 {len(want)}, 칸당 {args.draws} draw, {args.frames}프레임, "
      f"덤프 -> {args.dump}", flush=True)


def truth(v0):
    T._set(X0.clone(), v0, T.eye.clone(), torch.zeros_like(T.eye))
    xs = [X0.clone()]
    for _ in range(args.frames):
        for _sub in range(args.dt_mult):
            T.solver.p2g2p(None, sc0.sub_dt, device=T.wp_dev)
            if (_sub + 1) % 4 == 0 and not T._in_domain():
                return None
        xs.append(T.solver.export_particle_x_to_torch().clone())
    return torch.stack(xs)


def floor_of(ref, fitpt):
    sc, fit, cache = GEOM[fitpt]
    return torch.stack([fit.gaussian_pos(fit.project(ref[t], cache), cache)
                        for t in range(ref.shape[0])])


def student(fitpt, net, v0):
    sc, fit, cache = GEOM[fitpt]
    p = sc.anchor_canonical.clone()
    v = fit.project_v(v0, cache)
    out = [fit.gaussian_pos(p, cache)]
    while len(out) <= args.frames:
        bad = False
        for q, d in apply_step(net, p, v, None, dt_c, fit.fixed):
            p, v = q, d / dt_c
            if not torch.isfinite(p).all():
                bad = True; break
            out.append(fit.gaussian_pos(p, cache))
            if len(out) > args.frames: break
        if bad: break
    while len(out) <= args.frames:
        out.append(out[-1])
    return torch.stack(out[:args.frames + 1])


acc, kept, spans = {r: {} for r in ROWS}, [], {}
for ki, K in enumerate(args.ks):
    for rj, R in enumerate(args.rs):
        if f"{K}:{R:g}" not in want:
            continue
        per = {r: [] for r in ROWS}
        for dnum in range(args.draws):
            g = torch.Generator(device=dev)
            g.manual_seed(args.seed + 100000 * dnum + 1000 * ki + rj)
            f = sc0.random_multi_poke(g, K, R * spacing, BASE.norm().item() * args.mag)
            dv = sc0.impulse_dv(f)
            v0 = (T.w.unsqueeze(-1) * dv[T.idx]).sum(1).contiguous()
            ref = truth(v0)
            if ref is None:
                print(f"  K={K} r={R:g}x d{dnum}: MPM 이탈로 버림", flush=True)
                continue
            tag = f"K{K}_r{R:g}_d{dnum}"
            torch.save({"v0": v0.cpu(), "ref": ref.cpu().half(),
                        "K": K, "R": R, "draw": dnum},
                       os.path.join(args.dump, tag + ".pt"))
            kept.append(tag)
            span = float((ref - ref[0]).norm(dim=-1).max())
            spans.setdefault(f"{K}_{R}", []).append(span)
            def score(got):
                dist = float((got - ref).norm(dim=-1).mean())   # 정규화 전 평균 거리
                return (100 * dist / args.grid_lim,
                        100 * dist / max(span, 1e-12), dist)
            for fitpt in GEOM:
                key = f"floor({os.path.basename(fitpt).replace('.pt','')})"
                per[key].append(score(floor_of(ref, fitpt)))
            for name, fitpt, net in RUNS:
                per[name].append(score(student(fitpt, net, v0)))
        cell = f"{K}_{R}"
        for r in ROWS:
            if per[r]:
                acc[r][cell] = {"fixed": sum(x[0] for x in per[r]) / len(per[r]),
                                 "span": sum(x[1] for x in per[r]) / len(per[r]),
                                 "dist": sum(x[2] for x in per[r]) / len(per[r])}
            else:
                acc[r][cell] = None
        if per[ROWS[0]]:
            sp = sum(spans.get(cell, [0])) / max(len(spans.get(cell, [1])), 1)
            print(f"  K={K:2d} r={R:6.3f}x  변위 {sp:.4f}  " + "  ".join(
                f"{r} {acc[r][cell]['fixed']:5.2f}%(고정)/{acc[r][cell]['span']:6.1f}%(자체)"
                for r in ROWS), flush=True)

print(f"\n[덤프] {len(kept)}개 궤적 저장")
print(f"\n{'행':>16} {'고정 평균':>10} {'고정 최악':>10} {'자체 평균':>10} {'자체 최악':>10}")
for r in ROWS:
    v = [x for x in acc[r].values() if x is not None]
    fx = [x["fixed"] for x in v]; sp = [x["span"] for x in v]
    print(f"{r:>16} {sum(fx)/len(fx):9.3f}% {max(fx):9.3f}% "
          f"{sum(sp)/len(sp):9.2f}% {max(sp):9.2f}%")
if args.out:
    json.dump({"res": acc, "kept": kept, "spans": spans, "grid_lim": args.grid_lim, "draws": args.draws, "seed": args.seed,
               "frames": args.frames, "cells": args.cells}, open(args.out, "w"), indent=1)
    print(f"저장: {args.out}")
print("\nREF_DUMP_DONE")

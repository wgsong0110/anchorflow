"""(K, r) 격자에서 학생들을 한 자로 비교한다.

임펄스는 포크 개수 K 와 반경 r 로만 결정되고, 둘 다 로그균등으로 뽑히는 학습·기하와
같은 계열이다. 격자는 그 두 축을 기하급수 간격으로 훑는다.

**임펄스는 기하와 무관하게 만든다.** 초기 속도를 피팅된 앵커 집합에서 유도하면
기하가 다른 학생끼리 정답 궤적 자체가 달라져 비교가 성립하지 않는다. 그래서
eval_vs_mpm 과 같이 베이스 씬의 스키닝으로 입자 속도를 만들고, 각 학생은 그 **같은**
속도장을 자기 앵커로 접어서 출발한다.

오차는 전부 가우시안 공간에서 MPM 입자와 견준다 -- 인코더나 앵커 집합이 달라도
그 자는 변하지 않으므로, 기하가 다른 학생끼리도 직접 비교된다. 칸마다 표현 하한
(floor) 도 함께 내서, 오차 증가가 표현 탓인지 동역학 탓인지 나뉘게 한다.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_lib = os.path.join(os.path.dirname(__file__), "..", "lib")
sys.path.insert(0, _lib)

import torch
from tqdm import tqdm

from anchorflow import scene_setup

ap = argparse.ArgumentParser()
ap.add_argument("--ply", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--dreamphysics", default="/workspace/DreamPhysics")
ap.add_argument("--pair", action="append", required=True,
                 help="이름:기하.pt:학생.pt -- 여러 번 줄 수 있다. 학생마다 자기 "
                      "기하 위에서 굴러가고, 임펄스와 정답은 공유한다.")
ap.add_argument("--frames", type=int, default=60)
ap.add_argument("--dt_mult", type=int, default=40)
ap.add_argument("--seed", type=int, default=777)
ap.add_argument("--mag", type=float, default=1.0, help="기준 힘의 배수")
ap.add_argument("--n_rep", type=int, default=1,
                 help="칸마다 뽑을 임펄스 개수. 포크 중심과 방향이 무작위라 한 번만 "
                      "뽑으면 칸 사이 차이가 그 무작위성에 묻힌다 -- 평균과 함께 "
                      "표준편차도 내서 얼마나 묻히는지 보이게 한다.")
ap.add_argument("--norm", choices=("extent", "self"), default="extent",
                 help="eval_vs_mpm 과 같은 규약. 칸마다 변위가 자릿수로 다르므로 "
                      "자체 정규화를 쓰면 움직임이 거의 없는 칸이 최악으로 보인다.")
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

sc0 = scene_setup.build(args.ply, args.config, 512, 8, device=dev,
                        frozen_weights=True, rot_fallback=True, eig_floor=0.02)
T = MPMTeacher(sc0)                      # sparse 없음 -> 임펄스가 기하와 무관하다
X0 = T.pos_m.clone()
BASE = torch.tensor([-0.477, 0.0, 0.0], device=dev)
spacing = sc0.sim.radius
dt_c = args.dt_mult * sc0.sub_dt

KS = [1, 2, 4, 8, 16, 32]
RS = [0.125, 0.25, 0.5, 0.75, 1.0, 2.0, 5.0, 10.0]

# 기하는 한 번만 올린다 -- 같은 기하를 쓰는 학생끼리 공유
GEOM, RUNS = {}, []
for spec in args.pair:
    name, fitpt, ckpt = spec.split(":")
    if fitpt not in GEOM:
        sc, _ = load_fitted(sc0, fitpt, dev)
        GEOM[fitpt] = (sc, sc.fit, sc._cache)
    sc, fit, cache = GEOM[fitpt]
    net = net_from_ckpt(torch.load(ckpt, map_location=dev, weights_only=False), dev).eval()
    RUNS.append((name, fitpt, net))
EXTENT = float(sc0.extent)
print(f"[setup] 기하 {len(GEOM)}개, 학생 {len(RUNS)}개, 격자 K{KS} x r{RS}, "
      f"세기 {args.mag}x, {args.frames}프레임", flush=True)


def truth(force):
    """MPM 정답. 초기 속도는 베이스 씬에서 만들어 기하와 무관하다."""
    dv = sc0.impulse_dv(force)
    v0 = (T.w.unsqueeze(-1) * dv[T.idx]).sum(1).contiguous()
    T._set(X0.clone(), v0, T.eye.clone(), torch.zeros_like(T.eye))
    xs = [X0.clone()]
    for _ in range(args.frames):
        for _sub in range(args.dt_mult):
            T.solver.p2g2p(None, sc0.sub_dt, device=T.wp_dev)
            if (_sub + 1) % 4 == 0 and not T._in_domain():
                return None
        xs.append(T.solver.export_particle_x_to_torch().clone())
    return torch.stack(xs), v0


def floor_of(ref, fitpt):
    """그 기하가 담을 수 있는 것. 프레임마다 독립으로 접었다 편다."""
    sc, fit, cache = GEOM[fitpt]
    out = [fit.gaussian_pos(fit.project(ref[t], cache), cache)
           for t in range(ref.shape[0])]
    return torch.stack(out)


def student(name, fitpt, net, v0):
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


ROWS = [f"floor({os.path.basename(f).replace('.pt','')})" for f in GEOM] + [n for n, _, _ in RUNS]
res = {r: {} for r in ROWS}
dropped = []
sd = {r: {} for r in ROWS}
# 칸마다 프레임별 오차 곡선도 남긴다 -- 변수를 고정했을 때 오차가
# 처음부터 큰지 굴러가며 쌓이는지는 평균 하나로는 안 보인다.
curve = {r: {} for r in ROWS}
for ki, K in enumerate(KS):
    for rj, R in enumerate(RS):
        cell = f"{K}_{R}"
        acc = {r: [] for r in ROWS}
        cur = {r: [] for r in ROWS}
        nbad = 0
        for rep in range(args.n_rep):
            g = torch.Generator(device=dev)
            g.manual_seed(args.seed + 1000 * ki + 10 * rj + 100000 * rep)
            f = sc0.random_multi_poke(g, K, R * spacing, BASE.norm().item() * args.mag)
            t = truth(f)
            if t is None:
                nbad += 1
                continue
            ref, v0 = t
            ref_span = (ref - ref[0]).norm(dim=-1).max().clamp(min=1e-12)
            span = EXTENT if args.norm == "extent" else ref_span
            for fitpt in GEOM:
                key = f"floor({os.path.basename(fitpt).replace('.pt','')})"
                _e = (floor_of(ref, fitpt) - ref).norm(dim=-1).mean(-1) / span
                acc[key].append(100 * float(_e.mean()))
                cur[key].append((100 * _e).tolist())
            for name, fitpt, net in RUNS:
                got = student(name, fitpt, net, v0)
                _e = (got - ref).norm(dim=-1).mean(-1) / span
                acc[name].append(100 * float(_e.mean()))
                cur[name].append((100 * _e).tolist())
        if not acc[ROWS[0]]:
            dropped.append((K, R))
            for r in ROWS:
                res[r][cell] = sd[r][cell] = curve[r][cell] = None
            print(f"  K={K:2d} r={R:6.3f}x  MPM 이탈로 {nbad}/{args.n_rep} 전부 버림",
                  flush=True)
            continue
        for r in ROWS:
            v = acc[r]
            m = sum(v) / len(v)
            res[r][cell] = m
            sd[r][cell] = (sum((x - m) ** 2 for x in v) / len(v)) ** 0.5
            cs = cur[r]
            n_f = min(len(c) for c in cs)
            curve[r][cell] = [sum(c[t] for c in cs) / len(cs) for t in range(n_f)]
        print(f"  K={K:2d} r={R:6.3f}x  n={len(acc[ROWS[0]])}  " + "  ".join(
            f"{r} {res[r][cell]:6.2f}+-{sd[r][cell]:4.2f}%" for r in ROWS), flush=True)

print(f"\n[격자] MPM 이탈로 버린 칸 {len(dropped)}/{len(KS)*len(RS)}  {dropped}")
lo = [f"{K}_{R}" for K in KS for R in RS if R < 1.0]
hi = [f"{K}_{R}" for K in KS for R in RS if R >= 1.0]
for r in ROWS:
    v = [x for x in res[r].values() if x is not None]
    vl = [res[r][c] for c in lo if res[r].get(c) is not None]
    vh = [res[r][c] for c in hi if res[r].get(c) is not None]
    print(f"\n### {r}  전체 {sum(v)/len(v):.2f}%  최악 {max(v):.2f}%"
          f"  |  r<1 {sum(vl)/len(vl):.2f}%  |  r>=1 {sum(vh)/len(vh):.2f}%")
    print("  K\\r  " + "".join(f"{x:>8.3f}x" for x in RS))
    for K in KS:
        print(f"  {K:3d}  " + "".join(
            ("     n/a" if res[r][f'{K}_{x}'] is None else f"{res[r][f'{K}_{x}']:8.2f}")
            for x in RS))

if args.out:
    json.dump({"KS": KS, "RS": RS, "mag": args.mag, "seed": args.seed,
               "n_rep": args.n_rep, "sd": sd, "curve": curve,
               "spacing": float(spacing), "frames": args.frames,
               "pairs": args.pair, "res": res, "dropped": dropped},
              open(args.out, "w"), indent=1)
    print(f"\n저장: {args.out}")
print("\nEVAL_GRID_DONE")

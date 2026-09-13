"""A~D 와 i-PhysGaussian 을 하나의 자로 잰다.

같은 (K, r) 임펄스, 같은 MPM 정답, 같은 지표. 셋을 맞추는 것이 이 스크립트의 전부다.

**한 프로세스에서 하나의 정답을 쓴다.** DreamPhysics 와 i-PhysGaussian 은 같은 이름의
mpm_solver_warp 사본을 각각 갖고 있어 함께 import 하면 충돌한다. 두 사본을 대조해 보면
차이는 autograd 플래그(requires_grad)와 빈 줄, 그리고 동기화 한 줄뿐이고 p2g2p 의 물리는
같다. 그래서 i-PG 쪽 사본 하나로 정답(명시적)과 i-PG(암시적)를 모두 돌린다 -- 정답이
갈라지지 않는다.

**두 방법이 아끼는 것이 다르다.** 우리 학생은 상태를 줄이고(입자 171,553 -> 앵커 512)
타임스텝은 그대로다. i-PG 는 물리와 상태를 그대로 두고 타임스텝을 k 배로 키운다. k=1 은
정답 그 자체라 이탈이 정의상 0 이므로, 비교는 k>1 에서만 뜻이 있다. 같은 자 위에 놓되
"무엇을 지불했는가" 는 표에 함께 적는다.

칸마다 임펄스를 여러 개 뽑아 평균한다. 1 개일 때는 칸별 분산이 모델 간 차이보다 커서
(K=1, r=0.75 에서 네 모델의 퍼짐이 52.9%p) 어떤 대조도 읽히지 않았다.
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
ap.add_argument("--ipg", required=True, help="i-PhysGaussian 사본 (mpm_solver_warp 포함)")
ap.add_argument("--pair", action="append", default=[], help="이름:기하.pt:학생.pt")
ap.add_argument("--ipg_k", type=int, nargs="*", default=[4, 20],
                 help="i-PG 의 타임스텝 배율. k=1 은 정답이므로 제외한다.")
ap.add_argument("--draws", type=int, default=5, help="칸마다 뽑는 임펄스 수")
ap.add_argument("--frames", type=int, default=60)
ap.add_argument("--dt_mult", type=int, default=40)
ap.add_argument("--seed", type=int, default=777)
ap.add_argument("--mag", type=float, default=1.0)
ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
ap.add_argument("--rs", type=float, nargs="+",
                 default=[0.125, 0.25, 0.5, 0.75, 1.0, 2.0, 5.0, 10.0])
ap.add_argument("--cells", default=None,
                 help="K:R,K:R,... 만 실행. 시드는 전체 격자의 인덱스로 "
                      "매기므로, --ks/--rs 를 A~D 격자와 같게 두고 여기서 "
                      "고르면 그 격자와 완전히 같은 임펄스가 나온다.")
ap.add_argument("--out", default=None)
args = ap.parse_args()

# i-PG 사본을 먼저 올려 mpm_solver_warp 가 그쪽 것으로 잡히게 한다
sys.path.insert(0, args.ipg)
import warp as wp

dev = "cuda"
torch.set_grad_enabled(False)
wp.init()
from implicit_mpm_solver import ImplicitMPMSolver

from anchorflow.anchor_sparse import load_fitted
from anchorflow.mpm_teacher import MPMTeacher
from anchorflow.nextstate import apply_step, net_from_ckpt

sc0 = scene_setup.build(args.ply, args.config, 512, 8, device=dev,
                        frozen_weights=True, rot_fallback=True, eig_floor=0.02)

# E 스케일을 맞춘다. DreamPhysics 의 mpm_utils 는 mu/lam 을 만들 때 E 에 1e7 을
# 곱하고 i-PhysGaussian 의 사본은 곱하지 않는다. 이 스크립트는 sys.path 를 i-PG 쪽으로
# 두어 한 프로세스에서 정답과 i-PG 를 함께 돌리므로, 같은 config 를 그대로 주면 재질이
# 1e7 배 물러져 물체가 흩어지고 MPM 이 도메인을 벗어난다 -- A~D 격자에서 멀쩡하던 칸이
# 여기서만 이탈하는 것으로 드러났다. 여기서 한 번 곱해 두면 정답도 i-PG 도 A~D 와
# 같은 물성 위에서 돈다.
if "E" in sc0.cfg:
    sc0.cfg = dict(sc0.cfg)
    sc0.cfg["E"] = float(sc0.cfg["E"]) * 1e7
    print(f"[setup] i-PG 사본에 맞춰 E 를 1e7 배: {sc0.cfg['E']:.4g}", flush=True)

T = MPMTeacher(sc0)
X0 = T.pos_m.clone()
BASE = torch.tensor([-0.477, 0.0, 0.0], device=dev)
spacing = sc0.sim.radius
dt_c = args.dt_mult * sc0.sub_dt
N_GRID = int(getattr(sc0, "n_grid", None) or 100)

GEOM, RUNS = {}, []
for spec in args.pair:
    name, fitpt, ckpt = spec.split(":")
    if fitpt not in GEOM:
        sc, _ = load_fitted(sc0, fitpt, dev)
        GEOM[fitpt] = (sc, sc.fit, sc._cache)
    net = net_from_ckpt(torch.load(ckpt, map_location=dev, weights_only=False), dev).eval()
    RUNS.append((name, fitpt, net))

ROWS = ([f"floor({os.path.basename(f).replace('.pt','')})" for f in GEOM]
        + [n for n, _, _ in RUNS] + [f"i-PG k={k}" for k in args.ipg_k])
print(f"[setup] 기하 {len(GEOM)}, 학생 {len(RUNS)}, i-PG k={args.ipg_k}, "
      f"격자 {len(args.ks)}x{len(args.rs)}, 칸당 {args.draws} draw, "
      f"{args.frames}프레임 x {args.dt_mult} 서브스텝", flush=True)


def v0_of(force):
    """입자 초기 속도. 베이스 씬의 스키닝으로 만들어 기하와 무관하다."""
    dv = sc0.impulse_dv(force)
    return (T.w.unsqueeze(-1) * dv[T.idx]).sum(1).contiguous()


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


_IPG = {}
def ipg_run(v0, k):
    """i-PG 의 암시적 적분기를 k 배 타임스텝으로. 상태도 물리도 줄이지 않는다."""
    if k not in _IPG:
        s = ImplicitMPMSolver(T.n, n_grid=N_GRID, grid_lim=T.grid_lim)
        s.load_initial_data_from_torch(X0.clone(), T.vol_m.clone(),
                                        torch.zeros((T.n, 6), device=dev),
                                        n_grid=N_GRID, grid_lim=T.grid_lim)
        mp = {kk: sc0.cfg[kk] for kk in ("E", "nu", "density", "material") if kk in sc0.cfg}
        mp.update({"n_grid": N_GRID, "grid_lim": T.grid_lim,
                   "g": sc0.cfg.get("g", [0, 0, 0]),
                   "grid_v_damping_scale": sc0.cfg.get("grid_v_damping_scale", 1.0)})
        if "additional_material_params" in sc0.cfg:
            mp["additional_material_params"] = sc0.cfg["additional_material_params"]
        s.set_parameters_dict(mp)
        s.finalize_mu_lam()
        _IPG[k] = s
    s = _IPG[k]
    eye = torch.eye(3, device=dev).reshape(1, 9).repeat(T.n, 1).contiguous()
    s.import_particle_x_from_torch(X0.clone())
    s.import_particle_v_from_torch(v0.clone())
    s.import_particle_F_from_torch(eye.clone())
    s.import_particle_C_from_torch(torch.zeros_like(eye))
    sub = max(1, args.dt_mult // k)
    xs = [X0.clone()]
    step = 0
    for _ in range(args.frames):
        for _ in range(sub):
            s.p2g2p_implicit(step, sc0.sub_dt * k)
            step += 1
        x = s.export_particle_x_to_torch().clone()
        if not torch.isfinite(x).all():
            return None
        xs.append(x)
    return torch.stack(xs)


acc = {r: {} for r in ROWS}
dropped = 0
for ki, K in enumerate(args.ks):
    for rj, R in enumerate(args.rs):
        cell = f"{K}_{R}"
        if args.cells and f"{K}:{R:g}" not in args.cells.split(","):
            for r in ROWS: acc[r][cell] = None
            continue
        per = {r: [] for r in ROWS}
        for dnum in range(args.draws):
            g = torch.Generator(device=dev)
            g.manual_seed(args.seed + 100000 * dnum + 1000 * ki + rj)
            f = sc0.random_multi_poke(g, K, R * spacing, BASE.norm().item() * args.mag)
            v0 = v0_of(f)
            ref = truth(v0)
            if ref is None:
                dropped += 1
                continue
            span = (ref - ref[0]).norm(dim=-1).max().clamp(min=1e-12)
            for fitpt in GEOM:
                key = f"floor({os.path.basename(fitpt).replace('.pt','')})"
                per[key].append(100 * float((floor_of(ref, fitpt) - ref).norm(dim=-1).mean() / span))
            for name, fitpt, net in RUNS:
                got = student(fitpt, net, v0)
                per[name].append(100 * float((got - ref).norm(dim=-1).mean() / span))
            for k in args.ipg_k:
                got = ipg_run(v0, k)
                per[f"i-PG k={k}"].append(
                    float("nan") if got is None
                    else 100 * float((got - ref).norm(dim=-1).mean() / span))
        for r in ROWS:
            v = [x for x in per[r] if x == x]
            acc[r][cell] = (sum(v) / len(v)) if v else None
        print(f"  K={K:2d} r={R:6.3f}x  " + "  ".join(
            f"{r} {acc[r][cell]:6.2f}%" if acc[r][cell] is not None else f"{r} n/a"
            for r in ROWS), flush=True)

print(f"\n[격자] MPM 이탈로 버린 draw {dropped}/{len(args.ks)*len(args.rs)*args.draws}")
lo = [f"{K}_{R}" for K in args.ks for R in args.rs if R < 1.0]
hi = [f"{K}_{R}" for K in args.ks for R in args.rs if R >= 1.0]
k2 = [f"{K}_{R}" for K in args.ks for R in args.rs if K >= 2]
def m(r, cells):
    v = [acc[r][c] for c in cells if acc[r].get(c) is not None]
    return (sum(v) / len(v)) if v else float("nan")
print(f"\n{'행':>14} {'전체':>8} {'r<1':>8} {'r>=1':>8} {'K>=2':>8} {'최악':>8}")
for r in ROWS:
    v = [x for x in acc[r].values() if x is not None]
    worst = max(v) if v else float("nan")
    print(f"{r:>14} {m(r, list(acc[r])):7.2f}% {m(r, lo):7.2f}% {m(r, hi):7.2f}% "
          f"{m(r, k2):7.2f}% {worst:7.2f}%")
if args.out:
    json.dump({"ks": args.ks, "rs": args.rs, "draws": args.draws, "seed": args.seed,
               "ipg_k": args.ipg_k, "pairs": args.pair, "res": acc, "dropped": dropped},
              open(args.out, "w"), indent=1)
    print(f"\n저장: {args.out}")
print("\nEVAL_IPG_DONE")

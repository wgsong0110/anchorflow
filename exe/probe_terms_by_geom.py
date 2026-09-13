"""기하별로 왕복 손실의 세 항(x, v, F)이 어떻게 다른지 잰다.

피팅 로그는 고정창 진단에 합계만 찍어서 항별 변화를 되짚을 수 없다. 완성된
체크포인트들에 대해 같은 창·같은 규약으로 세 항을 따로 재서 나란히 놓는다.

속도 항은 모든 기하가 같은 복호기(project_v_ls + lift 의 형상 매칭 Fdot)를 쓰므로
거기서 나는 차이는 전적으로 앵커 배치 탓이다. 위치·F 항은 복호기가 기하마다
다르므로(형상 매칭 / 회전+신축 / 자유 F) 배치와 복호기 차이가 섞인다.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
import torch

from anchorflow import scene_setup

ap = argparse.ArgumentParser()
ap.add_argument("--ply", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--dreamphysics", default="/workspace/DreamPhysics")
ap.add_argument("--traj_cache", required=True)
ap.add_argument("--ckpt", nargs="+", required=True,
                help="이름:종류:경로 (종류 = shape | us | F)")
ap.add_argument("--n_win", type=int, default=8)
ap.add_argument("--frames", type=int, default=60)
ap.add_argument("--unroll", type=int, default=1)
ap.add_argument("--dt_mult", type=int, default=40)
ap.add_argument("--grid_lim", type=float, default=2.0)
ap.add_argument("--anchors", type=int, default=512)
ap.add_argument("--knn", type=int, default=8)
ap.add_argument("--sh_degree", type=int, default=3)
ap.add_argument("--init", required=True, help="학습 전 기하 체크포인트 (r_ref 를 여기서 잡는다)")
args = ap.parse_args()

sys.path.insert(0, args.dreamphysics)
import warp as wp

dev = "cuda"
torch.set_grad_enabled(False)
wp.init()
from anchorflow.anchor_sparse import Traj, load_fitted
from anchorflow.frame_encode import FrameState

sys.modules["__main__"].Traj = Traj

sc = scene_setup.build(args.ply, args.config, args.anchors, args.knn, device=dev,
                       frozen_weights=True, rot_fallback=True, eig_floor=0.02,
                       sh_degree=args.sh_degree)
MASS = sc.volume[sc.keep].clone() if sc.volume.shape[0] != int(sc.keep.sum()) \
    else sc.volume.clone()
DT_C = args.dt_mult * sc.sub_dt

# 피팅이 학습 시작 시점에 한 번 잡고 고정한 상수와 같은 값. 세 피팅이 모두 같은
# 학습 전 기하에서 출발했으므로 r_ref 도 하나다.
_f0 = load_fitted(sc, args.init, dev)[0].fit
R_REF = float((_f0.Xc - _f0.prepare()[1]).norm(dim=-1).pow(2).mean().sqrt())
del _f0
torch.cuda.empty_cache()
C_X, C_V, C_F = 1.0 / args.grid_lim, DT_C / args.grid_lim, R_REF / args.grid_lim
print(f"[setup] c_x {C_X:.5g}  c_v {C_V:.5g}  c_F {C_F:.5g}  (r_ref {R_REF:.5f})", flush=True)

FIT = torch.load(args.traj_cache, map_location=dev, weights_only=False)["fit"]
_g = torch.Generator(device="cpu")
_g.manual_seed(20260905)
_hi = max(1, args.frames - args.unroll)
WIN = [(int(torch.randint(len(FIT), (1,), generator=_g).item()),
        int(torch.randint(_hi, (1,), generator=_g).item()))
       for _ in range(args.n_win)]
print(f"[data] 궤적 {len(FIT)}, 평가창 {len(WIN)}", flush=True)


def mw(e2):
    return float(((MASS * e2).sum() / MASS.sum()).sqrt())


def measure(fit, kind):
    cache = fit.prepare()
    fac = fit.ls_factor(cache)
    w = cache[0]
    Yg = fit.Xc - cache[1]
    FS = None if kind == "shape" else FrameState(fit.pair_g, fit.pair_a, fit.N, fit.M, MASS)
    tot = [0.0, 0.0, 0.0]
    for i, t in WIN:
        fit.reset_carried()
        x0 = FIT[i][0][t].to(dev, torch.float32)
        v0 = FIT[i][1][t].to(dev, torch.float32)
        F0m = FIT[i][2][t].to(dev, torch.float32).view(-1, 3, 3)
        v = fit.project_v_ls(v0, cache, fac)
        if kind == "shape":
            p = fit.project_ls(x0, cache, fac)
            xh, vl, Fl, _ = fit.lift(p, v, cache)
            Fh = Fl.view(-1, 3, 3)
        elif kind == "us":
            p, u, s = FS.encode_closed(x0, F0m, w, Yg, fixed=fit.fixed, p_fix=fit.pos)
            xh, Fh = FS.decode_x(p, u, s, w, Yg)
            vl = fit.lift(p, v, cache)[1]
        else:
            p, Fa = FS.encode_closed_F(x0, F0m, w, Yg, fixed=fit.fixed, p_fix=fit.pos)
            xh, Fh = FS.decode_x_F(p, Fa, w, Yg)
            vl = fit.lift(p, v, cache)[1]
        tot[0] += mw((xh - x0).pow(2).sum(-1))
        tot[1] += mw((vl - v0).pow(2).sum(-1))
        tot[2] += mw((Fh - F0m).pow(2).sum((-1, -2)))
    return [a / len(WIN) for a in tot], fit.M, int(fit.pair_g.shape[0])


ROWS = []
for spec in args.ckpt:
    name, kind, path = spec.split(":", 2)
    if not os.path.exists(path):
        print(f"  [건너뜀] {name}: {path} 없음", flush=True)
        continue
    fit = load_fitted(sc, path, dev)[0].fit
    (ex, ev, ef), M, P = measure(fit, kind)
    ROWS.append((name, kind, M, P, ex, ev, ef))
    print(f"  {name}({kind}): x {ex:.4e}  v {ev:.4e}  F {ef:.4f}", flush=True)
    fit = None
    torch.cuda.empty_cache()

print(f"\n{'기하':<12}{'복호':>6}{'앵커':>6}{'mwRMS x':>12}{'mwRMS v':>12}{'mwRMS F':>10}")
for n, k, M, P, ex, ev, ef in ROWS:
    print(f"{n:<12}{k:>6}{M:>6}{ex:>12.4e}{ev:>12.4e}{ef:>10.4f}")

print(f"\n{'기하':<12}{'x 기여':>12}{'v 기여':>12}{'F 기여':>12}{'합계':>12}{'x %':>7}{'v %':>7}{'F %':>7}")
for n, k, M, P, ex, ev, ef in ROWS:
    a, b, c = C_X * ex, C_V * ev, C_F * ef
    s = a + b + c
    print(f"{n:<12}{a:>12.4e}{b:>12.4e}{c:>12.4e}{s:>12.4e}{100*a/s:>6.1f}%{100*b/s:>6.1f}%{100*c/s:>6.1f}%")

print("\n속도 항은 모든 기하가 같은 복호기를 쓰므로 그 차이는 앵커 배치 탓이다.")
print("TERMS_DONE")

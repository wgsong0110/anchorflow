"""프레임 피팅이 왜 부푸는 쪽으로 가는지 가른다.

기하를 고정해 놓고 지지 폭만 전역 배수로 훑으며 목적함수를 잰다. 각 지점에서
**두 번** 잰다 -- 피팅이 실제로 쓰는 약한 내부 풀이(GN 6/CG 20)와 충분히 수렴한
것(GN 20/CG 60).

  A) 수렴한 풀이에서 L 이 폭에 대해 최소점을 갖고 지금이 그 오른쪽이면
     -> 목적함수가 진짜로 어느 폭을 원하고, 피팅이 지나친 것이다.
  B) 수렴한 풀이는 평평/단조인데 약한 풀이만 넓은 쪽을 낮게 본다면
     -> 인코더가 수렴을 못 해 그래디언트가 틀린 것이다. 겹침이 늘수록 역블러링이
        나빠지므로 이것은 양의 되먹임이 된다.

정류성 |dPhi/dz| (자유 부분공간에서만) 을 함께 내서 B 를 직접 확인한다.
"""
from __future__ import annotations

import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
import torch
from anchorflow import scene_setup

ap = argparse.ArgumentParser()
ap.add_argument("--ply", required=True); ap.add_argument("--config", required=True)
ap.add_argument("--dreamphysics", default="/workspace/DreamPhysics")
ap.add_argument("--traj_cache", required=True)
ap.add_argument("--ckpt", nargs="+", required=True, help="이름=경로 (init 가능)")
ap.add_argument("--mults", default="0.6,0.8,1.0,1.25,1.5,2.0")
ap.add_argument("--closed", action="store_true", help="닫힌 형태 인코더로 잰다")
ap.add_argument("--n_win", type=int, default=8)
ap.add_argument("--frames", type=int, default=60)
ap.add_argument("--dt_mult", type=int, default=40)
ap.add_argument("--grid_lim", type=float, default=2.0)
ap.add_argument("--anchors", type=int, default=512)
args = ap.parse_args()

sys.path.insert(0, args.dreamphysics)
import warp as wp
dev = "cuda"; torch.set_grad_enabled(False); wp.init()
from anchorflow.anchor_sparse import AnchorSparse, Traj
from anchorflow.frame_encode import FrameState
sys.modules["__main__"].Traj = Traj

sc = scene_setup.build(args.ply, args.config, args.anchors, 8, device=dev,
                       frozen_weights=True, rot_fallback=True, eig_floor=0.02)
MASS = sc.volume[sc.keep].clone()
DT_C = args.dt_mult * sc.sub_dt
FIT = torch.load(args.traj_cache, map_location=dev, weights_only=False)["fit"]

# 피팅의 고정 평가창과 같은 방식·같은 씨앗
_g = torch.Generator(device="cpu"); _g.manual_seed(20260905)
_hi = max(1, args.frames - 1)
WIN = [(int(torch.randint(len(FIT), (1,), generator=_g).item()),
        int(torch.randint(_hi, (1,), generator=_g).item())) for _ in range(args.n_win)]
print(f"[setup] 평가창 {len(WIN)}개, 궤적 {len(FIT)}", flush=True)


def mw(e2): return ((MASS * e2).sum() / MASS.sum()).sqrt()


def measure(fit, gn, cg, r_ref):
    """L 과 정류성. 정류성은 고정 앵커의 p 를 뺀 자유 부분공간에서만 잰다."""
    cache = fit.prepare()
    fac = fit.ls_factor(cache)
    Yg = fit.Xc - cache[1]
    FS = FrameState(fit.pair_g, fit.pair_a, fit.N, fit.M, MASS)
    c_x, c_v, c_F = 1.0 / args.grid_lim, DT_C / args.grid_lim, r_ref / args.grid_lim
    tot = [0.0, 0.0, 0.0, 0.0]
    for i, t in WIN:
        x0 = FIT[i][0][t].to(dev, torch.float32)
        v0 = FIT[i][1][t].to(dev, torch.float32)
        F0m = FIT[i][2][t].to(dev, torch.float32).view(-1, 3, 3)
        if gn == 0:
            pj, uj, sj = FS.encode_closed(x0, F0m, cache[0], Yg,
                                           fixed=fit.fixed, p_fix=fit.pos)
        else:
            pj, uj, sj, _ = FS.encode_joint(x0, F0m, cache[0], Yg, c_x=c_x, c_F=c_F,
                                            fixed=fit.fixed, p_fix=fit.pos,
                                            iters=gn, cg_iters=cg)
        xh, Fh = FS.decode_x(pj, uj, sj, cache[0], Yg)
        vl = fit.lift(pj, fit.project_v_ls(v0, cache, fac), cache)[1]
        tot[0] += float(mw((xh - x0).pow(2).sum(-1)))
        tot[1] += float(mw((vl - v0).pow(2).sum(-1)))
        tot[2] += float(mw((Fh - F0m).pow(2).sum((-1, -2))))
        with torch.enable_grad():
            a, b, c = pj.clone().requires_grad_(True), uj.clone().requires_grad_(True), \
                      sj.clone().requires_grad_(True)
            xx, FF = FS.decode_x(a, b, c, cache[0], Yg)
            phi = c_x * mw((xx - x0).pow(2).sum(-1)) + c_F * mw((FF - F0m).pow(2).sum((-1, -2)))
            gp, gu, gs = torch.autograd.grad(phi, [a, b, c])
        gp = torch.where(fit.fixed.unsqueeze(-1), torch.zeros_like(gp), gp)
        tot[3] += float((gp.pow(2).sum() + gu.pow(2).sum() + gs.pow(2).sum()).sqrt())
    n = len(WIN)
    ex, ev, ef, gz = [q / n for q in tot]
    return c_x * ex + c_v * ev + c_F * ef, ex, ev, ef, gz


MU = [float(x) for x in args.mults.split(",")]
# 피팅은 r_ref 를 학습 시작 때 한 번 재고 고정한다. 폭을 바꾸면 실제 팔 길이는
# 변하지만 계수는 안 변한다 -- 그 규약을 그대로 재현해야 "피팅이 보던 목적함수"가
# 나온다. 비교를 위해 배수마다 다시 잰 규약도 함께 낸다.
_f0 = AnchorSparse(sc, c=0.25, eig_floor=0.02).to(dev); _f0.init_from_geometry()
R_FIX = float((_f0.Xc - _f0.prepare()[1]).norm(dim=-1).pow(2).mean().sqrt())
del _f0; torch.cuda.empty_cache()
print(f"[setup] 피팅이 쓰는 고정 r_ref = {R_FIX:.6f}")
print(f"\n{'기하':>7}{'배수':>6}{'짝':>12}{'n_eff':>7}{'r_ref':>9}"
      f"{'mwRMS x':>11}{'mwRMS v':>11}{'mwRMS F':>9}"
      f"{'L(고정 c_F)':>13}{'L(재계산)':>12}{'|dPhi/dz|':>11}")
for spec in args.ckpt:
    name, path = spec.split("=", 1)
    for mu in MU:
        try:
            fit = AnchorSparse(sc, c=0.25, eig_floor=0.02).to(dev)
            if path == "init":
                fit.init_from_geometry()
            else:
                b = torch.load(path, map_location=dev, weights_only=False)
                fit._rebuild(b["pos"].to(dev), b["quat"].to(dev), b["log_s"].to(dev),
                             b.get("log_k"), b.get("log_amp"))
            with torch.no_grad():
                fit.log_s.add_(float(torch.tensor(mu).log()))
            fit.refresh()
            cache0 = fit.prepare()
            r_now = float((fit.Xc - cache0[1]).norm(dim=-1).pow(2).mean().sqrt())
            w = cache0[0]
            w2 = torch.zeros(fit.N, device=dev).index_add_(0, fit.pair_g, w * w)
            neff = float((1.0 / w2.clamp(min=1e-20)).mean())
            P = int(fit.pair_g.shape[0])
            gn, cg = (0, 0) if args.closed else (20, 60)
            _, ex, ev, ef, gz = measure(fit, gn, cg, R_FIX)
            cx, cv = 1.0 / args.grid_lim, DT_C / args.grid_lim
            L_fix = cx * ex + cv * ev + (R_FIX / args.grid_lim) * ef
            L_now = cx * ex + cv * ev + (r_now / args.grid_lim) * ef
            print(f"{name:>7}{mu:>6.2f}{P:>12,}{neff:>7.1f}{r_now:>9.5f}"
                  f"{ex:>11.4e}{ev:>11.4e}{ef:>9.4f}{L_fix:>13.5e}{L_now:>12.5e}"
                  f"{gz:>11.3e}", flush=True)
        except torch.OutOfMemoryError:
            print(f"{name:>7}{mu:>6.2f}{'OOM':>12}", flush=True)
        finally:
            fit = None; torch.cuda.empty_cache()
print("\nINFLATION_DONE")

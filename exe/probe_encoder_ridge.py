"""인코더가 내놓는 앵커 상태가 학생이 배울 만한 것인지 본다.

u_g = sum_a w_ga u_a 를 뒤집는 최소제곱은 역블러링이라 병조건이다. 해가 크게
진동하면서 **블렌딩될 때 상쇄되는** 형태로 나오는데, 표현으로는 문제가 없어도
학생에게는 치명적이다 -- 앵커마다 그 큰 값을 정확히 재현해야 상쇄가 유지되고,
독립적인 오차가 섞이면 곧바로 깨진다.

릿지를 키우면 해가 매끄러워지는 대신 F 잔차가 는다. 그 맞바꿈을 잰다.
증폭비 |u_a| / |u_g| 가 학생이 감당할 크기인지 판단의 근거다.
"""
from __future__ import annotations

import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
import torch
from anchorflow import scene_setup

ap = argparse.ArgumentParser()
ap.add_argument("--ply", required=True); ap.add_argument("--config", required=True)
ap.add_argument("--dreamphysics", default="/workspace/DreamPhysics")
ap.add_argument("--fit", required=True); ap.add_argument("--traj_cache", required=True)
ap.add_argument("--ridges", default="1e-4,1e-3,1e-2,3e-2,1e-1,3e-1")
ap.add_argument("--n_win", type=int, default=6)
ap.add_argument("--frames", type=int, default=30)
ap.add_argument("--grid_lim", type=float, default=2.0)
args = ap.parse_args()

sys.path.insert(0, args.dreamphysics)
import warp as wp
dev = "cuda"; torch.set_grad_enabled(False); wp.init()
from anchorflow.anchor_sparse import Traj, load_fitted
from anchorflow.frame_encode import FrameState, polar_target
sys.modules["__main__"].Traj = Traj

sc = scene_setup.build(args.ply, args.config, 512, 8, device=dev,
                       frozen_weights=True, rot_fallback=True, eig_floor=0.02)
fit = load_fitted(sc, args.fit, dev)[0].fit
cache = fit.prepare(); w = cache[0]; Yg = fit.Xc - cache[1]
MASS = sc.volume[sc.keep].clone()
FS = FrameState(fit.pair_g, fit.pair_a, fit.N, fit.M, MASS)
FIT = torch.load(args.traj_cache, map_location=dev, weights_only=False)["fit"]
g = torch.Generator(device="cpu"); g.manual_seed(20260908)
WIN = [(int(torch.randint(len(FIT), (1,), generator=g).item()),
        int(torch.randint(args.frames, (1,), generator=g).item())) for _ in range(args.n_win)]
print(f"[setup] 앵커 {fit.M}, 평가창 {len(WIN)}", flush=True)


def mw(e2): return float(((MASS * e2).sum() / MASS.sum()).sqrt())


print(f"\n{'릿지':>8}{'|u_a|':>9}{'|u_g|':>9}{'증폭':>8}{'|s_a|':>9}{'|s_g|':>9}"
      f"{'증폭':>8}{'mwRMS F':>10}{'mwRMS x':>11}")
for r in [float(x) for x in args.ridges.split(",")]:
    ua_n = ug_n = sa_n = sg_n = ef = ex = 0.0
    for i, t in WIN:
        x0 = FIT[i][0][t].to(dev, torch.float32)
        F0m = FIT[i][2][t].to(dev, torch.float32).view(-1, 3, 3)
        tg = polar_target(F0m)
        p, u, s = FS.encode_closed(x0, F0m, w, Yg, fixed=fit.fixed, p_fix=fit.pos,
                                    ridge=r, targets=tg)
        ug, sg = FS.blend(u, s, w)
        xh, Fh = FS.decode_x(p, u, s, w, Yg)
        ua_n += float(u.norm(dim=-1).mean()); ug_n += float(ug.norm(dim=-1).mean())
        sa_n += float(s.norm(dim=-1).mean()); sg_n += float(sg.norm(dim=-1).mean())
        ef += mw((Fh - F0m).pow(2).sum((-1, -2)))
        ex += mw((xh - x0).pow(2).sum(-1))
    n = len(WIN)
    ua_n, ug_n, sa_n, sg_n, ef, ex = [q / n for q in (ua_n, ug_n, sa_n, sg_n, ef, ex)]
    print(f"{r:>8.0e}{ua_n:>9.3f}{ug_n:>9.4f}{ua_n/max(ug_n,1e-12):>8.1f}"
          f"{sa_n:>9.4f}{sg_n:>9.5f}{sa_n/max(sg_n,1e-12):>8.1f}"
          f"{ef:>10.4f}{ex:>11.4e}", flush=True)
print("\n증폭 = 앵커값 크기 / 가우시안값 크기. 클수록 상쇄에 의존하는 해이고,")
print("학생이 앵커마다 독립적으로 틀리면 그만큼 크게 깨진다.")
print("RIDGE_DONE")

"""안쪽 풀이가 정말 정류점에 있는가 -- 포락선 정리의 전제를 직접 검사한다.

방향미분 검사에서 하강 방향은 98.8% 로 맞았지만 무작위 방향 하나가 1.67 배로 어긋났다.
eps 를 바꿔도 값이 유지되므로 잡음이 아니라 계통 오차다. 후보는 하나다:
dL/dtheta = dPhi/dtheta 는 z* 에서 dPhi/dz = 0 일 때만 성립하는데, GN 이 수렴하지
않았다면 그 항이 통째로 빠진다.

그래서 안쪽 풀이의 강도를 올려 가며 (a) 정류성 |dPhi/dz| 가 초기 대비 얼마나
떨어지는지, (b) 어긋났던 방향의 유한차분 비가 1 로 가는지 함께 본다.
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
ap.add_argument("--grid_lim", type=float, default=2.0)
ap.add_argument("--eps", type=float, default=3e-4)
args = ap.parse_args()

sys.path.insert(0, args.dreamphysics)
import warp as wp
dev = "cuda"; wp.init()
from anchorflow.anchor_sparse import AnchorSparse, Traj
from anchorflow.frame_encode import FrameState
sys.modules["__main__"].Traj = Traj

sc = scene_setup.build(args.ply, args.config, 512, 8, device=dev,
                       frozen_weights=True, rot_fallback=True, eig_floor=0.02)
fit = AnchorSparse(sc, c=0.25, eig_floor=0.02).to(dev)
with torch.no_grad():
    fit.init_from_geometry()
MASS = sc.volume[sc.keep].clone()
with torch.no_grad():
    R_REF = float((fit.Xc - fit.prepare()[1]).norm(dim=-1).pow(2).mean().sqrt())
C_X, C_F = 1.0 / args.grid_lim, R_REF / args.grid_lim
FS = FrameState(fit.pair_g, fit.pair_a, fit.N, fit.M, MASS)
FITD = torch.load(args.traj_cache, map_location=dev, weights_only=False)["fit"]
x0 = FITD[0][0][5]; F0m = FITD[0][2][5].reshape(-1, 3, 3).float()
NAMES = ("pos", "log_s", "quat", "log_amp")
P = dict(fit.named_parameters())
print(f"[setup] 고정 앵커 {int(fit.fixed.sum())} / {fit.M}")


def mw(e2): return ((MASS * e2).sum() / MASS.sum()).sqrt()


def phi(p_, u_, s_, w_, Yg):
    xh, F = FS.decode_x(p_, u_, s_, w_, Yg)
    return C_X * mw((xh - x0).pow(2).sum(-1)) + C_F * mw((F - F0m).pow(2).sum((-1, -2)))


def gz(p_, u_, s_, w_, Yg):
    """|dPhi/dz| -- 앵커 상태에 대한 그래디언트 크기"""
    with torch.enable_grad():
        a, b, c = p_.clone().requires_grad_(True), u_.clone().requires_grad_(True), \
                  s_.clone().requires_grad_(True)
        g = torch.autograd.grad(phi(a, b, c, w_, Yg), [a, b, c])
    return float(sum(x.pow(2).sum() for x in g).sqrt())


def solve(gn, cg, grad=False):
    with torch.set_grad_enabled(grad):
        cache = fit.prepare()
        w_, Yg = cache[0], fit.Xc - cache[1]
        with torch.no_grad():
            pj, uj, sj, _ = FS.encode_joint(x0, F0m, w_.detach(), Yg.detach(),
                                            c_x=C_X, c_F=C_F, fixed=fit.fixed,
                                            p_fix=fit.pos.detach(), iters=gn, cg_iters=cg)
        return pj, uj, sj, w_, Yg


# 어긋났던 방향(probe_frame_grad 의 "무작위 2")을 같은 씨앗으로 되만든다
torch.manual_seed(1)
for _ in range(2):
    r = {n: torch.randn_like(P[n]) for n in NAMES}
rn = sum(float(r[n].pow(2).sum()) for n in NAMES) ** 0.5
D = {n: r[n] / rn for n in NAMES}

print(f"\n{'GN':>4}{'CG':>5}{'Phi':>14}{'|dPhi/dz| 초기':>16}{'-> 해':>14}{'비율':>9}"
      f"{'해석적':>14}{'유한차분':>14}{'비':>8}")
for gn, cg in ((6, 20), (20, 60), (60, 60), (60, 200), (150, 200)):
    pj, uj, sj, w_, Yg = solve(gn, cg)
    with torch.no_grad():
        cache0 = fit.prepare()
        w0, Y0 = cache0[0], fit.Xc - cache0[1]
        ug_t, sg_t = __import__("anchorflow.frame_encode", fromlist=["x"]).polar_target(F0m)
        Lg = FS.gram(w0)
        u0, s0 = FS._ls_init(ug_t, sg_t, w0, L=Lg)
        Fh = FS.decode(u0, s0, w0)
        p0 = FS._wls(x0 - torch.einsum("nij,nj->ni", Fh, Y0), w0, Lg)
    g_init = gz(p0, u0, s0, w0, Y0)
    g_sol = gz(pj, uj, sj, w_, Yg)
    # 해석적 방향미분
    pj, uj, sj, w_, Yg = solve(gn, cg, grad=True)
    L = phi(pj, uj, sj, w_, Yg)
    fit.zero_grad(set_to_none=True); L.backward()
    an = sum(float((P[n].grad * D[n]).sum()) for n in NAMES)
    # 유한차분
    vals = []
    for sgn in (1.0, -1.0):
        with torch.no_grad():
            for n in NAMES: P[n].add_(D[n], alpha=sgn * args.eps)
        pk, uk, sk, wk, Yk = solve(gn, cg)
        vals.append(float(phi(pk, uk, sk, wk, Yk)))
        with torch.no_grad():
            for n in NAMES: P[n].add_(D[n], alpha=-sgn * args.eps)
    fd = (vals[0] - vals[1]) / (2 * args.eps)
    print(f"{gn:>4}{cg:>5}{float(L):14.6e}{g_init:16.4e}{g_sol:14.4e}"
          f"{g_sol/max(g_init,1e-30):9.4f}{an:14.6e}{fd:14.6e}"
          f"{fd/an if an else 0:8.3f}", flush=True)
print("STAT_DONE")

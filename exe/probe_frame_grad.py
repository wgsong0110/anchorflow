"""프레임 왕복 손실의 그래디언트를 유한차분으로 검증한다.

인코더는 grad 없이 (p,u,s) 를 풀고 detach 한다. 그것으로 정확한 그래디언트가 나오는
근거는 포락선 정리 -- L(t) = min_z Phi(z,t) 이면 dL/dt = dPhi/dt|_{z*} -- 이고,
전제는 (1) 인코더가 여기서 재는 것과 **같은** 목적함수를 풀 것, (2) 안쪽 풀이가
수렴해 있을 것이다. 유한차분은 z* 의 변화까지 포함하므로 두 전제를 함께 검사한다.
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
ap.add_argument("--gn", type=int, default=20); ap.add_argument("--cg", type=int, default=60)
ap.add_argument("--n_coord", type=int, default=6)
ap.add_argument("--eps", type=float, default=2e-4)
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
print(f"[setup] c_x {C_X:.5g}, c_F {C_F:.5g}, GN {args.gn}/CG {args.cg}, eps {args.eps:g}")


def mw(e2): return ((MASS * e2).sum() / MASS.sum()).sqrt()


def loss(grad):
    with torch.set_grad_enabled(grad):
        cache = fit.prepare()
        w_, rc_ = cache[0], cache[1]
        Yg = fit.Xc - rc_
        with torch.no_grad():
            pj, uj, sj, _ = FS.encode_joint(x0, F0m, w_.detach(), Yg.detach(),
                                            c_x=C_X, c_F=C_F, fixed=fit.fixed,
                                            p_fix=fit.pos.detach(),
                                            iters=args.gn, cg_iters=args.cg)
        pj = torch.where(fit.fixed.unsqueeze(-1), fit.pos, pj)
        xh, Ff = FS.decode_x(pj, uj, sj, w_, Yg)
        return C_X * mw((xh - x0).pow(2).sum(-1)) + C_F * mw((Ff - F0m).pow(2).sum((-1, -2)))


L = loss(True)
fit.zero_grad(set_to_none=True)
L.backward()
G = {n: (p.grad.clone() if p.grad is not None else torch.zeros_like(p))
     for n, p in fit.named_parameters()}
print(f"[기준] L = {float(L):.8e}")

# 좌표별 유한차분은 float32 잡음 바닥에 걸린다: L ~ 7.5e-4 의 1 ulp 가 9e-11 이고
# 2·eps 로 나누면 FD 의 분해능이 2e-7 인데, 좌표 하나의 그래디언트가 대체로 그 크기다.
# 방향미분은 그 문제가 없다 -- d = g/|g| 로 잡으면 신호가 |g| 라 잡음의 수백 배다.
torch.manual_seed(1)
names = ("pos", "log_s", "quat", "log_amp")
P = dict(fit.named_parameters())
gn2 = sum(float(G[n].pow(2).sum()) for n in names)
gnorm = gn2 ** 0.5
print(f"[기준] |g| = {gnorm:.6e}")


def directional(d, eps):
    """중앙차분 (L(t+eps d) - L(t-eps d)) / 2 eps"""
    out = []
    for sgn in (1.0, -1.0):
        with torch.no_grad():
            for n in names:
                P[n].add_(d[n], alpha=sgn * eps)
        out.append(float(loss(False)))
        with torch.no_grad():
            for n in names:
                P[n].add_(d[n], alpha=-sgn * eps)
    return (out[0] - out[1]) / (2 * eps)


DIRS = [("g/|g| (하강 방향)", {n: G[n] / gnorm for n in names})]
for k in range(2):
    r = {n: torch.randn_like(P[n]) for n in names}
    rn = sum(float(r[n].pow(2).sum()) for n in names) ** 0.5
    DIRS.append((f"무작위 {k+1}", {n: r[n] / rn for n in names}))

print()
print(f"{'방향':<20}{'eps':>10}{'해석적 <g,d>':>16}{'유한차분':>16}{'비':>9}{'상대오차':>10}")
for nm, d in DIRS:
    an = sum(float((G[n] * d[n]).sum()) for n in names)
    for eps in (1e-4, 3e-4, 1e-3, 3e-3):
        fd = directional(d, eps)
        print(f"{nm:<20}{eps:10.0e}{an:16.6e}{fd:16.6e}{fd/an if an else 0:9.3f}"
              f"{abs(an-fd)/max(abs(an),1e-20):10.3f}")
print("FRAME_GRAD_DONE")

"""결합 인코더 (p, u, s) 검증: 수반 검사 + 피팅 손실 대조 + 비용.

인코더가 피팅과 **같은** 목적함수를 풀어야 포락선 정리로 출력을 detach 하고도
정확한 그래디언트가 나온다. 여기서 재는 것은 그 목적함수 자체:

    L = c_x · mwRMS(dx) + c_F · mwRMS(dF),   c_x = 1/grid_lim, c_F = r_ref/grid_lim
"""
from __future__ import annotations

import argparse, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
import torch
from anchorflow import scene_setup

ap = argparse.ArgumentParser()
ap.add_argument("--ply", required=True); ap.add_argument("--config", required=True)
ap.add_argument("--dreamphysics", default="/workspace/DreamPhysics")
ap.add_argument("--traj_cache", required=True); ap.add_argument("--fit", default=None)
ap.add_argument("--n_win", type=int, default=4)
ap.add_argument("--frames", type=int, default=30)
ap.add_argument("--grid_lim", type=float, default=2.0)
ap.add_argument("--gn", type=int, default=6); ap.add_argument("--cg", type=int, default=20)
ap.add_argument("--adam", type=int, default=800)
args = ap.parse_args()

sys.path.insert(0, args.dreamphysics)
import warp as wp
dev = "cuda"; torch.set_grad_enabled(False); wp.init()
from anchorflow.anchor_sparse import AnchorSparse, Traj, load_fitted
from anchorflow import frame_encode as fe
sys.modules["__main__"].Traj = Traj

sc = scene_setup.build(args.ply, args.config, 512, 8, device=dev,
                       frozen_weights=True, rot_fallback=True, eig_floor=0.02)
if args.fit:
    scw, _ = load_fitted(sc, args.fit, dev); fit, cache = scw.fit, scw._cache
else:
    fit = AnchorSparse(sc, c=0.25, eig_floor=0.02).to(dev); fit.init_from_geometry()
    cache = fit.prepare()
fac = fit.ls_factor(cache)
w, rc = cache[0], cache[1]
Yg = fit.Xc - rc
MASS = sc.volume[sc.keep].clone()
R_REF = float(Yg.norm(dim=-1).pow(2).mean().sqrt())
C_X, C_F = 1.0 / args.grid_lim, R_REF / args.grid_lim
FS = fe.FrameState(fit.pair_g, fit.pair_a, fit.N, fit.M, MASS)
print(f"[setup] 앵커 {fit.M}, c_x {C_X:.5g}, c_F {C_F:.5g}", flush=True)

# ---- 수반 검사 ------------------------------------------------------------
torch.manual_seed(0)
_u = torch.randn(fit.M, 3, device=dev) * 0.05
_s = torch.randn(fit.M, 3, device=dev) * 0.05
_F = FS.decode(_u, _s, w); _Jl = fe.left_jacobian(FS.blend(_u, _s, w)[0])
z = torch.randn(fit.M, 9, device=dev)
dx, dF = FS._apply(_F, _Jl, Yg, w, z)
gx = torch.randn(fit.N, 3, device=dev); gF = torch.randn(fit.N, 3, 3, device=dev)
lhs = float((dx * gx).sum() + (dF * gF).sum())
rhs = float((z * FS._applyT(_F, _Jl, Yg, w, gx, gF)).sum())
print(f"[검증] 수반 상대오차 {abs(lhs-rhs)/max(abs(lhs),1e-12):.3e}")

def mw(e2): return float(((MASS * e2).sum() / MASS.sum()).sqrt())
def loss_of(xh, F, x0, F0m):
    ex, ef = mw((xh - x0).pow(2).sum(-1)), mw((F - F0m).pow(2).sum((-1, -2)))
    return C_X * ex + C_F * ef, ex, ef

FIT = torch.load(args.traj_cache, map_location=dev, weights_only=False)["fit"]
g = torch.Generator(device="cpu"); g.manual_seed(20260905)
hi = max(1, args.frames - 12)
acc = {k: [0.0, 0.0, 0.0] for k in ("기존", "분리", "결합", "Adam")}
ts = {"분리": [], "결합": []}
for wi in range(args.n_win):
    i = int(torch.randint(len(FIT), (1,), generator=g).item())
    t = int(torch.randint(hi, (1,), generator=g).item())
    ent = FIT[i]; x0 = ent[0][t]; F0m = ent[2][t].reshape(-1, 3, 3).float()

    # 기존 상태 (p, v) + 형상 매칭
    p = fit.project_ls(x0, cache, fac)
    xl, _, Fl, _ = fit.lift(p, torch.zeros_like(p), cache)
    r = loss_of(xl, Fl.view(-1, 3, 3), x0, F0m)
    for k in range(3): acc["기존"][k] += r[k]

    # 분리 인코더: F 만 GN 으로 풀고 p 는 그 뒤 최소제곱
    torch.cuda.synchronize(); t0 = time.time()
    ug_t, sg_t = fe.polar_target(F0m)
    Lg = FS.gram(w)
    u0, s0 = FS._ls_init(ug_t, sg_t, w, L=Lg)
    u1, s1, _ = FS.encode(F0m, w, iters=args.gn, cg_iters=args.cg, init=(u0, s0))
    F1 = FS.decode(u1, s1, w)
    p1 = FS._wls(x0 - torch.einsum("nij,nj->ni", F1, Yg), w, Lg)
    xh1 = FS.decode_x(p1, u1, s1, w, Yg)[0]
    torch.cuda.synchronize(); ts["분리"].append(time.time() - t0)
    r = loss_of(xh1, F1, x0, F0m)
    for k in range(3): acc["분리"][k] += r[k]

    # 결합 인코더
    torch.cuda.synchronize(); t0 = time.time()
    p2, u2, s2, hist = FS.encode_joint(x0, F0m, w, Yg, c_x=C_X, c_F=C_F,
                                       iters=args.gn, cg_iters=args.cg,
                                       fixed=fit.fixed, p_fix=fit.pos,
                                       verbose=(wi == 0))
    torch.cuda.synchronize(); ts["결합"].append(time.time() - t0)
    xh2, F2 = FS.decode_x(p2, u2, s2, w, Yg)
    r = loss_of(xh2, F2, x0, F0m)
    for k in range(3): acc["결합"][k] += r[k]

    # 참조 하한: 같은 L 을 Adam 으로 더 내려본다
    with torch.enable_grad():
        pa = p2.clone().requires_grad_(True); ua = u2.clone().requires_grad_(True)
        sa = s2.clone().requires_grad_(True)
        opt = torch.optim.Adam([pa, ua, sa], lr=3e-3)
        for _ in range(args.adam):
            opt.zero_grad(set_to_none=True)
            xh, F = FS.decode_x(pa, ua, sa, w, Yg)
            ex = ((MASS * (xh - x0).pow(2).sum(-1)).sum() / MASS.sum()).sqrt()
            ef = ((MASS * (F - F0m).pow(2).sum((-1, -2))).sum() / MASS.sum()).sqrt()
            (C_X * ex + C_F * ef).backward()
            opt.step()
    xh3, F3 = FS.decode_x(pa.detach(), ua.detach(), sa.detach(), w, Yg)
    r = loss_of(xh3, F3, x0, F0m)
    for k in range(3): acc["Adam"][k] += r[k]
    print(f"  창 {wi+1}/{args.n_win} 완료", flush=True)

n = args.n_win
print(f"\n== 피팅 손실 L = c_x·mwRMS(dx) + c_F·mwRMS(dF)  (창 {n} 개 평균) ==")
print(f"{'인코더':<28}{'L':>12}{'기존 대비':>10}{'mwRMS x':>12}{'mwRMS F':>12}{'ms/창':>9}")
base = acc["기존"][0] / n
for k, nm in (("기존", "기존 상태 (p,v) 형상매칭"), ("분리", "분리: F 만 GN, p 는 사후 LS"),
              ("결합", "결합: (p,u,s) 동시 GN"), ("Adam", f"참조: 결합 + Adam {args.adam}")):
    L, ex, ef = [acc[k][j] / n for j in range(3)]
    tt = f"{1e3*sum(ts[k])/n:9.0f}" if k in ts else f"{'-':>9}"
    print(f"{nm:<28}{L:12.5e}{L/base:10.3f}{ex:12.4e}{ef:12.4f}{tt}")
print("\nJOINT_DONE")

"""새 프레임 인코더(회전벡터 + 목적함수 위 가우스-뉴턴)를 기존 것들과 견준다.

내는 것:
  (a) 야코비안 검증 -- 무행렬 K, K^T 를 오토그라드와 대조
  (b) F 잔차 사다리 -- 항등원 / 형상 매칭 / deconv / 새 인코더(LS 초기값, GN) /
      Adam 참조 하한 / 자유 9 자유도 최소제곱 절대 하한
  (c) 왕복 손실 세 항의 비중, 새 인코더 기준
  (d) 창당 소요 시간 -- 기하 피팅 루프에 넣을 수 있는지
"""
from __future__ import annotations

import argparse
import os
import sys
import time

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
ap.add_argument("--gn_iters", type=int, default=8)
ap.add_argument("--cg_iters", type=int, default=40)
ap.add_argument("--adam", type=int, default=600, help="참조 하한용 Adam 스텝 (0 이면 끔)")
ap.add_argument("--ridge", type=float, default=1e-6)
args = ap.parse_args()

sys.path.insert(0, args.dreamphysics)
import warp as wp

dev = "cuda"
torch.set_grad_enabled(False)
wp.init()
from anchorflow.anchor_sparse import AnchorSparse, Traj, load_fitted
from anchorflow.anchor_frame import AnchorFrame
from anchorflow import frame_encode as fe
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
w, rc, qq, Binv, blocked, _ = cache
Y = fit.Xc - rc
MASS = sc.volume[sc.keep].clone()
DT_C = args.dt_mult * sc.sub_dt
R_REF = float(Y.norm(dim=-1).pow(2).mean().sqrt())

FS = fe.FrameState(fit.pair_g, fit.pair_a, fit.N, fit.M, MASS)
frame = AnchorFrame(dev)

flat = fit.pair_g * fit.M + fit.pair_a
W = torch.zeros(fit.N * fit.M, device=dev).index_add_(0, flat, w).view(fit.N, fit.M)
free = ~fit.fixed
Gw = (W.t() @ W)[free][:, free]
Gw = Gw + args.ridge * Gw.diagonal().mean().clamp(min=1e-20) * torch.eye(int(free.sum()), device=dev)
Lw = torch.linalg.cholesky(Gw)
P_FIX = torch.where(fit.fixed.unsqueeze(-1), fit.pos, torch.zeros_like(fit.pos))
IDX_FREE = torch.nonzero(free, as_tuple=False).squeeze(-1)
# 자유 9 자유도 F 를 상태로 둘 때의 절대 하한에 쓰는 인수분해 (질량 가중 없음: 기존 프로브와 동일)
Gm = W.t() @ W
Gm = Gm + args.ridge * Gm.diagonal().mean().clamp(min=1e-20) * torch.eye(fit.M, device=dev)
Lm = torch.linalg.cholesky(Gm)

print(f"[setup] 앵커 {fit.M}, 가우시안 {fit.N}, dt_c {DT_C:.5g}, r_ref {R_REF:.5g}, "
      f"기하 {'피팅' if args.fit else '샘플링'}", flush=True)


def solve_p(x0, F_g):
    b = torch.einsum("nij,nj->ni", F_g, Y)
    rhs = (W.t() @ (x0 - b - W @ P_FIX))[free]
    return P_FIX.clone().index_put((IDX_FREE,), torch.cholesky_solve(rhs, Lw))


def mw(e2):
    return float(((MASS * e2).sum() / MASS.sum()).sqrt())


def mwF(F, F0m):
    return mw((F - F0m).pow(2).sum((-1, -2)))


# ---- (a) 야코비안 검증 ----------------------------------------------------
def check_jac():
    torch.manual_seed(0)
    n = 64
    F = torch.randn(n, 3, 3, device=dev) * 0.3 + torch.eye(3, device=dev)
    u = torch.randn(n, 3, device=dev) * 0.2
    Jl = fe.left_jacobian(u)
    z = torch.randn(n, 6, device=dev) * 0.1
    # K z 를 오토그라드의 방향미분과 대조: d/dt exp([u + t zu]x) diag(e^{s + t zs})
    s = torch.randn(n, 3, device=dev) * 0.1
    def Fof(uu, ss):
        return fe.expmap(uu) * ss.exp().unsqueeze(-2)
    with torch.enable_grad():
        t = torch.zeros((), device=dev, requires_grad=True)
        out = Fof(u + t * z[:, :3], s + t * z[:, 3:])
        seed = torch.randn_like(out)
        gd = torch.autograd.grad((out * seed).sum(), t)[0]
    Fb = Fof(u, s)
    pred = (FS._Kz(Fb, Jl, z) * seed).sum()
    e1 = abs(float(pred - gd)) / max(abs(float(gd)), 1e-12)
    # 수반: <K z, Y> == <z, K^T Y>
    Yr = torch.randn(n, 3, 3, device=dev)
    lhs = float((FS._Kz(Fb, Jl, z) * Yr).sum())
    rhs = float((z * FS._KTy(Fb, Jl, Yr)).sum())
    e2 = abs(lhs - rhs) / max(abs(lhs), 1e-12)
    print(f"[검증] K 방향미분 상대오차 {e1:.3e},  수반 <Kz,Y> vs <z,K^T Y> 상대오차 {e2:.3e}")
    return e1, e2


check_jac()

blob = torch.load(args.traj_cache, map_location=dev, weights_only=False)
FIT = blob["fit"]
print(f"[data] 궤적 {len(FIT)}", flush=True)

g = torch.Generator(device="cpu"); g.manual_seed(20260905)
hi = max(1, args.frames - 12)
rows, times = [], []
for wi in range(args.n_win):
    i = int(torch.randint(len(FIT), (1,), generator=g).item())
    t = int(torch.randint(hi, (1,), generator=g).item())
    ent = FIT[i]
    x0, v0 = ent[0][t], ent[1][t]
    F0 = ent[2][t].reshape(-1, 9).float()
    F0m = F0.view(-1, 3, 3)
    eye = torch.eye(3, device=dev).expand_as(F0m)

    p = fit.project_ls(x0, cache, fac)
    v = fit.project_v_ls(v0, cache, fac)
    xl, vl, Fl, _ = fit.lift(p, v, cache)
    e_shape = (mw((xl - x0).pow(2).sum(-1)), mw((vl - v0).pow(2).sum(-1)), mwF(Fl.view(-1, 3, 3), F0m))

    # 기존 deconv 인코더
    og_t, sg_t = AnchorFrame.target(F0m)
    o_a, s_a = AnchorFrame.deconv(og_t, sg_t, w, fit.pair_g, fit.pair_a, fit.M)
    F_dec = frame.blend(o_a, s_a, w, fit.pair_g, fit.pair_a, fit.N, "geo")[0]

    # 새 인코더
    ug_t, sg2 = fe.polar_target(F0m)
    torch.cuda.synchronize(); t0 = time.time()
    u0, s0 = FS._ls_init(ug_t, sg2, w)
    torch.cuda.synchronize(); t_ls = time.time() - t0
    F_ls = FS.decode(u0, s0, w)
    t0 = time.time()
    ug, sg, hist = FS.encode(F0m, w, iters=args.gn_iters, cg_iters=args.cg_iters,
                             init=(u0, s0))
    torch.cuda.synchronize(); t_gn = time.time() - t0
    F_gn = FS.decode(ug, sg, w)

    # Adam 참조 하한 (같은 파라미터화)
    F_ad = None
    if args.adam > 0:
        with torch.enable_grad():
            ua = ug.clone().requires_grad_(True); sa2 = sg.clone().requires_grad_(True)
            opt = torch.optim.Adam([ua, sa2], lr=0.01)
            for _ in range(args.adam):
                opt.zero_grad(set_to_none=True)
                Fg = FS.decode(ua, sa2, w)
                ((MASS * (Fg - F0m).pow(2).sum((-1, -2))).sum() / MASS.sum()).backward()
                opt.step()
        F_ad = FS.decode(ua.detach(), sa2.detach(), w)

    # 자유 9 자유도 절대 하한
    Fa = torch.cholesky_solve(W.t() @ F0, Lm)
    F_free = (W @ Fa).view(-1, 3, 3)

    # 새 인코더 기준 세 항
    x_hat = W @ solve_p(x0, F_gn) + torch.einsum("nij,nj->ni", F_gn, Y)
    e_new = (mw((x_hat - x0).pow(2).sum(-1)), e_shape[1], mwF(F_gn, F0m))

    rows.append(dict(shape=e_shape, new=e_new, ident=mwF(eye, F0m),
                     dec=mwF(F_dec, F0m), ls=mwF(F_ls, F0m), gn=mwF(F_gn, F0m),
                     adam=(mwF(F_ad, F0m) if F_ad is not None else None),
                     free=mwF(F_free, F0m), hist=hist))
    times.append((t_ls, t_gn))
    print(f"  창 {wi+1}/{args.n_win}  F: 항등 {rows[-1]['ident']:.4f} | 형상 "
          f"{e_shape[2]:.4f} | deconv {rows[-1]['dec']:.4f} | 새LS {rows[-1]['ls']:.4f} "
          f"| 새GN {rows[-1]['gn']:.4f} | Adam {rows[-1]['adam']}"
          f" | 자유F {rows[-1]['free']:.4f}   ({t_ls*1e3:.0f}+{t_gn*1e3:.0f} ms)", flush=True)

n = len(rows)
A = lambda k: sum(r[k] for r in rows) / n
ID = A("ident")
print(f"\n== F 잔차 사다리 (평균, 창 {n} 개) ==")
print(f"{'방식':<34}{'mwRMS':>10}{'항등원 대비':>12}")
lad = [("아무것도 안 함 (F = I)", ID),
       ("형상 매칭 (상태 = p, v)", A("shape") if False else sum(r["shape"][2] for r in rows) / n),
       ("기존 deconv 인코더 (쿼터니언)", A("dec")),
       ("새 인코더: 로그공간 최소제곱 초기값", A("ls")),
       (f"새 인코더: 가우스-뉴턴 {args.gn_iters} 회", A("gn"))]
if rows[0]["adam"] is not None:
    lad.append((f"참조: 위에서 Adam {args.adam} 스텝 더", A("adam")))
lad.append(("절대 하한: 자유 9 자유도 F 를 상태로", A("free")))
for nm, v in lad:
    print(f"{nm:<34}{v:10.4f}{100*v/ID:11.1f}%")

CO = (1.0 / args.grid_lim, DT_C / args.grid_lim, R_REF / args.grid_lim)
NAMES = ("x", "v", "F")


def table(title, raw):
    conv = [raw[k] * CO[k] for k in range(3)]
    tot = sum(conv)
    print(f"\n== {title} ==")
    print(f"{'항':>4} {'원시 mwRMS':>14} {'환산 계수':>14} {'손실 기여':>14} {'비중':>8}")
    for k in range(3):
        print(f"{NAMES[k]:>4} {raw[k]:13.5e} {CO[k]:13.5e} {conv[k]:13.5e} "
              f"{100*conv[k]/tot:7.1f}%")
    print(f"{'합':>4} {'':>13} {'':>13} {tot:13.5e} {100.0:7.1f}%")
    return tot


t_b = table("기존 상태 (p, v)", [sum(r["shape"][k] for r in rows) / n for k in range(3)])
t_n = table("확장 상태 (p, v, u, s) -- 새 인코더", [sum(r["new"][k] for r in rows) / n for k in range(3)])
print(f"\n왕복 손실 합: {t_b:.5e} -> {t_n:.5e}  ({t_b/max(t_n,1e-30):.2f} 배)")
tl = sum(t[0] for t in times) / n; tg = sum(t[1] for t in times) / n
print(f"인코더 시간(창당): 최소제곱 초기값 {tl*1e3:.0f} ms + 가우스-뉴턴 {tg*1e3:.0f} ms "
      f"= {(tl+tg)*1e3:.0f} ms")
h = rows[0]["hist"]
print("첫 창의 GN 손실: " + " -> ".join(f"{x:.4e}" for x in h))
print("\nFRAME_ENCODE_DONE")

"""앵커 상태에 쿼터니언과 신축을 추가하고, 왕복 손실 세 항의 크기를 다시 잰다.

지금까지 앵커가 든 것은 (p, v) 뿐이었고 가우시안의 F 는 앵커 위치장의 미분 --
형상 매칭 -- 으로 유도됐다. 그 F 는 항등원 대비 99.4% 로, 사실상 아무것도 담지
못한다. 여기서는 앵커마다 쿼터니언 o_a 와 로그 신축 s_a 를 상태로 두고

    o_g = blend(o_a),  s_g = sum_a w_ga s_a,  F_g = R(o_g) diag(e^{s_g})

로 F 를 조립한다. F 가 상태에서 오므로 위치 복호도 바뀐다:

    x_g = sum_a w_ga p_a + F_g (X_g - r_g)

즉 p 는 형상 매칭 기저가 아니라 이 복호에 대해 다시 최소제곱으로 푼다.

세 항을 (기존 상태 = p,v) 와 (확장 상태 = p,v,o,s) 에서 각각 재어 비중을 견준다.
인코더(target+deconv)가 약한 것인지 상태 자체가 약한 것인지 가르기 위해, 확장
상태는 인코더 그대로와 거기서 경사하강으로 다듬은 것 둘 다 낸다.
"""
from __future__ import annotations

import argparse
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
ap.add_argument("--traj_cache", required=True)
ap.add_argument("--fit", default=None)
ap.add_argument("--n_win", type=int, default=8)
ap.add_argument("--dt_mult", type=int, default=40)
ap.add_argument("--frames", type=int, default=30)
ap.add_argument("--grid_lim", type=float, default=2.0)
ap.add_argument("--mode", default="geo", choices=("geo", "nlerp"))
ap.add_argument("--refine", type=int, default=300, help="확장 상태를 다듬는 스텝 수 (0 이면 끔)")
ap.add_argument("--refine_lr", type=float, default=0.02)
ap.add_argument("--ridge", type=float, default=1e-6)
args = ap.parse_args()

sys.path.insert(0, args.dreamphysics)
import warp as wp

dev = "cuda"
torch.set_grad_enabled(False)
wp.init()
from anchorflow.anchor_sparse import AnchorSparse, Traj, load_fitted
from anchorflow.anchor_frame import AnchorFrame
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
Y = fit.Xc - rc                                     # [N,3]  X_g - r_g
MASS = sc.volume[sc.keep].clone()
DT_C = args.dt_mult * sc.sub_dt
R_REF = float(Y.norm(dim=-1).pow(2).mean().sqrt())
frame = AnchorFrame(dev)

# 프레임 복호에서 위치의 앵커 계수: x = W p + F Y  ->  W 는 그냥 가중치 행렬
flat = fit.pair_g * fit.M + fit.pair_a
W = torch.zeros(fit.N * fit.M, device=dev).index_add_(0, flat, w).view(fit.N, fit.M)
free = ~fit.fixed
Gw = (W.t() @ W)[free][:, free]
Gw = Gw + args.ridge * Gw.diagonal().mean().clamp(min=1e-20) * torch.eye(int(free.sum()), device=dev)
Lw = torch.linalg.cholesky(Gw)
P_FIX = torch.where(fit.fixed.unsqueeze(-1), fit.pos, torch.zeros_like(fit.pos))
IDX_FREE = torch.nonzero(free, as_tuple=False).squeeze(-1)

print(f"[setup] 앵커 {fit.M} (고정 {int(fit.fixed.sum())}), dt_c {DT_C:.5g}, "
      f"r_ref {R_REF:.5g}, grid_lim {args.grid_lim:g}, 블렌드 {args.mode}, "
      f"기하 {'피팅' if args.fit else '샘플링'}", flush=True)


def solve_p(x0, F_g):
    """x = W p + F_g Y 에 대한 최소제곱 p"""
    b = torch.einsum("nij,nj->ni", F_g, Y)
    rhs = (W.t() @ (x0 - b - W @ P_FIX))[free]
    pf = torch.cholesky_solve(rhs, Lw)
    return P_FIX.clone().index_put((IDX_FREE,), pf)


def mw(e2):
    return float(((MASS * e2).sum() / MASS.sum()).sqrt())


blob = torch.load(args.traj_cache, map_location=dev, weights_only=False)
FIT = blob["fit"]
if len(FIT[0]) < 3:
    raise SystemExit("F 가 없는 캐시다")
print(f"[data] 궤적 {len(FIT)}", flush=True)

g = torch.Generator(device="cpu"); g.manual_seed(20260905)
hi = max(1, args.frames - 12)
rows = []
for wi in range(args.n_win):
    i = int(torch.randint(len(FIT), (1,), generator=g).item())
    t = int(torch.randint(hi, (1,), generator=g).item())
    ent = FIT[i]
    x0, v0 = ent[0][t], ent[1][t]
    F0 = ent[2][t].reshape(-1, 9).float()
    F0m = F0.view(-1, 3, 3)

    # --- 기존 상태 (p, v): F 는 형상 매칭 ---------------------------------
    p = fit.project_ls(x0, cache, fac)
    v = fit.project_v_ls(v0, cache, fac)
    xl, vl, Fl, _ = fit.lift(p, v, cache)
    base = (mw((xl - x0).pow(2).sum(-1)), mw((vl - v0).pow(2).sum(-1)),
            mw((Fl - F0).pow(2).sum(-1)))

    # --- 확장 상태 (p, v, o, s): F 는 프레임에서 조립 ---------------------
    og_t, sg_t = AnchorFrame.target(F0m)
    o_a, s_a = AnchorFrame.deconv(og_t, sg_t, w, fit.pair_g, fit.pair_a, fit.M)
    F_enc = frame.blend(o_a, s_a, w, fit.pair_g, fit.pair_a, fit.N, args.mode)[0]
    x_enc = solve_p(x0, F_enc)
    x_enc = W @ x_enc + torch.einsum("nij,nj->ni", F_enc, Y)
    enc = (mw((x_enc - x0).pow(2).sum(-1)), base[1],
           mw((F_enc.reshape(-1, 9) - F0).pow(2).sum(-1)))

    # --- 확장 상태를 경사하강으로 다듬은 것 -------------------------------
    ref = None
    if args.refine > 0:
        with torch.enable_grad():
            oa = o_a.clone().requires_grad_(True)
            sa = s_a.clone().requires_grad_(True)
            opt = torch.optim.Adam([oa, sa], lr=args.refine_lr)
            for _ in range(args.refine):
                opt.zero_grad(set_to_none=True)
                on = oa / oa.norm(dim=-1, keepdim=True).clamp(min=1e-12)
                Fg = frame.blend(on, sa, w, fit.pair_g, fit.pair_a, fit.N, args.mode)[0]
                loss = (MASS * (Fg.reshape(-1, 9) - F0).pow(2).sum(-1)).sum() / MASS.sum()
                loss.backward()
                opt.step()
        with torch.no_grad():
            on = oa / oa.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            F_ref = frame.blend(on, sa.detach(), w, fit.pair_g, fit.pair_a, fit.N, args.mode)[0]
        x_ref = solve_p(x0, F_ref)
        x_ref = W @ x_ref + torch.einsum("nij,nj->ni", F_ref, Y)
        ref = (mw((x_ref - x0).pow(2).sum(-1)), base[1],
               mw((F_ref.reshape(-1, 9) - F0).pow(2).sum(-1)))

    eye = torch.eye(3, device=dev).reshape(1, 9).expand_as(F0)
    rows.append((base, enc, ref, mw((eye - F0).pow(2).sum(-1))))
    print(f"  창 {wi+1}/{args.n_win}  F: 형상매칭 {base[2]:.4f} / 인코더 "
          f"{enc[2]:.4f}" + (f" / 다듬음 {ref[2]:.4f}" if ref else ""), flush=True)

n = len(rows)


def avg(sel):
    return [sum(sel(r)[k] for r in rows) / n for k in range(3)]


F_ID = sum(r[3] for r in rows) / n
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


t_base = table("기존 상태 (p, v) -- F 는 형상 매칭", avg(lambda r: r[0]))
t_enc = table("확장 상태 (p, v, o, s) -- 인코더 그대로", avg(lambda r: r[1]))
if rows[0][2] is not None:
    t_ref = table(f"확장 상태 (p, v, o, s) -- {args.refine} 스텝 다듬음", avg(lambda r: r[2]))

b, e = avg(lambda r: r[0]), avg(lambda r: r[1])
print(f"\nF 원시: 형상매칭 {b[2]:.4f} (항등원 대비 {100*b[2]/F_ID:.1f}%), "
      f"프레임 인코더 {e[2]:.4f} ({100*e[2]/F_ID:.1f}%)", end="")
if rows[0][2] is not None:
    rr = avg(lambda r: r[2])
    print(f", 다듬음 {rr[2]:.4f} ({100*rr[2]/F_ID:.1f}%)", end="")
print(f", 항등원 {F_ID:.4f}")
print(f"왕복 손실 합: 기존 {t_base:.5e} -> 확장 {t_enc:.5e} "
      f"({t_base/max(t_enc,1e-30):.2f} 배 감소)")
print("\nFRAME_STATE_DONE")

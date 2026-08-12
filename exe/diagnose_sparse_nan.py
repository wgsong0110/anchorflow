"""Why does the sparse fit go to NaN around its 200th iteration?

The first NaN in this module was identified exactly -- the eigendecomposition's
backward divides by the gap between B's eigenvalues, and a compact support makes
that gap zero. The second was not: guards went in for the plausible causes
without establishing which one it was, and a guard on the wrong thing leaves the
run quietly wrong instead of loudly broken.

The log says one thing about it. At iteration 200 the held-out evaluation was
NaN while the training loss was still 83.3%, so the parameters were finite and a
ROLLOUT had blown up; by 240 the loss and the parameters had followed. A rollout
that blows up under an explicit integrator with a fixed substep is what happens
when the discretisation gets stiff enough that the substep exceeds the stability
limit -- and the discretisation is what is being fitted. That is a hypothesis,
and this reproduces the failure with the guards off to test it.

Every iteration records what would move if that were the cause: the smallest
anchor extent, the smallest anchor mass, and the largest acceleration seen. At
the first non-finite quantity the offending step is replayed substep by substep.
Growth that compounds every substep is a stability failure; a single spike is
not, and would point elsewhere.
"""
from __future__ import annotations

import argparse
import os
import sys

_lib = os.path.join(os.path.dirname(__file__), "..", "lib")
sys.path.insert(0, _lib)

import torch
from tqdm import tqdm

from anchorflow import scene_setup
from anchorflow.anchor_sparse import AnchorSparse

ap = argparse.ArgumentParser()
ap.add_argument("--ply", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--dreamphysics", default="/workspace/DreamPhysics")
ap.add_argument("--traj_cache", required=True)
ap.add_argument("--iters", type=int, default=320)
ap.add_argument("--batch", type=int, default=2)
ap.add_argument("--dt_mult", type=int, default=40)
ap.add_argument("--frames", type=int, default=60)
ap.add_argument("--warmup", type=int, default=80)
ap.add_argument("--c", type=float, default=0.25)
ap.add_argument("--eig_floor", type=float, default=0.02)
ap.add_argument("--polar_iters", type=int, default=6)
ap.add_argument("--lr_pos", type=float, default=3e-4)
ap.add_argument("--lr_scale", type=float, default=1e-2)
ap.add_argument("--lr_quat", type=float, default=1e-2)
ap.add_argument("--refresh_every", type=int, default=20)
ap.add_argument("--densify_every", type=int, default=60)
ap.add_argument("--no_bref", action="store_true",
                 help="remove the absolute floor under the scatter regulariser, "
                      "leaving it proportional to B as it was when the fit failed. "
                      "The controlled version of 'is that what it was'.")
ap.add_argument("--guards", action="store_true",
                 help="leave the guards on, to check they actually hold")
args = ap.parse_args()

sys.path.insert(0, args.dreamphysics)
import warp as wp

dev = "cuda"
torch.manual_seed(0)
wp.init()

sc = scene_setup.build(args.ply, args.config, 512, 8, device=dev, frozen_weights=True,
                        rot_fallback=True, eig_floor=args.eig_floor)
fit = AnchorSparse(sc, c=args.c, eig_floor=args.eig_floor,
                    polar_iters=args.polar_iters).to(dev)
if not args.guards:
    fit.s_lo, fit.s_hi, fit.polar_ridge = 1e-9, 1e9, 0.0
fit.init_from_geometry()
if args.no_bref:
    fit.B_ref.zero_()
    fit.set_B_ref = lambda *a, **k: 0.0
blob = torch.load(args.traj_cache, map_location=dev, weights_only=False)
FIT = blob["fit"]
print(f"[setup] guards {'on' if args.guards else 'OFF'}, B_ref "
      f"{'OFF' if args.no_bref else f'{float(fit.B_ref):.3e}'}, {fit.M} anchors, "
      f"{fit.pair_g.shape[0]} pairs, scale bounds "
      f"[{fit.s_lo:.2e}, {fit.s_hi:.2e}], polar ridge {fit.polar_ridge}")


def make_opt():
    return torch.optim.Adam([{"params": [fit.pos], "lr": args.lr_pos},
                             {"params": [fit.log_s], "lr": args.lr_scale},
                             {"params": [fit.quat], "lr": args.lr_quat}])


def trace(x0, v0, cache, n):
    """one coarse frame, substep by substep, recording what a stability failure
    would show: velocity and acceleration compounding rather than spiking"""
    w, rc, q, Binv, blocked, mass = cache
    m = mass.unsqueeze(-1)
    keep = (~fit.fixed).unsqueeze(-1).to(torch.float32)
    p, v = fit.project(x0, cache), fit.project_v(v0, cache)
    rows = []
    for k in range(n):
        a = fit.force(p, w, rc, q, Binv, blocked) / m
        v = (v + fit.dt * a) * fit.damping * keep
        p = p + fit.dt * v
        F, _ = fit.deformation(p, w, rc, q, Binv, blocked)
        rows.append((k, a.norm(dim=-1).max().item(), v.norm(dim=-1).max().item(),
                     torch.linalg.det(F).min().item(),
                     (p - fit.pos).norm(dim=-1).max().item()))
        if not torch.isfinite(p).all():
            break
    return rows


opt = make_opt()
grad_accum = torch.zeros(fit.M, device=dev)
print(f"\n{'it':>4} {'loss':>9} {'s min':>9} {'mass min':>10} {'|a| max':>10} "
      f"{'detF min':>9} {'M':>5}")
for it in range(1, args.iters + 1):
    if it > 1 and it % args.refresh_every == 0:
        fit.refresh()
    if args.densify_every and it % args.densify_every == 0 and it <= 0.7 * args.iters:
        fit.densify_and_prune(grad_accum, max_anchors=1024)
        opt = make_opt()
        grad_accum = torch.zeros(fit.M, device=dev)

    hi = max(2, int(min(1.0, it / args.warmup) * (args.frames - 1)))
    cache = fit.prepare()
    picks, loss = [], 0.0
    for _ in range(args.batch):
        X, V = FIT[torch.randint(len(FIT), (1,)).item()]
        t = torch.randint(hi, (1,)).item()
        picks.append((X, V, t))
        p = fit.project(X[t], cache)
        v = fit.project_v(V[t], cache)
        p, _ = fit.rollout(p, v, args.dt_mult, cache)
        got = fit.gaussian_pos(p, cache)
        d = (X[t + 1] - X[t]).norm(dim=-1).mean().clamp(min=1e-12)
        loss = loss + ((got - X[t + 1]).norm(dim=-1).mean() / d)
    loss = loss / args.batch

    with torch.no_grad():
        w, rc, q, Binv, blocked, mass = cache
        X, V, t = picks[0]
        a0 = fit.force(fit.project(X[t], cache), w, rc, q, Binv, blocked) / \
            mass.unsqueeze(-1)
        F0, _ = fit.deformation(fit.project(X[t], cache), w, rc, q, Binv, blocked)
        stats = (fit.log_s.exp().min().item(), mass.min().item(),
                 a0.norm(dim=-1).max().item(), torch.linalg.det(F0).min().item())

    bad = None
    if not torch.isfinite(loss):
        bad = "loss"
    else:
        opt.zero_grad(set_to_none=True)
        loss.backward()
        for n_, prm in fit.named_parameters():
            if prm.grad is not None and not torch.isfinite(prm.grad).all():
                bad = f"grad[{n_}]"
                break

    if it % 10 == 0 or bad:
        print(f"{it:4d} {loss.item():9.4f} {stats[0]:9.2e} {stats[1]:10.2e} "
              f"{stats[2]:10.2e} {stats[3]:9.4f} {fit.M:5d}", flush=True)
    if bad:
        print(f"\n[first failure] iteration {it}: {bad} is not finite")
        X, V, t = picks[0]
        print(f"  replaying the sample at frame {t}, substep by substep")
        print(f"  {'substep':>8} {'|a| max':>12} {'|v| max':>12} {'detF min':>10} "
              f"{'moved':>10}")
        for k, am, vm, dm, mv in trace(X[t], V[t], cache, args.dt_mult):
            if k < 4 or k % 4 == 0 or k >= args.dt_mult - 4:
                print(f"  {k:8d} {am:12.3e} {vm:12.3e} {dm:10.4f} {mv:10.4f}")
        with torch.no_grad():
            s = fit.log_s.exp()
            j = mass.argmin()
            print(f"\n  smallest anchor: mass {mass[j]:.3e}, extent "
                  f"{s[j].min():.3e} (started {sc.sim.radius:.4f}), "
                  f"holds {int((fit.pair_a == j).sum())} Gaussians")
            print(f"  extents: min {s.min():.3e}, 1% {s.flatten().quantile(0.01):.3e}, "
                  f"median {s.median():.3e}, max {s.max():.3e}")
            print(f"  masses:  min {mass.min():.3e}, 1% {mass.quantile(0.01):.3e}, "
                  f"median {mass.median():.3e}")
            # what the substep would have to be for this configuration
            print(f"\n  the explicit step is stable while dt < ~2/omega, and omega "
                  f"grows as an\n  anchor's mass falls: mass has moved by a factor "
                  f"{mass.median().item() / mass.min().item():.1f} across the set.")
        break
    torch.nn.utils.clip_grad_norm_(list(fit.parameters()), 1.0)
    opt.step()
    if args.guards:
        fit.clamp_()
    with torch.no_grad():
        if fit.pos.grad is not None:
            grad_accum += fit.pos.grad.norm(dim=-1)
else:
    print(f"\n[no failure] {args.iters} iterations, guards "
          f"{'on' if args.guards else 'off'}")

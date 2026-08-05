"""Train the anchor dynamics directly from explicit-simulator trajectories.

No physics in the loss, the targets or the output path: the target is where the
simulator's anchors actually were one coarse step later, and the loss is L2 on
the displacement. See lib/anchorflow/nextstate.py for why the state is a history
of positions rather than (v, a).

Evaluation is the only thing that matters here and it is done on the network's
OWN rollout, not on single steps: position error against the reference
trajectory, frame by frame, which is directly comparable across runs and does
not depend on any physics quantity.
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
from anchorflow.nextstate import NextStep, roll_history

ap = argparse.ArgumentParser()
ap.add_argument("--ply", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--n_anchors", type=int, default=512)
ap.add_argument("--K", type=int, default=8)
ap.add_argument("--dt_mult", type=int, default=40,
                 help="coarse step, in explicit substeps. One network call replaces this many.")
ap.add_argument("--n_traj", type=int, default=12)
ap.add_argument("--n_steps", type=int, default=60, help="coarse steps per trajectory")
ap.add_argument("--history", type=int, default=2)
ap.add_argument("--hidden", type=int, default=128)
ap.add_argument("--depth", type=int, default=4)
ap.add_argument("--heads", type=int, default=4)
ap.add_argument("--iters", type=int, default=30000)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--noise", type=float, default=0.0,
                 help="random-walk noise on the input history, as a fraction of the typical "
                      "displacement. GNS's fix for rollout error accumulation: the network "
                      "only ever sees clean simulator states otherwise, and has no idea what "
                      "to do once its own error has moved it off that manifold.")
ap.add_argument("--eval_every", type=int, default=5000)
ap.add_argument("--eval_frames", type=int, default=60)
ap.add_argument("--out", default=None)
args = ap.parse_args()

dev = "cuda"
torch.manual_seed(0)
sc = scene_setup.build(args.ply, args.config, args.n_anchors, args.K, device=dev)
AC, M, fixed = sc.anchor_canonical, sc.M, sc.fixed_mask
print(f"[setup] N={sc.N} M={M} pinned={int(fixed.sum())} "
      f"coarse dt={args.dt_mult * sc.sub_dt} ({args.dt_mult} substeps)")

# ---------------- data: coarse-sampled explicit trajectories ----------------
base_force = None
for bc in sc.cfg.get("boundary_conditions", []):
    if bc["type"] == "particle_impulse":
        base_force = torch.tensor(bc["force"], device=dev)
gen = torch.Generator(device=dev); gen.manual_seed(1234)


def rand_rot():
    q, r = torch.linalg.qr(torch.randn(3, 3, device=dev, generator=gen))
    q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def trajectory(force):
    """anchor positions sampled every dt_mult substeps -> [n_steps+1, M, 3]"""
    p, v = AC.clone(), sc.initial_velocity(force)
    gp = sc.pos.clone()
    out = [p.clone()]
    for _ in range(args.n_steps):
        p, v, gp = sc.explicit_step(p, v, gp, args.dt_mult)
        if not torch.isfinite(p).all():
            break
        out.append(p.clone())
    return torch.stack(out)


print(f"[data] {args.n_traj} trajectories x {args.n_steps} coarse steps...")
trajs = []
for t in tqdm(range(args.n_traj), desc="collect", ncols=90):
    if t == 0 or base_force is None:
        f = base_force
    else:
        s = 0.5 * (4.0 ** torch.rand(1, device=dev, generator=gen).item())
        f = (rand_rot() @ base_force) * s
    trajs.append(trajectory(f))
REF = trajs[0]                              # the config's own run, for evaluation

# (trajectory, step) pairs that have a full history behind them and a target ahead
pairs = [(ti, k) for ti, tr in enumerate(trajs)
         for k in range(args.history, tr.shape[0] - 1)]
DISP = [tr[1:] - tr[:-1] for tr in trajs]
DISP_SCALE = float(torch.cat([d.norm(dim=-1).flatten() for d in DISP]).mean())
ACC_SCALE = float(torch.cat([(d[1:] - d[:-1]).norm(dim=-1).flatten() for d in DISP]).mean())
print(f"[data] {len(pairs)} training pairs; typical coarse displacement = {DISP_SCALE:.5f}, "
      f"typical change in it = {ACC_SCALE:.5f}")
print(f"[data] reference peak anchor displacement = "
      f"{(REF - AC).norm(dim=-1).max().item():.5f}")

dt_coarse = args.dt_mult * sc.sub_dt
net = NextStep(args.hidden, args.depth, args.heads, args.history, DISP_SCALE,
                ACC_SCALE).to(dev)
opt = torch.optim.Adam(net.parameters(), lr=args.lr)
print(f"[setup] params: {sum(q.numel() for q in net.parameters())/1e3:.1f}k")


def history_of(tr, k):
    """the `history` most recent displacements ending at step k, newest first"""
    return torch.stack([tr[k - i] - tr[k - i - 1] for i in range(args.history)], dim=1)


@torch.no_grad()
def rollout(frames):
    """autoregressive from the reference's initial condition; returns positions"""
    p = REF[args.history].clone()
    hist = history_of(REF, args.history).clone()
    out = [p.clone()]
    for _ in range(frames):
        du = net(p, hist, dt_coarse)
        du = torch.where(fixed.unsqueeze(-1), torch.zeros_like(du), du)
        p = p + du
        hist = roll_history(hist, du)
        if not torch.isfinite(p).all():
            break
        out.append(p.clone())
    return torch.stack(out)


@torch.no_grad()
def rollout_error(frames):
    got = rollout(frames)
    n = min(got.shape[0], REF.shape[0] - args.history)
    ref = REF[args.history:args.history + n]
    err = (got[:n] - ref).norm(dim=-1).mean(-1)          # mean anchor error per frame
    span = (ref - AC).norm(dim=-1).max().clamp(min=1e-12)
    return n, err, (err / span)


hist_log = []
pbar = tqdm(range(1, args.iters + 1), desc="train", ncols=105)
for it in pbar:
    ti, k = pairs[torch.randint(len(pairs), (1,)).item()]
    tr = trajs[ti]
    p = tr[k]
    h = history_of(tr, k)
    if args.noise > 0:
        # random walk on the history, the way GNS does it: perturb the past
        # displacements and move the position consistently, so the (position,
        # history) pair stays a state the network could actually have produced
        nz = torch.randn(h.shape, device=dev, generator=gen) * (args.noise * DISP_SCALE)
        nz[fixed] = 0
        h = h + nz
        p = p + nz.sum(1)
    target = tr[k + 1] - tr[k]
    du = net(p, h, dt_coarse)
    du = torch.where(fixed.unsqueeze(-1), torch.zeros_like(du), du)
    loss = ((du - target) ** 2).mean()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    opt.step()

    if it % 50 == 0 or it == 1:
        rel = (du - target).norm() / target.norm().clamp(min=1e-20)
        hist_log.append((it, loss.item(), rel.item()))
        pbar.set_postfix(mse=f"{loss.item():.2e}", rel=f"{rel.item():.4f}")
    if it % args.eval_every == 0 or it == args.iters:
        n, err, rel = rollout_error(args.eval_frames)
        print(f"\n  [rollout] it={it} survived {n}/{args.eval_frames+1} frames; "
              f"mean anchor error vs reference: "
              f"frame10={err[min(10,n-1)]:.5f} frame{n-1}={err[n-1]:.5f} "
              f"({rel[n-1]*100:.1f}% of the reference's own motion)", flush=True)

print("\n[result]   iter        step MSE   rel step err")
for it, ls, rl in hist_log[::max(1, len(hist_log) // 20)]:
    print(f"  {it:8d}   {ls:.6e}   {rl:10.4f}")

n, err, rel = rollout_error(args.eval_frames)
print(f"\n[rollout] {n}/{args.eval_frames+1} frames survived")
print(f"  {'frame':>6} {'mean anchor err':>16} {'% of reference motion':>22}")
for i in range(0, n, max(1, n // 12)):
    print(f"  {i:6d} {err[i]:16.5f} {rel[i]*100:21.2f}%")

if args.out:
    torch.save({"model": net.state_dict(), "args": vars(args), "hist": hist_log,
                "disp_scale": DISP_SCALE, "acc_scale": ACC_SCALE}, args.out)
    print(f"[save] {args.out}")

"""What is different about the held-out trajectory the rollout runs away on?

D_field diverges on exactly one of twenty force-field trajectories -- 493% of
the reference's own motion, reproduced at a second seed (323%) -- and is fine on
the other nineteen and on all twenty uniform-impulse ones. Before trying to fix
that, it is worth knowing what distinguishes it, because the two obvious
candidates call for different work:

  * amplitude. If the failures are the trajectories that deform furthest, the
    training data does not reach them and the answer is coverage. (Reaching
    further was tried directly -- repeated impulses, C_stream -- and cost 6.6
    points on the uniform set, so "just add bigger ones" is already known not to
    work unmodified.)
  * spatial scale. If they are the ones forced at a particular correlation
    length, something about that length is unrepresented or genuinely harder.

So: every held-out field trajectory's forcing scale, how far it actually moves,
and what the rollout does on it -- against the same statistics for the
trajectories the network trained on. Plus, for the failures, the frame the
rollout leaves and where the state is by then relative to anything training saw.
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
from anchorflow.nextstate import NextStep
from anchorflow.streams import draw_impulse

ap = argparse.ArgumentParser()
ap.add_argument("--ply", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--ckpt", required=True)
ap.add_argument("--traj_cache", required=True,
                 help="the trajectories the checkpoint trained on, for comparison")
ap.add_argument("--n_field", type=int, default=20)
ap.add_argument("--n_holdout", type=int, default=5,
                 help="how many of the cached trajectories were held out, so the training "
                      "statistics are taken from the ones actually trained on")
ap.add_argument("--frames", type=int, default=60)
ap.add_argument("--dt_mult", type=int, default=40)
ap.add_argument("--n_anchors", type=int, default=512)
ap.add_argument("--K", type=int, default=8)
ap.add_argument("--impulse_range", type=float, default=4.0)
ap.add_argument("--runaway", type=float, default=1.5,
                 help="amplitude, as a fraction of the reference's own peak, above which a "
                      "rollout is called a runaway rather than an inaccurate one")
args = ap.parse_args()

dev = "cuda"
torch.set_grad_enabled(False)
torch.manual_seed(0)

ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
ta = ck["args"]
sc = scene_setup.build(args.ply, args.config, args.n_anchors, args.K, device=dev,
                        frozen_weights=bool(ta.get("frozen_weights", False)))
AC, fixed = sc.anchor_canonical, sc.fixed_mask
dt = args.dt_mult * sc.sub_dt
USE_A = not ta.get("no_accel", False)
net = NextStep(ta["hidden"], ta["depth"], ta["heads"], ck["disp_scale"], ck["vel_scale"],
               ck["acc_scale"], use_accel=USE_A).to(dev)
net.load_state_dict(ck["model"]); net.eval()
print(f"[setup] {os.path.basename(args.ckpt)}, frozen={ta.get('frozen_weights')}, "
      f"use_accel={USE_A}")

base_force = None
for bc in sc.cfg.get("boundary_conditions", []):
    if bc["type"] == "particle_impulse":
        base_force = torch.tensor(bc["force"], device=dev)

# ---- what training saw -----------------------------------------------------
blob = torch.load(args.traj_cache, map_location=dev, weights_only=False)
train = blob["trajs"][args.n_holdout:]
tp = torch.tensor([(t - AC)[:, ~fixed].norm(dim=-1).max().item() for t in train])
td = torch.cat([((t[1:] - t[:-1])[:, ~fixed].norm(dim=-1).max(-1).values) for t in train])
print(f"[training] {len(train)} trajectories")
print(f"  peak displacement : median {tp.median():.4f}, 90th {tp.quantile(0.9):.4f}, "
      f"max {tp.max():.4f}")
print(f"  per-step, max over anchors : median {td.median():.5f}, "
      f"99th {td.quantile(0.99):.5f}, max {td.max():.5f}")


def trajectory(force):
    p, v, gp = AC.clone(), sc.initial_velocity(force), sc.pos.clone()
    ps = [p.clone()]
    for _ in range(args.frames):
        p, v, gp = sc.explicit_step(p, v, gp, args.dt_mult)
        if not torch.isfinite(p).all():
            break
        ps.append(p.clone())
    return torch.stack(ps)


def rollout(r):
    p = r[1].clone()
    v = (r[1] - r[0]) / dt
    gp = sc.skin(p, sc.pos.clone()) if USE_A else None
    errs, amps, steps = [], [], []
    for k in range(r.shape[0] - 1):
        errs.append((p - r[1 + k])[~fixed].norm(dim=-1).mean().item())
        amps.append((p - AC)[~fixed].norm(dim=-1).max().item())
        a = sc.elastic_accel(p, gp) if USE_A else None
        du = net(p, v, a, dt)
        du = torch.where(fixed.unsqueeze(-1), torch.zeros_like(du), du)
        steps.append(du[~fixed].norm(dim=-1).max().item())
        p = p + du
        v = du / dt
        if not torch.isfinite(p).all():
            break
        if USE_A:
            gp = sc.skin(p, gp)
    return errs, amps, steps


gen = torch.Generator(device=dev); gen.manual_seed(5678)
rows = []
for i in tqdm(range(args.n_field), desc="field", ncols=90):
    f, sig = draw_impulse(sc, base_force, gen, args.impulse_range, field=True)
    r = trajectory(f)
    errs, amps, steps = rollout(r)
    span = (r - AC).norm(dim=-1).max().item()
    ref_step = (r[1:] - r[:-1])[:, ~fixed].norm(dim=-1).max().item()
    over = [k for k, a in enumerate(amps) if a > args.runaway * span]
    rows.append({"i": i, "sigma": sig, "span": span, "ref_step": ref_step,
                 "err": sum(errs) / len(errs) / span,
                 "amp": max(amps) / span, "left": (over[0] if over else None),
                 "steps": steps, "amps": amps, "errs": errs})

print(f"\n{'traj':>5} {'sigma':>8} {'x spacing':>10} {'ref peak':>9} {'vs train':>9} "
      f"{'err':>8} {'amp':>7} {'ran away':>9}")
for r in sorted(rows, key=lambda r: -r["amp"]):
    pct = 100 * float((tp < r["span"]).float().mean())
    print(f"{r['i']:>5} {r['sigma']:8.4f} {r['sigma']/sc.sim.radius:9.1f}x "
          f"{r['span']:9.5f} {pct:8.0f}% {100*r['err']:7.2f}% {100*r['amp']:6.0f}% "
          f"{('frame ' + str(r['left'])) if r['left'] is not None else '--':>9}")
print(f"  ('vs train' = the percentile this trajectory's peak displacement sits at "
      f"among the\n   trajectories the network trained on)")

# ---- correlate -------------------------------------------------------------
amp = torch.tensor([r["amp"] for r in rows])
span = torch.tensor([r["span"] for r in rows])
sig = torch.tensor([r["sigma"] for r in rows])


def corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    return (a * b).sum() / (a.norm() * b.norm()).clamp(min=1e-12)


print(f"\n[correlation with how far the rollout ran]")
print(f"  reference peak displacement : {corr(span, amp):+.2f}")
print(f"  forcing correlation length  : {corr(sig, amp):+.2f}")
print(f"  log of it                   : {corr(sig.log(), amp):+.2f}")

# ---- the failures, frame by frame -----------------------------------------
for r in rows:
    if r["left"] is None:
        continue
    print(f"\n[traj {r['i']}] ran away at frame {r['left']}; sigma {r['sigma']:.4f}, "
          f"reference peak {r['span']:.5f}")
    print(f"  {'frame':>6} {'err %':>8} {'amp %':>8} {'predicted step':>15} "
          f"{'vs train 99th':>14}")
    n = len(r["amps"])
    for k in list(range(0, min(r["left"] + 6, n), max(1, r["left"] // 8 or 1))) + \
             [min(n - 1, r["left"] + 10), n - 1]:
        if k >= n:
            continue
        print(f"  {k:6d} {100*r['errs'][k]/r['span']:7.1f}% "
              f"{100*r['amps'][k]/r['span']:7.0f}% {r['steps'][k]:15.5f} "
              f"{r['steps'][k]/td.quantile(0.99).item():13.1f}x")
    print(f"  (the reference's own largest single step here is {r['ref_step']:.5f}; "
          f"training's 99th\n   percentile step is {td.quantile(0.99):.5f})")

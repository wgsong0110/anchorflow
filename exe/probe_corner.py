"""Is it the corner, or is it that one trajectory?

The rollout runs away on exactly one of twenty held-out force-field
trajectories. Neither its forcing length nor its amplitude is unusual alone --
other trajectories match each and are fine -- but it is the only one that is
both localised and large, and it sits past every localised trajectory trained
on. That is a diagnosis drawn from a single instance, which is not a diagnosis.

So build more of them. Two groups, matched on peak displacement and differing
only in the forcing length:

  localised : sigma 2-4.4x the skinning radius, the band the failure is in
  spread    : sigma 8-24x, where trajectories of the same amplitude are fine

If the localised group diverges and the spread group does not, the corner is
the cause and filling it is the fix. If both are fine, something else about
that trajectory matters and the search starts again. If both diverge, it is
amplitude after all and the forcing length was a coincidence.

The amplitude is matched by measuring what a unit field actually moves and
rescaling, since a localised field of a given strength moves the object much
less -- which is why the corner was empty to begin with.
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
from anchorflow.nextstate import NextStep, apply_step
from anchorflow.streams import draw_field_shape

ap = argparse.ArgumentParser()
ap.add_argument("--ply", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--ckpt", nargs="+", required=True)
ap.add_argument("--n", type=int, default=8, help="trajectories per group")
ap.add_argument("--target", type=float, default=0.19086,
                 help="peak displacement to match, defaulting to the failing trajectory's")
ap.add_argument("--loc_lo", type=float, default=2.0)
ap.add_argument("--loc_hi", type=float, default=4.4)
ap.add_argument("--spread_lo", type=float, default=8.0)
ap.add_argument("--spread_hi", type=float, default=24.0,
                 help="forcing length bands, in multiples of the skinning radius")
ap.add_argument("--frames", type=int, default=60)
ap.add_argument("--dt_mult", type=int, default=40)
ap.add_argument("--n_anchors", type=int, default=512)
ap.add_argument("--K", type=int, default=8)
ap.add_argument("--seed", type=int, default=99991)
ap.add_argument("--calib_steps", type=int, default=25)
ap.add_argument("--cache", default=None,
                 help="where to keep the built trajectories. Regenerating them per invocation "
                      "leaves the same checkpoint scoring 4/8 in one run and 5/8 in the next: "
                      "the force kernel accumulates with atomicAdd, and a reference that "
                      "differs in the seventh digit is enough to flip a rollout that was going "
                      "to run away either way.")
args = ap.parse_args()

dev = "cuda"
torch.set_grad_enabled(False)
torch.manual_seed(0)

first = torch.load(args.ckpt[0], map_location="cpu", weights_only=False)["args"]
sc = scene_setup.build(args.ply, args.config, args.n_anchors, args.K, device=dev,
                        frozen_weights=bool(first.get("frozen_weights", False)))
AC, fixed = sc.anchor_canonical, sc.fixed_mask
dt = args.dt_mult * sc.sub_dt
R = sc.sim.radius
base_force = None
for bc in sc.cfg.get("boundary_conditions", []):
    if bc["type"] == "particle_impulse":
        base_force = torch.tensor(bc["force"], device=dev)
print(f"[setup] skinning radius {R:.4f}, object {sc.extent:.4f}, "
      f"matching peak displacement {args.target:.5f}")


def run(force, steps):
    p, v, gp = AC.clone(), sc.initial_velocity(force), sc.pos.clone()
    ps, peak = [p.clone()], 0.0
    for _ in range(steps):
        p, v, gp = sc.explicit_step(p, v, gp, args.dt_mult)
        if not torch.isfinite(p).all():
            break
        ps.append(p.clone())
        peak = max(peak, (p - AC)[~fixed].norm(dim=-1).max().item())
    return torch.stack(ps), peak


def build(gen, lo, hi):
    shape, sigma = draw_field_shape(sc, gen, lo * R, hi * R)
    m0 = base_force.norm().item()
    _, d0 = run(shape * m0, args.calib_steps)
    k = min(max(args.target / max(d0, 1e-9), 0.05), 30.0)
    traj, peak = run(shape * m0 * k, args.frames)
    return traj, sigma, peak


GROUPS = {"localised": (args.loc_lo, args.loc_hi),
          "spread": (args.spread_lo, args.spread_hi)}
key = f"{args.seed}_{args.n}_{args.target:.5f}_{args.frames}_{args.loc_lo}_{args.loc_hi}" \
      f"_{args.spread_lo}_{args.spread_hi}"
sets = None
if args.cache and os.path.exists(args.cache):
    blob = torch.load(args.cache, map_location=dev, weights_only=False)
    if blob.get("key") == key:
        sets = blob["sets"]
        print(f"[cache] {args.cache}")
if sets is None:
    gen = torch.Generator(device=dev); gen.manual_seed(args.seed)
    sets = {}
    for name, (lo, hi) in GROUPS.items():
        out = []
        for _ in tqdm(range(args.n), desc=name, ncols=90):
            out.append(build(gen, lo, hi))
        sets[name] = out
    if args.cache:
        torch.save({"sets": sets, "key": key}, args.cache)
        print(f"[cache] written to {args.cache}")

nets = []
for path in args.ckpt:
    ck = torch.load(path, map_location=dev, weights_only=False)
    ta = ck["args"]
    ua = not ta.get("no_accel", False)
    net = NextStep(ta["hidden"], ta["depth"], ta["heads"], ck["disp_scale"],
                   ck["vel_scale"], ck["acc_scale"], use_accel=ua,
                   chunk=ta.get("chunk", 1)).to(dev)
    net.load_state_dict(ck["model"]); net.eval()
    nets.append((os.path.basename(path).replace(".pt", ""), net, ua))


def rollout(net, use_a, r):
    p = r[1].clone()
    v = (r[1] - r[0]) / dt
    gp = sc.skin(p, sc.pos.clone()) if use_a else None
    errs, amps = [], []
    n = r.shape[0] - 1
    k = 0
    while k < n:
        a = sc.elastic_accel(p, gp) if use_a else None
        bad = False
        for q, d in apply_step(net, p, v, a, dt, fixed):
            errs.append((p - r[1 + k])[~fixed].norm(dim=-1).mean().item())
            amps.append((p - AC)[~fixed].norm(dim=-1).max().item())
            p, v = q, d / dt
            k += 1
            if not torch.isfinite(p).all():
                bad = True
                break
            if k >= n:
                break
        if bad:
            break
        if use_a:
            gp = sc.skin(p, gp)
    span = (r - AC).norm(dim=-1).max().item()
    return sum(errs) / len(errs) / span, max(amps) / span


for name, group in sets.items():
    lo, hi = GROUPS[name]
    print(f"\n[{name}] sigma {lo}-{hi}x the skinning radius, {len(group)} trajectories")
    print(f"  {'#':>3} {'sigma':>7} {'x R':>7} {'peak':>8}", end="")
    for n, _, _ in nets:
        print(f" | {n[:16]:>16}", end="")
    print()
    ran = {n: 0 for n, _, _ in nets}
    for i, (traj, sigma, peak) in enumerate(group):
        print(f"  {i:>3} {sigma:7.4f} {sigma/R:6.1f}x {peak:8.5f}", end="")
        for n, net, ua in nets:
            e, a = rollout(net, ua, traj)
            if a > 1.5:
                ran[n] += 1
            print(f" | {100*e:8.1f}% {100*a:5.0f}{'!' if a > 1.5 else ' '}", end="")
        print()
    print(f"  ran away (over 1.5x the reference's own motion): " +
          ", ".join(f"{n} {ran[n]}/{len(group)}" for n in ran))

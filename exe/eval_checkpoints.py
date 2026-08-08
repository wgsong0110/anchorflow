"""Score trained checkpoints on a measure that can tell them apart.

Training reports one number per evaluation: the mean anchor error at the final
frame, over five held-out trajectories. That swings about a percentage point
between consecutive evaluations of the same run -- B finished 13.9, 14.4, 13.7,
13.6 and C_cap 15.7, 13.7, 12.9, 13.2 -- so the 0.36 point between them means
nothing, and a checkpoint that happened to be evaluated on a good draw looks
better than one that was not. C_stream produced an 11.4% at one evaluation and a
54% at the next.

Two changes, both cheap because they only cost rollouts:

  * average the error over every frame rather than reading the last one. A
    60-step rollout provides 60 measurements and the training loop throws away
    59 of them.
  * more held-out trajectories.

Reports both the uniform-impulse trajectories every earlier run was scored on
and the force-field ones, since a model can be good at being shoved and bad at
being poked, and the uniform set alone cannot say which.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

_lib = os.path.join(os.path.dirname(__file__), "..", "lib")
sys.path.insert(0, _lib)

import torch
from tqdm import tqdm

from anchorflow import scene_setup
from anchorflow.nextstate import NextStep
from anchorflow.streams import rand_rot, draw_impulse

ap = argparse.ArgumentParser()
ap.add_argument("--ply", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--ckpt", nargs="+", required=True,
                 help="checkpoint files, or globs")
ap.add_argument("--n_uniform", type=int, default=20,
                 help="held-out trajectories with the config's impulse rotated and rescaled. "
                      "The first five are the ones every run was trained against holding "
                      "out, so those numbers stay comparable; the rest are new.")
ap.add_argument("--n_field", type=int, default=20,
                 help="held-out trajectories driven by a smooth random force field")
ap.add_argument("--frames", type=int, default=60)
ap.add_argument("--dt_mult", type=int, default=40)
ap.add_argument("--n_anchors", type=int, default=512)
ap.add_argument("--K", type=int, default=8)
ap.add_argument("--impulse_range", type=float, default=4.0)
args = ap.parse_args()

dev = "cuda"
torch.set_grad_enabled(False)
torch.manual_seed(0)

paths = []
for p in args.ckpt:
    paths.extend(sorted(glob.glob(p)) if any(c in p for c in "*?[") else [p])
paths = [p for p in paths if os.path.exists(p)]
assert paths, "no checkpoints found"

# every checkpoint has to be scored on the same trajectories, so the scene is
# built once. Runs differ in whether their weights were frozen, and that changes
# the physics -- a model trained under one convention is not being asked the same
# question under the other, so refuse rather than quietly compare across it.
first = torch.load(paths[0], map_location="cpu", weights_only=False)["args"]
frozen = bool(first.get("frozen_weights", False))
for p in paths[1:]:
    a = torch.load(p, map_location="cpu", weights_only=False)["args"]
    if bool(a.get("frozen_weights", False)) != frozen:
        raise SystemExit(f"{p} was trained with frozen_weights="
                         f"{a.get('frozen_weights')}, the others with {frozen}; "
                         f"the simulator itself differs, so the scores are not comparable")

sc = scene_setup.build(args.ply, args.config, args.n_anchors, args.K, device=dev,
                        frozen_weights=frozen)
AC, fixed = sc.anchor_canonical, sc.fixed_mask
dt = args.dt_mult * sc.sub_dt
print(f"[setup] N={sc.N} M={sc.M} frozen_weights={frozen}, {len(paths)} checkpoints")

base_force = None
for bc in sc.cfg.get("boundary_conditions", []):
    if bc["type"] == "particle_impulse":
        base_force = torch.tensor(bc["force"], device=dev)


def trajectory(force):
    p, v, gp = AC.clone(), sc.initial_velocity(force), sc.pos.clone()
    ps = [p.clone()]
    for _ in range(args.frames):
        p, v, gp = sc.explicit_step(p, v, gp, args.dt_mult)
        if not torch.isfinite(p).all():
            break
        ps.append(p.clone())
    return torch.stack(ps)


def build_set(n, field):
    """the uniform set reproduces the training script's draw order exactly, so
    its first five trajectories are the ones every run was held out on"""
    gen = torch.Generator(device=dev)
    gen.manual_seed(1234 if not field else 5678)
    out = []
    for t in tqdm(range(n), desc="field" if field else "uniform", ncols=90):
        if field:
            f, _ = draw_impulse(sc, base_force, gen, args.impulse_range, field=True)
        elif t == 0:
            f = base_force
        else:
            s = 0.5 * (args.impulse_range ** torch.rand(1, device=dev, generator=gen).item())
            s = s if args.impulse_range > 1 else 1.0
            f = (rand_rot(gen, dev) @ base_force) * s
        out.append(trajectory(f))
    m = min(t.shape[0] for t in out)
    return torch.stack([t[:m] for t in out])


SETS = {}
if args.n_uniform > 0:
    SETS["uniform"] = build_set(args.n_uniform, False)
if args.n_field > 0 and base_force is not None:
    SETS["field"] = build_set(args.n_field, True)


def score(net, use_a, REF):
    """mean error over every frame, and over the final frame, per trajectory"""
    every, final, amp = [], [], []
    for w in range(REF.shape[0]):
        r = REF[w]
        p = r[1].clone()
        v = (r[1] - r[0]) / dt
        gp = sc.skin(p, sc.pos.clone()) if use_a else None
        n = r.shape[0] - 1
        errs, moved = [], []
        for k in range(n):
            errs.append((p - r[1 + k])[~fixed].norm(dim=-1).mean())
            moved.append((p - AC)[~fixed].norm(dim=-1).max())
            a = sc.elastic_accel(p, gp) if use_a else None
            du = net(p, v, a, dt)
            du = torch.where(fixed.unsqueeze(-1), torch.zeros_like(du), du)
            p = p + du
            v = du / dt
            if not torch.isfinite(p).all():
                break
            if use_a:
                gp = sc.skin(p, gp)
        e = torch.stack(errs)
        span = (r - AC).norm(dim=-1).max().clamp(min=1e-12)
        every.append((e.mean() / span).item())
        final.append((e[-1] / span).item())
        amp.append((torch.stack(moved).max() / span).item())
    return every, final, amp


def stats(v):
    t = torch.tensor(v)
    # the spread over trajectories is what the single number was hiding: it is
    # also roughly the uncertainty on the mean of five of them
    return t.mean().item(), t.std().item(), t.max().item()


rows = []
for path in paths:
    ck = torch.load(path, map_location=dev, weights_only=False)
    ta = ck["args"]
    use_a = not ta.get("no_accel", False)
    net = NextStep(ta["hidden"], ta["depth"], ta["heads"], ck["disp_scale"],
                   ck["vel_scale"], ck["acc_scale"], use_accel=use_a).to(dev)
    net.load_state_dict(ck["model"]); net.eval()
    row = {"name": os.path.basename(path).replace(".pt", ""), "iter": ck.get("iter", 0)}
    for kind, REF in SETS.items():
        ev, fi, am = score(net, use_a, REF)
        row[kind] = (stats(ev), stats(fi), stats(am))
    rows.append(row)
    print(f"  scored {row['name']}", flush=True)

print(f"\n{'checkpoint':>22} {'iter':>7}", end="")
for kind in SETS:
    print(f" | {kind + ' all-frame':>17} {'final':>14} {'amp':>6}", end="")
print()
print("-" * (30 + 42 * len(SETS)))
for r in rows:
    print(f"{r['name']:>22} {r['iter']:>7}", end="")
    for kind in SETS:
        (em, es, _), (fm, fs, _), (am, _, _) = r[kind]
        print(f" | {100*em:7.2f}% +-{100*es:5.2f} {100*fm:7.2f}% +-{100*fs:5.2f} "
              f"{100*am:5.0f}%", end="")
    print()

print(f"\n[note] +- is the spread ACROSS trajectories, which is also roughly the "
      f"uncertainty\n       on a mean of five of them -- the training loop's number.")
print(f"[note] all-frame averages {args.frames} measurements per trajectory instead of "
      f"reading\n       only the last, which is where most of the training metric's "
      f"variance came from.")

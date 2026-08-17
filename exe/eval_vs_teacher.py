"""Each student against its own teacher, on one impulse set.

exe/eval_vs_mpm.py answers "how far is the chain from the truth". This answers
the different question of "how well did the student learn what it was shown",
which is the one that decides whether the student stage is the bottleneck. The
two had been mixed: the teacher-relative numbers quoted so far come from each
run's own training log, on that run's own held-out draw, at impulse_range 4.0 --
while the MPM-relative table uses a separate set at 16.0. Numbers from those two
sources were being compared, and they are not comparable.

So every student is re-measured here on one set, against whichever simulator it
was trained to imitate: the sampled anchor simulator for a run with no --fit, the
fitted one for a run with it. No MPM anywhere, which also makes this cheap --
there is no reference to generate, only rollouts.

Errors are in Gaussian space and normalised by the TEACHER's own peak
displacement, so students of different anchor counts stay comparable: what is
being asked of each is to reproduce its own teacher's motion.
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

ap = argparse.ArgumentParser()
ap.add_argument("--ply", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--dreamphysics", default="/workspace/DreamPhysics")
ap.add_argument("--ckpt", nargs="+", required=True)
ap.add_argument("--fit", default=None,
                 help="the fitted anchor set, for checkpoints that name one. Their "
                      "stored path is from the instance they trained on.")
ap.add_argument("--n_uniform", type=int, default=8)
ap.add_argument("--n_field", type=int, default=8)
ap.add_argument("--n_poke", type=int, default=8)
ap.add_argument("--impulse_range", type=float, default=16.0)
ap.add_argument("--n_anchors", type=int, default=512)
ap.add_argument("--K", type=int, default=8)
ap.add_argument("--eig_floor", type=float, default=0.02)
ap.add_argument("--frames", type=int, default=60)
ap.add_argument("--dt_mult", type=int, default=40)
ap.add_argument("--seed", type=int, default=20260811)
ap.add_argument("--per_traj", action="store_true")
args = ap.parse_args()

sys.path.insert(0, args.dreamphysics)
import warp as wp

from anchorflow.anchor_sparse import load_fitted
from anchorflow.nextstate import apply_step, net_from_ckpt
from anchorflow.streams import draw_impulse, rand_rot

dev = "cuda"
torch.set_grad_enabled(False)
wp.init()

sc = scene_setup.build(args.ply, args.config, args.n_anchors, args.K, device=dev,
                        frozen_weights=True, rot_fallback=True,
                        eig_floor=args.eig_floor)
mat = torch.nonzero(sc.keep, as_tuple=False).squeeze(-1)
base = next(torch.tensor(bc["force"], device=dev)
            for bc in sc.cfg["boundary_conditions"]
            if bc["type"] == "particle_impulse")
dt = args.dt_mult * sc.sub_dt

# one impulse set, drawn exactly as eval_vs_mpm.py draws its own, so the two
# tables at least share a distribution even though they score against different
# references
g = torch.Generator(device=dev); g.manual_seed(args.seed)
IMP = {}
for kind, n in (("uniform", args.n_uniform), ("field", args.n_field),
                 ("poke", args.n_poke)):
    fs_ = []
    for _ in range(n):
        if kind == "uniform":
            s = 0.5 * (args.impulse_range ** torch.rand(1, device=dev, generator=g).item())
            fs_.append((rand_rot(g, dev) @ base) * s)
        elif kind == "field":
            fs_.append(draw_impulse(sc, base, g, args.impulse_range, field=True)[0])
        else:
            rad = sc.sim.radius * (2.0 ** (1.0 + 2.0 * torch.rand(1, device=dev, generator=g).item()))
            fs_.append(sc.random_poke(g, rad, base.norm().item())[0])
    IMP[kind] = fs_
print(f"[setup] {sum(len(v) for v in IMP.values())} impulses, "
      f"impulse_range {args.impulse_range}, {args.frames} frames")

FITTED = None
if args.fit:
    FITTED, _fb = load_fitted(sc, args.fit, dev)
    print(f"[fit] {args.fit}: {FITTED.M} anchors")


def teacher_run(scene, force):
    """the simulator the student was asked to imitate, as a Gaussian cloud"""
    p, v = scene.anchor_canonical.clone(), scene.initial_velocity(force)
    gp = scene.pos.clone()
    out = [gp[mat].clone()]
    for _ in range(args.frames):
        p, v, gp = scene.explicit_step(p, v, gp, args.dt_mult)
        if not torch.isfinite(p).all():
            return None
        # the fitted wrapper advances anchors only; the sampled one returns the
        # cloud already updated
        if FITTED is not None and scene is FITTED:
            gp = scene.skin(p, gp)
        out.append(gp[mat].clone())
    return torch.stack(out)


def student_run(net, scene, force, chunk):
    p, v = scene.anchor_canonical.clone(), scene.initial_velocity(force)
    gp = scene.pos.clone()
    out = [gp[mat].clone()]
    k = 0
    while k < args.frames:
        for q, d in apply_step(net, p, v, None, dt, scene.fixed_mask):
            p, v = q, d / dt
            if not torch.isfinite(p).all():
                return None
            gp = scene.skin(p, gp)
            out.append(gp[mat].clone())
            k += 1
            if k >= args.frames:
                break
    return torch.stack(out)


rows = []
for path in args.ckpt:
    ck = torch.load(path, map_location=dev, weights_only=False)
    ta = ck["args"]
    uses_fit = ta.get("fit") is not None
    if uses_fit and FITTED is None:
        print(f"  {os.path.basename(path)}: needs --fit, skipped")
        continue
    scene = FITTED if uses_fit else sc
    net = net_from_ckpt(ck, dev)
    name = os.path.basename(path).replace(".pt", "")
    tag = "fitted" if uses_fit else "sampled"
    res = {}
    for kind, fl in IMP.items():
        errs, lost = [], 0
        for f in tqdm(fl, desc=f"{name} {kind}", ncols=88, leave=False):
            tr = teacher_run(scene, f)
            if tr is None:
                lost += 1
                continue
            st = student_run(net, scene, f, ta.get("chunk", 1))
            if st is None:
                lost += 1
                errs.append(float("nan"))
                continue
            span = (tr - tr[0]).norm(dim=-1).max().clamp(min=1e-12)
            errs.append(float((st - tr).norm(dim=-1).mean(-1).mean() / span))
        res[kind] = (errs, lost)
    rows.append((name, tag, ta.get("dagger", False), ta.get("hidden"), res))
    fin = "  ".join(
        f"{k} {100 * (sum(e for e in v[0] if e == e) / max(sum(1 for e in v[0] if e == e), 1)):.2f}%"
        for k, v in res.items())
    print(f"  {name:16} [{tag}, dagger={ta.get('dagger')}, hidden={ta.get('hidden')}]  {fin}")

print(f"\n{'run':>16} {'teacher':>9} {'dag':>5} {'hid':>5} "
       f"{'uniform':>9} {'field':>9} {'poke':>9} {'diverged':>9}")
for name, tag, dag, hid, res in rows:
    cells, lost_tot = [], 0
    for kind in ("uniform", "field", "poke"):
        e, lost = res[kind]
        ok = [x for x in e if x == x]
        lost_tot += lost
        cells.append(f"{100 * sum(ok) / max(len(ok), 1):8.2f}%")
    print(f"{name:>16} {tag:>9} {str(dag):>5} {hid:>5} " + " ".join(cells)
          + f" {lost_tot:9d}")

if args.per_traj:
    for name, tag, dag, hid, res in rows:
        print(f"\n[{name}] per trajectory, % of the teacher's own motion")
        for kind in ("uniform", "field", "poke"):
            e, _ = res[kind]
            print(f"  {kind:8} " + " ".join(
                ("  div" if x != x else f"{100 * x:5.1f}") for x in e))
print("\n[note] normalised by each TEACHER's peak displacement, so a student of the\n"
       "       fitted set and one of the sampled set are both being asked to\n"
       "       reproduce their own teacher. MPM is not involved.")

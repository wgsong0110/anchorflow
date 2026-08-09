"""How much of the speedup is available without learning anything?

The learned stepper replaces 40 explicit substeps of 1e-4 with one network
evaluation and is 7.9x faster than them. But nobody has measured what the
explicit simulator does at a larger substep. If it stays stable and accurate at
8 substeps, five of that eight is free and the learned model's advantage is
really 1.6x, not 7.9x. Explicit integration has a stability limit set by the
stiffest mode, so there is some floor -- the question is where it is, and what
the accuracy costs on the way down.

Same total simulated time throughout: 60 coarse steps of 4e-3, taken with n
substeps of 4e-3/n each. Error is measured against a finer reference than the
40 the project calls ground truth, since if 40 is not itself converged then
every error in this project is measured against the wrong thing.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

_lib = os.path.join(os.path.dirname(__file__), "..", "lib")
sys.path.insert(0, _lib)

import torch
from tqdm import tqdm

from anchorflow import scene_setup
from anchorflow.nextstate import NextStep
from anchorflow.streams import draw_field_shape

ap = argparse.ArgumentParser()
ap.add_argument("--ply", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--ckpt", default=None,
                 help="a learned stepper to place on the same axes")
ap.add_argument("--coarse_dt", type=float, default=4e-3)
ap.add_argument("--frames", type=int, default=60)
ap.add_argument("--n_ref", type=int, default=320,
                 help="substeps per coarse step for the reference. Has to be finer than the "
                      "40 this project treats as ground truth, or the question of whether 40 "
                      "is converged cannot be asked.")
ap.add_argument("--substeps", type=int, nargs="+",
                 default=[160, 80, 40, 20, 10, 8, 5, 4, 2, 1])
ap.add_argument("--n_anchors", type=int, default=512)
ap.add_argument("--K", type=int, default=8)
ap.add_argument("--field_sigma", type=float, default=0.0,
                 help="if set, drive with a force field of this correlation length instead of "
                      "the config's uniform impulse -- localised forcing excites stiffer modes "
                      "and should hit the stability limit sooner")
ap.add_argument("--repeat", type=int, default=5)
args = ap.parse_args()

dev = "cuda"
torch.set_grad_enabled(False)
torch.manual_seed(0)
sc = scene_setup.build(args.ply, args.config, args.n_anchors, args.K, device=dev,
                        frozen_weights=True)
AC, fixed = sc.anchor_canonical, sc.fixed_mask

base_force = None
for bc in sc.cfg.get("boundary_conditions", []):
    if bc["type"] == "particle_impulse":
        base_force = torch.tensor(bc["force"], device=dev)
force = base_force
if args.field_sigma > 0:
    gen = torch.Generator(device=dev); gen.manual_seed(4242)
    shape, _ = draw_field_shape(sc, gen, args.field_sigma, args.field_sigma)
    force = shape * base_force.norm().item()
    print(f"[force] field, correlation length {args.field_sigma:.4f} "
          f"({args.field_sigma / sc.sim.radius:.1f}x the skinning radius)")
else:
    print(f"[force] the config's uniform impulse")

print(f"[setup] {args.frames} coarse steps of {args.coarse_dt:g}; the project's own "
      f"ground truth is {int(args.coarse_dt / sc.sub_dt)} substeps")


def run(n_sub):
    """n_sub explicit substeps per coarse step, same total time"""
    sub = args.coarse_dt / n_sub
    old = sc.sub_dt
    sc.sub_dt = sub
    p, v, gp = AC.clone(), sc.initial_velocity(force), sc.pos.clone()
    ps, ok = [p.clone()], True
    for _ in range(args.frames):
        p, v, gp = sc.explicit_step(p, v, gp, n_sub)
        if not torch.isfinite(p).all():
            ok = False
            break
        ps.append(p.clone())
    sc.sub_dt = old
    return torch.stack(ps), ok


def timeit(n_sub):
    sub = args.coarse_dt / n_sub
    old = sc.sub_dt
    sc.sub_dt = sub
    p, v, gp = AC.clone(), sc.initial_velocity(force), sc.pos.clone()
    for _ in range(2):
        sc.explicit_step(p.clone(), v.clone(), gp.clone(), n_sub)
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(args.repeat):
        sc.explicit_step(p.clone(), v.clone(), gp.clone(), n_sub)
    e1.record(); torch.cuda.synchronize()
    sc.sub_dt = old
    return e0.elapsed_time(e1) / args.repeat


REF, ok = run(args.n_ref)
assert ok, f"the reference itself diverged at {args.n_ref} substeps"
span = (REF - AC).norm(dim=-1).max().item()
print(f"[reference] {args.n_ref} substeps, peak anchor displacement {span:.5f}\n")

rows = []
for n in tqdm(args.substeps, desc="substeps", ncols=90):
    traj, ok = run(n)
    ms = timeit(n)
    m = min(traj.shape[0], REF.shape[0])
    err = (traj[:m] - REF[:m])[:, ~fixed].norm(dim=-1).mean(-1)
    amp = (traj[:m] - AC)[:, ~fixed].norm(dim=-1).max().item()
    rows.append({"n": n, "ok": ok, "frames": traj.shape[0] - 1,
                 "err": err.mean().item() / span, "amp": amp / span, "ms": ms})

base_ms = next(r["ms"] for r in rows if r["n"] == int(args.coarse_dt / sc.sub_dt))
print(f"\n{'substeps':>9} {'substep dt':>11} {'error':>9} {'amp':>7} {'survived':>9} "
      f"{'ms/step':>9} {'vs 40':>7}")
for r in rows:
    surv = "all" if r["ok"] else f"{r['frames']}/{args.frames}"
    print(f"{r['n']:>9} {args.coarse_dt / r['n']:11.2e} {100*r['err']:8.2f}% "
          f"{100*r['amp']:6.0f}% {surv:>9} {r['ms']:9.3f} {base_ms / r['ms']:6.2f}x")

if args.ckpt:
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    ta = ck["args"]
    ua = not ta.get("no_accel", False)
    net = NextStep(ta["hidden"], ta["depth"], ta["heads"], ck["disp_scale"],
                   ck["vel_scale"], ck["acc_scale"], use_accel=ua).to(dev)
    net.load_state_dict(ck["model"]); net.eval()
    dt = args.coarse_dt
    p = REF[1].clone()
    v = (REF[1] - REF[0]) / dt
    gp = sc.skin(p, sc.pos.clone()) if ua else None
    errs, amps = [], []
    for k in range(REF.shape[0] - 1):
        errs.append((p - REF[1 + k])[~fixed].norm(dim=-1).mean().item())
        amps.append((p - AC)[~fixed].norm(dim=-1).max().item())
        a = sc.elastic_accel(p, gp) if ua else None
        du = net(p, v, a, dt)
        du = torch.where(fixed.unsqueeze(-1), torch.zeros_like(du), du)
        p = p + du; v = du / dt
        if not torch.isfinite(p).all():
            break
        if ua:
            gp = sc.skin(p, gp)

    def learned():
        a_ = sc.elastic_accel(p, gp) if ua else None
        q = net(p, v, a_, dt)
        if ua:
            sc.skin(p + q, gp)

    for _ in range(3):
        learned()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(20):
        learned()
    e1.record(); torch.cuda.synchronize()
    ms = e0.elapsed_time(e1) / 20
    print(f"{'learned':>9} {'--':>11} {100*sum(errs)/len(errs)/span:8.2f}% "
          f"{100*max(amps)/span:6.0f}% {'all':>9} {ms:9.3f} {base_ms / ms:6.2f}x")

print(f"\n[note] error is against the {args.n_ref}-substep reference, averaged over every "
      f"frame,\n       as a fraction of that reference's own peak displacement.")

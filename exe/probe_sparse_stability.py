"""Is the sparse fit's NaN a stability failure, and what sets the limit?

The failing run's parameters were still finite when a rollout first went NaN, so
something blew up inside the forty substeps of a coarse frame rather than in the
optimiser. An explicit integrator with a fixed substep does that when the thing
it is integrating gets stiff enough, and the discretisation -- which is what
decides that stiffness -- is exactly what the fit is free to change.

Replaying two hundred iterations to catch it is slow and the host keeps dying,
so this tests the mechanism directly instead. Anchor extents are shrunk by a
uniform factor, which is the direction the fit was heading, and for each factor
one coarse frame is run substep by substep. Then the substep is cut at a factor
that fails: an explicit stability limit is fixed by halving dt, and anything
that survives that is not one.

Reported alongside is the acceleration each configuration produces, which is
what the limit actually depends on -- the fit reached |a| of 2e4 by its tenth
iteration against the 175 the original simulator runs at.
"""
from __future__ import annotations

import argparse
import os
import sys

_lib = os.path.join(os.path.dirname(__file__), "..", "lib")
sys.path.insert(0, _lib)

import torch

from anchorflow import scene_setup
from anchorflow.anchor_sparse import AnchorSparse

ap = argparse.ArgumentParser()
ap.add_argument("--ply", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--dreamphysics", default="/workspace/DreamPhysics")
ap.add_argument("--traj_cache", required=True)
ap.add_argument("--frame", type=int, default=10)
ap.add_argument("--dt_mult", type=int, default=40)
ap.add_argument("--c", type=float, default=0.25)
ap.add_argument("--eig_floor", type=float, default=0.02)
ap.add_argument("--shrink", type=float, nargs="+",
                 default=[1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2])
ap.add_argument("--substep_mults", type=int, nargs="+", default=[1, 2, 4, 8, 16])
args = ap.parse_args()

sys.path.insert(0, args.dreamphysics)
import warp as wp

dev = "cuda"
torch.set_grad_enabled(False)
torch.manual_seed(0)
wp.init()

sc = scene_setup.build(args.ply, args.config, 512, 8, device=dev, frozen_weights=True,
                        rot_fallback=True, eig_floor=args.eig_floor)
fit = AnchorSparse(sc, c=args.c, eig_floor=args.eig_floor).to(dev)
fit.s_lo, fit.s_hi = 1e-9, 1e9
fit.init_from_geometry()
S0 = fit.log_s.detach().clone()
blob = torch.load(args.traj_cache, map_location=dev, weights_only=False)
X, V = blob["fit"][0]
x0, v0 = X[args.frame], V[args.frame]
print(f"[setup] {fit.M} anchors, radius {sc.sim.radius:.4f}, frame {args.frame}")


def run(n_sub, dt_scale, trace=False):
    """one coarse frame at dt/dt_scale, with dt_scale times as many substeps"""
    cache = fit.prepare()
    w, rc, q, Binv, blocked, mass = cache
    m = mass.unsqueeze(-1)
    keep = (~fit.fixed).unsqueeze(-1).to(torch.float32)
    dt = fit.dt / dt_scale
    p, v = fit.project(x0, cache), fit.project_v(v0, cache)
    peak_a = 0.0
    rows = []
    for k in range(n_sub * dt_scale):
        a = fit.force(p, w, rc, q, Binv, blocked) / m
        peak_a = max(peak_a, a.norm(dim=-1).max().item())
        v = (v + dt * a) * fit.damping * keep
        p = p + dt * v
        if trace and (k < 3 or k % max(1, (n_sub * dt_scale) // 8) == 0):
            rows.append((k, a.norm(dim=-1).max().item(), v.norm(dim=-1).max().item()))
        if not torch.isfinite(p).all():
            return None, peak_a, rows, k
    return p, peak_a, rows, None


print(f"\n[1. shrinking every anchor] one coarse frame, {args.dt_mult} substeps")
print(f"  {'factor':>7} {'min extent':>11} {'min mass':>10} {'|a| max':>11} "
      f"{'result':>22}")
first_bad = None
for f in args.shrink:
    fit.log_s.copy_(S0 + float(torch.log(torch.tensor(f))))
    fit.refresh()
    cache = fit.prepare()
    p, pa, _, k = run(args.dt_mult, 1)
    s = fit.log_s.exp()
    res = f"diverged at substep {k}" if p is None else \
        f"moved {(p - fit.pos).norm(dim=-1).max():.4f}"
    print(f"  {f:7.2f} {s.min():11.2e} {cache[5].min():10.2e} {pa:11.3e} {res:>22}")
    if p is None and first_bad is None:
        first_bad = f

if first_bad is None:
    print(f"\n[2.] nothing diverged, so shrinking the extents is not by itself the "
          f"mechanism")
else:
    print(f"\n[2. cutting the substep] at factor {first_bad}, where it first failed")
    print(f"  {'substeps':>9} {'dt':>11} {'|a| max':>11} {'result':>22}")
    fit.log_s.copy_(S0 + float(torch.log(torch.tensor(first_bad))))
    fit.refresh()
    for mlt in args.substep_mults:
        p, pa, _, k = run(args.dt_mult, mlt)
        res = f"diverged at substep {k}" if p is None else \
            f"moved {(p - fit.pos).norm(dim=-1).max():.4f}"
        print(f"  {args.dt_mult * mlt:9d} {fit.dt / mlt:11.2e} {pa:11.3e} {res:>22}")
    print(f"\n  A failure that a smaller step fixes is the explicit stability limit, "
          f"and the\n  limit moved because the fit changed the discretisation. One "
          f"that survives every\n  step size is not, and the extents are not the "
          f"mechanism.")

print(f"\n[3. what the initialisation already hands it]")
fit.log_s.copy_(S0)
fit.refresh()
cache = fit.prepare()
s = fit.log_s.exp()
mass = cache[5]
print(f"  extents: min {s.min():.3e}  1% {s.flatten().quantile(0.01):.3e}  "
      f"median {s.median():.3e}  (radius {sc.sim.radius:.4f})")
print(f"  masses:  min {mass.min():.3e}  1% {mass.quantile(0.01):.3e}  "
      f"median {mass.median():.3e}")
print(f"  the same for the simulator this replaces: mass min {sc.mass.min():.3e}, "
      f"median {sc.mass.median():.3e}")
_, pa, rows, _ = run(args.dt_mult, 1, trace=True)
print(f"  |a| max over one frame {pa:.3e}; the original simulator runs at about 175")

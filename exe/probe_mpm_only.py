"""What distinguishes the case the fit fails on, measured without the fit.

Of eight held-out field impulses the fitted discretisation lost one, #5, and the
two signals that separated it -- the rollout overshooting to 147% of MPM's motion
and the only nonzero CFL overshoot in the set -- are both properties of our
simulator. That makes them symptoms. If the cause is visible at all it has to be
visible in MPM's own trajectory, before any anchor is involved.

So nothing here touches the fitted simulator, or any anchor set:

  peak speed        the largest per-frame particle displacement. This is the
                    quantity a CFL condition is about, and it is NOT what was
                    measured before: total displacement over sixty frames says
                    nothing about how fast the object moves at its fastest, and
                    the four control cases were rescaled to match the total.
  when it peaks     a case that is fastest at frame 1 and one fastest at frame 30
                    stress an explicit scheme differently
  initial kick      the velocity the impulse imparts, at the particles
  MPM's own strain  ||F - R|| from the solver's own per-particle deformation
                    gradient and its polar factor. Distance to the nearest
                    rotation, so rigid motion is subtracted -- unlike the ||F - I||
                    reported earlier, which was dominated by rotation and
                    confounded with amplitude.
  the impulse       how spread out and how concentrated

The twelve cases are the eight held-out ones plus the four drawn at #5's
amplitude, which all succeeded. A variable that explains the failure has to
separate #5 from those four as well, not merely from the seven smaller ones.
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
ap.add_argument("--caches", nargs="+", required=True,
                 help="LABEL=PATH, e.g. holdout=eval_ref.pt large=amp_ref.pt")
ap.add_argument("--kind", default="field")
ap.add_argument("--failed", nargs="*", default=["holdout:5"],
                 help="LABEL:INDEX of the cases the fitted simulator lost, so the "
                      "table can be split. Labels only -- nothing here recomputes them.")
ap.add_argument("--n_anchors", type=int, default=512)
ap.add_argument("--K", type=int, default=8)
ap.add_argument("--eig_floor", type=float, default=0.02)
ap.add_argument("--frames", type=int, default=60)
ap.add_argument("--dt_mult", type=int, default=40)
ap.add_argument("--n_grid", type=int, default=100)
args = ap.parse_args()

sys.path.insert(0, args.dreamphysics)
import warp as wp

from anchorflow.mpm_teacher import MPMTeacher

dev = "cuda"
torch.set_grad_enabled(False)
wp.init()

sc = scene_setup.build(args.ply, args.config, args.n_anchors, args.K, device=dev,
                        frozen_weights=True, rot_fallback=True,
                        eig_floor=args.eig_floor)
T = MPMTeacher(sc, n_grid=args.n_grid)
mat = T.mat
Xc = sc.pos[mat]
extent = float((Xc.max(0).values - Xc.min(0).values).norm())
spacing = sc.sim.radius
print(f"[setup] extent {extent:.4f}, anchor spacing {spacing:.4f}, "
      f"substep dt {sc.sub_dt:g}, {args.dt_mult} substeps per frame")
print(f"        speeds below are per FRAME; a substep covers 1/{args.dt_mult} of one")

FAILED = set(args.failed)


def force_shape(f):
    fm = f[mat] if f.shape[0] != mat.shape[0] else f
    w = fm.norm(dim=-1)
    tot = w.sum().clamp(min=1e-20)
    c = (w.unsqueeze(-1) * Xc).sum(0) / tot
    sigma = ((w * (Xc - c).pow(2).sum(-1)).sum() / tot).sqrt()
    k = max(1, int(0.1 * w.shape[0]))
    return float(sigma) / extent, float(torch.topk(w, k).values.sum() / tot)


def mpm_run(force):
    """MPM from rest, returning its own deformation as well as its path.

    F and R come from the solver, not from any reconstruction: ||F - R|| is then
    the strain MPM itself carries, with rigid motion removed.
    """
    dv = sc.impulse_dv(force)
    v0 = (T.w.unsqueeze(-1) * dv[T.idx]).sum(1).contiguous()
    T._set(T.pos_m.clone(), v0, T.eye.clone(), torch.zeros_like(T.eye))
    xs = [T.pos_m.clone()]
    for _ in range(args.frames):
        for k in range(args.dt_mult):
            T.solver.p2g2p(None, sc.sub_dt, device=T.wp_dev)
            if (k + 1) % 8 == 0 and not T._in_domain():
                return None, None, None
        xs.append(T.solver.export_particle_x_to_torch().clone())
    F = T.solver.export_particle_F_to_torch().reshape(-1, 3, 3)
    R = T.solver.export_particle_R_to_torch().reshape(-1, 3, 3)
    return torch.stack(xs), F, R


rows = []
for spec in args.caches:
    label, path = spec.split("=", 1)
    blob = torch.load(path, map_location=dev, weights_only=False)
    for i, f in enumerate(blob["force"][args.kind]):
        truth = blob["ref"][args.kind][i][: args.frames + 1]
        span = float((truth - truth[0]).norm(dim=-1).max())

        # per-frame displacement: the speed the integrator actually sees
        step = (truth[1:] - truth[:-1]).norm(dim=-1)              # [frames, Nm]
        per_frame_max = step.max(dim=-1).values                    # [frames]
        peak = float(per_frame_max.max())
        peak_at = int(per_frame_max.argmax())
        p99 = float(step.reshape(-1).quantile(0.99))

        dv = sc.impulse_dv(f)
        kick = float((T.w.unsqueeze(-1) * dv[T.idx]).sum(1).norm(dim=-1).max())

        _x, Fm, Rm = mpm_run(f)
        strain = (Fm - Rm).reshape(-1, 9).norm(dim=-1) if Fm is not None else None
        sig, top10 = force_shape(f)

        rows.append(dict(
            case=f"{label}:{i}", span=span, peak=peak, peak_at=peak_at, p99=p99,
            kick=kick, sig=sig, top10=top10,
            strain=float(strain.quantile(0.99)) if strain is not None else float("nan"),
            strain_max=float(strain.max()) if strain is not None else float("nan"),
            bad=f"{label}:{i}" in FAILED))

print(f"\n{'case':>10} {'total':>8} {'peak/frame':>11} {'as substeps':>12} "
      f"{'at frame':>9} {'p99/frame':>10} {'kick':>8} {'force len':>10} "
      f"{'top10':>6} {'p99 |F-R|':>10} {'max |F-R|':>10}")
for r in sorted(rows, key=lambda r: -r["peak"]):
    # a substep covers 1/dt_mult of a frame, so this is the fraction of the
    # anchor spacing a particle crosses in one substep at its fastest
    frac = r["peak"] / args.dt_mult / spacing
    mark = "  <-- the fit loses here" if r["bad"] else ""
    print(f"{r['case']:>10} {r['span']:8.4f} {r['peak']:11.5f} {frac:11.1%} "
          f"{r['peak_at']:9d} {r['p99']:10.5f} {r['kick']:8.4f} {r['sig']:9.2f}x "
          f"{r['top10']:6.2f} {r['strain']:10.3f} {r['strain_max']:10.3f}{mark}")

bad = [r for r in rows if r["bad"]]
good = [r for r in rows if not r["bad"]]
if bad and good:
    print(f"\n{'':26} {'the loss':>10} {'the other ' + str(len(good)):>14} {'ratio':>8}")
    for name, key in (("total displacement", "span"), ("peak per-frame", "peak"),
                       ("p99 per-frame", "p99"), ("initial kick", "kick"),
                       ("force length", "sig"), ("concentration", "top10"),
                       ("p99 MPM strain |F-R|", "strain"),
                       ("max MPM strain |F-R|", "strain_max")):
        b = sum(r[key] for r in bad) / len(bad)
        g = sum(r[key] for r in good) / len(good)
        print(f"  {name:26} {b:10.4g} {g:14.4g} {b / max(g, 1e-20):7.2f}x")
    print("\n  A cause has to separate the loss from ALL of the others, including the\n"
          "  ones drawn at its own amplitude. A ratio near 1 rules the column out.")

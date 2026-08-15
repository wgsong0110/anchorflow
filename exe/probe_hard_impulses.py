"""What makes two of the held-out impulses harder than the other four?

Impulses 0 and 1 cost the original anchor simulator 18.64% and 28.41% against
MPM where the rest cost 7-10%, and they stay the largest after fitting. Either
they are simply bigger -- more motion, more room to disagree -- or they excite
something the anchor discretisation represents badly, in which case the residual
has a shape and the fit has somewhere left to go.

Four properties, measured against the error each impulse produces:

  amplitude      how far MPM moves the object
  spatial scale  a smooth force field has a length over which it varies, and a
                 short one bends the object rather than pushing it
  concentration  how much of the force lands on how little of the object
  strain         how far MPM's own deformation gradient gets from the identity,
                 which is the thing shape matching has to reproduce

Amplitude alone would be the dull answer and is worth ruling out first: it is
already divided out of the error, so if it still explains everything then the
normalisation is doing less than it looks.
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
ap.add_argument("--frames", type=int, default=60)
ap.add_argument("--dt_mult", type=int, default=40)
ap.add_argument("--errors", type=float, nargs="+",
                 default=[18.64, 28.41, 7.51, 6.95, 10.11, 7.10],
                 help="the 8-NN simulator's error on each held-out impulse")
ap.add_argument("--fitted", type=float, nargs="+",
                 default=[6.24, 7.96, 3.36, 4.82, 4.42, 2.74],
                 help="and the fitted simulator's, for the same impulses")
args = ap.parse_args()

sys.path.insert(0, args.dreamphysics)
import warp as wp

from anchorflow.anchor_sparse import AnchorSparse
from anchorflow.mpm_teacher import MPMTeacher

dev = "cuda"
torch.set_grad_enabled(False)
torch.manual_seed(0)
wp.init()

sc = scene_setup.build(args.ply, args.config, 512, 8, device=dev, frozen_weights=True,
                        rot_fallback=True, eig_floor=0.02)
T = MPMTeacher(sc)
fit = AnchorSparse(sc, c=0.25, eig_floor=0.02).to(dev)
cache = fit.prepare()
blob = torch.load(args.traj_cache, map_location=dev, weights_only=False)
CHK, FORCES = blob["chk"], blob["forces"]["chk"]
mat = T.mat
Xc = sc.pos[mat]
extent = float((Xc.max(0).values - Xc.min(0).values).norm())
print(f"[setup] object extent {extent:.4f}, anchor spacing {sc.sim.radius:.4f}, "
      f"{len(CHK)} held-out impulses")


def force_shape(f):
    """the length over which the force varies, and how concentrated it is.

    The length comes from the force-weighted spread of where the force acts: a
    field that pushes one branch has a small one, a uniform push has the whole
    object. Concentration is the share of the total force carried by the tenth
    of Gaussians carrying the most.
    """
    if f.dim() == 1:
        return extent, 0.1, 1.0
    fm = f[mat] if f.shape[0] != mat.shape[0] else f
    w = fm.norm(dim=-1)
    tot = w.sum().clamp(min=1e-20)
    c = (w.unsqueeze(-1) * Xc).sum(0) / tot
    sigma = ((w * (Xc - c).pow(2).sum(-1)).sum() / tot).sqrt()
    k = max(1, int(0.1 * w.shape[0]))
    top = torch.topk(w, k).values.sum() / tot
    return float(sigma), float(top), float(w.mean() / w.max().clamp(min=1e-20))


rows = []
for i, (X, V) in enumerate(CHK[:len(args.errors)]):
    Xg = torch.stack([X[k] for k in range(args.frames + 1)])
    span = (Xg - Xg[0]).norm(dim=-1).max()
    sigma, top10, flat = force_shape(FORCES[i])

    # how far MPM's own deformation gets from a rigid motion, which is what the
    # anchors have to track. Taken through the anchor state, since that is the
    # only thing the simulator sees
    p_last = fit.project(X[args.frames], cache)
    F, _ = fit.deformation(p_last, *cache[:5])
    I = torch.eye(3, device=dev)
    strain = (F - I).reshape(-1, 9).norm(dim=-1)
    rows.append((i, float(span), sigma / extent, top10, float(strain.mean()),
                 float(strain.quantile(0.99)), args.errors[i], args.fitted[i]))

print(f"\n{'imp':>4} {'MPM moved':>10} {'force scale':>12} {'top 10%':>8} "
      f"{'strain':>8} {'p99':>8} {'8-NN err':>9} {'fitted':>8}")
for r in rows:
    print(f"{r[0]:4d} {r[1]:10.4f} {r[2]:11.2f}x {r[3]:8.2f} {r[4]:8.3f} "
          f"{r[5]:8.3f} {r[6]:8.2f}% {r[7]:7.2f}%")


def corr(a, b):
    a = torch.tensor(a); b = torch.tensor(b)
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm()).clamp(min=1e-20))


print(f"\n{'':22} {'vs 8-NN error':>15} {'vs fitted error':>17}")
for name, idx in (("amplitude", 1), ("force scale (small=local)", 2),
                   ("concentration", 3), ("mean strain", 4), ("p99 strain", 5)):
    v = [r[idx] for r in rows]
    print(f"  {name:22} {corr(v, [r[6] for r in rows]):14.2f} "
          f"{corr(v, [r[7] for r in rows]):16.2f}")
print(f"\n  Six points, so a correlation here is a direction to look rather than "
      f"a finding.\n  What matters is whether the same property explains both "
      f"columns: one that does\n  is a property of the discretisation, one that "
      f"only explains the first was fixed.")

"""A reference set at one chosen amplitude, in eval_vs_mpm.py's cache format.

Of the eight held-out field impulses, the fitted discretisation lost exactly
one -- #5, at 22.43% against the sampled simulator's 16.41% -- and #5 is also
the largest of the eight: MPM moves the object 1.107 where the others sit
between 0.525 and 0.656. One trajectory cannot separate "the fit fails at large
amplitude" from "the fit failed on that trajectory", so this draws more at that
amplitude and lets the same evaluation and the same renderer answer it.

Amplitude is targeted rather than drawn, because it is what is being controlled
for. The field SHAPE is drawn as usual and normalised to unit RMS; the magnitude
is then set by one MPM run and a rescale, since displacement is close to linear
in the impulse over this range, and confirmed by a second run.

Large amplitude is also where MPM's own domain runs out -- two of ten uniform
impulses in the standard set left the grid -- so rejections are counted and
reported rather than quietly resampled.
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
ap.add_argument("--n", type=int, default=6, help="cases to accept")
ap.add_argument("--target", type=float, default=1.107,
                 help="MPM peak displacement to aim for. The default is field #5's.")
ap.add_argument("--band", type=float, default=0.15,
                 help="fractional tolerance around the target")
ap.add_argument("--n_anchors", type=int, default=512)
ap.add_argument("--K", type=int, default=8)
ap.add_argument("--eig_floor", type=float, default=0.02)
ap.add_argument("--frames", type=int, default=60)
ap.add_argument("--dt_mult", type=int, default=40)
ap.add_argument("--n_grid", type=int, default=100)
ap.add_argument("--seed", type=int, default=71)
ap.add_argument("--max_tries", type=int, default=40)
ap.add_argument("--out", required=True)
args = ap.parse_args()

sys.path.insert(0, args.dreamphysics)
import warp as wp

from anchorflow.mpm_teacher import MPMTeacher
from anchorflow.streams import draw_field_shape

dev = "cuda"
torch.set_grad_enabled(False)
wp.init()

sc = scene_setup.build(args.ply, args.config, args.n_anchors, args.K, device=dev,
                        frozen_weights=True, rot_fallback=True,
                        eig_floor=args.eig_floor)
T = MPMTeacher(sc, n_grid=args.n_grid)
base = next(torch.tensor(bc["force"], device=dev)
            for bc in sc.cfg["boundary_conditions"]
            if bc["type"] == "particle_impulse")
xw = sc.pos[sc.keep]
extent = float((xw.max(0).values - xw.min(0).values).norm())
print(f"[setup] object extent {extent:.4f}, target displacement {args.target:.4f} "
      f"({100 * args.target / extent:.0f}% of the object)")


def mpm_ref(force):
    """MPM from rest, or None if it left the grid. warp does not bounds-check its
    grid writes, so a particle that leaves takes the process with it -- this is
    checked mid-frame, not just per frame."""
    dv = sc.impulse_dv(force)
    v0 = (T.w.unsqueeze(-1) * dv[T.idx]).sum(1).contiguous()
    T._set(T.pos_m.clone(), v0, T.eye.clone(), torch.zeros_like(T.eye))
    xs = [T.pos_m.clone()]
    for _ in range(args.frames):
        for k in range(args.dt_mult):
            T.solver.p2g2p(None, sc.sub_dt, device=T.wp_dev)
            if (k + 1) % 8 == 0 and not T._in_domain():
                return None
        xs.append(T.solver.export_particle_x_to_torch().clone())
    return torch.stack(xs)


def peak(x):
    return float((x - x[0]).norm(dim=-1).max())


gen = torch.Generator(device=dev); gen.manual_seed(args.seed)
refs, forces, sigmas = [], [], []
left_grid = missed = 0
bar = tqdm(total=args.n, desc="cases", ncols=90)
tries = 0
while len(refs) < args.n and tries < args.max_tries:
    tries += 1
    shape, sigma = draw_field_shape(sc, gen)
    f = shape * base.norm().item()

    x = mpm_ref(f)
    if x is None:
        left_grid += 1
        continue
    d0 = peak(x)
    if d0 < 1e-6:
        continue
    # displacement is close to linear in the impulse here, so one measurement
    # sets the scale; the second run is the one that counts
    f = f * (args.target / d0)
    x = mpm_ref(f)
    if x is None:
        left_grid += 1
        continue
    d = peak(x)
    if abs(d - args.target) / args.target > args.band:
        missed += 1
        continue
    refs.append(x); forces.append(f); sigmas.append(sigma)
    bar.update(1)
bar.close()

print(f"[draw] {len(refs)} accepted from {tries} attempts: {left_grid} left MPM's grid, "
      f"{missed} missed the amplitude band")
for i, (x, s) in enumerate(zip(refs, sigmas)):
    print(f"   #{i}: displacement {peak(x):.4f}, correlation length {s:.4f} "
          f"({s / sc.sim.radius:.1f} anchor spacings)")

# eval_vs_mpm.py's format, so the same renderer and the same scoring read it.
# The key is deliberately not one eval_vs_mpm would generate: this set is not
# its held-out set and must not be mistaken for it.
key = f"amp{args.target}_{args.n}_{args.frames}_{args.dt_mult}_seed{args.seed}"
torch.save({"ref": {"field": refs}, "force": {"field": forces}, "key": key}, args.out)
print(f"[saved] {args.out}")

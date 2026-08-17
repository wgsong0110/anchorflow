"""Why does the fitted discretisation lose exactly one of eight field impulses?

It wins seven and loses #5, at 22.43% against the sampled simulator's 16.41%.
Amplitude was the obvious explanation and is ruled out: #5 is the largest of the
eight, but four fresh impulses drawn at that same amplitude are won by 2-3x
(exe/make_amp_cache.py). So something else about #5.

Everything measurable about the eight, side by side, in the hope that #5 is an
outlier in one column rather than in none:

  the force        how spread out it is, how concentrated, and which material it
                   lands on -- ficus is a stiff trunk under a canopy ten times
                   softer, and a force on one or the other is a different problem
  the error        per frame, not averaged, because "gradually worse" and "fine
                   until frame 30" have different causes and the same mean
  the motion       how far each simulator moves the object against MPM. Falling
                   short and overshooting are different failures
  the integration  whether the fitted anchors outrun their substep, which is the
                   failure the CFL term in the fit was added to prevent and the
                   one that took earlier runs to NaN
  the deformation  how far F gets from a rotation, which is what shape matching
                   has to reproduce and where it degrades first
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
ap.add_argument("--cache", required=True)
ap.add_argument("--fit", required=True)
ap.add_argument("--kind", default="field")
ap.add_argument("--n_anchors", type=int, default=512)
ap.add_argument("--K", type=int, default=8)
ap.add_argument("--eig_floor", type=float, default=0.02)
ap.add_argument("--frames", type=int, default=60)
ap.add_argument("--dt_mult", type=int, default=40)
args = ap.parse_args()

sys.path.insert(0, args.dreamphysics)
import warp as wp

from anchorflow.anchor_sparse import load_fitted
from anchorflow.mpm_teacher import MPMTeacher

dev = "cuda"
torch.set_grad_enabled(False)
wp.init()

sc = scene_setup.build(args.ply, args.config, args.n_anchors, args.K, device=dev,
                        frozen_weights=True, rot_fallback=True,
                        eig_floor=args.eig_floor)
T = MPMTeacher(sc)
mat = T.mat
Xc = sc.pos[mat]
extent = float((Xc.max(0).values - Xc.min(0).values).norm())
mu_m = sc.mu[mat]
soft = mu_m < mu_m.median()                       # the canopy, ten times softer

fs, fb = load_fitted(sc, args.fit, dev)
fit = fs.fit
cache = fit.prepare()
I3 = torch.eye(3, device=dev)

blob = torch.load(args.cache, map_location=dev, weights_only=False)
refs, forces = blob["ref"][args.kind], blob["force"][args.kind]
print(f"[setup] {len(refs)} {args.kind} cases, object extent {extent:.4f}, "
      f"anchor spacing {sc.sim.radius:.4f}, {fit.M} fitted anchors")
print(f"        canopy is {int(soft.sum())} of {mat.shape[0]} material Gaussians, "
      f"mu {float(mu_m[soft].mean()):.4g} against the trunk's "
      f"{float(mu_m[~soft].mean()):.4g}")


def force_shape(f):
    """the length over which the force varies, how concentrated it is, and how
    much of it lands on the soft canopy rather than the trunk"""
    fm = f[mat] if f.shape[0] != mat.shape[0] else f
    w = fm.norm(dim=-1)
    tot = w.sum().clamp(min=1e-20)
    c = (w.unsqueeze(-1) * Xc).sum(0) / tot
    sigma = ((w * (Xc - c).pow(2).sum(-1)).sum() / tot).sqrt()
    k = max(1, int(0.1 * w.shape[0]))
    top10 = torch.topk(w, k).values.sum() / tot
    return float(sigma), float(top10), float(w[soft].sum() / tot)


def run_anchor(force):
    p, v, gp = sc.anchor_canonical.clone(), sc.initial_velocity(force), sc.pos.clone()
    out = [gp[mat].clone()]
    for _ in range(args.frames):
        p, v, gp = sc.explicit_step(p, v, gp, args.dt_mult)
        out.append(gp[mat].clone())
    return torch.stack(out), 0.0


def run_fitted(force):
    """the fitted rollout, and the worst CFL overshoot seen along the way"""
    p, v = fit.pos.detach().clone(), fit.impulse_dv(force, cache)
    gp = sc.pos.clone()
    out = [gp[mat].clone()]
    worst = 0.0
    for _ in range(args.frames):
        for _ in range(args.dt_mult):
            p, v, over = fit.substep(p, v, *cache[:5], cache[5].unsqueeze(-1),
                                      (~fit.fixed).unsqueeze(-1).to(p.dtype))
            worst = max(worst, float(over))
        gp = fs.skin(p, gp)
        out.append(gp[mat].clone())
    return torch.stack(out), worst


def strain(x_particles):
    """how far the fitted anchors' implied F is from a rotation, at this state"""
    p = fit.project(x_particles, cache)
    F, _ = fit.deformation(p, *cache[:5])
    return (F - I3).reshape(-1, 9).norm(dim=-1)


rows = []
for i, (truth_all, f) in enumerate(zip(refs, forces)):
    truth = truth_all[: args.frames + 1]
    span = (truth - truth[0]).norm(dim=-1).max().clamp(min=1e-12)
    sigma, top10, soft_share = force_shape(f)

    xa, _ = run_anchor(f)
    xf, cfl = run_fitted(f)
    ea = (xa - truth).norm(dim=-1).mean(-1) / span
    ef = (xf - truth).norm(dim=-1).mean(-1) / span
    ampa = ((xa - xa[0]).norm(dim=-1).max() / span).item()
    ampf = ((xf - xf[0]).norm(dim=-1).max() / span).item()

    # where the fitted error overtakes the sampled one, if it does
    ahead = (ef > ea).nonzero().squeeze(-1)
    cross = int(ahead[0]) if ahead.numel() else -1

    s = strain(truth[-1])
    rows.append(dict(i=i, d=float(span), sigma=sigma / extent, top10=top10,
                      soft=soft_share, ea=float(ea.mean()), ef=float(ef.mean()),
                      eaf=float(ea[-1]), eff=float(ef[-1]), ampa=ampa, ampf=ampf,
                      cfl=cfl, s99=float(s.quantile(0.99)), cross=cross,
                      ef_curve=ef))

hdr = (f"\n{'#':>2} {'MPM moved':>10} {'force len':>10} {'top10%':>7} {'on canopy':>10} "
       f"{'anchor':>8} {'fitted':>8} {'amp a':>7} {'amp f':>7} {'CFL':>9} "
       f"{'p99 strain':>11} {'overtakes':>10}")
print(hdr)
for r in rows:
    mark = "  <-- the loss" if r["ef"] > r["ea"] else ""
    print(f"{r['i']:2d} {r['d']:10.4f} {r['sigma']:9.2f}x {r['top10']:7.2f} "
          f"{100 * r['soft']:9.0f}% {100 * r['ea']:7.2f}% {100 * r['ef']:7.2f}% "
          f"{100 * r['ampa']:6.0f}% {100 * r['ampf']:6.0f}% {r['cfl']:9.2e} "
          f"{r['s99']:11.3f} {r['cross']:10d}{mark}")

bad = [r for r in rows if r["ef"] > r["ea"]]
good = [r for r in rows if r["ef"] <= r["ea"]]
if bad and good:
    print(f"\n{'':22} {'the loss':>12} {'the other ' + str(len(good)):>16}")
    for name, key in (("MPM moved", "d"), ("force length / extent", "sigma"),
                       ("top 10% share", "top10"), ("share on canopy", "soft"),
                       ("fitted amplitude", "ampf"), ("CFL overshoot", "cfl"),
                       ("p99 strain", "s99")):
        b = sum(r[key] for r in bad) / len(bad)
        g = sum(r[key] for r in good) / len(good)
        print(f"  {name:22} {b:12.4g} {g:16.4g}")

for r in bad:
    e = r["ef_curve"]
    marks = " ".join(f"{100 * float(e[t]):.0f}" for t in range(0, args.frames + 1, 6))
    print(f"\n[case {r['i']}] fitted error every 6th frame (%): {marks}")
    print(f"           overtakes the sampled simulator at frame {r['cross']}, "
          f"CFL overshoot {r['cfl']:.2e}")

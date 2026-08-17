"""Does the fitted discretisation fail where the diagnosis says it should?

Measuring MPM alone said the one field failure is a speed failure, not a size
one: #5 has 2.1x the peak per-frame speed of the other eleven cases and 2.2x the
initial kick, while its force shape and MPM's own strain are indistinguishable
from theirs. That is a prediction -- impulses at that kick should fail, impulses
below it should not -- and a prediction is worth more than the diagnosis it came
from.

Kick is exact to target and free to compute: impulse_dv is linear in the force,
so one multiplication puts a drawn field at any kick without running anything.
That is the whole reason to use it rather than displacement, which needed an MPM
run to measure and a rescale that missed its band 36 times out of 40.

Each case gets MPM for the reference, the sampled simulator, and the fitted one,
so what comes out is not "does the fitted simulator look bad" but "does it lose
to the discretisation it was supposed to improve on, and at what kick".

MPM's own domain is the ceiling here -- at large enough impulse the reference
itself leaves the grid -- so rejections are counted and reported. If the fitted
simulator only fails past the point where MPM cannot be asked, that is a
different and much less interesting result.
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
ap.add_argument("--fit", required=True)
ap.add_argument("--kicks", type=float, nargs="+",
                 default=[6.0, 10.0, 14.0, 16.5, 20.0, 25.0],
                 help="initial-kick levels to draw at. 16.5 is field #5's; the "
                      "eleven cases the fit wins sit between 3.2 and 12.5.")
ap.add_argument("--per_kick", type=int, default=3)
ap.add_argument("--n_anchors", type=int, default=512)
ap.add_argument("--K", type=int, default=8)
ap.add_argument("--eig_floor", type=float, default=0.02)
ap.add_argument("--frames", type=int, default=60)
ap.add_argument("--dt_mult", type=int, default=40)
ap.add_argument("--n_grid", type=int, default=100)
ap.add_argument("--seed", type=int, default=909)
ap.add_argument("--out", default=None, help="save the accepted cases for rendering")
args = ap.parse_args()

sys.path.insert(0, args.dreamphysics)
import warp as wp

from anchorflow.anchor_sparse import load_fitted
from anchorflow.mpm_teacher import MPMTeacher
from anchorflow.streams import draw_field_shape

dev = "cuda"
torch.set_grad_enabled(False)
wp.init()

sc = scene_setup.build(args.ply, args.config, args.n_anchors, args.K, device=dev,
                        frozen_weights=True, rot_fallback=True,
                        eig_floor=args.eig_floor)
T = MPMTeacher(sc, n_grid=args.n_grid)
mat = T.mat
base = next(torch.tensor(bc["force"], device=dev)
            for bc in sc.cfg["boundary_conditions"]
            if bc["type"] == "particle_impulse")

fs, fb = load_fitted(sc, args.fit, dev)
fit = fs.fit
cache = fit.prepare()
m_col = cache[5].unsqueeze(-1)
keep_col = (~fit.fixed).unsqueeze(-1).to(fit.pos.dtype)
print(f"[setup] {fit.M} fitted anchors, spacing {sc.sim.radius:.4f}, "
      f"CFL limit {fit.cfl_limit:.5f} per substep")


def kick_of(force):
    """the largest particle speed the impulse imparts, before anything moves"""
    dv = sc.impulse_dv(force)
    return float((T.w.unsqueeze(-1) * dv[T.idx]).sum(1).norm(dim=-1).max())


def mpm_ref(force):
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


def run_anchor(force):
    p, v, gp = sc.anchor_canonical.clone(), sc.initial_velocity(force), sc.pos.clone()
    out = [gp[mat].clone()]
    for _ in range(args.frames):
        p, v, gp = sc.explicit_step(p, v, gp, args.dt_mult)
        if not torch.isfinite(p).all():
            return None
        out.append(gp[mat].clone())
    return torch.stack(out)


def run_fitted(force):
    """the fitted rollout, and the worst CFL overshoot along the way"""
    p, v = fit.pos.detach().clone(), fit.impulse_dv(force, cache)
    gp = sc.pos.clone()
    out = [gp[mat].clone()]
    worst = 0.0
    for _ in range(args.frames):
        for _ in range(args.dt_mult):
            p, v, over = fit.substep(p, v, *cache[:5], m_col, keep_col)
            worst = max(worst, float(over))
        if not torch.isfinite(p).all():
            return None, worst
        gp = fs.skin(p, gp)
        out.append(gp[mat].clone())
    return torch.stack(out), worst


gen = torch.Generator(device=dev); gen.manual_seed(args.seed)
rows, keep_ref, keep_force = [], [], []
bar = tqdm(total=len(args.kicks) * args.per_kick, desc="cases", ncols=90)
for target in args.kicks:
    for j in range(args.per_kick):
        shape, _sigma = draw_field_shape(sc, gen)
        f = shape * base.norm().item()
        k0 = kick_of(f)
        # linear in the force, so this lands exactly
        f = f * (target / max(k0, 1e-12))

        truth = mpm_ref(f)
        if truth is None:
            rows.append(dict(kick=target, ok=False))
            bar.update(1)
            continue
        span = (truth - truth[0]).norm(dim=-1).max().clamp(min=1e-12)
        step = (truth[1:] - truth[:-1]).norm(dim=-1).max(dim=-1).values

        xa = run_anchor(f)
        xf, cfl = run_fitted(f)
        ea = float((xa - truth).norm(dim=-1).mean(-1).mean() / span) if xa is not None else float("nan")
        ef = float((xf - truth).norm(dim=-1).mean(-1).mean() / span) if xf is not None else float("nan")
        ampf = float((xf - xf[0]).norm(dim=-1).max() / span) if xf is not None else float("nan")
        rows.append(dict(kick=target, ok=True, real_kick=kick_of(f),
                          span=float(span), peak=float(step.max()),
                          frac=float(step.max()) / args.dt_mult / sc.sim.radius,
                          ea=ea, ef=ef, ampf=ampf, cfl=cfl,
                          lost=(ef != ef) or (ef > ea)))
        keep_ref.append(truth); keep_force.append(f)
        bar.update(1)
bar.close()

print(f"\n{'kick':>6} {'MPM moved':>10} {'peak/frame':>11} {'substep':>8} "
      f"{'anchor':>8} {'fitted':>8} {'amp f':>7} {'CFL':>10} {'result':>10}")
for r in rows:
    if not r["ok"]:
        print(f"{r['kick']:6.1f} {'--':>10} {'MPM left the grid':>31}")
        continue
    res = "FITTED LOSES" if r["lost"] else "fitted wins"
    ef = "diverged" if r["ef"] != r["ef"] else f"{100 * r['ef']:7.2f}%"
    print(f"{r['kick']:6.1f} {r['span']:10.4f} {r['peak']:11.5f} {r['frac']:7.1%} "
          f"{100 * r['ea']:7.2f}% {ef:>8} {100 * r['ampf']:6.0f}% {r['cfl']:10.2e} "
          f"{res:>12}")

print(f"\n{'kick':>6} {'ran':>5} {'MPM ok':>7} {'fitted wins':>12} {'mean CFL':>10}")
for target in args.kicks:
    g = [r for r in rows if r["kick"] == target]
    ok = [r for r in g if r["ok"]]
    wins = sum(1 for r in ok if not r["lost"])
    mc = sum(r["cfl"] for r in ok) / len(ok) if ok else float("nan")
    print(f"{target:6.1f} {len(g):5d} {len(ok):7d} {wins:6d}/{len(ok):<5d} {mc:10.2e}")

if args.out and keep_ref:
    torch.save({"ref": {"field": keep_ref}, "force": {"field": keep_force},
                 "key": f"cfl_{args.seed}_{args.frames}_{args.dt_mult}"}, args.out)
    print(f"\n[saved] {args.out} ({len(keep_ref)} cases, in eval_vs_mpm's format)")

"""Does inverting the decoder put the anchors somewhere the simulator lives?

project() takes a weighted average of the displacements around each anchor. It
never consults skin(), which is the map that will read the result back, and the
two disagree: the anchor states it produces carry fifteen times the internal
stress of any state the simulator reaches on its own trajectory
(exe/probe_accel_residual.py). Every training sample that is not a DAgger state
starts from one of those.

project_ls() solves for the anchor state whose DECODING is closest to the
particles instead. Same anchors, same weights, no new parameters, no training --
the difference is an adjoint replaced by a least-squares inverse.

Three numbers decide whether that was the problem:

  reconstruction   |skin(p) - x| against how far MPM moved. The least-squares
                   solution minimises exactly this, so it cannot lose; the
                   question is by how much.
  stress           the median |f/m| the force law produces at the projected
                   state, against what the simulator generates for itself. This
                   is the number the whole exercise is aimed at.
  rest             project(Xc) must return the anchors unmoved. A projection
                   that fails this is wrong however good the other two look.
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
ap.add_argument("--ckpt", action="append", default=[])
ap.add_argument("--n_anchors", type=int, default=512)
ap.add_argument("--K", type=int, default=8)
ap.add_argument("--c", type=float, default=0.25)
ap.add_argument("--eig_floor", type=float, default=0.02)
ap.add_argument("--n_traj", type=int, default=4)
ap.add_argument("--frames", type=int, default=60)
ap.add_argument("--dt_mult", type=int, default=40)
ap.add_argument("--every", type=int, default=10)
ap.add_argument("--impulse_range", type=float, default=10.0)
ap.add_argument("--seed", type=int, default=777)
args = ap.parse_args()

sys.path.insert(0, args.dreamphysics)

import warp as wp

from anchorflow.anchor_sparse import AnchorSparse
from anchorflow.mpm_teacher import MPMTeacher
from anchorflow.streams import draw_impulse

dev = "cuda"
torch.manual_seed(0)
wp.init()

sc = scene_setup.build(args.ply, args.config, args.n_anchors, args.K, device=dev,
                        frozen_weights=True, rot_fallback=True,
                        eig_floor=args.eig_floor)
T = MPMTeacher(sc)
base = next(torch.tensor(bc["force"], device=dev)
            for bc in sc.cfg["boundary_conditions"]
            if bc["type"] == "particle_impulse")

fit = AnchorSparse(sc, c=args.c, eig_floor=args.eig_floor).to(dev)
fit.init_from_geometry()
FRESH = {"pos": fit.pos.detach().clone(), "quat": fit.quat.detach().clone(),
         "log_s": fit.log_s.detach().clone(), "log_k": fit.log_k.detach().clone()}


def load(which):
    if which == "untrained":
        fit._rebuild(FRESH["pos"], FRESH["quat"], FRESH["log_s"], FRESH["log_k"])
        return "untrained (geometry init)"
    b = torch.load(which, map_location=dev, weights_only=False)
    fit._rebuild(b["pos"].to(dev), b["quat"].to(dev), b["log_s"].to(dev),
                  b["log_k"].to(dev) if "log_k" in b else None)
    return f"{os.path.basename(which)} at iteration {b.get('iter')}"


fit._rebuild(FRESH["pos"], FRESH["quat"], FRESH["log_s"], FRESH["log_k"])
cache0 = fit.prepare()
g = torch.Generator(device=dev); g.manual_seed(args.seed)
forces = [draw_impulse(sc, base, g, args.impulse_range, field=True)[0]
          for _ in range(args.n_traj)]


@torch.no_grad()
def collect(force):
    dv = fit.impulse_dv(force, cache0)
    v0 = torch.zeros(fit.N, 3, device=dev).index_add_(
        0, fit.pair_g, cache0[0].unsqueeze(-1) * dv[fit.pair_a])
    T._set(T.pos_m.clone(), v0.contiguous(), T.eye.clone(), torch.zeros_like(T.eye))
    got = []
    for f in range(args.frames + 1):
        if f % args.every == 0:
            got.append(T.solver.export_particle_x_to_torch().clone().cpu())
        if f == args.frames:
            break
        for k in range(args.dt_mult):
            T.solver.p2g2p(None, sc.sub_dt, device=T.wp_dev)
            if (k + 1) % 8 == 0 and not T._in_domain():
                return None
    return got


STATES = []
with torch.no_grad():
    for f_ in tqdm(forces, desc="MPM held-out", ncols=90):
        out = collect(f_)
        if out is not None:
            STATES += out
print(f"[data] {len(STATES)} held-out states, seed {args.seed}")
_lo, _hi = fit.Xc.min(0).values, fit.Xc.max(0).values
_ctr = fit.Xc.mean(0)
print(f"[geom] bbox min  {_lo[0]:.4f} {_lo[1]:.4f} {_lo[2]:.4f}")
print(f"[geom] bbox max  {_hi[0]:.4f} {_hi[1]:.4f} {_hi[2]:.4f}")
print(f"[geom] bbox size {(_hi-_lo)[0]:.4f} {(_hi-_lo)[1]:.4f} {(_hi-_lo)[2]:.4f}   "
      f"diagonal {(_hi-_lo).norm():.4f}")
print(f"[geom] centroid  {_ctr[0]:.4f} {_ctr[1]:.4f} {_ctr[2]:.4f}")
_d = (fit.Xc - _ctr).abs()
print(f"[geom] |x-centroid|  mean {_d.mean(0)[0]:.4f} {_d.mean(0)[1]:.4f} {_d.mean(0)[2]:.4f}"
      f"   max {_d.max(0).values[0]:.4f} {_d.max(0).values[1]:.4f} {_d.max(0).values[2]:.4f}")
print(f"[geom] std       {fit.Xc.std(0)[0]:.4f} {fit.Xc.std(0)[1]:.4f} {fit.Xc.std(0)[2]:.4f}")
print(f"[geom] {fit.N} material Gaussians, anchor spacing {sc.sim.radius:.4f}\n")


@torch.no_grad()
def own_stress(force):
    """what the force law produces on the simulator's own trajectory"""
    cache = fit.prepare()
    w, rc, q, Binv, blocked, mass = cache
    free = ~fit.fixed
    p, v = fit.pos.detach().clone(), fit.impulse_dv(force, cache)
    out = []
    for f in range(args.frames + 1):
        if f % args.every == 0 and f > 0:
            a = (fit.force(p, w, rc, q, Binv, blocked) / mass.unsqueeze(-1))[free]
            out.append(float(a.norm(dim=-1).median()))
        if f == args.frames:
            break
        p, v, _ = fit.rollout(p, v, args.dt_mult, cache)
    return sum(out) / max(len(out), 1)


@torch.no_grad()
def measure():
    cache = fit.prepare()
    w, rc, q, Binv, blocked, mass = cache
    free = ~fit.fixed
    fac = fit.ls_factor(cache)

    # rest check: the anchors must come back unmoved
    rest_avg = float((fit.project(fit.Xc, cache) - fit.pos).abs().max())
    rest_ls = float((fit.project_ls(fit.Xc, cache, fac) - fit.pos).abs().max())

    r_avg = r_ls = s_avg = s_ls = 0.0
    abs_avg = abs_ls = moved_abs = 0.0
    n = 0
    for x_c in STATES:
        x = x_c.to(dev)
        # the rest state is in the list and it has not moved: normalising by its
        # own displacement divides by zero. Skipped, and the ruler is the
        # largest displacement in the state rather than the mean, so a state
        # that has barely started is not scored on almost nothing.
        moved = (x - fit.Xc).norm(dim=-1).max()
        if float(moved) < 1e-4:
            continue
        for tag, p in (("avg", fit.project(x, cache)),
                        ("ls", fit.project_ls(x, cache, fac))):
            ae = float((fit.gaussian_pos(p, cache) - x).norm(dim=-1).mean())
            e = ae / float(moved)
            a = float((fit.force(p, w, rc, q, Binv, blocked)
                       / mass.unsqueeze(-1))[free].norm(dim=-1).median())
            if tag == "avg":
                r_avg += e; s_avg += a; abs_avg += ae
            else:
                r_ls += e; s_ls += a; abs_ls += ae
        moved_abs += float(moved)
        n += 1
    assert n, "every state was at rest"
    k = max(n, 1)
    own = sum(own_stress(f_) for f_ in forces) / len(forces)
    return (100 * r_avg / k, 100 * r_ls / k, s_avg / k, s_ls / k, own,
            rest_avg, rest_ls, abs_avg / k, abs_ls / k, moved_abs / k)


print(f"{'checkpoint':26} {'|err| avg':>10} {'|err| LS':>10} {'MPM moved':>10} | "
      f"{'rel avg':>8} {'rel LS':>8} | {'stress avg':>10} {'stress LS':>9} {'own':>7}")
for which in (args.ckpt or ["untrained"]):
    name = load(which)
    ra, rl, sa, sl, own, qa, ql, aa, al, mv = measure()
    print(f"{name[:26]:26} {aa:10.3e} {al:10.3e} {mv:10.3e} | "
          f"{ra:7.2f}% {rl:7.2f}% | {sa:10.1f} {sl:9.1f} {own:7.1f}")

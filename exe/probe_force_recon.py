"""How much of MPM's force survives the trip through the anchors?

The earlier comparison -- the force law at a projected MPM state against the
force law on the simulator's own trajectory -- was not a reconstruction error.
Those two configurations are deformed by different amounts, so their ratio mixes
what the compression loses with how far the two trajectories have drifted apart.

This measures the compression alone. One MPM state, one moment in time, and
every quantity computed twice from it:

  MPM's own      F comes straight out of the solver, where it was accumulated
                 per particle by F <- (I + dt grad v) F.
  through anchors x is encoded to anchor positions and F is DERIVED from them by
                 shape matching -- which is all the anchor simulator ever has.

Then the same material law is applied to both, and the resulting anchor forces
are gathered the same way, so any difference is the discretisation and nothing
else. Three levels are reported because they do not have to agree: F can be off
while the stress is not (the law saturates), and the stress can be off while the
gathered force is not (the error is in a direction the gather cancels).

Both encoders are run, since the least-squares one reconstructs POSITIONS six
times better and the question is whether that buys anything for the forces.
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

from anchorflow.anchor_fit import det3, inv3
from anchorflow.anchor_sparse import AnchorSparse
from anchorflow.anchor_fit import closest_rotation
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
        return "untrained"
    b = torch.load(which, map_location=dev, weights_only=False)
    fit._rebuild(b["pos"].to(dev), b["quat"].to(dev), b["log_s"].to(dev),
                  b["log_k"].to(dev) if "log_k" in b else None)
    return f"{os.path.basename(which)} it {b.get('iter')}"


fit._rebuild(FRESH["pos"], FRESH["quat"], FRESH["log_s"], FRESH["log_k"])
cache0 = fit.prepare()
g = torch.Generator(device=dev); g.manual_seed(args.seed)
forces = [draw_impulse(sc, base, g, args.impulse_range, field=True)[0]
          for _ in range(args.n_traj)]


@torch.no_grad()
def collect(force):
    """MPM's positions AND its own deformation gradient, at sampled frames"""
    dv = fit.impulse_dv(force, cache0)
    v0 = torch.zeros(fit.N, 3, device=dev).index_add_(
        0, fit.pair_g, cache0[0].unsqueeze(-1) * dv[fit.pair_a])
    T._set(T.pos_m.clone(), v0.contiguous(), T.eye.clone(), torch.zeros_like(T.eye))
    got = []
    for f in range(args.frames + 1):
        if f % args.every == 0 and f > 0:
            got.append((T.solver.export_particle_x_to_torch().clone().cpu(),
                        T.solver.export_particle_F_to_torch()
                        .reshape(-1, 3, 3).clone().cpu()))
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
print(f"[data] {len(STATES)} held-out states, seed {args.seed}\n")


@torch.no_grad()
def stress(F, w):
    """Fixed Corotated PK1, exactly as force() forms it"""
    R = closest_rotation(F, fit.polar_iters, fit.polar_ridge)
    J = det3(F)
    n = F.reshape(-1, 9).norm(dim=-1).clamp(min=1e-12).reshape(-1, 1, 1)
    Finv_T = inv3(F + 1e-6 * n * torch.eye(3, device=dev),
                   eps=1e-30).transpose(-1, -2)
    k = fit.stiffness(w).unsqueeze(-1).unsqueeze(-1)
    mu = fit.mu.unsqueeze(-1).unsqueeze(-1) * k
    lam = fit.lam.unsqueeze(-1).unsqueeze(-1) * k
    return 2 * mu * (F - R) + lam * (J - 1).unsqueeze(-1).unsqueeze(-1) * \
        J.unsqueeze(-1).unsqueeze(-1) * Finv_T


@torch.no_grad()
def gather(P, w, q, Binv):
    """PK1 -> anchor force, the same reduction force() ends with"""
    PB = (P @ Binv)[fit.pair_g]
    contrib = -(fit.vol[fit.pair_g] * w).unsqueeze(-1) * \
        torch.einsum("pij,pj->pi", PB, q)
    return torch.zeros(fit.M, 3, device=dev).index_add_(0, fit.pair_a, contrib)


def rel(a, b):
    """|a - b| against |b|, particle-wise then averaged"""
    d = (a - b).reshape(a.shape[0], -1).norm(dim=-1)
    s = b.reshape(b.shape[0], -1).norm(dim=-1).clamp(min=1e-20)
    return float((d / s).median())


@torch.no_grad()
def measure(encoder):
    cache = fit.prepare()
    w, rc, q, Binv, blocked, mass = cache
    fac = fit.ls_factor(cache) if encoder == "ls" else None
    free = ~fit.fixed
    eF = eP = ef = 0.0
    mag_a = mag_m = 0.0
    n = 0
    for x_c, Fm_c in STATES:
        x, Fm = x_c.to(dev), Fm_c.to(dev)
        p = fit.project_ls(x, cache, fac) if encoder == "ls" else fit.project(x, cache)
        Fa, _ = fit.deformation(p, w, rc, q, Binv, blocked)

        # F: measured against how far MPM's own F is from the identity, since a
        # relative error against F itself would be flattered by F ~ I
        eye = torch.eye(3, device=dev)
        d = (Fa - Fm).reshape(-1, 9).norm(dim=-1)
        s = (Fm - eye).reshape(-1, 9).norm(dim=-1).clamp(min=1e-20)
        eF += float((d / s).median())

        Pm, Pa = stress(Fm, w), stress(Fa, w)
        eP += rel(Pa, Pm)

        fm, fa = gather(Pm, w, q, Binv), gather(Pa, w, q, Binv)
        ef += float(((fa - fm).norm(dim=-1) / fm.norm(dim=-1).clamp(min=1e-20))[free].median())
        mag_m += float((fm / mass.unsqueeze(-1))[free].norm(dim=-1).median())
        mag_a += float((fa / mass.unsqueeze(-1))[free].norm(dim=-1).median())
        n += 1
    k = max(n, 1)
    return 100 * eF / k, 100 * eP / k, 100 * ef / k, mag_m / k, mag_a / k


print(f"{'checkpoint':22} {'enc':>4} {'F err':>9} {'stress err':>11} {'force err':>10} "
      f"{'|a| MPM-F':>11} {'|a| anchor-F':>13}")
for which in (args.ckpt or ["untrained"]):
    name = load(which)
    for enc in ("avg", "ls"):
        eF, eP, ef, mm, ma = measure(enc)
        print(f"{name[:22]:22} {enc:>4} {eF:8.2f}% {eP:10.2f}% {ef:9.2f}% "
              f"{mm:11.1f} {ma:13.1f}")

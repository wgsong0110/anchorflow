"""Can the anchors hold MPM's acceleration field, and can they predict it?

Two different questions that a trajectory loss never separates, and the answer
decides what to do next:

  representation  MPM's acceleration, pushed onto the anchors and pulled back
                  through the simulator's OWN encode/decode (project_v, and the
                  velocity path of lift()). No force law involved. This is a
                  floor: whatever the anchors cannot hold, no stress model can
                  predict.

  dynamics        the acceleration the anchor simulator computes for itself,
                  from its own Fixed Corotated force at the projected state,
                  against MPM's acceleration projected onto the same anchors.
                  Both live in anchor space, so nothing is skinned back and the
                  reconstruction offset that swamps a substep position error
                  never enters. This is what a force-matching loss would train,
                  and log_k -- invisible to any representation loss -- is in it.

Reported together with the cosine between the two anchor fields and the ratio of
their magnitudes, because a force law that points the right way and is 30% too
weak is a stiffness that needs rescaling, while one that points elsewhere is a
different problem.

The scene has no gravity (g = [0,0,0]) and the base is a sticky cuboid, so the
only body force is elastic and the fixed anchors are excluded. MPM's grid
velocity damping (0.9999 a substep) rides along in the measured acceleration and
is left there: it is one part in ten thousand.

  python exe/probe_accel_residual.py --ply ... --config ... \
      --ckpt untrained --ckpt /workspace/N_decay.pt --ckpt /workspace/k_rl.pt
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


# ---- states, collected once with the fresh anchors ---------------------------
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
            got.append(tuple(t.clone().cpu() for t in (
                T.solver.export_particle_x_to_torch(),
                T.solver.export_particle_v_to_torch(),
                T.solver.export_particle_F_to_torch().reshape(-1, 9),
                T.solver.export_particle_C_to_torch().reshape(-1, 9))))
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


@torch.no_grad()
def mpm_accel(state):
    """MPM's acceleration at one of its own states, on two time scales.

    One substep is what a force law is compared against, but MPM's particle
    velocity after one substep comes back off the grid, so the difference
    carries the transfer's spatial smoothing as well as the force. Over a whole
    coarse frame that noise averages out and what is left is the acceleration
    the coarse simulator is actually asked to reproduce. Both are returned,
    because whether they differ IS the measurement.
    """
    x, v, F, C = (t.to(dev, torch.float32) for t in state)
    T._set(x.contiguous(), v.contiguous(), F.contiguous(), C.contiguous())
    T.solver.p2g2p(None, sc.sub_dt, device=T.wp_dev)
    if not T._in_domain():
        return None
    a1 = (T.solver.export_particle_v_to_torch() - v) / sc.sub_dt
    for _ in range(args.dt_mult - 1):
        T.solver.p2g2p(None, sc.sub_dt, device=T.wp_dev)
        if not T._in_domain():
            return None
    aN = (T.solver.export_particle_v_to_torch() - v) / (args.dt_mult * sc.sub_dt)
    return x, v, a1, aN


ACC = []
with torch.no_grad():
    for s in tqdm(STATES, desc="MPM accelerations", ncols=90):
        a = mpm_accel(s)
        if a is not None:
            ACC.append(tuple(t.cpu() for t in a))
print(f"[data] {len(ACC)} held-out states, seed {args.seed} "
      f"(not the fit's 4242 or the check set's 31337)\n")


@torch.no_grad()
def decode(u, p, cache):
    """an anchor vector field -> per particle, the way lift() reconstructs velocity"""
    w, rc, q, Binv, blocked, _ = cache
    F, _ = fit.deformation(p, w, rc, q, Binv, blocked)
    ua = u[fit.pair_a]
    uc = torch.zeros(fit.N, 3, device=dev).index_add_(
        0, fit.pair_g, w.unsqueeze(-1) * ua)
    du = ua - uc[fit.pair_g]
    Ad = torch.zeros(fit.N, 3, 3, device=dev).index_add_(
        0, fit.pair_g, w.reshape(-1, 1, 1) * (du.unsqueeze(-1) * q.unsqueeze(-2)))
    return uc + torch.einsum("nij,nj->ni", Ad @ Binv, fit.Xc - rc)


@torch.no_grad()
def measure(coarse):
    """rep, dyn, cos, |a_anchor|/|a_mpm|, and the three magnitudes themselves.

    The magnitudes are reported because a ratio of 750 and a cosine of 0.03 have
    two very different explanations -- a force law that is wrong, or a target
    that cancelled itself when it was averaged onto the anchors -- and only the
    magnitudes tell them apart.
    """
    cache = fit.prepare()
    w, rc, q, Binv, blocked, mass = cache
    free = ~fit.fixed
    rep = dyn = cos = ratio = 0.0
    mp = mt = ma = hi = 0.0
    n = 0
    for row in ACC:
        x = row[0].to(dev)
        a = row[3 if coarse else 2].to(dev)
        p = fit.project(x, cache)
        ua = fit.project_v(a, cache)                      # MPM's acceleration, on anchors

        # (a) can the anchors hold it at all
        back = decode(ua, p, cache)
        amag = a.norm(dim=-1).mean().clamp(min=1e-20)
        rep += float((back - a).norm(dim=-1).mean() / amag)

        # (b) does the force law predict it
        f = fit.force(p, w, rc, q, Binv, blocked)
        aa = (f / mass.unsqueeze(-1))[free]
        tt = ua[free]
        # per anchor, then the median: a handful of anchors sitting on a nearly
        # degenerate neighbourhood carry forces two orders above the rest, and
        # any mean or global dot product is a report about those few
        na, nt = aa.norm(dim=-1), tt.norm(dim=-1)
        dyn += float(((aa - tt).norm(dim=-1) / nt.clamp(min=1e-20)).median())
        cos += float(((aa * tt).sum(-1) / (na * nt).clamp(min=1e-20)).median())
        ratio += float((na / nt.clamp(min=1e-20)).median())
        mp += float(amag)
        mt += float(nt.median())
        ma += float(na.median())
        hi += float(na.quantile(0.99) / na.median().clamp(min=1e-20))
        n += 1
    k = max(n, 1)
    return (100 * rep / k, 100 * dyn / k, cos / k, ratio / k,
            mp / k, mt / k, ma / k, hi / k)


@torch.no_grad()
def self_accel(force):
    """what this simulator's own force law produces on its OWN trajectory.

    The control the two residuals above are meaningless without: if the force at
    a projected MPM state is a hundred times what the simulator ever generates
    for itself, the projection is putting the anchors somewhere they never go,
    and no comparison made there says anything about the force law.
    """
    cache = fit.prepare()
    w, rc, q, Binv, blocked, mass = cache
    free = ~fit.fixed
    p = fit.pos.detach().clone()
    v = fit.impulse_dv(force, cache)
    out = []
    for f in range(args.frames + 1):
        if f % args.every == 0:
            a = (fit.force(p, w, rc, q, Binv, blocked) / mass.unsqueeze(-1))[free]
            out.append(float(a.norm(dim=-1).median()))
        if f == args.frames:
            break
        p, v, _ = fit.rollout(p, v, args.dt_mult, cache)
    return out


print(f"[dt] the simulator integrates at {fit.dt:.2e}, MPM at {sc.sub_dt:.2e}\n")
print(f"{'checkpoint':30} {'|a| on its own trajectory (median per anchor)':>50}")
for which in (args.ckpt or ["untrained"]):
    name = load(which)
    rows = [self_accel(f_) for f_ in forces]
    med = [sum(r[k] for r in rows) / len(rows) for k in range(len(rows[0]))]
    print(f"{name[:30]:30} " + " ".join(f"{m:8.1f}" for m in med))

for coarse in (False, True):
    print(f"\n=== target: {'one coarse frame (40 substeps)' if coarse else 'one substep'} ===")
    print(f"{'checkpoint':30} {'(a)rep':>9} {'(b)dyn':>10} {'cos':>7} {'ratio':>8} "
          f"{'|a_mpm|':>10} {'|a_proj|':>10} {'|a_anc|':>10} {'p99/med':>8}")
    for which in (args.ckpt or ["untrained"]):
        name = load(which)
        rep, dyn, cos, ratio, mp, mt, ma, hi = measure(coarse)
        print(f"{name[:30]:30} {rep:8.2f}% {dyn:9.2f}% {cos:7.3f} {ratio:8.2f} "
              f"{mp:10.3e} {mt:10.3e} {ma:10.3e} {hi:8.1f}")

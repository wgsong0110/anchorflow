"""Does a fit trained on twelve substeps hold up on twelve substeps it never saw?

The substep-supervised fit (--fine_steps 12) did not improve the sixty-frame
rollout, and that was recorded as "it did not learn". But that conflates two
different failures, and the fix differs:

  the loss never moved       -- twelve substeps of an object that has barely
                                deformed carry no signal, and the fit is no
                                better than the discretisation it started from
                                even on its own horizon.

  the loss moved, the        -- the same failure --unroll 1 has: what is learned
  horizon did not transfer      on a short horizon says nothing about a long one.

So this measures the short horizon directly, on impulses drawn from a stream
neither the fit set (seed 4242) nor the check set (seed 31337) came from. MPM is
restarted from its OWN recorded state (x, v, F, C) rather than from a lifted
anchor state, so what is compared is the simulator, not the lift.

Error at substep k is against how far MPM has moved from the start by substep k,
which is the same normalisation the fine loss trained against -- so the number
here is on the same scale as the loss that run was minimising.

  python exe/probe_fine_generalize.py --ply ... --config ... \
      --ckpt untrained --ckpt /workspace/I_fine.pt --ckpt /workspace/k_rl.pt
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
ap.add_argument("--ckpt", action="append", default=[],
                 help="a fitted checkpoint, or the literal 'untrained' for the "
                      "discretisation as it comes out of scene_setup. Repeatable; "
                      "every one is measured on the same states.")
ap.add_argument("--n_anchors", type=int, default=512)
ap.add_argument("--K", type=int, default=8)
ap.add_argument("--c", type=float, default=0.25)
ap.add_argument("--eig_floor", type=float, default=0.02)
ap.add_argument("--n_traj", type=int, default=6, help="held-out impulses")
ap.add_argument("--frames", type=int, default=60)
ap.add_argument("--dt_mult", type=int, default=40)
ap.add_argument("--every", type=int, default=10, help="sample a state every N frames")
ap.add_argument("--n_sub", type=int, default=12, help="substeps to walk, the fine horizon")
ap.add_argument("--impulse_range", type=float, default=10.0)
ap.add_argument("--seed", type=int, default=777, help="NOT 4242 (fit) or 31337 (check)")
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
FRESH = {k: v.detach().clone() for k, v in
         (("pos", fit.pos), ("quat", fit.quat), ("log_s", fit.log_s),
          ("log_k", fit.log_k))}


def load(which):
    """put one set of anchors in place, and report what it is"""
    if which == "untrained":
        fit._rebuild(FRESH["pos"], FRESH["quat"], FRESH["log_s"], FRESH["log_k"])
        return "untrained (geometry init)"
    b = torch.load(which, map_location=dev, weights_only=False)
    fit._rebuild(b["pos"].to(dev), b["quat"].to(dev), b["log_s"].to(dev),
                  b["log_k"].to(dev) if "log_k" in b else None)
    return f"{os.path.basename(which)} at iteration {b.get('iter')}"


# ---- the states, collected once and reused for every checkpoint --------------
#
# Collected with the FRESH anchors, because the impulse is applied through the
# anchors' weights and a checkpoint that moved them would give each checkpoint a
# different trajectory to be scored on.
fit._rebuild(FRESH["pos"], FRESH["quat"], FRESH["log_s"], FRESH["log_k"])
cache0 = fit.prepare()

g = torch.Generator(device=dev); g.manual_seed(args.seed)
forces = [draw_impulse(sc, base, g, args.impulse_range, field=True)[0]
          for _ in range(args.n_traj)]

STATES = []          # (x, v, F, C) at every sampled frame, on the CPU


@torch.no_grad()
def collect(force):
    dv = fit.impulse_dv(force, cache0)
    v0 = torch.zeros(fit.N, 3, device=dev).index_add_(
        0, fit.pair_g, cache0[0].unsqueeze(-1) * dv[fit.pair_a])
    T._set(T.pos_m.clone(), v0.contiguous(), T.eye.clone(), torch.zeros_like(T.eye))
    got = []
    for f in range(args.frames + 1):
        if f % args.every == 0:
            got.append((T.solver.export_particle_x_to_torch().clone(),
                        T.solver.export_particle_v_to_torch().clone(),
                        T.solver.export_particle_F_to_torch().reshape(-1, 9).clone(),
                        T.solver.export_particle_C_to_torch().reshape(-1, 9).clone()))
        if f == args.frames:
            break
        for k in range(args.dt_mult):
            T.solver.p2g2p(None, sc.sub_dt, device=T.wp_dev)
            if (k + 1) % 8 == 0 and not T._in_domain():
                return None                      # MPM left its grid: unusable
    return got


with torch.no_grad():
    for f_ in tqdm(forces, desc="MPM held-out", ncols=90):
        out = collect(f_)
        if out is None:
            continue
        STATES += [tuple(t.to(torch.float16).cpu() for t in s) for s in out]
print(f"[data] {len(STATES)} held-out states from {args.n_traj} impulses, "
      f"seed {args.seed}")


# ---- and the measurement ----------------------------------------------------
@torch.no_grad()
def truth(state):
    """MPM's own next n_sub substeps from one of its own states"""
    x, v, F, C = (t.to(dev, torch.float32) for t in state)
    T._set(x.contiguous(), v.contiguous(), F.contiguous(), C.contiguous())
    out = []
    for _ in range(args.n_sub):
        T.solver.p2g2p(None, sc.sub_dt, device=T.wp_dev)
        if not T._in_domain():
            return None
        out.append(T.solver.export_particle_x_to_torch().to(torch.float16).cpu())
    # kept on the CPU in half: twelve substeps of 171k particles is 25 MB a
    # state in float32, and this runs beside a fit that owns most of the card
    return x.to(torch.float16).cpu(), torch.stack(out)


TRUTH = []
with torch.no_grad():
    for s in tqdm(STATES, desc="MPM substeps", ncols=90):
        t_ = truth(s)
        if t_ is not None:
            TRUTH.append((s, t_[0], t_[1]))
print(f"[data] {len(TRUTH)} states MPM would continue from\n")


@torch.no_grad()
def measure():
    """per-substep error, averaged over states, in percent"""
    cache = fit.prepare()
    acc = torch.zeros(args.n_sub, device=dev)
    n = 0
    for state, x0_c, tgt_c in TRUTH:
        x0 = x0_c.to(dev, torch.float32)
        tgt = tgt_c.to(dev, torch.float32)
        x = state[0].to(dev, torch.float32)
        v = state[1].to(dev, torch.float32)
        p, vv = fit.project(x, cache), fit.project_v(v, cache)
        row = []
        for k in range(args.n_sub):
            p, vv, _ = fit.rollout(p, vv, 1, cache)
            got = fit.gaussian_pos(p, cache)
            d = (tgt[k] - x0).norm(dim=-1).mean().clamp(min=1e-12)
            row.append((got - tgt[k]).norm(dim=-1).mean() / d)
        r = torch.stack(row)
        if torch.isfinite(r).all():
            acc += r
            n += 1
    return 100 * acc / max(n, 1), n


print(f"{'checkpoint':38} {'sub 1':>8} {'sub 4':>8} {'sub 8':>8} "
      f"{'sub 12':>8} {'mean':>8}  states")
rows = {}
for which in (args.ckpt or ["untrained"]):
    name = load(which)
    curve, n = measure()
    rows[which] = curve
    print(f"{name[:38]:38} {curve[0]:8.2f} {curve[3]:8.2f} {curve[7]:8.2f} "
          f"{curve[-1]:8.2f} {curve.mean():8.2f}  {n}")

if len(rows) > 1:
    print("\nper substep, in full:")
    print(f"{'substep':>8} " + " ".join(f"{os.path.basename(k)[:14]:>14}" for k in rows))
    for k in range(args.n_sub):
        print(f"{k + 1:>8} " + " ".join(f"{v[k]:14.2f}" for v in rows.values()))

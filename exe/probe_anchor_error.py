"""Does the student fail at the anchors whose properties it cannot see?

The fitted set is strongly heterogeneous -- stiffness spread over 10x, extents
over 11x, a median 2:1 anisotropy and rotations past 90 degrees -- while the
student's input is each anchor's position and velocity and nothing else. Two
anchors at the same place moving the same way but with different stiffness must
move apart on the next step, and nothing in the input distinguishes them. The
sampled set has no such problem: every anchor there is isotropic with stiffness
one, so position and velocity determine the dynamics completely.

If that partial observability is what costs the student, its error should be
concentrated at the anchors whose hidden properties are furthest from typical.
This checks that without training anything: roll the student and its teacher out
side by side, take the error PER ANCHOR, and correlate it with each anchor's
stiffness, extent, anisotropy, rotation and local crowding.

A flat correlation would be the more useful answer -- it would rule out the
explanation that the next experiment is built on.
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
ap.add_argument("--ckpt", required=True)
ap.add_argument("--fit", required=True)
ap.add_argument("--n_traj", type=int, default=12)
ap.add_argument("--impulse_range", type=float, default=16.0)
ap.add_argument("--n_anchors", type=int, default=512)
ap.add_argument("--K", type=int, default=8)
ap.add_argument("--eig_floor", type=float, default=0.02)
ap.add_argument("--frames", type=int, default=60)
ap.add_argument("--dt_mult", type=int, default=40)
ap.add_argument("--seed", type=int, default=31337)
args = ap.parse_args()

sys.path.insert(0, args.dreamphysics)
import warp as wp

from anchorflow.anchor_fit import quat_to_R
from anchorflow.anchor_sparse import load_fitted
from anchorflow.nextstate import apply_step, net_from_ckpt
from anchorflow.streams import draw_impulse

dev = "cuda"
torch.set_grad_enabled(False)
wp.init()

sc = scene_setup.build(args.ply, args.config, args.n_anchors, args.K, device=dev,
                        frozen_weights=True, rot_fallback=True,
                        eig_floor=args.eig_floor)
fs, _fb = load_fitted(sc, args.fit, dev)
fit = fs.fit
base = next(torch.tensor(bc["force"], device=dev)
            for bc in sc.cfg["boundary_conditions"]
            if bc["type"] == "particle_impulse")
dt = args.dt_mult * sc.sub_dt

ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
ta = ck["args"]
net = net_from_ckpt(ck, dev)
print(f"[setup] {os.path.basename(args.ckpt)} on {fit.M} fitted anchors, "
      f"{args.n_traj} impulses, {args.frames} frames")

# ---- the properties the student cannot see ---------------------------------
k = fit.log_k.exp()
s = fit.log_s.exp()
qn = fit.quat / fit.quat.norm(dim=-1, keepdim=True)
props = {
    "stiffness k": k,
    "|log k| from median": (fit.log_k - fit.log_k.median()).abs(),
    "mean extent": s.mean(-1),
    "anisotropy": s.max(-1).values / s.min(-1).values.clamp(min=1e-9),
    "rotation angle": 2 * qn[:, 0].abs().clamp(max=1.0).acos(),
}
# crowding is visible to the student in principle -- it is geometry -- so it is
# the control: a correlation here does not support the hidden-property story
d = torch.cdist(fit.pos.detach(), fit.pos.detach())
d.fill_diagonal_(float("inf"))
props["nearest neighbour"] = d.min(-1).values
props["pairs held"] = torch.bincount(fit.pair_a, minlength=fit.M).float()


def rollouts(force):
    """teacher and student on the same anchors from the same start, in ANCHOR
    space -- per-anchor error is the point, so nothing is skinned"""
    p_t, v_t = fit.pos.detach().clone(), fs.initial_velocity(force)
    p_s, v_s = p_t.clone(), v_t.clone()
    et = torch.zeros(fit.M, device=dev)
    gp = sc.pos.clone()
    T = [p_t.clone()]
    for _ in range(args.frames):
        p_t, v_t, gp = fs.explicit_step(p_t, v_t, gp, args.dt_mult)
        if not torch.isfinite(p_t).all():
            return None, None
        T.append(p_t.clone())
    S, kf = [p_s.clone()], 0
    while kf < args.frames:
        for q, dd in apply_step(net, p_s, v_s, None, dt, fs.fixed_mask):
            p_s, v_s = q, dd / dt
            if not torch.isfinite(p_s).all():
                return None, None
            S.append(p_s.clone())
            kf += 1
            if kf >= args.frames:
                break
    T, S = torch.stack(T), torch.stack(S[: args.frames + 1])
    # each anchor's error, averaged over frames, against how far the TEACHER
    # moved that anchor -- an anchor that never moves cannot be got wrong
    moved = (T - T[0]).norm(dim=-1).max(0).values
    et = (S - T).norm(dim=-1).mean(0)
    return et, moved


g = torch.Generator(device=dev); g.manual_seed(args.seed)
err_sum = torch.zeros(fit.M, device=dev)
moved_sum = torch.zeros(fit.M, device=dev)
n = 0
for _ in tqdm(range(args.n_traj), desc="rollouts", ncols=88):
    f, _ = draw_impulse(sc, base, g, args.impulse_range, field=True)
    e, mv = rollouts(f)
    if e is None:
        continue
    err_sum += e; moved_sum += mv; n += 1
print(f"[ran] {n} of {args.n_traj} impulses completed")

# relative error per anchor: how badly the student tracks that anchor, against
# how much that anchor actually moves. Anchors the teacher barely moves would
# otherwise dominate the correlation with near-zero errors
rel = err_sum / moved_sum.clamp(min=1e-9)
mobile = moved_sum / n > 0.02 * float((moved_sum / n).max())
print(f"[anchors] {int(mobile.sum())} of {fit.M} move enough to be scored")


def corr(a, b):
    a = a[mobile].float(); b = b[mobile].float()
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm()).clamp(min=1e-20))


def spearman(a, b):
    """rank correlation, so one extreme anchor cannot carry the result"""
    ra = a[mobile].float().argsort().argsort().float()
    rb = b[mobile].float().argsort().argsort().float()
    return corr_raw(ra, rb)


def corr_raw(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm()).clamp(min=1e-20))


print(f"\n{'property':>24} {'pearson':>9} {'spearman':>9}   {'lowest quartile':>15} "
      f"{'highest quartile':>16}")
for name, v in props.items():
    vm = v[mobile].float()
    rm = rel[mobile]
    q1, q3 = vm.quantile(0.25), vm.quantile(0.75)
    lo = float(rm[vm <= q1].mean()) * 100
    hi = float(rm[vm >= q3].mean()) * 100
    print(f"{name:>24} {corr(rel, v):9.3f} {spearman(rel, v):9.3f}   "
          f"{lo:14.2f}% {hi:15.2f}%")

print(f"\n  overall relative error {100 * float(rel[mobile].mean()):.2f}%")
print("  'quartile' columns are the mean relative error of the anchors in the\n"
       "  bottom and top quarter of that property. A property that matters should\n"
       "  separate them by more than the noise between rollouts.")

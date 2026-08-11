"""Fit the anchor discretisation so the simulator follows MPM.

The anchor simulator is 5.5% from PhysGaussian's MPM over a rollout, and a
student trained on it inherits that -- measured: the student trained against it
lands within 2 sigma of its own teacher on every impulse family, so improving
the student cannot help and improving the teacher is the whole lever.

What is fitted is the discretisation, not the material. Each anchor carries a
position, an orientation and three extents, which is what turns the isotropic
kernel into a Mahalanobis one; E, nu and density stay exactly what the config
says. That distinction matters: an anchor set that reproduces MPM by quietly
softening the material has reproduced nothing.

The loss is one coarse step from MPM's own states. Instantaneous forces cannot
be matched -- the simulator's |f/m| is around 175 on its own trajectory against
MPM's 19, because the projection averages thousands of particles and erases a
high-frequency component that averages out over a frame anyway -- but what
survives a frame is exactly what the project cares about.

The step is relative: divided by how far MPM moved over it. Late frames move
less as the motion damps out, and an absolute loss would spend all its weight on
the first few frames, which are the ones already correct.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

_lib = os.path.join(os.path.dirname(__file__), "..", "lib")
sys.path.insert(0, _lib)

import torch
from tqdm import tqdm

from anchorflow import scene_setup
from anchorflow.anchor_fit import AnchorFit
from anchorflow.streams import rand_rot

ap = argparse.ArgumentParser()
ap.add_argument("--ply", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--dreamphysics", default="/workspace/DreamPhysics")
ap.add_argument("--n_fit", type=int, default=3, help="MPM trajectories to fit on")
ap.add_argument("--n_check", type=int, default=2, help="held out")
ap.add_argument("--frames", type=int, default=60)
ap.add_argument("--dt_mult", type=int, default=40)
ap.add_argument("--n_anchors", type=int, default=512)
ap.add_argument("--K", type=int, default=8)
ap.add_argument("--eig_floor", type=float, default=0.02)
ap.add_argument("--polar_iters", type=int, default=6)
ap.add_argument("--iters", type=int, default=400)
ap.add_argument("--batch", type=int, default=2, help="states per iteration")
ap.add_argument("--lr_pos", type=float, default=3e-4)
ap.add_argument("--lr_scale", type=float, default=1e-2)
ap.add_argument("--lr_quat", type=float, default=1e-2)
ap.add_argument("--warmup", type=int, default=80,
                 help="iterations spent on the low-deformation end before the "
                      "whole trajectory is sampled")
ap.add_argument("--no_geom_init", action="store_true")
ap.add_argument("--freeze", nargs="*", default=[], choices=["pos", "log_s", "quat"])
ap.add_argument("--out", default=None)
ap.add_argument("--state", default=None,
                 help="where the full fit state lives, so a host taking its GPU back "
                      "mid-run costs minutes rather than the whole fit. Defaults to "
                      "--out with .state appended.")
ap.add_argument("--resume", action="store_true")
ap.add_argument("--eval_every", type=int, default=50)
args = ap.parse_args()

sys.path.insert(0, args.dreamphysics)
import warp as wp

from anchorflow.mpm_teacher import MPMTeacher

dev = "cuda"
torch.manual_seed(0)
wp.init()

sc = scene_setup.build(args.ply, args.config, args.n_anchors, args.K, device=dev,
                        frozen_weights=True, rot_fallback=True, eig_floor=args.eig_floor)
AC, fixed = sc.anchor_canonical, sc.fixed_mask
T = MPMTeacher(sc)
base = None
for bc in sc.cfg.get("boundary_conditions", []):
    if bc["type"] == "particle_impulse":
        base = torch.tensor(bc["force"], device=dev)

fit = AnchorFit(sc, eig_floor=args.eig_floor, polar_iters=args.polar_iters).to(dev)
if not args.no_geom_init:
    thin = fit.init_from_geometry()
    s_ = fit.log_s.exp()
    print(f"[init] oriented along the local material; axis ratio median "
          f"{(s_.max(-1).values / s_.min(-1).values).median():.2f}, "
          f"{thin} anchors left isotropic")
for n in args.freeze:
    getattr(fit, n).requires_grad_(False)
print(f"[setup] {sc.M} anchors, fitting "
      f"{sum(p.numel() for p in fit.parameters() if p.requires_grad)} parameters")


@torch.no_grad()
def states(force):
    """MPM's trajectory as anchor states: (p, v) per frame"""
    dv = sc.impulse_dv(force)
    v0 = (T.w.unsqueeze(-1) * dv[T.idx]).sum(1).contiguous()
    T._set(T.pos_m.clone(), v0, T.eye.clone(), torch.zeros_like(T.eye))
    ps, vs = [AC.clone()], [sc.initial_velocity(force)]
    for _ in range(args.frames):
        for k in range(args.dt_mult):
            T.solver.p2g2p(None, sc.sub_dt, device=T.wp_dev)
            if (k + 1) % 8 == 0 and not T._in_domain():
                return None
        ps.append(T.project(T.solver.export_particle_x_to_torch()))
        vs.append(T.project_v(T.solver.export_particle_v_to_torch()))
    return torch.stack(ps), torch.stack(vs)


def draw(g, n):
    out = [base]
    while len(out) < n:
        k = 0.5 * (4.0 ** torch.rand(1, device=dev, generator=g).item())
        out.append((rand_rot(g, dev) @ base) * k)
    return out[:n]


gen = torch.Generator(device=dev); gen.manual_seed(4242)
FIT = [s for s in (states(f) for f in tqdm(draw(gen, args.n_fit), desc="MPM fit", ncols=90))
       if s is not None]
gen2 = torch.Generator(device=dev); gen2.manual_seed(31337)
CHK = [s for s in (states(f) for f in tqdm(draw(gen2, args.n_check + 1)[1:],
                                            desc="MPM check", ncols=90)) if s is not None]
print(f"[data] {len(FIT)} fit and {len(CHK)} held-out trajectories x {args.frames} frames")


def step_error(sets, tmax=None, every=1):
    """how far one coarse step lands from MPM's, relative to the step itself"""
    tot, n = 0.0, 0
    cache = fit.prepare()
    for P, V in sets:
        hi = P.shape[0] - 1 if tmax is None else min(tmax, P.shape[0] - 1)
        for t in range(0, hi, every):
            q, _ = fit.rollout(P[t].clone(), V[t].clone(), args.dt_mult, cache=cache)
            d = (P[t + 1] - P[t])[~fixed].norm(dim=-1).mean().clamp(min=1e-12)
            tot += ((q - P[t + 1])[~fixed].norm(dim=-1).mean() / d).item()
            n += 1
    return 100 * tot / max(n, 1)


@torch.no_grad()
def report(tag):
    a = step_error(FIT, every=10)
    b = step_error(CHK, every=10)
    print(f"  [{tag}] one-step error  fit {a:7.1f}%   held out {b:7.1f}%", flush=True)
    return b


groups = [{"params": [fit.pos], "lr": args.lr_pos},
          {"params": [fit.log_s], "lr": args.lr_scale},
          {"params": [fit.quat], "lr": args.lr_quat}]
opt = torch.optim.Adam([g for g in groups if g["params"][0].requires_grad])
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.iters)

STATE = args.state or (args.out + ".state" if args.out else None)
start_it = 1
print(f"\n[before]")
best = report("init")
if args.resume and STATE and os.path.exists(STATE):
    blob = torch.load(STATE, map_location=dev, weights_only=False)
    with torch.no_grad():
        fit.pos.copy_(blob["pos"]); fit.log_s.copy_(blob["log_s"])
        fit.quat.copy_(blob["quat"])
    opt.load_state_dict(blob["opt"]); sched.load_state_dict(blob["sched"])
    start_it, best = blob["iter"] + 1, blob["best"]
    print(f"[resume] {STATE} at iteration {blob['iter']}, best {best:.1f}%")
    report("resumed")
t0 = time.time()
bar = tqdm(range(start_it, args.iters + 1), desc="fit", ncols=90)
for it in bar:
    # the low-deformation end first: that is where the simulator and MPM still
    # agree on direction, so the gradient there is a correction rather than a
    # push in an arbitrary direction. The window opens as the fit takes hold
    frac = min(1.0, it / max(args.warmup, 1))
    hi = max(2, int(frac * (args.frames - 1)))
    cache = fit.prepare()
    loss = 0.0
    for _ in range(args.batch):
        P, V = FIT[torch.randint(len(FIT), (1,)).item()]
        t = torch.randint(hi, (1,)).item()
        q, _ = fit.rollout(P[t].clone(), V[t].clone(), args.dt_mult, cache=cache)
        d = (P[t + 1] - P[t])[~fixed].norm(dim=-1).mean().clamp(min=1e-12)
        loss = loss + ((q - P[t + 1])[~fixed].norm(dim=-1).mean() / d)
    loss = loss / args.batch
    opt.zero_grad(set_to_none=True)
    loss.backward()
    with torch.no_grad():
        if fit.pos.grad is not None:
            fit.pos.grad[fixed] = 0            # the pinned anchors are boundary
    torch.nn.utils.clip_grad_norm_([p for p in fit.parameters() if p.requires_grad], 1.0)
    opt.step()
    sched.step()
    bar.set_postfix(loss=f"{loss.item():.3f}", window=hi)
    if it % args.eval_every == 0 or it == args.iters:
        with torch.no_grad():
            b = report(f"it {it}")
        if args.out and b < best:
            best = b
            torch.save({"pos": fit.pos.detach().cpu(), "log_s": fit.log_s.detach().cpu(),
                         "quat": fit.quat.detach().cpu(), "iter": it,
                         "eig_floor": args.eig_floor, "K": args.K,
                         "n_anchors": args.n_anchors}, args.out)
        if STATE:
            torch.save({"pos": fit.pos.detach(), "log_s": fit.log_s.detach(),
                         "quat": fit.quat.detach(), "opt": opt.state_dict(),
                         "sched": sched.state_dict(), "iter": it, "best": best}, STATE)

print(f"\n[done] {time.time() - t0:.0f}s")
with torch.no_grad():
    print(f"[final] by deformation, held out:")
    print(f"  {'frames':>10} {'one-step error':>16}")
    for lo, hi in ((1, 10), (10, 25), (25, 45), (45, args.frames)):
        tot, n = 0.0, 0
        cache = fit.prepare()
        for P, V in CHK:
            for t in range(lo, min(hi, P.shape[0] - 1), 3):
                q, _ = fit.rollout(P[t].clone(), V[t].clone(), args.dt_mult, cache=cache)
                d = (P[t + 1] - P[t])[~fixed].norm(dim=-1).mean().clamp(min=1e-12)
                tot += ((q - P[t + 1])[~fixed].norm(dim=-1).mean() / d).item(); n += 1
        print(f"  {f'{lo}-{hi}':>10} {100 * tot / max(n,1):15.1f}%")
    s_ = fit.log_s.exp()
    print(f"[params] moved {(fit.pos - AC).norm(dim=-1).median():.4f} (median), "
          f"axis ratio {(s_.max(-1).values / s_.min(-1).values).median():.2f}, "
          f"size {s_.mean():.4f} vs radius {sc.sim.radius:.4f}")
if args.out:
    print(f"[saved ] {args.out}")

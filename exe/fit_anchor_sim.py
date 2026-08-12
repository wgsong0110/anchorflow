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

Fitted on MPM's states alone this halves the one-step error and makes the
ROLLOUT worse -- 12.5% to 14.1% against MPM's particles, with the motion falling
to 70% of MPM's. The reason is the one this project already met at the level of
the learned student: the loss is measured where MPM goes and the simulator has
to run where it goes itself. At MPM's states the anchor simulator always
overshoots, so a one-step loss there teaches it to move less, and sixty steps of
that is a simulator that damps.

So the states are also drawn from the simulator's own rollout, with MPM asked
what happens from there -- which needs an expert that can be started from an
arbitrary anchor state, and that is what anchorflow.mpm_teacher's lift is for.
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
ap.add_argument("--save_every", type=int, default=10,
                 help="iterations between state saves. These hosts reclaim their GPUs "
                      "without warning, so this is the granularity of what a loss costs.")
ap.add_argument("--traj_cache", default=None,
                 help="the MPM reference trajectories. Regenerating them on every "
                      "restart is minutes of GPU each time and they are identical, "
                      "being drawn from fixed seeds.")
ap.add_argument("--dagger_every", type=int, default=40,
                 help="iterations between rounds of collecting the simulator's own "
                      "states and labelling them with MPM. Zero fits on MPM's states "
                      "alone, which damps the rollout.")
ap.add_argument("--dagger_traj", type=int, default=2)
ap.add_argument("--dagger_frames", type=int, default=25)
ap.add_argument("--dagger_stride", type=int, default=3)
ap.add_argument("--dagger_frac", type=float, default=0.5,
                 help="share of each batch drawn from the simulator's own states")
ap.add_argument("--dagger_cap", type=float, default=3.0,
                 help="rollouts are abandoned past this multiple of the reference's "
                      "own motion; past there the state is not one MPM can answer for")
ap.add_argument("--r2", default=None,
                 help="rclone destination for the state, so a destroyed instance "
                      "costs the run and not the fit")
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


TRAJ_KEY = f"{args.n_fit}_{args.n_check}_{args.frames}_{args.dt_mult}_{args.n_anchors}_{args.K}"
FIT = CHK = None
if args.traj_cache and os.path.exists(args.traj_cache):
    blob = torch.load(args.traj_cache, map_location=dev, weights_only=False)
    if blob.get("key") == TRAJ_KEY:
        FIT, CHK = blob["fit"], blob["chk"]
        print(f"[data] {args.traj_cache}")
if FIT is None:
    gen = torch.Generator(device=dev); gen.manual_seed(4242)
    FIT = [s for s in (states(f) for f in tqdm(draw(gen, args.n_fit),
                                                desc="MPM fit", ncols=90)) if s is not None]
    gen2 = torch.Generator(device=dev); gen2.manual_seed(31337)
    CHK = [s for s in (states(f) for f in tqdm(draw(gen2, args.n_check + 1)[1:],
                                                desc="MPM check", ncols=90)) if s is not None]
    if args.traj_cache:
        torch.save({"fit": FIT, "chk": CHK, "key": TRAJ_KEY}, args.traj_cache)
        print(f"[data] cached to {args.traj_cache}")
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


POOL = {"p": [], "v": [], "tgt": []}


@torch.no_grad()
def collect_dagger():
    """roll the simulator out and ask MPM what happens from where it got to"""
    added = skipped = 0
    cache = fit.prepare()
    cap = args.dagger_cap * max((P - AC)[:, ~fixed].norm(dim=-1).max().item()
                                 for P, _ in FIT)
    for i in range(args.dagger_traj):
        P, V = FIT[i % len(FIT)]
        p, v = P[0].clone(), V[0].clone()
        for t in range(args.dagger_frames):
            if (p - AC)[~fixed].norm(dim=-1).max() > cap or not torch.isfinite(p).all():
                break
            if t % args.dagger_stride == 0:
                got = T.query(p, v, 1, args.dt_mult)
                if got is None:
                    skipped += 1
                else:
                    POOL["p"].append(p.clone()); POOL["v"].append(v.clone())
                    POOL["tgt"].append(got[0].clone()); added += 1
            p, v = fit.rollout(p, v, args.dt_mult, cache=cache)
    return added, skipped


groups = [{"params": [fit.pos], "lr": args.lr_pos},
          {"params": [fit.log_s], "lr": args.lr_scale},
          {"params": [fit.quat], "lr": args.lr_quat}]
opt = torch.optim.Adam([g for g in groups if g["params"][0].requires_grad])
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.iters)

STATE = args.state or (args.out + ".state" if args.out else None)


def save_state(it, best):
    """written to a temporary file and renamed, because a process killed
    mid-write leaves a half-file that resume cannot read -- which is the same
    as having no resume at all"""
    if not STATE:
        return
    tmp = STATE + ".tmp"
    torch.save({"pos": fit.pos.detach(), "log_s": fit.log_s.detach(),
                 "quat": fit.quat.detach(), "opt": opt.state_dict(),
                 "sched": sched.state_dict(), "iter": it, "best": best,
                 "rng": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state(),
                 "pool": POOL, "args": vars(args)}, tmp)
    os.replace(tmp, STATE)
    if args.r2:
        os.system(f"rclone copy {STATE} {args.r2} 2>/dev/null &")


start_it, best = 1, None
if args.resume and STATE:
    if not os.path.exists(STATE) and args.r2:
        os.system(f"rclone copy {args.r2}/{os.path.basename(STATE)} "
                   f"{os.path.dirname(STATE) or '.'} 2>/dev/null")
    if os.path.exists(STATE):
        blob = torch.load(STATE, map_location=dev, weights_only=False)
        with torch.no_grad():
            fit.pos.copy_(blob["pos"]); fit.log_s.copy_(blob["log_s"])
            fit.quat.copy_(blob["quat"])
        opt.load_state_dict(blob["opt"]); sched.load_state_dict(blob["sched"])
        torch.set_rng_state(blob["rng"].cpu())
        torch.cuda.set_rng_state(blob["rng_cuda"].cpu())
        start_it, best = blob["iter"] + 1, blob["best"]
        for k in POOL:
            POOL[k] = [q.to(dev) for q in blob.get("pool", {}).get(k, [])]
        print(f"\n[resume] {STATE} at iteration {blob['iter']}, best {best:.1f}%, "
              f"{len(POOL['p'])} collected states")
    else:
        print(f"\n[resume] nothing at {STATE}; starting from the beginning")
if best is None:
    # only when there is nothing to resume from: evaluating the initial
    # parameters costs a minute and says nothing new on a restart
    print(f"\n[before]")
    best = report("init")
t0 = time.time()
bar = tqdm(range(start_it, args.iters + 1), desc="fit", ncols=90)
for it in bar:
    # the low-deformation end first: that is where the simulator and MPM still
    # agree on direction, so the gradient there is a correction rather than a
    # push in an arbitrary direction. The window opens as the fit takes hold
    frac = min(1.0, it / max(args.warmup, 1))
    hi = max(2, int(frac * (args.frames - 1)))
    if args.dagger_every and (it == start_it or it % args.dagger_every == 0):
        a_, s_ = collect_dagger()
        print(f"\n  [dagger] it={it}: +{a_} states from the simulator's own rollout "
              f"({s_} MPM could not answer for), pool {len(POOL['p'])}", flush=True)
    cache = fit.prepare()
    loss = 0.0
    for b_ in range(args.batch):
        own = POOL["p"] and (torch.rand(1).item() < args.dagger_frac)
        if own:
            j = torch.randint(len(POOL["p"]), (1,)).item()
            p0, v0, tgt = POOL["p"][j], POOL["v"][j], POOL["tgt"][j]
        else:
            P, V = FIT[torch.randint(len(FIT), (1,)).item()]
            t = torch.randint(hi, (1,)).item()
            p0, v0, tgt = P[t], V[t], P[t + 1]
        q, _ = fit.rollout(p0.clone(), v0.clone(), args.dt_mult, cache=cache)
        d = (tgt - p0)[~fixed].norm(dim=-1).mean().clamp(min=1e-12)
        loss = loss + ((q - tgt)[~fixed].norm(dim=-1).mean() / d)
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
        save_state(it, best)
    elif it % args.save_every == 0:
        save_state(it, best)

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

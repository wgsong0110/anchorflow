"""Fit the anchor set to MPM, letting its support and its size both change.

The fixed-neighbour fit halved the one-step error and left the rollout where it
was -- 12.50% to 12.30% against MPM's particles on uniform impulses, worse on
the other two families. Two things it could not do, both structural rather than
a matter of more iterations: an anchor could only redistribute weight among the
eight Gaussians assigned to it at the start, and there were always exactly 512
anchors wherever the error happened to be.

Here the support is the region G(x) > c with weight G(x) - c, so membership
follows the parameters continuously, and anchors are split where the fit pushes
hardest and dropped where they hold nothing.

The loss moves to Gaussian space. It has to: with the anchor count changing
there is no fixed anchor-space target to compare against, and a rollout is
scored on particles anyway. So the state is projected onto whatever anchors
currently exist, stepped one coarse frame, skinned back out, and compared with
MPM's own particles.
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
from anchorflow.anchor_sparse import AnchorSparse
from anchorflow.streams import draw_impulse, rand_rot

ap = argparse.ArgumentParser()
ap.add_argument("--ply", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--dreamphysics", default="/workspace/DreamPhysics")
ap.add_argument("--n_fit", type=int, default=115,
                 help="MPM trajectories to fit on. The student that imitates this "
                      "simulator was trained on 125 and its divergence was traced to "
                      "the narrowness of the force distribution, not to model size; "
                      "the fit was being run on three.")
ap.add_argument("--n_check", type=int, default=10)
ap.add_argument("--field", action="store_true", default=True,
                 help="draw smooth random force fields rather than one uniform push, "
                      "as the student's data does")
ap.add_argument("--uniform_check", type=int, default=5,
                 help="held-out trajectories kept on the config's uniform impulse, so "
                      "the headline number stays comparable across runs")
ap.add_argument("--impulse_range", type=float, default=16.0)
ap.add_argument("--frames", type=int, default=60)
ap.add_argument("--dt_mult", type=int, default=40)
ap.add_argument("--n_anchors", type=int, default=512)
ap.add_argument("--K", type=int, default=8)
ap.add_argument("--c", type=float, default=0.25,
                 help="the kernel value that bounds an anchor's support. Smaller "
                      "reaches further and costs more pairs.")
ap.add_argument("--eig_floor", type=float, default=0.02)
ap.add_argument("--polar_iters", type=int, default=6)
ap.add_argument("--iters", type=int, default=400)
ap.add_argument("--batch", type=int, default=2)
ap.add_argument("--lr_pos", type=float, default=3e-4)
ap.add_argument("--lr_scale", type=float, default=1e-2)
ap.add_argument("--lr_quat", type=float, default=1e-2)
ap.add_argument("--warmup", type=int, default=80)
ap.add_argument("--refresh_every", type=int, default=20,
                 help="iterations between rebuilding the candidate pairs. The list "
                      "is padded to stay a superset in between; missing a pair is a "
                      "wrong loss, not a rough one.")
ap.add_argument("--densify_every", type=int, default=60)
ap.add_argument("--densify_until", type=float, default=0.7,
                 help="fraction of the run after which the anchor set is left alone")
ap.add_argument("--split_frac", type=float, default=0.05)
ap.add_argument("--prune_share", type=float, default=0.05)
ap.add_argument("--max_anchors", type=int, default=1024)
ap.add_argument("--dagger_every", type=int, default=40)
ap.add_argument("--dagger_traj", type=int, default=2)
ap.add_argument("--dagger_frames", type=int, default=25)
ap.add_argument("--dagger_stride", type=int, default=3)
ap.add_argument("--dagger_frac", type=float, default=0.5)
ap.add_argument("--dagger_cap", type=float, default=3.0)
ap.add_argument("--no_geom_init", action="store_true")
ap.add_argument("--params", default="shape",
                 choices=["shape", "stiff", "all"],
                 help="what the fit is allowed to move. 'shape' is where the anchors "
                      "are, which way they point and how far they reach; 'stiff' is a "
                      "per-anchor stiffness multiplier and nothing else, which is the "
                      "parameterisation an earlier attempt failed to fit at all "
                      "because the loss it used was meaningless; 'all' is both.")
ap.add_argument("--lr_stiff", type=float, default=3e-2)
ap.add_argument("--out", default=None)
ap.add_argument("--state", default=None)
ap.add_argument("--resume", action="store_true")
ap.add_argument("--save_every", type=int, default=10)
ap.add_argument("--traj_cache", default=None)
ap.add_argument("--r2", default=None)
ap.add_argument("--eval_every", type=int, default=40)
ap.add_argument("--cfl_frac", type=float, default=0.05,
                 help="how far an anchor may travel in one substep, as a fraction of "
                      "the anchor spacing. The starting discretisation sits at 1.4%%; "
                      "the configurations the fit blew up on reach 50%%.")
ap.add_argument("--lambda_cfl", type=float, default=1.0)
ap.add_argument("--unroll", type=int, default=1,
                 help="coarse frames per training sample. One frame is what the fit "
                      "has always optimised and it does not transfer: the one-step "
                      "error halves while the sixty-frame rollout does not move. "
                      "Unrolling makes the loss measure what is actually wanted, at "
                      "the cost of that many times the compute per sample.")
ap.add_argument("--eval_rollout", type=int, default=0,
                 help="score the held-out set on a full rollout at each evaluation "
                      "rather than on one step, so the number being tracked is the "
                      "number being asked for")
ap.add_argument("--no_guards", action="store_true",
                 help="run without the scale bounds, the polar ridge, the scatter "
                      "floor or the skip, and stop at the first non-finite quantity "
                      "with a substep-by-substep replay. For finding out what the "
                      "failure is rather than surviving it.")
ap.add_argument("--final_rollout", type=int, default=3,
                 help="impulses to roll out fully against MPM at the end")
args = ap.parse_args()

sys.path.insert(0, args.dreamphysics)
import warp as wp

from anchorflow.mpm_teacher import MPMTeacher

dev = "cuda"
torch.manual_seed(0)
wp.init()

sc = scene_setup.build(args.ply, args.config, args.n_anchors, args.K, device=dev,
                        frozen_weights=True, rot_fallback=True, eig_floor=args.eig_floor)
T = MPMTeacher(sc)
base = None
for bc in sc.cfg.get("boundary_conditions", []):
    if bc["type"] == "particle_impulse":
        base = torch.tensor(bc["force"], device=dev)

fit = AnchorSparse(sc, c=args.c, eig_floor=args.eig_floor,
                    polar_iters=args.polar_iters, cfl_frac=args.cfl_frac).to(dev)
if args.no_guards:
    fit.s_lo, fit.s_hi, fit.polar_ridge = 1e-9, 1e9, 0.0
print(f"[setup] fitting {args.params}: {', '.join(TRAIN)}")
print(f"[setup] {fit.M} anchors, {fit.N} material Gaussians, support at "
      f"{fit.mahal_radius:.2f} sigma, {fit.pair_g.shape[0]} pairs "
      f"({fit.pair_g.shape[0] / fit.N:.1f} anchors per Gaussian)")
if args.no_guards:
    fit.B_ref.zero_()
    fit.set_B_ref = lambda *a, **k: 0.0
if not args.no_geom_init:
    thin = fit.init_from_geometry()
    s_ = fit.log_s.exp()
    print(f"[init] oriented; axis ratio median "
          f"{(s_.max(-1).values / s_.min(-1).values).median():.2f}, {thin} left round, "
          f"{fit.pair_g.shape[0]} pairs")


class Traj:
    """a trajectory held on the CPU in half precision, handed out a frame at a
    time in the precision the physics wants.

    At the student's data volume the set is 15.7 GB: it does not belong on a
    24 GB card, and a single frame is 2 MB, so the transfer is nothing next to
    the forty substeps it feeds.
    """

    def __init__(self, t):
        self.t = t

    def __getitem__(self, i):
        return self.t[i].to(dev, torch.float32, non_blocking=True)

    def __len__(self):
        return self.t.shape[0]

    @property
    def shape(self):
        return self.t.shape


@torch.no_grad()
def mpm_states(force):
    """MPM's own particles and velocities, frame by frame"""
    cache = fit.prepare()
    dv = fit.impulse_dv(force, cache)
    v0 = torch.zeros(fit.N, 3, device=dev).index_add_(
        0, fit.pair_g, cache[0].unsqueeze(-1) * dv[fit.pair_a])
    T._set(T.pos_m.clone(), v0.contiguous(), T.eye.clone(), torch.zeros_like(T.eye))
    xs = [T.pos_m.to(torch.float16).cpu()]
    vs = [v0.to(torch.float16).cpu()]
    for _ in range(args.frames):
        for k in range(args.dt_mult):
            T.solver.p2g2p(None, sc.sub_dt, device=T.wp_dev)
            if (k + 1) % 8 == 0 and not T._in_domain():
                return None
        xs.append(T.solver.export_particle_x_to_torch().to(torch.float16).cpu())
        vs.append(T.solver.export_particle_v_to_torch().to(torch.float16).cpu())
    return Traj(torch.stack(xs)), Traj(torch.stack(vs))


def draw(g, n, field=True, n_uniform=0):
    """the student's impulse distribution: one uniform push, then force fields.

    A field varies over the object rather than pushing all of it the same way,
    and its spatial frequency is drawn per trajectory. Fitting on uniform pushes
    alone is what the student's own failure was traced to.
    """
    out = [base]
    while len(out) < n:
        if not field or len(out) < n_uniform:
            k = 0.5 * (args.impulse_range ** torch.rand(1, device=dev, generator=g).item())
            out.append((rand_rot(g, dev) @ base) * k)
        else:
            out.append(draw_impulse(sc, base, g, args.impulse_range, field=True)[0])
    return out[:n]


KEY = f"{args.n_fit}_{args.n_check}_{args.frames}_{args.dt_mult}_{args.c}"
FIT = CHK = FORCES = None
if args.traj_cache and os.path.exists(args.traj_cache):
    blob = torch.load(args.traj_cache, map_location=dev, weights_only=False)
    if blob.get("key") == KEY:
        FIT, CHK, FORCES = blob["fit"], blob["chk"], blob["forces"]
        print(f"[data] {args.traj_cache}")
if FIT is None:
    g1 = torch.Generator(device=dev); g1.manual_seed(4242)
    ff = draw(g1, args.n_fit, field=args.field)
    FIT = [s for s in (mpm_states(f) for f in tqdm(ff, desc="MPM fit", ncols=90))
           if s is not None]
    g2 = torch.Generator(device=dev); g2.manual_seed(31337)
    fc = draw(g2, args.n_check + 1, field=args.field,
               n_uniform=args.uniform_check + 1)[1:]
    CHK = [s for s in (mpm_states(f) for f in tqdm(fc, desc="MPM check", ncols=90))
           if s is not None]
    FORCES = {"fit": ff[:len(FIT)], "chk": fc[:len(CHK)]}
    if args.traj_cache:
        torch.save({"fit": FIT, "chk": CHK, "forces": FORCES, "key": KEY}, args.traj_cache)
print(f"[data] {len(FIT)} fit and {len(CHK)} held-out MPM trajectories")


def one_step(x0, v0, cache):
    """MPM particle state -> one coarse frame of this simulator -> particles,
    and how far the substeps were from being able to carry it"""
    p = fit.project(x0, cache)
    v = fit.project_v(v0, cache)
    p, _, cfl = fit.rollout(p, v, args.dt_mult, cache)
    return fit.gaussian_pos(p, cache), cfl


def unrolled(X, V, t, n, cache):
    """n coarse frames from one MPM state, scored against MPM at every frame.

    The simulator runs on its own output after the first frame, which is the
    regime it will be used in and the one a single step says nothing about.
    Each frame is divided by how far MPM moved from the start, so a later frame
    is not weighted down for having drifted further.
    """
    p = fit.project(X[t], cache)
    v = fit.project_v(V[t], cache)
    loss = pen = 0.0
    hi = min(n, X.shape[0] - 1 - t)
    for j in range(hi):
        p, v, cfl = fit.rollout(p, v, args.dt_mult, cache)
        got = fit.gaussian_pos(p, cache)
        d = (X[t + j + 1] - X[t]).norm(dim=-1).mean().clamp(min=1e-12)
        loss = loss + (got - X[t + j + 1]).norm(dim=-1).mean() / d
        pen = pen + cfl
    return loss / max(hi, 1), pen / max(hi, 1)


@torch.no_grad()
def step_error(sets, every=10):
    cache = fit.prepare()
    tot, n = 0.0, 0
    for X, V in sets:
        for t in range(0, X.shape[0] - 1, every):
            got, _ = one_step(X[t], V[t], cache)
            d = (X[t + 1] - X[t]).norm(dim=-1).mean().clamp(min=1e-12)
            tot += ((got - X[t + 1]).norm(dim=-1).mean() / d).item(); n += 1
    return 100 * tot / max(n, 1)


@torch.no_grad()
def rollout_error(sets):
    """the whole trajectory, from rest, as the evaluation actually asks for it"""
    cache = fit.prepare()
    tot = []
    for X, V in sets:
        p, v = fit.project(X[0], cache), fit.project_v(V[0], cache)
        out = [fit.gaussian_pos(p, cache)]
        for _ in range(args.frames):
            p, v, _ = fit.rollout(p, v, args.dt_mult, cache)
            out.append(fit.gaussian_pos(p, cache))
        G = torch.stack(out)
        Xg = torch.stack([X[k] for k in range(args.frames + 1)])
        span = (Xg - Xg[0]).norm(dim=-1).max().clamp(min=1e-12)
        tot.append(((G - Xg).norm(dim=-1).mean(-1) / span).mean().item())
    return 100 * sum(tot) / max(len(tot), 1)


@torch.no_grad()
def report(tag):
    a, b = step_error(FIT), step_error(CHK)
    extra = ""
    if args.eval_rollout:
        extra = f"   rollout {rollout_error(CHK):6.2f}%"
    print(f"  [{tag}] one-step  fit {a:7.1f}%   held out {b:7.1f}%{extra}   "
          f"{fit.M} anchors, {fit.pair_g.shape[0]} pairs", flush=True)
    return b


POOL = {"x": [], "v": [], "tgt": []}


@torch.no_grad()
def collect_dagger():
    cache = fit.prepare()
    cap = args.dagger_cap * max((X[args.frames] - X[0]).norm(dim=-1).max().item()
                                 for X, _ in FIT[:8])
    added = skipped = 0
    for i in range(args.dagger_traj):
        X, V = FIT[i % len(FIT)]
        p, v = fit.project(X[0], cache), fit.project_v(V[0], cache)
        for t in range(args.dagger_frames):
            gp = fit.gaussian_pos(p, cache)
            if not torch.isfinite(p).all() or (gp - X[0]).norm(dim=-1).max() > cap:
                break
            if t % args.dagger_stride == 0:
                x, vx, F, C = fit.lift(p, v, cache)
                ok = torch.isfinite(x).all() and x.min() > T.margin and \
                    x.max() < T.grid_lim - T.margin
                if ok:
                    T._set(x, vx, F, C)
                    out = T._advance(1, args.dt_mult)
                    if out is None:
                        ok = False
                if ok:
                    POOL["x"].append(x.clone()); POOL["v"].append(vx.clone())
                    POOL["tgt"].append(T.solver.export_particle_x_to_torch().clone())
                    added += 1
                else:
                    skipped += 1
            p, v, _ = fit.rollout(p, v, args.dt_mult, cache)
    return added, skipped


SHAPE = ("pos", "log_s", "quat")
STIFF = ("log_k",)
TRAIN = SHAPE if args.params == "shape" else (
    STIFF if args.params == "stiff" else SHAPE + STIFF)
LRS = {"pos": args.lr_pos, "log_s": args.lr_scale, "quat": args.lr_quat,
       "log_k": args.lr_stiff}


def make_opt():
    for n, prm in fit.named_parameters():
        prm.requires_grad_(n in TRAIN)
    return torch.optim.Adam([{"params": [getattr(fit, n)], "lr": LRS[n]}
                             for n in TRAIN])


opt = make_opt()
grad_accum = torch.zeros(fit.M, device=dev)
STATE = args.state or (args.out + ".state" if args.out else None)


def save_state(it, best):
    """A state that is not finite is not a state to come back to. The previous
    run wrote one and then failed to resume from it four times in a row."""
    if not STATE:
        return
    if not all(torch.isfinite(q).all() for q in
               (fit.pos, fit.log_s, fit.quat)):
        print(f"  [state] iteration {it} is not finite; keeping the last good one",
              flush=True)
        return
    torch.save({"pos": fit.pos.detach(), "log_s": fit.log_s.detach(),
                 "quat": fit.quat.detach(), "log_k": fit.log_k.detach(),
                 "iter": it, "best": best,
                 "opt": opt.state_dict(), "grad_accum": grad_accum,
                 "rng": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state(),
                 "pool": POOL, "c": args.c, "args": vars(args)}, STATE + ".tmp")
    os.replace(STATE + ".tmp", STATE)
    if args.r2:
        os.system(f"rclone copy {STATE} {args.r2} 2>/dev/null &")


start_it, best = 1, None
if args.resume and STATE:
    if not os.path.exists(STATE) and args.r2:
        os.system(f"rclone copy {args.r2}/{os.path.basename(STATE)} "
                   f"{os.path.dirname(STATE) or '.'} 2>/dev/null")
    if os.path.exists(STATE):
        blob = torch.load(STATE, map_location=dev, weights_only=False)
        fit._rebuild(blob["pos"], blob["quat"], blob["log_s"], blob.get("log_k"))
        opt = make_opt(); opt.load_state_dict(blob["opt"])
        grad_accum = blob["grad_accum"].to(dev)
        torch.set_rng_state(blob["rng"].cpu()); torch.cuda.set_rng_state(blob["rng_cuda"].cpu())
        for k in POOL:
            POOL[k] = [q.to(dev) for q in blob.get("pool", {}).get(k, [])]
        start_it, best = blob["iter"] + 1, blob["best"]
        print(f"\n[resume] {STATE} at iteration {blob['iter']}, {fit.M} anchors, "
              f"best {best:.1f}%, {len(POOL['x'])} collected states")
if best is None:
    print(f"\n[before]")
    best = report("init")

t0 = time.time()
n_skip = 0
bar = tqdm(range(start_it, args.iters + 1), desc="fit", ncols=90)
for it in bar:
    if it > start_it and args.refresh_every and it % args.refresh_every == 0:
        fit.refresh()
    if args.densify_every and it % args.densify_every == 0 and \
            it <= args.densify_until * args.iters:
        dead, split = fit.densify_and_prune(grad_accum, args.split_frac,
                                             args.prune_share,
                                             max_anchors=args.max_anchors)
        # the parameters are new tensors, so Adam's moments no longer refer to
        # anything; kept simple by restarting them rather than reindexing
        opt = make_opt()
        grad_accum = torch.zeros(fit.M, device=dev)
        print(f"\n  [density] it={it}: -{dead} +{split} -> {fit.M} anchors, "
              f"{fit.pair_g.shape[0]} pairs", flush=True)
    if args.dagger_every and (it == start_it or it % args.dagger_every == 0):
        a_, s_ = collect_dagger()
        print(f"\n  [dagger] it={it}: +{a_} states ({s_} MPM could not answer for), "
              f"pool {len(POOL['x'])}", flush=True)

    frac = min(1.0, it / max(args.warmup, 1))
    hi = max(2, int(frac * (args.frames - 1)))
    cache = fit.prepare()
    loss, pen, bad_sample = 0.0, 0.0, None
    for _ in range(args.batch):
        src = None
        if POOL["x"] and torch.rand(1).item() < args.dagger_frac:
            # a DAgger state has one labelled frame after it and nothing more, so
            # it stays a single step whatever the unroll length is
            j = torch.randint(len(POOL["x"]), (1,)).item()
            x0, v0, tgt = POOL["x"][j], POOL["v"][j], POOL["tgt"][j]
        else:
            X, V = FIT[torch.randint(len(FIT), (1,)).item()]
            t = torch.randint(hi, (1,)).item()
            x0, v0, tgt = X[t], V[t], X[t + 1]
            src = (X, V, t)
        if src is not None and args.unroll > 1:
            X_, V_, t_ = src
            contrib, cfl = unrolled(X_, V_, t_, args.unroll, cache)
        else:
            got, cfl = one_step(x0, v0, cache)
            d = (tgt - x0).norm(dim=-1).mean().clamp(min=1e-12)
            contrib = (got - tgt).norm(dim=-1).mean() / d
        if args.no_guards and not torch.isfinite(contrib):
            bad_sample = (x0, v0)
        loss = loss + contrib
        pen = pen + cfl
    loss, pen = loss / args.batch, pen / args.batch
    total = loss + args.lambda_cfl * pen
    skipped = 0
    if not torch.isfinite(total):
        # one bad sample -- a rollout that ran away, a configuration the polar
        # factor cannot handle -- should cost that iteration, not the run
        skipped = 1
    else:
        opt.zero_grad(set_to_none=True)
        total.backward()
        with torch.no_grad():
            if fit.pos.grad is not None:
                grad_accum += torch.nan_to_num(fit.pos.grad).norm(dim=-1)
                fit.pos.grad[fit.fixed] = 0
        if all(getattr(fit, n).grad is None or
               torch.isfinite(getattr(fit, n).grad).all() for n in TRAIN):
            torch.nn.utils.clip_grad_norm_([getattr(fit, n) for n in TRAIN], 1.0)
            opt.step()
            fit.clamp_()
        else:
            skipped = 1
    n_skip += skipped
    if skipped and args.no_guards:
        # what a survivable run hides: which quantity went first, and whether the
        # step that produced it grew every substep or spiked once
        print(f"\n[first failure] iteration {it}, loss {loss.item()}", flush=True)
        if bad_sample is not None:
            x0, v0 = bad_sample
        w_, rc_, q_, Binv_, blocked_, mass_ = cache
        with torch.no_grad():
            F_, _ = fit.deformation(fit.project(x0, cache), w_, rc_, q_, Binv_, blocked_)
            det = torch.linalg.det(F_)
            print(f"  mass: min {mass_.min():.3e}, at the clamp "
                  f"{int((mass_ <= 1.0000001e-12).sum())} of {fit.M}")
            print(f"  detF: min {det.min():.4f}, inverted {int((det <= 0).sum())} of "
                  f"{fit.N}, extents min {fit.log_s.exp().min():.3e}")
            m_ = mass_.unsqueeze(-1)
            keep_ = (~fit.fixed).unsqueeze(-1).to(torch.float32)
            p_, v_ = fit.project(x0, cache), fit.project_v(v0, cache)
            print(f"  {'substep':>8} {'|a| max':>12} {'|v| max':>12} {'detF min':>10}")
            for k_ in range(args.dt_mult):
                a_ = fit.force(p_, w_, rc_, q_, Binv_, blocked_) / m_
                v_ = (v_ + fit.dt * a_) * fit.damping * keep_
                p_ = p_ + fit.dt * v_
                F2, _ = fit.deformation(p_, w_, rc_, q_, Binv_, blocked_)
                if k_ < 4 or k_ % 4 == 0 or not torch.isfinite(p_).all():
                    print(f"  {k_:8d} {a_.norm(dim=-1).max():12.3e} "
                          f"{v_.norm(dim=-1).max():12.3e} "
                          f"{torch.linalg.det(F2).min():10.4f}")
                if not torch.isfinite(p_).all():
                    break
        break
    bar.set_postfix(loss=f"{loss.item():.3f}", cfl=f"{float(pen):.2e}", M=fit.M,
                     win=hi, skip=n_skip)
    if it % args.eval_every == 0 or it == args.iters:
        with torch.no_grad():
            b = report(f"it {it}")
        if args.out and b < best:
            best = b
            torch.save({"pos": fit.pos.detach().cpu(), "log_s": fit.log_s.detach().cpu(),
                         "quat": fit.quat.detach().cpu(),
                         "log_k": fit.log_k.detach().cpu(), "c": args.c, "iter": it,
                         "eig_floor": args.eig_floor}, args.out)
        save_state(it, best)
    elif it % args.save_every == 0:
        save_state(it, best)

print(f"\n[done] {time.time() - t0:.0f}s, {fit.M} anchors, {n_skip} iterations skipped "
      f"as non-finite")

# ---- the number the project actually asks for ------------------------------
with torch.no_grad():
    print(f"\n[rollout] {args.frames} frames against MPM's particles, "
          f"{args.final_rollout} held-out impulses")
    cache = fit.prepare()
    # the simulator this replaces, on the same impulses: without it the number
    # above is only comparable to other runs of this script
    print(f"  {'impulse':>8} {'error':>9} {'final':>9} {'motion':>8} {'8-NN sim':>10}")
    tot, tot0 = [], []
    for i, (X, V) in enumerate(CHK[:args.final_rollout]):
        f0 = FORCES["chk"][i]
        p0, v0, g0 = sc.anchor_canonical.clone(), sc.initial_velocity(f0), sc.pos.clone()
        base_out = [g0[fit.mat].clone()]
        for _ in range(args.frames):
            p0, v0, g0 = sc.explicit_step(p0, v0, g0, args.dt_mult)
            base_out.append(g0[fit.mat].clone())
        G0 = torch.stack(base_out)
        Xb = torch.stack([X[k] for k in range(args.frames + 1)])
        span0 = (Xb - Xb[0]).norm(dim=-1).max().clamp(min=1e-12)
        e0 = ((G0 - Xb).norm(dim=-1).mean(-1) / span0).mean().item()
        tot0.append(e0)
        p = fit.project(X[0], cache)
        v = fit.project_v(V[0], cache)
        out = [fit.gaussian_pos(p, cache)]
        for _ in range(args.frames):
            p, v, _ = fit.rollout(p, v, args.dt_mult, cache)
            out.append(fit.gaussian_pos(p, cache))
        G = torch.stack(out)
        Xg = torch.stack([X[k] for k in range(args.frames + 1)])
        span = (Xg - Xg[0]).norm(dim=-1).max().clamp(min=1e-12)
        e = (G - Xg).norm(dim=-1).mean(-1) / span
        tot.append(e.mean().item())
        print(f"  {i:8d} {100*e.mean():8.2f}% {100*e[-1]:8.2f}% "
              f"{100*(G - G[0]).norm(dim=-1).max()/span:7.0f}% {100*e0:9.2f}%")
    print(f"  {'mean':>8} {100*sum(tot)/max(len(tot),1):8.2f}% {'':>9} {'':>8} "
          f"{100*sum(tot0)/max(len(tot0),1):9.2f}%")
    s_, k_ = fit.log_s.exp(), fit.log_k.exp()
    print(f"\n[params] {fit.M} anchors, {fit.pair_g.shape[0]} pairs, axis ratio "
          f"{(s_.max(-1).values / s_.min(-1).values).median():.2f}, "
          f"size {s_.mean():.4f}, stiffness {k_.min():.3f}..{k_.max():.3f} "
          f"(median {k_.median():.3f})")
if args.out:
    print(f"[saved ] {args.out}")

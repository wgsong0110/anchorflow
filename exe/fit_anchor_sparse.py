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

import math
import argparse
import os
import subprocess
import sys
import time

_lib = os.path.join(os.path.dirname(__file__), "..", "lib")
sys.path.insert(0, _lib)

import torch
from tqdm import tqdm

from anchorflow import scene_setup
from anchorflow.anchor_fit import det3
from anchorflow.anchor_sparse import AnchorSparse, Traj
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
ap.add_argument("--accum", type=int, default=1,
                 help="how many batches to average before stepping. The graph of "
                      "an unrolled sample is 30 GB on the torch path, so --batch "
                      "cannot be raised past 2 on a 48 GB card; accumulating N "
                      "batches costs the same memory and cuts the gradient noise "
                      "by sqrt(N), which is the actual point of a bigger batch.")
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
ap.add_argument("--dagger_pool_max", type=int, default=512,
                 help="states to keep, oldest evicted. Each is a full particle state, "
                      "so an uncapped pool over a long run is tens of gigabytes. Held "
                      "on the CPU in half precision and left out of the state file: it "
                      "refills in a few collections, and writing a gigabyte every save "
                      "would cost more than regenerating it.")
ap.add_argument("--no_geom_init", action="store_true")
ap.add_argument("--params", default="shape",
                 choices=["shape", "stiff", "all"],
                 help="what the fit is allowed to move. 'shape' is where the anchors "
                      "are, which way they point and how far they reach; 'stiff' is a "
                      "per-anchor stiffness multiplier and nothing else, which is the "
                      "parameterisation an earlier attempt failed to fit at all "
                      "because the loss it used was meaningless; 'all' is both.")
ap.add_argument("--lr_stiff", type=float, default=3e-2)
ap.add_argument("--reg", type=float, default=0.0,
                 help="pull the parameters back toward where they started. The knob "
                      "on how much the discretisation is allowed to become: at zero "
                      "the fit reaches its best rollout a third of the way through "
                      "and then doubles it while the loss keeps falling.")
ap.add_argument("--init_from", default=None,
                 help="start from a saved fit rather than from the sampled anchors. "
                      "With --iters 0 this scores an existing one on the same "
                      "held-out set as everything else, which is the only way three "
                      "parameterisations fitted at different times become comparable.")
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
ap.add_argument("--encoder", default="avg", choices=("avg", "ls"),
                 help="how an MPM state becomes an anchor state. avg is the "
                      "weighted average of the displacements around each anchor; "
                      "ls solves for the anchor state whose DECODING is closest "
                      "to the particles, which is the least-squares inverse of "
                      "the same skinning the simulator reads back through. "
                      "Measured (exe/probe_encoder.py): the reconstruction error "
                      "falls from 7.6e-3 to 1.2e-3, and every training sample "
                      "that is not a DAgger state starts from one of these.")
ap.add_argument("--grad_frames", type=int, default=0,
                 help="truncate the gradient to the last N coarse frames of the "
                      "unroll. The LOSS is unchanged -- all --unroll frames are "
                      "still scored -- but the state is detached before each "
                      "earlier frame, so no gradient travels more than N frames "
                      "back. --unroll changes the horizon and the chain length "
                      "together; this changes only the chain, which is what "
                      "separates a gradient pathology from a loss whose optimum "
                      "really does look like this. 0 leaves the full chain.")
ap.add_argument("--eval_rollout", type=int, default=0,
                 help="score the held-out set on a full rollout at each evaluation "
                      "rather than on one step, so the number being tracked is the "
                      "number being asked for")
ap.add_argument("--no_guards", action="store_true",
                 help="run without the scale bounds, the polar ridge, the scatter "
                      "floor or the skip, and stop at the first non-finite quantity "
                      "with a substep-by-substep replay. For finding out what the "
                      "failure is rather than surviving it.")
ap.add_argument("--n_fine", type=int, default=40,
                 help="trajectories for which MPM's own F and C are stored as well, so "
                      "a fine-step sample can restart MPM from its own state instead of "
                      "from a lifted one. Eighteen more floats per particle per frame is "
                      "377 MB a trajectory, so this is not all of them.")
ap.add_argument("--fine_steps", type=int, default=0,
                 help="score every SUBSTEP over this many, instead of every coarse "
                      "frame over --unroll of them. 480 differentiable substeps to "
                      "produce twelve numbers of supervision is what made an iteration "
                      "cost two minutes; MPM steps at the same 1e-4, so the two can be "
                      "walked together and compared throughout. The horizon then has to "
                      "come from DAgger rather than from rolling forward, which is what "
                      "it is for.")
ap.add_argument("--fine_end", type=int, default=0,
                 help="anneal the fine horizon from --fine_steps down to this over the "
                      "run. A long horizon shows error compounding, which is what a "
                      "short one cannot see and what the fit exists to remove; a short "
                      "one is cheap. Starting long and ending short spends the expensive "
                      "steps where the discretisation is still far off.")
ap.add_argument("--lr_cosine", type=int, default=0,
                 help="anneal every learning rate to zero over --iters on a "
                      "cosine. Without it the step size never shrinks, and once "
                      "the gradient is noise the parameters random-walk at full "
                      "stride: at the defaults that is 14%% of the anchor "
                      "spacing, 28%% of the anchor size and a factor 2.1 in "
                      "stiffness over 600 iterations.")
ap.add_argument("--ema", type=float, default=0.0,
                 help="evaluate an exponential moving average of the parameters "
                      "rather than the parameters themselves. Averages the walk "
                      "out without touching the optimisation; 0 disables.")
ap.add_argument("--lr_decay_at", type=int, default=0,
                 help="drop every learning rate by --lr_decay_by at this iteration. "
                      "Five fits have reached their best rollout between 125 and 250 "
                      "and never improved after, whatever the regularisation or the "
                      "length; if a smaller step keeps moving, that plateau is the "
                      "optimiser's and not the discretisation's.")
ap.add_argument("--lr_decay_by", type=float, default=0.1)
ap.add_argument("--grad_log", default=None,
                 help="write the gradient norm per parameter group, and the cosine "
                      "between consecutive iterations, to this file. A cosine near "
                      "zero means the gradient coming back through 480 stiff substeps "
                      "is noise, which is a different problem from a bad objective.")
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
SHAPE = ("pos", "log_s", "quat")
STIFF = ("log_k",)
TRAIN = SHAPE if args.params == "shape" else (
    STIFF if args.params == "stiff" else SHAPE + STIFF)
LRS = {"pos": args.lr_pos, "log_s": args.lr_scale, "quat": args.lr_quat,
       "log_k": args.lr_stiff}


print(f"[setup] fitting {args.params}: {', '.join(TRAIN)}")
print(f"[setup] {fit.M} anchors, {fit.N} material Gaussians, support at "
      f"{fit.mahal_radius:.2f} sigma, {fit.pair_g.shape[0]} pairs "
      f"({fit.pair_g.shape[0] / fit.N:.1f} anchors per Gaussian)")
if args.no_guards:
    fit.B_ref.zero_()
    fit.set_B_ref = lambda *a, **k: 0.0
if args.init_from:
    _b = torch.load(args.init_from, map_location=dev, weights_only=False)
    fit._rebuild(_b["pos"].to(dev), _b["quat"].to(dev), _b["log_s"].to(dev),
                  _b["log_k"].to(dev) if "log_k" in _b else None)
    print(f"[init] {args.init_from} at iteration {_b.get('iter')}, {fit.M} anchors")
elif not args.no_geom_init:
    thin = fit.init_from_geometry()
    s_ = fit.log_s.exp()
    print(f"[init] oriented; axis ratio median "
          f"{(s_.max(-1).values / s_.min(-1).values).median():.2f}, {thin} left round, "
          f"{fit.pair_g.shape[0]} pairs")


_LSF = {"w": None, "fac": None}


def ls_fac(cache):
    """the Cholesky of C^T C for this rest state, reused across one iteration"""
    key = cache[0].data_ptr(), int(fit.M), int(fit.pair_g.shape[0])
    if _LSF["w"] != key:
        _LSF["w"], _LSF["fac"] = key, fit.ls_factor(cache)
    return _LSF["fac"]


def enc(x, cache):
    if args.encoder == "avg":
        return fit.project(x, cache)
    return fit.project_ls(x, cache, ls_fac(cache))


def enc_v(v, cache):
    if args.encoder == "avg":
        return fit.project_v(v, cache)
    return fit.project_v_ls(v, cache, ls_fac(cache))


@torch.no_grad()
def mpm_states(force, keep_fc=False):
    """MPM's own particles and velocities, frame by frame.

    With keep_fc, its deformation gradient and affine velocity too. Those are
    what MPM needs to be restarted, and storing them is the difference between
    walking alongside MPM from its own state and having to reconstruct one by
    lifting ours -- which is lossy, and which MPM refuses outright often enough
    that three samples in four were being thrown away.
    """
    cache = fit.prepare()
    dv = fit.impulse_dv(force, cache)
    v0 = torch.zeros(fit.N, 3, device=dev).index_add_(
        0, fit.pair_g, cache[0].unsqueeze(-1) * dv[fit.pair_a])
    T._set(T.pos_m.clone(), v0.contiguous(), T.eye.clone(), torch.zeros_like(T.eye))
    xs = [T.pos_m.to(torch.float16).cpu()]
    vs = [v0.to(torch.float16).cpu()]
    fs_ = [T.eye.to(torch.float16).cpu()] if keep_fc else None
    cs_ = [torch.zeros_like(T.eye).to(torch.float16).cpu()] if keep_fc else None
    for _ in range(args.frames):
        for k in range(args.dt_mult):
            T.solver.p2g2p(None, sc.sub_dt, device=T.wp_dev)
            if (k + 1) % 8 == 0 and not T._in_domain():
                return None
        xs.append(T.solver.export_particle_x_to_torch().to(torch.float16).cpu())
        vs.append(T.solver.export_particle_v_to_torch().to(torch.float16).cpu())
        if keep_fc:
            fs_.append(T.solver.export_particle_F_to_torch().reshape(-1, 9)
                       .to(torch.float16).cpu())
            cs_.append(T.solver.export_particle_C_to_torch().reshape(-1, 9)
                       .to(torch.float16).cpu())
    out = [Traj(torch.stack(xs)), Traj(torch.stack(vs))]
    if keep_fc:
        out += [Traj(torch.stack(fs_)), Traj(torch.stack(cs_))]
    return tuple(out)


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


KEY = f"{args.n_fit}_{args.n_check}_{args.frames}_{args.dt_mult}_{args.c}_fc{args.n_fine}"
FIT = CHK = FORCES = None
if args.traj_cache and args.r2 and not os.path.exists(args.traj_cache):
    # the trajectories are twenty minutes of MPM and they were not being
    # mirrored, so an instance going away cost them every time even though the
    # fit state itself survived. Pulled back before anything else is decided.
    print(f"[data] fetching {os.path.basename(args.traj_cache)} from {args.r2}",
          flush=True)
    os.system(f"rclone copy {args.r2}/{os.path.basename(args.traj_cache)} "
               f"{os.path.dirname(args.traj_cache) or '.'} 2>/dev/null")
if args.traj_cache and os.path.exists(args.traj_cache):
    blob = torch.load(args.traj_cache, map_location=dev, weights_only=False)
    if blob.get("key") == KEY:
        FIT, CHK, FORCES = blob["fit"], blob["chk"], blob["forces"]
        print(f"[data] {args.traj_cache}")
if FIT is None:
    g1 = torch.Generator(device=dev); g1.manual_seed(4242)
    ff = draw(g1, args.n_fit, field=args.field)
    FIT = [s for s in (mpm_states(f, keep_fc=(i < args.n_fine))
                       for i, f in enumerate(tqdm(ff, desc="MPM fit", ncols=90)))
           if s is not None]
    g2 = torch.Generator(device=dev); g2.manual_seed(31337)
    fc = draw(g2, args.n_check + 1, field=args.field,
               n_uniform=args.uniform_check + 1)[1:]
    CHK = [s for s in (mpm_states(f) for f in tqdm(fc, desc="MPM check", ncols=90))
           if s is not None]
    FORCES = {"fit": ff[:len(FIT)], "chk": fc[:len(CHK)]}
    if args.traj_cache:
        torch.save({"fit": FIT, "chk": CHK, "forces": FORCES, "key": KEY}, args.traj_cache)
        if args.r2:
            print(f"[data] mirroring the trajectories to {args.r2}", flush=True)
            os.system(f"rclone copy {args.traj_cache} {args.r2} 2>/dev/null &")
print(f"[data] {len(FIT)} fit and {len(CHK)} held-out MPM trajectories")


def one_step(x0, v0, cache):
    """MPM particle state -> one coarse frame of this simulator -> particles,
    and how far the substeps were from being able to carry it"""
    p = enc(x0, cache)
    v = enc_v(v0, cache)
    p, _, cfl = fit.rollout(p, v, args.dt_mult, cache)
    return fit.gaussian_pos(p, cache), cfl


def unrolled(X, V, t, n, cache):
    """n coarse frames from one MPM state, scored against MPM at every frame.

    The simulator runs on its own output after the first frame, which is the
    regime it will be used in and the one a single step says nothing about.
    Each frame is divided by how far MPM moved from the start, so a later frame
    is not weighted down for having drifted further.
    """
    p = enc(X[t], cache)
    v = enc_v(V[t], cache)
    loss = pen = 0.0
    hi = min(n, X.shape[0] - 1 - t)
    # a frame earlier than this hands the next one a detached state, so no
    # gradient travels more than --grad_frames frames back. The loss is
    # untouched: every frame is still scored.
    cut = hi - args.grad_frames if args.grad_frames else 0
    for j in range(hi):
        if cut > 0 and j < cut:
            p, v = p.detach(), v.detach()
        p, v, cfl = fit.rollout(p, v, args.dt_mult, cache)
        got = fit.gaussian_pos(p, cache)
        d = (X[t + j + 1] - X[t]).norm(dim=-1).mean().clamp(min=1e-12)
        loss = loss + (got - X[t + j + 1]).norm(dim=-1).mean() / d
        pen = pen + cfl
    return loss / max(hi, 1), pen / max(hi, 1)



def fine_loss(x0, v0, cache, n_sub, fc0=None):
    """n SUBSTEPS from one state, scored against MPM at every one of them.

    The unrolled loss above walks twelve coarse frames -- 480 differentiable
    substeps -- and compares at twelve of them. That is 480 forward evaluations
    producing twelve numbers of supervision, with a gradient that has to travel
    back through all of them, and it is the reason a fit iteration cost two
    minutes.

    MPM's own step is the same 1e-4 as this simulator's, so the two can be walked
    side by side and compared at every substep instead. The same compute then
    yields forty times the supervision, and a short horizon becomes affordable:
    what the long unroll was buying is states the simulator reaches on its own,
    and DAgger supplies those directly rather than by rolling forward from an MPM
    state every iteration.

    MPM is restarted from the lifted anchor state rather than from its own stored
    frame, because the cache holds coarse frames only and storing substeps would
    be 490 GB. The lift reproduces MPM's continuation to 2e-4 of a 0.59 span
    (exe/verify_mpm_teacher.py), and it is what makes a DAgger state usable at
    all -- so both sources of start state go through the same door.
    """
    p = enc(x0, cache)
    v = enc_v(v0, cache)
    with torch.no_grad():
        if fc0 is not None:
            # MPM's own recorded state: exact, and never refused
            T._set(x0.contiguous(), v0.contiguous(),
                   fc0[0].contiguous(), fc0[1].contiguous())
        else:
            x, vx, F, C = fit.lift(p, v, cache)
            if not (torch.isfinite(x).all() and x.min() > T.margin
                    and x.max() < T.grid_lim - T.margin):
                return None, None
            det = det3(F.reshape(-1, 3, 3))
            if det.min() < 0.05 or det.max() > 20.0:
                return None, None
            T._set(x, vx, F, C)

    loss = pen = 0.0
    for _ in range(n_sub):
        p, v, cfl = fit.rollout(p, v, 1, cache)
        with torch.no_grad():
            T.solver.p2g2p(None, sc.sub_dt, device=T.wp_dev)
            if not T._in_domain():
                return None, None
            tgt = T.solver.export_particle_x_to_torch()
        got = fit.gaussian_pos(p, cache)
        # against how far MPM has moved from the start, so an early substep is
        # not weighted down for having barely moved
        d = (tgt - x0).norm(dim=-1).mean().clamp(min=1e-12)
        loss = loss + (got - tgt).norm(dim=-1).mean() / d
        pen = pen + cfl
    return loss / n_sub, pen / n_sub


@torch.no_grad()
def step_error(sets, every=10):
    cache = fit.prepare()
    tot, n = 0.0, 0
    for X, V, *_ in sets:
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
    for X, V, *_ in sets:
        p, v = enc(X[0], cache), enc_v(V[0], cache)
        out = [fit.gaussian_pos(p, cache)]
        for _ in range(args.frames):
            p, v, _ = fit.rollout(p, v, args.dt_mult, cache)
            out.append(fit.gaussian_pos(p, cache))
        G = torch.stack(out)
        Xg = torch.stack([X[k] for k in range(args.frames + 1)])
        span = (Xg - Xg[0]).norm(dim=-1).max().clamp(min=1e-12)
        tot.append(((G - Xg).norm(dim=-1).mean(-1) / span).mean().item())
    return 100 * sum(tot) / max(len(tot), 1)


# scoring every fit trajectory costs more than training on them: 106 of them at
# one sample per ten frames is 636 rollouts of forty substeps, ten minutes an
# evaluation. A fixed subset says the same thing about whether the fit is moving
N_REPORT = 10


@torch.no_grad()
def report(tag):
    with ema_weights():
        return _report(tag)


@torch.no_grad()
def _report(tag):
    a, b = step_error(FIT[:N_REPORT]), step_error(CHK)
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
                                 for X, *_ in FIT[:8])
    added = skipped = 0
    for i in range(args.dagger_traj):
        X, V, *_ = FIT[i % len(FIT)]
        p, v = enc(X[0], cache), enc_v(V[0], cache)
        for t in range(args.dagger_frames):
            gp = fit.gaussian_pos(p, cache)
            if not torch.isfinite(p).all() or (gp - X[0]).norm(dim=-1).max() > cap:
                break
            if t % args.dagger_stride == 0:
                x, vx, F, C = fit.lift(p, v, cache)
                ok = torch.isfinite(x).all() and x.min() > T.margin and \
                    x.max() < T.grid_lim - T.margin
                if ok and not args.fine_steps:
                    # the coarse loss compares against a stored frame; the fine
                    # one walks MPM alongside and needs no target, which also
                    # saves forty MPM substeps per state collected
                    T._set(x, vx, F, C)
                    out = T._advance(1, args.dt_mult)
                    if out is None:
                        ok = False
                if ok:
                    POOL["x"].append(x.to(torch.float16).cpu())
                    POOL["v"].append(vx.to(torch.float16).cpu())
                    if not args.fine_steps:
                        POOL["tgt"].append(
                            T.solver.export_particle_x_to_torch().to(torch.float16).cpu())
                    added += 1
                else:
                    skipped += 1
    # oldest out first, so the pool tracks where the simulator is now rather than
    # where it was at the start
    for k_ in POOL:
        while len(POOL[k_]) > args.dagger_pool_max:
            POOL[k_].pop(0)
            p, v, _ = fit.rollout(p, v, args.dt_mult, cache)
    return added, skipped


def make_opt():
    for n, prm in fit.named_parameters():
        prm.requires_grad_(n in TRAIN)
    return torch.optim.Adam([{"params": [getattr(fit, n)], "lr": LRS[n]}
                             for n in TRAIN])


opt = make_opt()
grad_accum = torch.zeros(fit.M, device=dev)
STATE = args.state or (args.out + ".state" if args.out else None)
LR0 = {n: LRS[n] for n in TRAIN}
EMA = {}


def ema_update():
    """track an average of the parameters, restarted when their shape changes"""
    if not args.ema:
        return
    for n in TRAIN:
        p_ = getattr(fit, n).detach()
        if n not in EMA or EMA[n].shape != p_.shape:
            EMA[n] = p_.clone()
        else:
            EMA[n].mul_(args.ema).add_(p_, alpha=1 - args.ema)


class ema_weights:
    """evaluate with the average, then put the live parameters back"""

    def __enter__(self):
        self.saved = None
        if args.ema and EMA:
            self.saved = {n: getattr(fit, n).detach().clone() for n in TRAIN}
            with torch.no_grad():
                for n in TRAIN:
                    getattr(fit, n).copy_(EMA[n])

    def __exit__(self, *a):
        if self.saved is not None:
            with torch.no_grad():
                for n in TRAIN:
                    getattr(fit, n).copy_(self.saved[n])


# the mirror is the backup, so a mirror that has quietly stopped working is no
# backup at all. A vast host started intercepting TLS mid-run and every copy
# failed with a certificate error for an hour and a half without a word, because
# the upload was a backgrounded shell command with stderr sent to /dev/null. It
# stays asynchronous -- training should not wait on an upload -- but the previous
# copy's exit status is now collected before the next one starts.
_mirror_proc = None
_mirror_fails = 0


def _mirror(path):
    global _mirror_proc, _mirror_fails
    if _mirror_proc is not None:
        rc = _mirror_proc[0].poll()
        if rc is None:
            pass
        elif rc != 0:
            _mirror_fails += 1
            err = _mirror_proc[0].stderr.read().decode(errors="replace").strip()
            print(f"\n[mirror] FAILED to copy {os.path.basename(_mirror_proc[1])} "
                  f"to {args.r2} (rclone exit {rc}, {_mirror_fails} in a row)\n"
                  f"[mirror] {err.splitlines()[-1] if err else 'no output'}\n"
                  f"[mirror] this run is NOT backed up", flush=True)
        else:
            if _mirror_fails:
                print(f"\n[mirror] recovered after {_mirror_fails} failures", flush=True)
            _mirror_fails = 0
        _mirror_proc = None
    p = subprocess.Popen(["rclone", "copy", path, args.r2],
                          stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    _mirror_proc = (p, path)


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
                 # the pool stays out: it is a gigabyte of particle states that
                 # refills in a few collections, so writing it every save would
                 # cost more than regenerating it
                 "c": args.c, "args": vars(args)}, STATE + ".tmp")
    os.replace(STATE + ".tmp", STATE)
    if args.r2:
        _mirror(STATE)


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
            POOL[k] = []          # not persisted; refills in a few collections
        start_it, best = blob["iter"] + 1, blob["best"]
        print(f"\n[resume] {STATE} at iteration {blob['iter']}, {fit.M} anchors, "
              f"best {best:.1f}%, {len(POOL['x'])} collected states")
if best is None:
    print(f"\n[before]")
    best = report("init")

# where the fit started, for the regulariser to refer to. Re-taken after every
# density change, since the anchors are not the same set any more
P0, S0, Q0 = fit.pos.detach().clone(), fit.log_s.detach().clone(), fit.quat.detach().clone()
t0 = time.time()
n_skip = 0
GLOG = open(args.grad_log, "a") if args.grad_log else None
if GLOG:
    GLOG.write("# iter " + " ".join(f"|g_{n}| cos_{n}" for n in TRAIN) + "\n")
PREV_G = {}

bar = tqdm(range(start_it, args.iters + 1), desc="fit", ncols=90)
for it in bar:
    if args.lr_cosine:
        # cosine from the given rate to zero across the whole run
        f = 0.5 * (1.0 + math.cos(math.pi * min(it / max(args.iters, 1), 1.0)))
        for gp, n in zip(opt.param_groups, TRAIN):
            gp["lr"] = LR0[n] * f
    if args.lr_decay_at and it == args.lr_decay_at:
        for gp in opt.param_groups:
            gp["lr"] *= args.lr_decay_by
        print(f"\n  [lr] it={it}: every rate multiplied by {args.lr_decay_by}",
              flush=True)
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
        P0 = fit.pos.detach().clone(); S0 = fit.log_s.detach().clone()
        Q0 = fit.quat.detach().clone()
        print(f"\n  [density] it={it}: -{dead} +{split} -> {fit.M} anchors, "
              f"{fit.pair_g.shape[0]} pairs", flush=True)
    if args.dagger_every and (it == start_it or it % args.dagger_every == 0):
        a_, s_ = collect_dagger()
        print(f"\n  [dagger] it={it}: +{a_} states ({s_} MPM could not answer for), "
              f"pool {len(POOL['x'])}", flush=True)

    frac = min(1.0, it / max(args.warmup, 1))
    hi = max(2, int(frac * (args.frames - 1)))
    # the fine horizon, annealed if asked. Geometric rather than linear: what
    # matters is the ratio between what a sample sees and what a rollout does,
    # and that is a scale
    n_sub_now = args.fine_steps
    if args.fine_steps and args.fine_end:
        u = min(1.0, (it - 1) / max(args.iters - 1, 1))
        n_sub_now = max(1, int(round(args.fine_steps
                                      * (args.fine_end / args.fine_steps) ** u)))
    cache = fit.prepare()
    opt.zero_grad(set_to_none=True)
    acc_loss, acc_pen, acc_n, skipped = 0.0, 0.0, 0, 0
    for _accum in range(args.accum):
      loss, pen, bad_sample, n_ok = 0.0, 0.0, None, 0
      for _ in range(args.batch):
          src, fc0 = None, None
          if POOL["x"] and torch.rand(1).item() < args.dagger_frac:
              # a DAgger state has one labelled frame after it and nothing more, so
              # it stays a single step whatever the unroll length is
              j = torch.randint(len(POOL["x"]), (1,)).item()
              x0 = POOL["x"][j].to(dev, torch.float32)
              v0 = POOL["v"][j].to(dev, torch.float32)
              tgt = POOL["tgt"][j].to(dev, torch.float32) if POOL["tgt"] else None
          else:
              ent = FIT[torch.randint(len(FIT), (1,)).item()]
              X, V = ent[0], ent[1]
              t = torch.randint(hi, (1,)).item()
              x0, v0, tgt = X[t], V[t], X[t + 1]
              src = (X, V, t)
              # MPM's own state at that frame, when it was kept: then the fine loss
              # walks alongside MPM from where MPM actually was, with no lift
              if len(ent) == 4:
                  fc0 = (ent[2][t], ent[3][t])
          if args.fine_steps:
              contrib, cfl = fine_loss(x0, v0, cache, n_sub_now, fc0)
              if contrib is None:      # MPM cannot be asked from here
                  continue
          elif src is not None and args.unroll > 1:
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
          n_ok += 1
      if n_ok == 0:
        # every sample in the batch started somewhere MPM will not answer for --
        # a lifted state outside its grid, or a deformation gradient no material
        # is in. Nothing to learn from, and dividing by the batch would leave a
        # float where a tensor is expected
        continue
      loss, pen = loss / n_ok, pen / n_ok
      total = loss + args.lambda_cfl * pen
      if args.reg > 0:
        # measured against the anchor spacing and against unit scale, so one
        # number covers parameters that do not share units
        h = sc.sim.radius
        r = ((fit.pos - P0) / h).pow(2).mean() + (fit.log_s - S0).pow(2).mean() \
            + (fit.quat - Q0).pow(2).mean() + fit.log_k.pow(2).mean()
        total = total + args.reg * r
      if not torch.isfinite(total):
        # one bad sample -- a rollout that ran away, a configuration the polar
        # factor cannot handle -- should cost that iteration, not the run
        skipped = 1
        break
      # backward here rather than at the end: the graph of this batch is freed
      # before the next one is built, which is the whole reason accumulation
      # buys a bigger effective batch on a card that cannot hold one
      (total / args.accum).backward()
      acc_loss += float(loss); acc_pen += float(pen); acc_n += 1
    if acc_n == 0:
        n_skip += 1
        continue
    loss = torch.tensor(acc_loss / acc_n)
    pen = acc_pen / acc_n
    if not skipped:
      if True:
        with torch.no_grad():
            if fit.pos.grad is not None:
                grad_accum += torch.nan_to_num(fit.pos.grad).norm(dim=-1)
                fit.pos.grad[fit.fixed] = 0
        if all(getattr(fit, n).grad is None or
               torch.isfinite(getattr(fit, n).grad).all() for n in TRAIN):
            torch.nn.utils.clip_grad_norm_([getattr(fit, n) for n in TRAIN], 1.0)
            opt.step()
            fit.clamp_()
            ema_update()
        else:
            skipped = 1
    if GLOG and not skipped:
        parts = []
        for n in TRAIN:
            g_ = getattr(fit, n).grad
            if g_ is None:
                parts += ["nan", "nan"]
                continue
            g_ = g_.detach().reshape(-1)
            c = float("nan")
            # densify_and_prune changes the anchor count, and a gradient of a
            # different length is a gradient for a different parameter set --
            # there is no angle between them to report
            if n in PREV_G and PREV_G[n].shape == g_.shape:
                c = float((g_ * PREV_G[n]).sum()
                          / (g_.norm() * PREV_G[n].norm()).clamp(min=1e-30))
            PREV_G[n] = g_.clone()
            parts += [f"{float(g_.norm()):.6g}", f"{c:.4f}"]
        GLOG.write(f"{it} " + " ".join(parts) + "\n")
        GLOG.flush()
    n_skip += skipped
    if skipped and args.no_guards:
        # what a survivable run hides: which quantity went first, and whether the
        # step that produced it grew every substep or spiked once
        print(f"\n[first failure] iteration {it}, loss {loss.item()}", flush=True)
        if bad_sample is not None:
            x0, v0 = bad_sample
        w_, rc_, q_, Binv_, blocked_, mass_ = cache
        with torch.no_grad():
            F_, _ = fit.deformation(enc(x0, cache), w_, rc_, q_, Binv_, blocked_)
            det = torch.linalg.det(F_)
            print(f"  mass: min {mass_.min():.3e}, at the clamp "
                  f"{int((mass_ <= 1.0000001e-12).sum())} of {fit.M}")
            print(f"  detF: min {det.min():.4f}, inverted {int((det <= 0).sum())} of "
                  f"{fit.N}, extents min {fit.log_s.exp().min():.3e}")
            m_ = mass_.unsqueeze(-1)
            keep_ = (~fit.fixed).unsqueeze(-1).to(torch.float32)
            p_, v_ = enc(x0, cache), enc_v(v0, cache)
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
            if args.eval_rollout:
                # the one-step error keeps falling while the rollout doubles, so
                # keeping the best by one-step keeps the wrong checkpoint
                b = rollout_error(CHK)
        if args.out and b < best:
            best = b
            # args go in too: the runs that produced the fitted sets in use
            # saved only parameters, so months later there was no way to tell
            # what --iters, --reg or DAgger setting had produced them
            torch.save({"pos": fit.pos.detach().cpu(), "log_s": fit.log_s.detach().cpu(),
                         "quat": fit.quat.detach().cpu(),
                         "log_k": fit.log_k.detach().cpu(), "c": args.c, "iter": it,
                         "eig_floor": args.eig_floor, "score": float(b),
                         "args": vars(args)}, args.out)
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
        p = enc(X[0], cache)
        v = enc_v(V[0], cache)
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

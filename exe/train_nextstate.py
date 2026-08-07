"""Train the anchor dynamics directly from explicit-simulator trajectories.

No physics in the loss, the targets or the integration: the target is where the
simulator's anchors actually were one coarse step later, and the loss is L2 on
the displacement. The one physical quantity involved is the elastic acceleration
f_int(p)/m, which the network takes as an INPUT and which is recomputed from
whatever configuration the network has produced -- so unlike the retired line,
it means the same thing in training and in rollout.

Evaluation is the only thing that matters here and it is done on the network's
OWN rollout, not on single steps: position error against the reference
trajectory, frame by frame, which is directly comparable across runs and does
not depend on any physics quantity.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

_lib = os.path.join(os.path.dirname(__file__), "..", "lib")
sys.path.insert(0, _lib)

import torch
from tqdm import tqdm

from anchorflow import scene_setup
from anchorflow.nextstate import NextStep

ap = argparse.ArgumentParser()
ap.add_argument("--ply", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--n_anchors", type=int, default=512)
ap.add_argument("--K", type=int, default=8)
ap.add_argument("--dt_mult", type=int, default=40,
                 help="coarse step, in explicit substeps. One network call replaces this many.")
ap.add_argument("--n_traj", type=int, default=12)
ap.add_argument("--n_steps", type=int, default=60, help="coarse steps per trajectory")
ap.add_argument("--hidden", type=int, default=128)
ap.add_argument("--depth", type=int, default=4)
ap.add_argument("--heads", type=int, default=4)
ap.add_argument("--iters", type=int, default=30000)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--zero_init", action="store_true",
                 help="zero the decoder's last layer. Off by default: it came from the "
                      "retired implicit model, where it meant starting at an explicit step, "
                      "and here it only means starting frozen with no gradient reaching any "
                      "earlier layer until the decoder grows.")
ap.add_argument("--n_holdout", type=int, default=5,
                 help="trajectories kept out of training and used for the rollout score. One "
                      "is not enough: a 59-step chain amplifies whichever way the per-step "
                      "error happens to be biased, so the same run scored 4.6%% and 179%% on "
                      "consecutive evaluations while its single-step error never moved. "
                      "Selecting a checkpoint on one such number selects noise.")
ap.add_argument("--cosine", action="store_true",
                 help="decay the learning rate to zero over the run (cosine). Held flat, the "
                      "rollout measure swung between 4.6%% and 179%% on consecutive "
                      "evaluations of the same run -- the step size was still large enough to "
                      "walk out of whatever it had found.")
ap.add_argument("--batch", type=int, default=8,
                 help="states per iteration. With one state per step the gradient carried the "
                      "state-to-state difficulty spread directly -- the relative step error "
                      "swung 0.21-0.28 around its own mean for 29k iterations without the "
                      "trend moving.")
ap.add_argument("--noise", type=float, default=0.0,
                 help="random-walk noise on the input history, as a fraction of the typical "
                      "displacement. GNS's fix for rollout error accumulation: the network "
                      "only ever sees clean simulator states otherwise, and has no idea what "
                      "to do once its own error has moved it off that manifold.")
ap.add_argument("--eval_every", type=int, default=5000)
ap.add_argument("--eval_frames", type=int, default=60)
ap.add_argument("--compile", action="store_true",
                 help="wrap the model in torch.compile. Off by default -- it is the obvious "
                      "answer to a launch-overhead bottleneck, but it changes what is being "
                      "compared between runs and the batching fix below already removes ~50 "
                      "of the per-iteration ops.")
ap.add_argument("--impulse_range", type=float, default=4.0,
                 help="the random impulse strength spans [0.5, 0.5*this] times the config's. "
                      "Set to 1 to hold every trajectory at the config's own strength: the "
                      "output and loss are normalised by the mean displacement over the whole "
                      "set, so a wider spread coarsens the units for the modest trajectory the "
                      "rollout is scored on.")
ap.add_argument("--traj_cache", default=None,
                 help="file to keep the collected trajectories in. Collection is ~1 s per "
                      "trajectory of explicit substeps, so a data-scaling sweep re-simulates "
                      "the same runs several times over; the RNG is seeded, so trajectory i is "
                      "the same in every run and a cache of N serves any n_traj <= N.")
ap.add_argument("--out", default=None)
ap.add_argument("--ckpt_every", type=int, default=2000,
                 help="write a resumable checkpoint this often. These instances stop on an "
                      "idle watchdog and can be reclaimed at any time; a run that can only "
                      "save at the end loses everything when that happens.")
ap.add_argument("--stream_len", type=int, default=0,
                 help="if set, training data is this many coarse steps of a CONTINUOUS run "
                      "with impulses arriving along the way, instead of n_traj separate runs "
                      "from rest. Every trajectory from rest explores the same narrow set of "
                      "states -- what one impulse reaches before it decays -- and the later "
                      "steps of all of them are near rest. A running object that keeps being "
                      "hit reaches configurations no single impulse produces, which is where "
                      "every run so far fails.")
ap.add_argument("--n_stream", type=int, default=4,
                 help="how many such runs, so consecutive samples in a batch are not all from "
                      "the same stretch of one history")
ap.add_argument("--impulse_every", type=int, default=20,
                 help="mean coarse steps between impulses, jittered by half")
ap.add_argument("--stream_amp_cap", type=float, default=3.0,
                 help="skip an impulse while the object is already displaced more than this "
                      "many times the config impulse's peak. Impulses land on top of each "
                      "other by design, and the shape-matching fit degrades once the local "
                      "neighbourhoods are far from their rest arrangement.")
ap.add_argument("--frozen_weights", action="store_true",
                 help="hold the blend weights at their canonical values. Without this the "
                      "simulator is not a function of its state -- the same anchors and "
                      "velocities step differently depending on the cloud the weights were "
                      "read from -- so part of the target is not determined by the inputs and "
                      "no network can fit it. Costs 0.24%% of peak motion at ficus's own "
                      "impulse, 1.4%% at three times it.")
ap.add_argument("--no_accel", action="store_true",
                 help="drop the elastic acceleration from the network's input. Under frozen "
                      "weights it is a function of the positions, so this measures what the "
                      "physics feature is worth as computation rather than as information -- "
                      "and removes both the force evaluation and the per-step skinning from "
                      "the rollout, since the cloud exists only to evaluate the force against.")
ap.add_argument("--reset_best", action="store_true",
                 help="forget the resumed run's best score. The rollout measure changed when it "
                      "started penalising a rollout for motion it fails to produce, so a score "
                      "carried over from before is on a different scale and no later evaluation "
                      "can beat it.")
ap.add_argument("--snapshot_every", type=int, default=0,
                 help="also keep a checkpoint named after its iteration, so the run leaves a "
                      "trail instead of one overwritten file. The rollout measure swings hard "
                      "between evaluations, and without these the only way back to a good "
                      "iterate is to train to it again.")
ap.add_argument("--r2", default=None,
                 help="rclone destination to copy every checkpoint to as it is written, e.g. "
                      "r2:storage/result/anchorflow/ckpt/NAME. The instance can be reclaimed "
                      "at any time, so nothing that matters should live only on its disk.")
ap.add_argument("--resume", default=None,
                 help="resume from a checkpoint written by --out (model, optimiser, iteration "
                      "and RNG). The collected trajectories are regenerated, not stored -- "
                      "they are deterministic given the seed and take seconds.")
args = ap.parse_args()

dev = "cuda"
torch.manual_seed(0)
sc = scene_setup.build(args.ply, args.config, args.n_anchors, args.K, device=dev,
                        frozen_weights=args.frozen_weights)
USE_A = not args.no_accel
# in stream mode the from-rest trajectories are only there to be held out, so
# only the held-out ones are worth generating
N_TRAJ = args.n_holdout if args.stream_len > 0 else args.n_traj
AC, M, fixed = sc.anchor_canonical, sc.M, sc.fixed_mask
print(f"[setup] N={sc.N} M={M} pinned={int(fixed.sum())} "
      f"coarse dt={args.dt_mult * sc.sub_dt} ({args.dt_mult} substeps)")

# ---------------- data: coarse-sampled explicit trajectories ----------------
base_force = None
for bc in sc.cfg.get("boundary_conditions", []):
    if bc["type"] == "particle_impulse":
        base_force = torch.tensor(bc["force"], device=dev)
gen = torch.Generator(device=dev); gen.manual_seed(1234)


def rand_rot():
    q, r = torch.linalg.qr(torch.randn(3, 3, device=dev, generator=gen))
    q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def trajectory(force):
    """coarse-sampled explicit run -> (positions [T,M,3], elastic accel [T,M,3])

    The Gaussian cloud is 2.4 MB a snapshot and is only needed to evaluate the
    force, so the force is evaluated during collection and the cloud discarded.
    """
    p, v = AC.clone(), sc.initial_velocity(force)
    gp = sc.pos.clone()
    ps, accs = [p.clone()], [sc.elastic_accel(p, gp)]
    for _ in range(args.n_steps):
        p, v, gp = sc.explicit_step(p, v, gp, args.dt_mult)
        if not torch.isfinite(p).all():
            break
        ps.append(p.clone()); accs.append(sc.elastic_accel(p, gp))
    return torch.stack(ps), torch.stack(accs)


def stream(n_steps, cap):
    """one continuous run, impulses arriving along the way.

    Returns (positions, accelerations, contaminated) where contaminated[k] marks
    a step whose target carries an impulse the inputs cannot see: the kick lands
    between p_k and p_{k+1}, so no function of (p_k, v_k, a_k) reaches the
    answer. One sample per impulse, dropped. Every later step is fine -- by then
    the kick is in the positions, and the velocity is read off them.
    """
    p, v = AC.clone(), torch.zeros(M, 3, device=dev)
    gp = sc.pos.clone()
    ps, acc, bad = [p.clone()], [sc.elastic_accel(p, gp)], []
    due = 0
    for k in range(n_steps):
        fired = False
        if k >= due and base_force is not None:
            amp = (p - AC)[~fixed].norm(dim=-1).max().item()
            if amp < cap:
                s = 0.5 * (args.impulse_range ** torch.rand(1, device=dev, generator=gen).item())
                s = s if args.impulse_range > 1 else 1.0
                v = v + sc.impulse_dv((rand_rot() @ base_force) * s)
                fired = True
            jitter = 0.5 + torch.rand(1, device=dev, generator=gen).item()
            due = k + max(1, int(round(args.impulse_every * jitter)))
        bad.append(fired)
        p, v, gp = sc.explicit_step(p, v, gp, args.dt_mult)
        if not torch.isfinite(p).all():
            print(f"  [stream] diverged at step {k}, truncating", flush=True)
            bad.pop()
            break
        ps.append(p.clone()); acc.append(sc.elastic_accel(p, gp))
    return torch.stack(ps), torch.stack(acc), torch.tensor(bad, device=dev)


trajs, accs = [], []
if args.traj_cache and os.path.exists(args.traj_cache):
    blob = torch.load(args.traj_cache, map_location=dev, weights_only=False)
    trajs, accs = blob["trajs"], blob["accs"]
    print(f"[data] {len(trajs)} trajectories from {args.traj_cache}")
if len(trajs) < N_TRAJ:
    print(f"[data] collecting {N_TRAJ - len(trajs)} more trajectories "
          f"x {args.n_steps} coarse steps...")
    # the generator is advanced once per trajectory, so extending a cache
    # reproduces exactly the trajectories a fresh run of this size would give
    for t in range(N_TRAJ):
        if t == 0 or base_force is None:
            f = base_force
        else:
            s = 0.5 * (args.impulse_range ** torch.rand(1, device=dev, generator=gen).item())
            s = s if args.impulse_range > 1 else 1.0
            f = (rand_rot() @ base_force) * s
        if t < len(trajs):
            continue
        ps, ac_ = trajectory(f)
        trajs.append(ps); accs.append(ac_)
        if (t + 1) % 25 == 0 or t + 1 == N_TRAJ:
            print(f"  {t + 1}/{N_TRAJ}", flush=True)
    if args.traj_cache:
        torch.save({"trajs": trajs, "accs": accs}, args.traj_cache)
        print(f"[data] cached to {args.traj_cache}")
trajs, accs = trajs[:N_TRAJ], accs[:N_TRAJ]
# trajectory 0 is the config's own impulse and is HELD OUT: it is the rollout
# every run is scored on, so leaving it in the training set turns that score into
# a memorisation test. It also silently biases any comparison across dataset
# sizes -- with 12 trajectories it is 8.3% of the data, with 120 it is 0.8%, so
# fewer trajectories look better for a reason that has nothing to do with
# generalisation.
# the first n_holdout trajectories are HELD OUT. Trajectory 0 is the config's own
# impulse and is the headline rollout; the rest are there because one 59-step
# chain is a very noisy number (see --n_holdout).
# in stream mode the training data is elsewhere, so every from-rest trajectory
# can be held out; in classic mode at least one has to be left to train on
HOLD = args.n_holdout if args.stream_len > 0 else min(args.n_holdout, len(trajs) - 1)
REF, REF_A = trajs[0], accs[0]
# one tensor, indexed on the GPU: building a batch out of a python list of
# slices was ~50 separate microsecond-scale ops per iteration, and at this model
# size that launch overhead is what the GPU waits on rather than the arithmetic
n_min = min(t.shape[0] for t in trajs)
# the held-out trajectories stay what they always were -- from rest, one
# impulse each -- so the rollout number remains comparable across every run in
# this project, whatever the training data was drawn from
HTRAJ = torch.stack([t[:n_min] for t in trajs[:HOLD]])
HACCS = torch.stack([a[:n_min] for a in accs[:HOLD]])

if args.stream_len > 0:
    cap = args.stream_amp_cap * (trajs[0] - AC).norm(dim=-1).max().item()
    print(f"[data] {args.n_stream} continuous runs x {args.stream_len} coarse steps, "
          f"an impulse every ~{args.impulse_every}, skipped above "
          f"{cap:.4f} of displacement", flush=True)
    st, sa, sb = [], [], []
    for i in range(args.n_stream):
        a_, b_, c_ = stream(args.stream_len, cap)
        st.append(a_); sa.append(b_); sb.append(c_)
        peak = (a_ - AC)[:, ~fixed].norm(dim=-1).max().item()
        print(f"  {i + 1}/{args.n_stream}: {a_.shape[0]} steps, "
              f"{int(c_.sum())} impulses, peak displacement {peak:.4f}", flush=True)
    s_min = min(t.shape[0] for t in st)
    TRAJ = torch.stack([t[:s_min] for t in st])
    ACCS = torch.stack([a[:s_min] for a in sa])
    BAD = torch.stack([b[:s_min] for b in sb])
    IDX = torch.tensor([(ti, k) for ti in range(TRAJ.shape[0])
                        for k in range(1, s_min - 1) if not bool(BAD[ti, k])], device=dev)
    print(f"[data] {int(BAD.sum())} samples dropped as impulse-contaminated")
else:
    TRAJ = torch.stack([t[:n_min] for t in trajs])       # [n_traj, T, M, 3]
    ACCS = torch.stack([a[:n_min] for a in accs])
    IDX = torch.tensor([(ti, k) for ti in range(HOLD, TRAJ.shape[0])
                        for k in range(1, n_min - 1)], device=dev)
pairs = IDX
dt_coarse = args.dt_mult * sc.sub_dt
SRC = [TRAJ[i] for i in range(TRAJ.shape[0])] if args.stream_len > 0 else trajs
SRC_A = [ACCS[i] for i in range(ACCS.shape[0])] if args.stream_len > 0 else accs
DISP = [tr[1:] - tr[:-1] for tr in SRC]
DISP_SCALE = float(torch.cat([d.norm(dim=-1).flatten() for d in DISP]).mean())
VEL_SCALE = DISP_SCALE / dt_coarse
ACC_SCALE = float(torch.cat([a.norm(dim=-1).flatten() for a in SRC_A]).mean())
DEV = [d[1:] - d[:-1] for d in DISP]
DEV_SCALE = float(torch.cat([q.norm(dim=-1).flatten() for q in DEV]).mean())
print(f"[data] {len(pairs)} training pairs (trajectory 0 held out for evaluation); typical coarse displacement = {DISP_SCALE:.5f}, "
      f"deviation from inertia = {DEV_SCALE:.5f} ({100*DEV_SCALE/DISP_SCALE:.1f}% of it), "
      f"velocity scale = {VEL_SCALE:.4f}, elastic accel scale = {ACC_SCALE:.2f}")

# baselines, on the same relative-error scale the training loop reports
with torch.no_grad():
    num = den = 0.0
    for d in DISP:                       # "the next displacement equals this one"
        num += float(((d[1:] - d[:-1]) ** 2).sum()); den += float((d[1:] ** 2).sum())
    print(f"[baseline] persistence (du_next = du): rel err = {(num / den) ** 0.5:.4f}")
    print(f"[baseline] zero      (du_next = 0):    rel err = 1.0000")
print(f"[data] reference peak anchor displacement = "
      f"{(REF - AC).norm(dim=-1).max().item():.5f}")

net = NextStep(args.hidden, args.depth, args.heads, DISP_SCALE, VEL_SCALE,
                ACC_SCALE, args.zero_init, use_accel=USE_A).to(dev)
opt = torch.optim.Adam(net.parameters(), lr=args.lr, fused=True)
if args.compile:
    net = torch.compile(net)
print(f"[setup] params: {sum(q.numel() for q in net.parameters())/1e3:.1f}k")


@torch.no_grad()
def rollout(frames, ref=None):
    """autoregressive from the reference's initial condition; returns positions.

    The elastic acceleration is re-evaluated every step from the network's own
    anchors, via the Gaussian cloud it has skinned -- the same call the training
    data was built with.
    """
    ref = REF if ref is None else ref
    p = ref[1].clone()
    v = (ref[1] - ref[0]) / dt_coarse
    gp = sc.skin(p, sc.pos.clone())
    out = [p.clone()]
    for _ in range(frames):
        # without the acceleration input there is nothing to evaluate a force
        # against, so the cloud is never skinned during the rollout either
        a = sc.elastic_accel(p, gp) if USE_A else None
        du = net(p, v, a, dt_coarse)
        du = torch.where(fixed.unsqueeze(-1), torch.zeros_like(du), du)
        p = p + du
        v = du / dt_coarse
        if not torch.isfinite(p).all():
            break
        if USE_A:
            gp = sc.skin(p, gp)
        out.append(p.clone())
    return torch.stack(out)


@torch.no_grad()
def rollout_error(frames, which=0):
    r = HTRAJ[which]
    got = rollout(frames, r)
    n = min(got.shape[0], r.shape[0] - 1)
    ref = r[1:1 + n]
    # over the anchors that actually move: the 129 pinned ones are identical on
    # both sides by construction and averaging them in scales the error down by
    # a quarter for free
    err = (got[:n] - ref)[:, ~fixed].norm(dim=-1).mean(-1)
    span = (ref - AC).norm(dim=-1).max().clamp(min=1e-12)
    # how far the rollout moved the anchors at all, against how far the reference
    # did. Position error alone cannot tell an accurate rollout from a frozen
    # one: the reference oscillates back through its start, so a model that
    # predicts almost nothing stays near it and scores well. One lr=1e-2 run did
    # exactly that -- 10% of the reference's motion, best score in the sweep.
    moved = (got[:n] - AC)[:, ~fixed].norm(dim=-1).max(-1).values
    return n, err, (err / span), (moved.max() / span)


@torch.no_grad()
def rollout_score(frames):
    """mean final error over every held-out trajectory"""
    vals, mv = [], []
    for w in range(HOLD):
        n, _, rel, m = rollout_error(frames, w)
        vals.append(float(rel[n - 1])); mv.append(float(m))
    err = sum(vals) / len(vals)
    amp = sum(mv) / len(mv)
    # penalise a rollout for the motion it is missing as well as for where it
    # put things, so standing still cannot win. Overshoot is already punished by
    # the error term, so only the shortfall counts here.
    return err + max(0.0, 1.0 - amp), vals, amp


hist_log = []
start_it = 1
BEST = {"score": float("inf"), "iter": 0}


def save(path, it):
    torch.save({"model": getattr(net, "_orig_mod", net).state_dict(), "opt": opt.state_dict(), "iter": it,
                "args": vars(args), "hist": hist_log, "disp_scale": DISP_SCALE,
                "dev_scale": DEV_SCALE,
                "vel_scale": VEL_SCALE, "acc_scale": ACC_SCALE,
                "best": BEST,
                "rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(),
                "gen": gen.get_state()}, path)
    if args.r2:
        subprocess.Popen(["rclone", "copy", path, args.r2],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


if args.resume and os.path.exists(args.resume):
    ck = torch.load(args.resume, map_location=dev, weights_only=False)
    getattr(net, "_orig_mod", net).load_state_dict(ck["model"])
    opt.load_state_dict(ck["opt"])
    hist_log = ck.get("hist", [])
    start_it = ck["iter"] + 1
    if not args.reset_best:
        BEST.update(ck.get("best", {}) or {})
    torch.set_rng_state(ck["rng"].cpu())
    torch.cuda.set_rng_state(ck["cuda_rng"].cpu())
    gen.set_state(ck["gen"].cpu())
    # the optimiser state carries the learning rate it was annealed down to, and
    # the schedule reads its base rate from the same place. Left alone, a resumed
    # run's rate is decided by the run it continues rather than by the flags it
    # was given, which is not what --lr on the command line means.
    for g in opt.param_groups:
        g["lr"] = args.lr
        g.pop("initial_lr", None)
    print(f"[resume] {args.resume} at iter {ck['iter']}, lr set to {args.lr:g}"
          + ("" if not args.reset_best else " (best score forgotten)"))

sched = None
if args.cosine:
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.iters - start_it + 1, eta_min=args.lr * 1e-2)
pbar = tqdm(range(start_it, args.iters + 1), desc="train", ncols=105, initial=start_it - 1,
             total=args.iters)
for it in pbar:
    sel = IDX[torch.randint(IDX.shape[0], (args.batch,), device=dev)]
    ti, k = sel[:, 0], sel[:, 1]
    p = TRAJ[ti, k]
    v = (p - TRAJ[ti, k - 1]) / dt_coarse
    a = ACCS[ti, k]
    if args.noise > 0:
        # GNS-style: perturb the position, move the velocity consistently, and
        # take the target to the TRUE next position -- so the network is taught
        # to come back onto the trajectory rather than to carry the error.
        nz = torch.randn(p.shape, device=dev, generator=gen) * (args.noise * DISP_SCALE)
        nz[:, fixed] = 0
        p = p + nz
        v = v + nz / dt_coarse
        # and the acceleration is re-evaluated at the perturbed configuration.
        # Leaving it at the clean one puts a state in front of the network whose
        # force does not belong to its positions, which is exactly the train/
        # rollout mismatch that sank the previous line of work: in rollout `a` is
        # always the force at wherever the network currently is.
        if USE_A:
            a = torch.stack([sc.elastic_accel(p[b], sc.skin(p[b], sc.pos.clone()))
                             for b in range(p.shape[0])])
    target = TRAJ[ti, k + 1] - p
    du = net(p, v, a, dt_coarse)
    du = torch.where(fixed.view(1, -1, 1), torch.zeros_like(du), du)
    # in units of the typical displacement. The raw-unit loss was ~5e-6, which
    # put the gradient reaching the decoder output at ~1e-9 -- the same order as
    # Adam's epsilon, so the term that is supposed to make Adam scale-invariant
    # was throttling it instead.
    loss = (((du - target) / DISP_SCALE) ** 2).mean()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    opt.step()
    if sched is not None:
        sched.step()

    if it % 50 == 0 or it == 1:
        rel = (du - target).norm() / target.norm().clamp(min=1e-20)
        hist_log.append((it, loss.item(), rel.item()))
        pbar.set_postfix(mse=f"{loss.item():.2e}", rel=f"{rel.item():.4f}")
    if args.out and (it % args.ckpt_every == 0 or it == args.iters):
        save(args.out, it)
    if args.out and args.snapshot_every and it % args.snapshot_every == 0:
        save(args.out.replace(".pt", f"_it{it:06d}.pt"), it)
    if it % args.eval_every == 0 or it == args.iters:
        score, vals, amp = rollout_score(args.eval_frames)
        n, err, rel, _ = rollout_error(args.eval_frames)
        # keep the best rollout separately. The measure swings hard between
        # evaluations -- 4.6% at 225k and 179% at 250k on the same run -- so the
        # last checkpoint is a lottery ticket, and the rollout is the number
        # anyone would select on.
        if args.out and score < BEST["score"]:
            BEST["score"], BEST["iter"] = score, it
            save(args.out.replace(".pt", "_best.pt"), it)
            print(f"  [best] score {100*score:.1f} at it={it}", flush=True)
        print(f"\n  [rollout] it={it} over {HOLD} held-out trajectories: "
              f"err {100*sum(vals)/len(vals):.1f}%  motion {100*amp:.0f}% of reference"
              f"  -> score {100*score:.1f}  (traj0 {100*vals[0]:.1f}%, "
              f"spread {100*min(vals):.1f}-{100*max(vals):.1f}%)", flush=True)

print("\n[result]   iter        step MSE   rel step err")
for it, ls, rl in hist_log[::max(1, len(hist_log) // 20)]:
    print(f"  {it:8d}   {ls:.6e}   {rl:10.4f}")

score, vals, amp = rollout_score(args.eval_frames)
n, err, rel, _ = rollout_error(args.eval_frames)
print(f"\n[rollout] mean over {HOLD} held-out trajectories: "
      f"err {100*sum(vals)/len(vals):.2f}%, motion {100*amp:.0f}% of reference, "
      f"score {100*score:.2f}  (per trajectory: "
      f"{', '.join(f'{100*v:.1f}%' for v in vals)})")
print(f"[rollout] trajectory 0, {n}/{args.eval_frames+1} frames survived")
print(f"  {'frame':>6} {'mean anchor err':>16} {'% of reference motion':>22}")
for i in range(0, n, max(1, n // 12)):
    print(f"  {i:6d} {err[i]:16.5f} {rel[i]*100:21.2f}%")

if args.out:
    save(args.out, args.iters)
    print(f"[save] {args.out}")

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
from anchorflow.streams import stream as _stream, draw_impulse, draw_field_shape

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
ap.add_argument("--field", action="store_true",
                 help="impulses become smooth random force FIELDS instead of one vector on "
                      "the whole object. Every impulse in this project has been the config's "
                      "particle_impulse -- the same force everywhere -- so the data contains "
                      "only the lowest spatial frequency there is, and nothing the network "
                      "has seen tells bending one branch apart from shoving all of it. The "
                      "correlation length is drawn log-uniformly from the anchor spacing to "
                      "the object's size, so the uniform push is one end of the range.")
ap.add_argument("--rollout_steps", type=int, default=1,
                 help="train on this many consecutive predicted steps, with the network fed "
                      "its own output in between and the gradient carried through all of "
                      "them. At 1 this is what the project has always done: fit one step from "
                      "a state on the true trajectory. That loss never sees a rollout, so the "
                      "failure mode it produces -- error accumulating until the state is "
                      "somewhere training never went, then a runaway -- is invisible to it. "
                      "This is the covariate shift behavioural cloning is known for, and "
                      "unrolling the loss is the cheap half of the standard answer (the other "
                      "half being to relabel the states the policy actually visits).")
ap.add_argument("--dagger", action="store_true",
                 help="periodically roll the current network out, and label the states it "
                      "actually visits by asking the simulator what happens next from there. "
                      "Unrolling the loss teaches the network to recover from its own error "
                      "over a few steps; this changes which states it is taught on at all. "
                      "The expert is available at any state and costs a few milliseconds, "
                      "which is what makes DAgger cheap here -- and it is well defined only "
                      "because freezing the blend weights made the step a function of the "
                      "state. Under the lagged weights, 'what would the simulator do from "
                      "here' had no answer.")
ap.add_argument("--dagger_every", type=int, default=5000)
ap.add_argument("--dagger_traj", type=int, default=8,
                 help="rollouts per collection round")
ap.add_argument("--dagger_stride", type=int, default=3,
                 help="keep every n-th visited state; consecutive ones are nearly the same "
                      "state and each costs rollout_steps explicit steps to label")
ap.add_argument("--dagger_frac", type=float, default=0.3,
                 help="fraction of each batch drawn from collected states")
ap.add_argument("--dagger_cap", type=float, default=4.0,
                 help="stop a collection rollout once it is displaced this many times the "
                      "reference's peak. Past that the shape-matching fit is not something to "
                      "trust as a label, so the runaway's own tail is not fed back in as truth.")
ap.add_argument("--target_amp", action="store_true",
                 help="draw a target peak displacement for each training trajectory and scale "
                      "the force to reach it, instead of drawing a force strength. Strength "
                      "and correlation length are otherwise independent, and the RMS "
                      "normalisation makes a localised field move the object far less, so the "
                      "corner that is both localised and large held 2 of 120 trajectories -- "
                      "and the one held-out rollout that runs away sits 10%% past the largest "
                      "localised trajectory trained on. Held-out trajectories are unaffected.")
ap.add_argument("--amp_lo", type=float, default=0.25)
ap.add_argument("--amp_hi", type=float, default=2.0,
                 help="target peak displacement range, as multiples of the config impulse's "
                      "own peak, drawn log-uniformly")
ap.add_argument("--calib_steps", type=int, default=25,
                 help="coarse steps of a probe run used to measure what a unit field actually "
                      "moves, before rescaling it to the target")
ap.add_argument("--n_field_holdout", type=int, default=5,
                 help="held-out trajectories driven by a force field, scored separately. "
                      "Without these the change is invisible: the existing held-out set is "
                      "all uniform impulses, so it cannot tell whether the network learned "
                      "anything about being pushed locally.")
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
ap.add_argument("--seed", type=int, default=0,
                 help="seeds the model's initialisation, the batch order and the injected "
                      "noise -- but NOT the trajectories, which stay on their own generator "
                      "so two seeds see identical data. What separates a failure that belongs "
                      "to the method from one that belongs to the run.")
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
torch.manual_seed(args.seed)
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
    Under --no_accel nothing ever reads that force, so it is not computed --
    it is a fused-kernel call per step and a [T,M,3] tensor per trajectory.
    """
    p, v = AC.clone(), sc.initial_velocity(force)
    gp = sc.pos.clone()
    ps = [p.clone()]
    accs = [sc.elastic_accel(p, gp)] if USE_A else None
    for _ in range(args.n_steps):
        p, v, gp = sc.explicit_step(p, v, gp, args.dt_mult)
        if not torch.isfinite(p).all():
            break
        ps.append(p.clone())
        if USE_A:
            accs.append(sc.elastic_accel(p, gp))
    return torch.stack(ps), (torch.stack(accs) if USE_A else None)


def stream(n_steps, cap):
    """see anchorflow.streams.stream -- shared so the renderer draws this data"""
    return _stream(sc, n_steps, args.dt_mult, base_force, gen, cap,
                    impulse_every=args.impulse_every, impulse_range=args.impulse_range,
                    field=args.field, keep_accel=USE_A)

def peak_of(force, steps):
    """how far a force actually moves the object, over a short probe run"""
    p, v, gp = AC.clone(), sc.initial_velocity(force), sc.pos.clone()
    peak = 0.0
    for _ in range(steps):
        p, v, gp = sc.explicit_step(p, v, gp, args.dt_mult)
        if not torch.isfinite(p).all():
            return float("inf")
        peak = max(peak, (p - AC)[~fixed].norm(dim=-1).max().item())
    return peak


def amplitude_targeted(target):
    """a field scaled to reach `target` peak displacement, and what it took.

    The force-to-displacement map is close enough to linear at these strains
    that one probe and a rescale lands near the target; the scale is clamped so
    a field that barely moves the object cannot ask for an absurd force.
    """
    shape, sig = draw_field_shape(sc, gen)
    m0 = base_force.norm().item()
    d0 = peak_of(shape * m0, args.calib_steps)
    k = min(max(target / max(d0, 1e-9), 0.05), 30.0)
    return shape * m0 * k, sig, d0, k


trajs, accs = [], []
if args.traj_cache and os.path.exists(args.traj_cache):
    blob = torch.load(args.traj_cache, map_location=dev, weights_only=False)
    trajs, accs = blob["trajs"], blob["accs"]
    if USE_A and (not accs or accs[0] is None):
        print(f"[data] {args.traj_cache} was written without accelerations; regenerating")
        trajs, accs = [], []
    elif bool(blob.get("field", False)) != bool(args.field) or \
            bool(blob.get("target_amp", False)) != bool(args.target_amp) or \
            int(blob.get("n_holdout", args.n_holdout)) != args.n_holdout:
        # different impulses -- these are not the trajectories this run wants
        print(f"[data] {args.traj_cache} has field={blob.get('field', False)}, "
              f"n_holdout={blob.get('n_holdout')}; regenerating")
        trajs, accs = [], []
    else:
        print(f"[data] {len(trajs)} trajectories from {args.traj_cache}")
REF_PEAK = None
if args.target_amp and base_force is not None and len(trajs) < N_TRAJ:
    REF_PEAK = peak_of(base_force, args.n_steps)
    print(f"[data] the config impulse peaks at {REF_PEAK:.5f}; targets drawn "
          f"log-uniformly over {args.amp_lo}x to {args.amp_hi}x that", flush=True)
if len(trajs) < N_TRAJ:
    print(f"[data] collecting {N_TRAJ - len(trajs)} more trajectories "
          f"x {args.n_steps} coarse steps...")
    # the generator is advanced once per trajectory, so extending a cache
    # reproduces exactly the trajectories a fresh run of this size would give
    for t in range(N_TRAJ):
        if t == 0 or base_force is None:
            f = base_force
        elif t < args.n_holdout or not args.field:
            # the held-out trajectories keep the uniform impulse they have always
            # had, down to the draw order, so [rollout] measures the same thing
            # in this run as in every run before it and the two stay comparable
            s = 0.5 * (args.impulse_range ** torch.rand(1, device=dev, generator=gen).item())
            s = s if args.impulse_range > 1 else 1.0
            f = (rand_rot() @ base_force) * s
        elif args.target_amp:
            u = torch.rand(1, device=dev, generator=gen).item()
            tgt = REF_PEAK * (args.amp_lo * ((args.amp_hi / args.amp_lo) ** u))
            f, sg, d0, k = amplitude_targeted(tgt)
            if t < args.n_holdout + 3 or (t + 1) % 25 == 0:
                print(f"    t={t}: sigma {sg / sc.sim.radius:.1f}x spacing, probe moved "
                      f"{d0:.4f}, scaled {k:.2f}x for target {tgt:.4f}", flush=True)
        else:
            f, _ = draw_impulse(sc, base_force, gen, args.impulse_range, field=True)
        if t < len(trajs):
            continue
        ps, ac_ = trajectory(f)
        trajs.append(ps); accs.append(ac_)
        if (t + 1) % 25 == 0 or t + 1 == N_TRAJ:
            print(f"  {t + 1}/{N_TRAJ}", flush=True)
    if args.traj_cache:
        torch.save({"trajs": trajs, "accs": accs, "field": bool(args.field),
                     "n_holdout": args.n_holdout,
                     "target_amp": bool(args.target_amp)}, args.traj_cache)
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
FTRAJ = None
if args.n_field_holdout > 0 and base_force is not None:
    fgen = torch.Generator(device=dev); fgen.manual_seed(5678)
    ft = []
    print(f"[data] {args.n_field_holdout} field-driven held-out trajectories", flush=True)
    for _ in range(args.n_field_holdout):
        f, sig = draw_impulse(sc, base_force, fgen, args.impulse_range, field=True)
        ps_, _ = trajectory(f)
        ft.append(ps_)
        print(f"  sigma {sig:.4f} ({sig / sc.sim.radius:.1f} anchor spacings, "
              f"{100 * sig / sc.extent:.0f}% of the object), peak "
              f"{(ps_ - AC).norm(dim=-1).max().item():.5f}", flush=True)
    fm = min(t.shape[0] for t in ft)
    FTRAJ = torch.stack([t[:fm] for t in ft])
REF, REF_A = trajs[0], (accs[0] if USE_A else None)
# one tensor, indexed on the GPU: building a batch out of a python list of
# slices was ~50 separate microsecond-scale ops per iteration, and at this model
# size that launch overhead is what the GPU waits on rather than the arithmetic
n_min = min(t.shape[0] for t in trajs)
# the held-out trajectories stay what they always were -- from rest, one
# impulse each -- so the rollout number remains comparable across every run in
# this project, whatever the training data was drawn from
HTRAJ = torch.stack([t[:n_min] for t in trajs[:HOLD]])
HACCS = torch.stack([a[:n_min] for a in accs[:HOLD]]) if USE_A else None

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
    ACCS = torch.stack([a[:s_min] for a in sa]) if USE_A else None
    BAD = torch.stack([b[:s_min] for b in sb])
    IDX = torch.tensor([(ti, k) for ti in range(TRAJ.shape[0])
                        for k in range(1, s_min - args.rollout_steps)
                        if not any(bool(BAD[ti, k + j]) for j in range(args.rollout_steps))],
                        device=dev)
    print(f"[data] {int(BAD.sum())} samples dropped as impulse-contaminated")
else:
    TRAJ = torch.stack([t[:n_min] for t in trajs])       # [n_traj, T, M, 3]
    ACCS = torch.stack([a[:n_min] for a in accs]) if USE_A else None
    IDX = torch.tensor([(ti, k) for ti in range(HOLD, TRAJ.shape[0])
                        for k in range(1, n_min - args.rollout_steps)], device=dev)
pairs = IDX
dt_coarse = args.dt_mult * sc.sub_dt
SRC = [TRAJ[i] for i in range(TRAJ.shape[0])] if args.stream_len > 0 else trajs
SRC_A = ([ACCS[i] for i in range(ACCS.shape[0])] if args.stream_len > 0 else accs) if USE_A else None
DISP = [tr[1:] - tr[:-1] for tr in SRC]
DISP_SCALE = float(torch.cat([d.norm(dim=-1).flatten() for d in DISP]).mean())
VEL_SCALE = DISP_SCALE / dt_coarse
ACC_SCALE = float(torch.cat([a.norm(dim=-1).flatten() for a in SRC_A]).mean()) if USE_A else 1.0
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
def rollout_error(frames, which=0, src=None):
    r = (HTRAJ if src is None else src)[which]
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
def rollout_score(frames, src=None):
    """mean final error over every held-out trajectory"""
    vals, mv = [], []
    for w in range((HOLD if src is None else src.shape[0])):
        n, _, rel, m = rollout_error(frames, w, src)
        vals.append(float(rel[n - 1])); mv.append(float(m))
    err = sum(vals) / len(vals)
    amp = sum(mv) / len(mv)
    # penalise a rollout for the motion it is missing as well as for where it
    # put things, so standing still cannot win. Overshoot is already punished by
    # the error term, so only the shortfall counts here.
    return err + max(0.0, 1.0 - amp), vals, amp


# DAgger pool: states the network itself reached, and what the simulator says
# happens next from each. Kept as flat tensors so a batch is one index_select.
POOL = {"p": None, "v": None, "tgt": None}


@torch.no_grad()
def collect_dagger(n_traj, k_steps):
    """roll the network out, label where it went"""
    ps, vs, tg = [], [], []
    cap = args.dagger_cap * (REF - AC).norm(dim=-1).max().item()
    order = torch.randperm(TRAJ.shape[0] - (0 if args.stream_len > 0 else HOLD),
                            device=dev)[:n_traj]
    for oi in order.tolist():
        ti = oi if args.stream_len > 0 else HOLD + oi
        r = TRAJ[ti]
        p = r[1].clone()
        v = (r[1] - r[0]) / dt_coarse
        gp = sc.skin(p, sc.pos.clone()) if USE_A else None
        for step in range(min(args.eval_frames, r.shape[0] - 1)):
            if (p - AC)[~fixed].norm(dim=-1).max() > cap:
                break
            if step % args.dagger_stride == 0:
                # what the simulator does from exactly here, k_steps of it
                q, w = p.clone(), v.clone()
                g2 = sc.skin(q, sc.pos.clone())
                fut = []
                for _ in range(k_steps):
                    q, w, g2 = sc.explicit_step(q, w, g2, args.dt_mult)
                    if not torch.isfinite(q).all():
                        break
                    fut.append(q.clone())
                if len(fut) == k_steps:
                    ps.append(p.clone()); vs.append(v.clone()); tg.append(torch.stack(fut))
            a_ = sc.elastic_accel(p, gp) if USE_A else None
            du = net(p, v, a_, dt_coarse)
            du = torch.where(fixed.unsqueeze(-1), torch.zeros_like(du), du)
            p = p + du
            v = du / dt_coarse
            if not torch.isfinite(p).all():
                break
            if USE_A:
                gp = sc.skin(p, gp)
    if not ps:
        return 0
    add = {"p": torch.stack(ps), "v": torch.stack(vs), "tgt": torch.stack(tg)}
    for key in POOL:
        POOL[key] = add[key] if POOL[key] is None else torch.cat([POOL[key], add[key]])
    return len(ps)


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
# the trajectories are drawn by now, so the generator can move off the seed they
# were built on without changing them
gen.manual_seed(1234 + 7919 * args.seed)
pbar = tqdm(range(start_it, args.iters + 1), desc="train", ncols=105, initial=start_it - 1,
             total=args.iters)
for it in pbar:
    if args.dagger and (it == 1 or it % args.dagger_every == 0):
        n_new = collect_dagger(args.dagger_traj, args.rollout_steps)
        print(f"\n  [dagger] it={it}: +{n_new} states from the network's own rollouts, "
              f"pool {0 if POOL['p'] is None else POOL['p'].shape[0]}", flush=True)
    n_pool = 0
    if args.dagger and POOL["p"] is not None:
        n_pool = min(int(round(args.batch * args.dagger_frac)), POOL["p"].shape[0])
    sel = IDX[torch.randint(IDX.shape[0], (args.batch - n_pool,), device=dev)]
    ti, k = sel[:, 0], sel[:, 1]
    p = TRAJ[ti, k]
    v = (p - TRAJ[ti, k - 1]) / dt_coarse
    a = ACCS[ti, k] if USE_A else None
    # targets for the on-trajectory part, so both sources are scored the same way
    TG = torch.stack([TRAJ[ti, k + 1 + j] for j in range(args.rollout_steps)], 1)
    if n_pool:
        pi = torch.randint(POOL["p"].shape[0], (n_pool,), device=dev)
        p = torch.cat([p, POOL["p"][pi]])
        v = torch.cat([v, POOL["v"][pi]])
        TG = torch.cat([TG, POOL["tgt"][pi]])
        if USE_A:
            a = torch.cat([a, torch.stack(
                [sc.elastic_accel(q, sc.skin(q, sc.pos.clone())) for q in POOL["p"][pi]])])
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
    # Unrolled: the network is fed its own output and scored against the true
    # trajectory at every step, with the gradient carried through the chain. At
    # rollout_steps=1 the two forms are identical -- p + du against TRAJ[k+1] is
    # the same as du against TRAJ[k+1] - p -- so nothing changes by default.
    #
    # The scale matters here as it did before: the raw-unit loss was ~5e-6, which
    # put the gradient reaching the decoder output at ~1e-9, the same order as
    # Adam's epsilon, so the term meant to make Adam scale-invariant throttled it.
    loss = 0.0
    P0 = p
    for j in range(args.rollout_steps):
        du = net(p, v, a, dt_coarse)
        du = torch.where(fixed.view(1, -1, 1), torch.zeros_like(du), du)
        p = p + du
        v = du / dt_coarse
        loss = loss + (((p - TG[:, j]) / DISP_SCALE) ** 2).mean()
        if USE_A and j + 1 < args.rollout_steps:
            # the force belongs to wherever the network has got to, and is an
            # input rather than something to differentiate through -- the same
            # convention the single-step path uses for the perturbed state
            with torch.no_grad():
                a = torch.stack([sc.elastic_accel(p[b].detach(),
                                                   sc.skin(p[b].detach(), sc.pos.clone()))
                                 for b in range(p.shape[0])])
    loss = loss / args.rollout_steps
    # the reported relative error is over the whole chain: how far it moved
    # against how far it should have
    du = p - P0
    target = TG[:, -1] - P0
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
        fs = fv = fa = None
        if FTRAJ is not None:
            fs, fv, fa = rollout_score(args.eval_frames, FTRAJ)
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
        if fs is not None:
            # the uniform-impulse set above cannot see whether the network can
            # be pushed locally, which is what this project is ultimately for
            print(f"  [field]   it={it} over {len(fv)} field-driven trajectories: "
                  f"err {100*sum(fv)/len(fv):.1f}%  motion {100*fa:.0f}%  "
                  f"spread {100*min(fv):.1f}-{100*max(fv):.1f}%", flush=True)

print("\n[result]   iter        step MSE   rel step err")
for it, ls, rl in hist_log[::max(1, len(hist_log) // 20)]:
    print(f"  {it:8d}   {ls:.6e}   {rl:10.4f}")

score, vals, amp = rollout_score(args.eval_frames)
n, err, rel, _ = rollout_error(args.eval_frames)
if FTRAJ is not None:
    fs, fv, fa = rollout_score(args.eval_frames, FTRAJ)
    print(f"\n[field] mean over {len(fv)} field-driven held-out trajectories: "
          f"err {100*sum(fv)/len(fv):.2f}%, motion {100*fa:.0f}%  (per trajectory: "
          f"{', '.join(f'{100*v:.1f}%' for v in fv)})")
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

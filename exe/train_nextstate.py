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
ap.add_argument("--out", default=None)
ap.add_argument("--ckpt_every", type=int, default=2000,
                 help="write a resumable checkpoint this often. These instances stop on an "
                      "idle watchdog and can be reclaimed at any time; a run that can only "
                      "save at the end loses everything when that happens.")
ap.add_argument("--resume", default=None,
                 help="resume from a checkpoint written by --out (model, optimiser, iteration "
                      "and RNG). The collected trajectories are regenerated, not stored -- "
                      "they are deterministic given the seed and take seconds.")
args = ap.parse_args()

dev = "cuda"
torch.manual_seed(0)
sc = scene_setup.build(args.ply, args.config, args.n_anchors, args.K, device=dev)
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


print(f"[data] {args.n_traj} trajectories x {args.n_steps} coarse steps...")
trajs, accs = [], []
for t in tqdm(range(args.n_traj), desc="collect", ncols=90):
    if t == 0 or base_force is None:
        f = base_force
    else:
        s = 0.5 * (4.0 ** torch.rand(1, device=dev, generator=gen).item())
        f = (rand_rot() @ base_force) * s
    ps, ac_ = trajectory(f)
    trajs.append(ps); accs.append(ac_)
REF, REF_A = trajs[0], accs[0]              # the config's own run, for evaluation

# steps with a previous position behind them (for velocity) and a target ahead
pairs = [(ti, k) for ti, tr in enumerate(trajs) for k in range(1, tr.shape[0] - 1)]
dt_coarse = args.dt_mult * sc.sub_dt
DISP = [tr[1:] - tr[:-1] for tr in trajs]
DISP_SCALE = float(torch.cat([d.norm(dim=-1).flatten() for d in DISP]).mean())
VEL_SCALE = DISP_SCALE / dt_coarse
ACC_SCALE = float(torch.cat([a.norm(dim=-1).flatten() for a in accs]).mean())
print(f"[data] {len(pairs)} training pairs; typical coarse displacement = {DISP_SCALE:.5f}, "
      f"velocity scale = {VEL_SCALE:.4f}, elastic accel scale = {ACC_SCALE:.2f}")
print(f"[data] reference peak anchor displacement = "
      f"{(REF - AC).norm(dim=-1).max().item():.5f}")

net = NextStep(args.hidden, args.depth, args.heads, DISP_SCALE, VEL_SCALE,
                ACC_SCALE).to(dev)
opt = torch.optim.Adam(net.parameters(), lr=args.lr)
print(f"[setup] params: {sum(q.numel() for q in net.parameters())/1e3:.1f}k")


@torch.no_grad()
def rollout(frames):
    """autoregressive from the reference's initial condition; returns positions.

    The elastic acceleration is re-evaluated every step from the network's own
    anchors, via the Gaussian cloud it has skinned -- the same call the training
    data was built with.
    """
    p = REF[1].clone()
    v = (REF[1] - REF[0]) / dt_coarse
    gp = sc.skin(p, sc.pos.clone())
    out = [p.clone()]
    for _ in range(frames):
        a = sc.elastic_accel(p, gp)
        du = net(p, v, a, dt_coarse)
        du = torch.where(fixed.unsqueeze(-1), torch.zeros_like(du), du)
        p = p + du
        v = du / dt_coarse
        if not torch.isfinite(p).all():
            break
        gp = sc.skin(p, gp)
        out.append(p.clone())
    return torch.stack(out)


@torch.no_grad()
def rollout_error(frames):
    got = rollout(frames)
    n = min(got.shape[0], REF.shape[0] - 1)
    ref = REF[1:1 + n]
    err = (got[:n] - ref).norm(dim=-1).mean(-1)          # mean anchor error per frame
    span = (ref - AC).norm(dim=-1).max().clamp(min=1e-12)
    return n, err, (err / span)


hist_log = []
start_it = 1


def save(path, it):
    torch.save({"model": net.state_dict(), "opt": opt.state_dict(), "iter": it,
                "args": vars(args), "hist": hist_log, "disp_scale": DISP_SCALE,
                "vel_scale": VEL_SCALE, "acc_scale": ACC_SCALE,
                "rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(),
                "gen": gen.get_state()}, path)


if args.resume and os.path.exists(args.resume):
    ck = torch.load(args.resume, map_location=dev, weights_only=False)
    net.load_state_dict(ck["model"])
    opt.load_state_dict(ck["opt"])
    hist_log = ck.get("hist", [])
    start_it = ck["iter"] + 1
    torch.set_rng_state(ck["rng"].cpu())
    torch.cuda.set_rng_state(ck["cuda_rng"].cpu())
    gen.set_state(ck["gen"].cpu())
    print(f"[resume] {args.resume} at iter {ck['iter']}")

pbar = tqdm(range(start_it, args.iters + 1), desc="train", ncols=105, initial=start_it - 1,
             total=args.iters)
for it in pbar:
    idx = torch.randint(len(pairs), (args.batch,))
    sel = [pairs[i] for i in idx.tolist()]
    p = torch.stack([trajs[ti][k] for ti, k in sel])
    v = torch.stack([(trajs[ti][k] - trajs[ti][k - 1]) / dt_coarse for ti, k in sel])
    a = torch.stack([accs[ti][k] for ti, k in sel])
    if args.noise > 0:
        # GNS-style: perturb the position and move the velocity consistently, so
        # the pair stays a state the network could have produced. The elastic
        # acceleration is NOT perturbed to match -- it would need the Gaussian
        # cloud at the perturbed configuration, which is not stored; the mismatch
        # is second order in the noise and is the price of not keeping 1.8 GB of
        # clouds around.
        nz = torch.randn(p.shape, device=dev, generator=gen) * (args.noise * DISP_SCALE)
        nz[:, fixed] = 0
        p = p + nz
        v = v + nz / dt_coarse
    target = torch.stack([trajs[ti][k + 1] - trajs[ti][k] for ti, k in sel])
    du = net(p, v, a, dt_coarse)
    du = torch.where(fixed.view(1, -1, 1), torch.zeros_like(du), du)
    loss = ((du - target) ** 2).mean()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    opt.step()

    if it % 50 == 0 or it == 1:
        rel = (du - target).norm() / target.norm().clamp(min=1e-20)
        hist_log.append((it, loss.item(), rel.item()))
        pbar.set_postfix(mse=f"{loss.item():.2e}", rel=f"{rel.item():.4f}")
    if args.out and (it % args.ckpt_every == 0 or it == args.iters):
        save(args.out, it)
    if it % args.eval_every == 0 or it == args.iters:
        n, err, rel = rollout_error(args.eval_frames)
        print(f"\n  [rollout] it={it} survived {n}/{args.eval_frames+1} frames; "
              f"mean anchor error vs reference: "
              f"frame10={err[min(10,n-1)]:.5f} frame{n-1}={err[n-1]:.5f} "
              f"({rel[n-1]*100:.1f}% of the reference's own motion)", flush=True)

print("\n[result]   iter        step MSE   rel step err")
for it, ls, rl in hist_log[::max(1, len(hist_log) // 20)]:
    print(f"  {it:8d}   {ls:.6e}   {rl:10.4f}")

n, err, rel = rollout_error(args.eval_frames)
print(f"\n[rollout] {n}/{args.eval_frames+1} frames survived")
print(f"  {'frame':>6} {'mean anchor err':>16} {'% of reference motion':>22}")
for i in range(0, n, max(1, n // 12)):
    print(f"  {i:6d} {err[i]:16.5f} {rel[i]*100:21.2f}%")

if args.out:
    save(args.out, args.iters)
    print(f"[save] {args.out}")

"""Fit a coarse MPM to the fine one, at the anchor set's own budget.

The chain's middle stage has been an anchor simulator: a few hundred anchors
carrying positions and velocities, with every Gaussian's deformation gradient
rebuilt from the arrangement each step. Fitted, it reaches 6.61% of MPM on the
field impulses, and a floor under that was measured directly -- branch MPM with
identical positions and velocities but the anchor-reconstructed F and the
trajectories separate by 2.6-5.0% within thirty frames. F is the one thing MPM
carries forward and that state cannot.

A coarse MPM has the same state and accumulates F the same way. What it gives up
is resolution, not a kind of information, so the question it faces is a different
one -- and this asks it at the same budget: 512 particles against 588 anchors,
the same impulses, the same ruler.

What is fitted, and what is not:

  volume     what each coarse particle stands for. It scales that particle's
             contribution to the grid's momentum through the stress.
  stiffness  a per-particle multiplier on mu and lam. The config's E, nu and
             density are untouched and keep their spatial pattern, exactly as
             the anchor fit leaves them alone.
  position   an offset on the rest position, which moves both where the particle
             starts and the offsets the Gaussians are skinned from.

  mass       held fixed. It is density times volume physically, but the adjoint
             does not carry it, and letting volume move while mass does not is a
             clearer knob than an inconsistent pair.

Skinning uses each coarse particle's own F -- x = sum_i w_i [x_i + F_i (X - X_i)]
-- because carrying F is the whole point of preferring this to shape matching.
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

ap = argparse.ArgumentParser()
ap.add_argument("--ply", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--dreamphysics", default="/workspace/DreamPhysics")
ap.add_argument("--traj_cache", required=True, help="the cache carrying MPM's F and C")
ap.add_argument("--n_coarse", type=int, default=512)
ap.add_argument("--n_grid", type=int, default=0,
                 help="grid resolution. Zero picks one from the particle count: MPM "
                      "needs several particles per cell, and 512 particles on the "
                      "reference's 100^3 leaves 0.075 of them per active cell, which "
                      "diverges inside a rollout before anything is fitted.")
ap.add_argument("--per_cell", type=float, default=4.0,
                 help="target particles per occupied cell, when --n_grid is picked")
ap.add_argument("--K", type=int, default=8, help="coarse particles each Gaussian follows")
ap.add_argument("--unroll", type=int, default=12)
ap.add_argument("--dt_mult", type=int, default=40)
ap.add_argument("--frames", type=int, default=60)
ap.add_argument("--batch", type=int, default=2)
ap.add_argument("--iters", type=int, default=400)
ap.add_argument("--lr_vol", type=float, default=3e-2)
ap.add_argument("--lr_stiff", type=float, default=3e-2)
ap.add_argument("--lr_pos", type=float, default=3e-4)
ap.add_argument("--no_pos", action="store_true", help="hold the rest positions")
ap.add_argument("--reg", type=float, default=0.05)
ap.add_argument("--warmup", type=int, default=80)
ap.add_argument("--eval_every", type=int, default=20)
ap.add_argument("--save_every", type=int, default=10)
ap.add_argument("--n_anchors", type=int, default=512)
ap.add_argument("--eig_floor", type=float, default=0.02)
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--out", default=None)
ap.add_argument("--state", default=None)
ap.add_argument("--resume", action="store_true")
ap.add_argument("--r2", default=None)
args = ap.parse_args()

sys.path.insert(0, args.dreamphysics)
import warp as wp

import mpmstep
from anchorflow.anchor_sparse import Traj
from anchorflow.mpm_teacher import MPMTeacher

dev = "cuda"
wp.init()
torch.manual_seed(args.seed)
sc = scene_setup.build(args.ply, args.config, args.n_anchors, 8, device=dev,
                        frozen_weights=True, rot_fallback=True,
                        eig_floor=args.eig_floor)
if not args.n_grid:
    # the object fills a fraction of the domain, so size the grid from how many
    # cells the particles would occupy rather than from the domain
    _T = MPMTeacher(sc, n_grid=100)
    _x = _T.pos_m
    _ext = float((_x.max(0).values - _x.min(0).values).max())
    _cells = max(1.0, args.n_coarse / max(args.per_cell, 1e-6))
    _h = _ext / max(_cells ** (1.0 / 3.0), 1.0)
    args.n_grid = max(8, int(round(2.0 / _h)))
    del _T
T = MPMTeacher(sc, n_grid=args.n_grid)
mat = T.mat
Xfull = T.pos_m                       # canonical material positions
Nm = Xfull.shape[0]

g = torch.Generator(device=dev).manual_seed(args.seed)
sel = torch.randperm(Nm, device=dev, generator=g)[: args.n_coarse].sort().values
X0 = Xfull[sel].contiguous()
vol0 = (T.vol_m[sel] * (float(T.vol_m.sum()) / float(T.vol_m[sel].sum()))).contiguous()

# the reference's own moduli and mass for these particles, so the starting point
# is the coarse simulator with nothing fitted
from mpm_solver_warp.mpm_solver_warp import MPM_Simulator_WARP

_c = MPM_Simulator_WARP(10)
_c.load_initial_data_from_torch(X0, vol0, torch.zeros((args.n_coarse, 6), device=dev),
                                 n_grid=args.n_grid, grid_lim=T.grid_lim)
_mp = {k: sc.cfg[k] for k in ("E", "nu", "density", "material") if k in sc.cfg}
_mp.update({"n_grid": args.n_grid, "grid_lim": T.grid_lim, "g": sc.cfg.get("g", [0, 0, 0]),
            "grid_v_damping_scale": sc.cfg.get("grid_v_damping_scale", 1.0)})
if "additional_material_params" in sc.cfg:
    _mp["additional_material_params"] = sc.cfg["additional_material_params"]
_c.set_parameters_dict(_mp)
_c.finalize_mu_lam()
MU0 = torch.from_numpy(_c.mpm_model.mu.numpy()).to(dev).contiguous()
LAM0 = torch.from_numpy(_c.mpm_model.lam.numpy()).to(dev).contiguous()
MASS = torch.from_numpy(_c.mpm_state.particle_mass.numpy()).to(dev).contiguous()
del _c

DX = float(T.grid_lim) / args.n_grid
GRAV = torch.tensor(sc.cfg.get("g", [0.0, 0.0, 0.0]), device=dev, dtype=torch.float32)
FIXED = torch.zeros(0, dtype=torch.uint8, device=dev)
DAMP = float(sc.cfg.get("grid_v_damping_scale", 1.0))
DAMP = DAMP if DAMP < 1.0 else 1.0
print(f"[setup] {args.n_coarse} coarse particles for {Nm} material, "
      f"{args.n_grid}^3 grid, damping {DAMP}")
print(f"        mu {float(MU0.min()):.4g}..{float(MU0.max()):.4g}, "
      f"volume {float(vol0.median()):.4g}")

# ---- what is fitted ---------------------------------------------------------
log_vol = torch.zeros(args.n_coarse, device=dev, requires_grad=True)
log_k = torch.zeros(args.n_coarse, device=dev, requires_grad=True)
dpos = torch.zeros(args.n_coarse, 3, device=dev, requires_grad=not args.no_pos)
TRAIN = {"log_vol": log_vol, "log_k": log_k}
if not args.no_pos:
    TRAIN["dpos"] = dpos
opt = torch.optim.Adam([{"params": [log_vol], "lr": args.lr_vol},
                        {"params": [log_k], "lr": args.lr_stiff}]
                       + ([{"params": [dpos], "lr": args.lr_pos}] if not args.no_pos else []))
print(f"[setup] fitting {sum(t.numel() for t in TRAIN.values())} parameters: "
      f"{', '.join(TRAIN)}")

# ---- skinning, fixed at the canonical configuration -------------------------
from scipy.spatial import cKDTree

_tree = cKDTree(X0.cpu().numpy())
_d, _i = _tree.query(Xfull.cpu().numpy(), k=args.K)
NN = torch.from_numpy(_i).long().to(dev)
_w = 1.0 / torch.from_numpy(_d).float().to(dev).clamp(min=1e-6)
W = (_w / _w.sum(-1, keepdim=True)).contiguous()


def skin(xc, Fc, rest):
    """the full cloud from the coarse state, through each particle's own F"""
    off = Xfull.unsqueeze(1) - rest[NN]
    moved = xc[NN] + torch.einsum("nkij,nkj->nki", Fc[NN], off)
    return (W.unsqueeze(-1) * moved).sum(1)


# ---- the reference trajectories --------------------------------------------
blob = torch.load(args.traj_cache, map_location=dev, weights_only=False)
FIT = [e for e in blob["fit"] if len(e) == 4]
CHK = [e for e in blob["chk"] if len(e) == 4] or FIT[:4]
print(f"[data] {len(FIT)} trajectories carrying MPM's F and C, {len(CHK)} held out")
if not FIT:
    raise SystemExit("the cache has no F/C -- regenerate it with --n_fine > 0")


def coarse_roll(state, n_frames, grad=True):
    """n coarse frames of the fitted simulator, as (positions, F) per frame"""
    x, v, C, F = state
    vol = vol0 * log_vol.exp()
    k = log_k.exp()
    rest = X0 + dpos
    out = []
    for _ in range(n_frames):
        x, v, C, F = mpmstep.rollout(x, v, C, F, vol, MASS, MU0 * k, LAM0 * k,
                                      GRAV, FIXED, args.n_grid, DX, sc.sub_dt,
                                      args.dt_mult, damp=DAMP,
                                      checkpoint=grad)
        out.append((x, F.reshape(-1, 3, 3)))
    return out, rest


def sample_loss(entry, t, n):
    """from MPM's own state at frame t, scored against MPM at every frame after"""
    X, V, Fm, Cm = entry
    st = (X[t][sel].contiguous(), V[t][sel].contiguous(),
          Cm[t][sel].contiguous(), Fm[t][sel].contiguous())
    hi = min(n, X.shape[0] - 1 - t)
    if hi < 1:
        return None
    frames, rest = coarse_roll(st, hi)
    loss = 0.0
    for j, (xc, Fc) in enumerate(frames):
        tgt = X[t + j + 1]
        d = (tgt - X[t]).norm(dim=-1).mean().clamp(min=1e-12)
        loss = loss + (skin(xc, Fc, rest) - tgt).norm(dim=-1).mean() / d
    return loss / hi


@torch.no_grad()
def rollout_error(sets, frames=None):
    """the number this is for: from rest, all the way, against MPM's particles"""
    frames = frames or args.frames
    tot = []
    for X, V, Fm, Cm in sets:
        st = (X[0][sel].contiguous(), V[0][sel].contiguous(),
              Cm[0][sel].contiguous(), Fm[0][sel].contiguous())
        fr, rest = coarse_roll(st, frames, grad=False)
        span = (X[frames] - X[0]).norm(dim=-1).max().clamp(min=1e-12)
        e = []
        for j, (xc, Fc) in enumerate(fr):
            if not torch.isfinite(xc).all():
                e = [float("inf")]
                break
            e.append(float((skin(xc, Fc, rest) - X[j + 1]).norm(dim=-1).mean() / span))
        tot.append(sum(e) / len(e))
    return sum(tot) / max(len(tot), 1)


STATE = args.state or (args.out + ".state" if args.out else None)
start_it, best = 1, None
if args.resume and STATE and os.path.exists(STATE):
    b = torch.load(STATE, map_location=dev, weights_only=False)
    with torch.no_grad():
        log_vol.copy_(b["log_vol"]); log_k.copy_(b["log_k"]); dpos.copy_(b["dpos"])
    opt.load_state_dict(b["opt"]); start_it, best = b["iter"] + 1, b["best"]
    print(f"[resume] {STATE} at iteration {b['iter']}, best {b['best']}")

if best is None:
    best = rollout_error(CHK)
    print(f"[init] rollout {100 * best:.2f}%  (nothing fitted yet)", flush=True)

bar = tqdm(range(start_it, args.iters + 1), desc="fit", ncols=95, initial=start_it - 1,
            total=args.iters)
for it in bar:
    frac = min(1.0, it / max(args.warmup, 1))
    hi = max(2, int(frac * (args.frames - 1 - args.unroll)))
    loss, n_ok = 0.0, 0
    for _ in range(args.batch):
        e = FIT[torch.randint(len(FIT), (1,)).item()]
        t = torch.randint(max(hi, 1), (1,)).item()
        c = sample_loss(e, t, args.unroll)
        if c is None or not torch.isfinite(c):
            continue
        loss = loss + c; n_ok += 1
    if n_ok == 0:
        continue
    loss = loss / n_ok
    total = loss
    if args.reg > 0:
        total = total + args.reg * (log_vol.pow(2).mean() + log_k.pow(2).mean()
                                     + (dpos / sc.sim.radius).pow(2).mean())
    opt.zero_grad(set_to_none=True)
    total.backward()
    if all(t.grad is None or torch.isfinite(t.grad).all() for t in TRAIN.values()):
        torch.nn.utils.clip_grad_norm_(list(TRAIN.values()), 1.0)
        opt.step()
    bar.set_postfix(loss=f"{float(loss):.3f}",
                    vol=f"{float(log_vol.exp().median()):.2f}",
                    k=f"{float(log_k.exp().median()):.2f}")
    if it % args.eval_every == 0 or it == args.iters:
        r = rollout_error(CHK)
        print(f"\n  [it {it}] rollout {100 * r:.2f}%   volume "
              f"{float(log_vol.exp().min()):.2f}..{float(log_vol.exp().max()):.2f}   "
              f"stiffness {float(log_k.exp().min()):.2f}..{float(log_k.exp().max()):.2f}",
              flush=True)
        if args.out and r < best:
            best = r
            torch.save({"sel": sel.cpu(), "log_vol": log_vol.detach().cpu(),
                         "log_k": log_k.detach().cpu(), "dpos": dpos.detach().cpu(),
                         "iter": it, "score": r, "args": vars(args)}, args.out)
    if STATE and (it % args.save_every == 0 or it == args.iters):
        torch.save({"log_vol": log_vol.detach(), "log_k": log_k.detach(),
                     "dpos": dpos.detach(), "opt": opt.state_dict(),
                     "iter": it, "best": best, "args": vars(args)}, STATE + ".tmp")
        os.replace(STATE + ".tmp", STATE)
        if args.r2:
            subprocess.Popen(["rclone", "copy", STATE, args.r2],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

print(f"\n[done] best rollout {100 * best:.2f}%")

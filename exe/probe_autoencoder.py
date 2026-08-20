"""What could a LEARNED code carry that shape matching cannot?

The measured wall is not the force law. Given MPM's own deformation gradient the
existing stress model and gather reproduce MPM's acceleration (36.8 against a
measured 27-47); given the one shape matching derives from anchor positions they
are off by two orders. That F recovers only 17% of MPM's deviation from identity
is the whole gap (exe/probe_force_recon.py).

So before building any latent dynamics, this asks the only question that decides
whether the idea is worth building: with the SAME anchors and the SAME
connectivity, but an encoder and decoder that are learned rather than derived,
how much of F comes back?

  MPM (x, F) --encoder--> z [M, d] --decoder--> (x, F)

No dynamics, no rollout, no rendering. The encoder aggregates onto anchors with
the existing weights and the decoder reads back through the same pairs, so the
comparison against the current scheme is like for like: what changes is only
what the anchors are allowed to carry, and how it is read.

d is swept, because "learned" and "wider" are two different claims and the
current scheme is d = 3 (a position, and nothing else).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

_lib = os.path.join(os.path.dirname(__file__), "..", "lib")
sys.path.insert(0, _lib)

import torch
import torch.nn as nn
from tqdm import tqdm

from anchorflow import scene_setup

ap = argparse.ArgumentParser()
ap.add_argument("--ply", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--dreamphysics", default="/workspace/DreamPhysics")
ap.add_argument("--ckpt", default=None, help="anchors to use; default the sampled set")
ap.add_argument("--n_anchors", type=int, default=512)
ap.add_argument("--K", type=int, default=8)
ap.add_argument("--c", type=float, default=0.25)
ap.add_argument("--eig_floor", type=float, default=0.02)
ap.add_argument("--n_traj", type=int, default=20)
ap.add_argument("--n_check", type=int, default=4)
ap.add_argument("--frames", type=int, default=60)
ap.add_argument("--dt_mult", type=int, default=40)
ap.add_argument("--every", type=int, default=5)
ap.add_argument("--impulse_range", type=float, default=10.0)
ap.add_argument("--dims", default="3,8,16,32")
ap.add_argument("--hidden", type=int, default=128)
ap.add_argument("--w_F", type=float, default=1.0,
                 help="how much the code is asked to spend on F rather than on x. "
                      "At 1.0 it gives away the position accuracy shape matching "
                      "already had; the question here is about F.")
ap.add_argument("--steps", type=int, default=1500)
ap.add_argument("--lr", type=float, default=3e-3)
ap.add_argument("--seed", type=int, default=777)
ap.add_argument("--cache", default="/workspace/ae_states.pt")
ap.add_argument("--save", default=None, help="write the trained code here")
args = ap.parse_args()

sys.path.insert(0, args.dreamphysics)

import warp as wp

from anchorflow.anchor_fit import closest_rotation, det3, inv3
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
if args.ckpt:
    b = torch.load(args.ckpt, map_location=dev, weights_only=False)
    fit._rebuild(b["pos"].to(dev), b["quat"].to(dev), b["log_s"].to(dev),
                  b["log_k"].to(dev) if "log_k" in b else None)
cache = fit.prepare()
w, rc, q, Binv, blocked, mass = (t.detach() for t in cache)
P, N, M = int(fit.pair_g.shape[0]), fit.N, fit.M
print(f"[setup] {M} anchors, {N} Gaussians, {P} pairs, spacing {sc.sim.radius:.4f}")


# ---- states -----------------------------------------------------------------
@torch.no_grad()
def collect(force):
    dv = fit.impulse_dv(force, cache)
    v0 = torch.zeros(N, 3, device=dev).index_add_(
        0, fit.pair_g, w.unsqueeze(-1) * dv[fit.pair_a])
    T._set(T.pos_m.clone(), v0.contiguous(), T.eye.clone(), torch.zeros_like(T.eye))
    got = []
    for f in range(args.frames + 1):
        if f % args.every == 0 and f > 0:
            got.append((T.solver.export_particle_x_to_torch().to(torch.float16).cpu(),
                        T.solver.export_particle_F_to_torch().reshape(-1, 9)
                        .to(torch.float16).cpu()))
        if f == args.frames:
            break
        for k in range(args.dt_mult):
            T.solver.p2g2p(None, sc.sub_dt, device=T.wp_dev)
            if (k + 1) % 8 == 0 and not T._in_domain():
                return None
    return got


if os.path.exists(args.cache):
    blob = torch.load(args.cache, map_location="cpu", weights_only=False)
    TRAIN, CHECK = blob["train"], blob["check"]
    print(f"[data] {args.cache}")
else:
    g = torch.Generator(device=dev); g.manual_seed(args.seed)
    ff = [draw_impulse(sc, base, g, args.impulse_range, field=True)[0]
          for _ in range(args.n_traj + args.n_check)]
    got = []
    for f_ in tqdm(ff, desc="MPM", ncols=90):
        o = collect(f_)
        got.append(o if o is not None else [])
    TRAIN = [s for o in got[:args.n_traj] for s in o]
    CHECK = [s for o in got[args.n_traj:] for s in o]
    torch.save({"train": TRAIN, "check": CHECK}, args.cache)
print(f"[data] {len(TRAIN)} train states, {len(CHECK)} held-out states\n")

EYE9 = torch.eye(3, device=dev).reshape(1, 9)
XS = float(torch.stack([(s[0].to(dev).float() - fit.Xc).norm(dim=-1).mean()
                        for s in TRAIN[::7]]).mean())
FS = float(torch.stack([(s[1].to(dev).float() - EYE9).norm(dim=-1).mean()
                        for s in TRAIN[::7]]).mean())
print(f"[scale] mean |x - Xc| {XS:.4e}, mean |F - I| {FS:.4e}\n")


# ---- the current scheme, for reference ---------------------------------------
@torch.no_grad()
def baseline():
    fac = fit.ls_factor(cache)
    ex = ef = 0.0
    for x_c, F_c in CHECK:
        x, Fm = x_c.to(dev).float(), F_c.to(dev).float().reshape(-1, 3, 3)
        p = fit.project_ls(x, cache, fac)
        Fa, _ = fit.deformation(p, w, rc, q, Binv, blocked)
        ex += float((fit.gaussian_pos(p, cache) - x).norm(dim=-1).mean())
        d = (Fa - Fm).reshape(-1, 9).norm(dim=-1)
        s = (Fm.reshape(-1, 9) - EYE9).norm(dim=-1).clamp(min=1e-20)
        ef += float((d / s).median())
    k = len(CHECK)
    return ex / k, 100 * ef / k


# ---- a learned code on the same graph ----------------------------------------
QN = (q / sc.sim.radius).detach()


class AE(nn.Module):
    """encode per-Gaussian state onto anchors, decode it back through the same pairs"""

    def __init__(self, d, h):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(12 + 3, h), nn.SiLU(),
                                  nn.Linear(h, h), nn.SiLU(), nn.Linear(h, d))
        self.dec = nn.Sequential(nn.Linear(d + 3, h), nn.SiLU(),
                                  nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 12))
        self.d = d

    def encode(self, x, F):
        f = torch.cat([(x - fit.Xc) / XS, (F - EYE9) / FS], -1)      # [N,12]
        e = self.enc(torch.cat([f[fit.pair_g], QN], -1))              # [P,d]
        z = torch.zeros(M, self.d, device=dev).index_add_(
            0, fit.pair_a, w.unsqueeze(-1) * e)
        den = torch.zeros(M, device=dev).index_add_(
            0, fit.pair_a, w).clamp(min=1e-12)
        return z / den.unsqueeze(-1)

    def decode(self, z):
        o = self.dec(torch.cat([z[fit.pair_a], QN], -1))              # [P,12]
        y = torch.zeros(N, 12, device=dev).index_add_(
            0, fit.pair_g, w.unsqueeze(-1) * o)
        return fit.Xc + y[:, :3] * XS, EYE9 + y[:, 3:] * FS


# ---- and what that F is worth as physics -------------------------------------
#
# An F error of 30% is a number about a tensor, not about a simulation. The
# quantity that matters is what the stress model and the gather make of it: at
# 105% they turned a true anchor acceleration of 37 into 4400. The map from F to
# force is not linear, so 30% has to be pushed through it rather than scaled.


@torch.no_grad()
def stress(F):
    """Fixed Corotated PK1, exactly as force() forms it"""
    R = closest_rotation(F, fit.polar_iters, fit.polar_ridge)
    J = det3(F)
    n = F.reshape(-1, 9).norm(dim=-1).clamp(min=1e-12).reshape(-1, 1, 1)
    Finv_T = inv3(F + 1e-6 * n * torch.eye(3, device=dev),
                   eps=1e-30).transpose(-1, -2)
    k = fit.stiffness(w).unsqueeze(-1).unsqueeze(-1)
    return 2 * (fit.mu.unsqueeze(-1).unsqueeze(-1) * k) * (F - R) + \
        (fit.lam.unsqueeze(-1).unsqueeze(-1) * k) * \
        (J - 1).unsqueeze(-1).unsqueeze(-1) * J.unsqueeze(-1).unsqueeze(-1) * Finv_T


@torch.no_grad()
def accel(F):
    """PK1 -> anchor force -> acceleration, the reduction force() ends with"""
    PB = (stress(F) @ Binv)[fit.pair_g]
    contrib = -(fit.vol[fit.pair_g] * w).unsqueeze(-1) * \
        torch.einsum("pij,pj->pi", PB, q)
    f = torch.zeros(M, 3, device=dev).index_add_(0, fit.pair_a, contrib)
    return f / mass.unsqueeze(-1)


@torch.no_grad()
def physics(net=None):
    """|a| from MPM's own F, from shape matching, and from the learned code"""
    free = ~fit.fixed
    fac = fit.ls_factor(cache)
    am = ash = ale = 0.0
    esh = ele = 0.0
    for x_c, F_c in CHECK:
        x, F = x_c.to(dev).float(), F_c.to(dev).float()
        a_m = accel(F.reshape(-1, 3, 3))
        p = fit.project_ls(x, cache, fac)
        Fsh, _ = fit.deformation(p, w, rc, q, Binv, blocked)
        a_sh = accel(Fsh)
        am += float(a_m[free].norm(dim=-1).median())
        ash += float(a_sh[free].norm(dim=-1).median())
        esh += float(((a_sh - a_m).norm(dim=-1)
                      / a_m.norm(dim=-1).clamp(min=1e-20))[free].median())
        if net is not None:
            _, Fh = net.decode(net.encode(x, F))
            a_le = accel(Fh.reshape(-1, 3, 3))
            ale += float(a_le[free].norm(dim=-1).median())
            ele += float(((a_le - a_m).norm(dim=-1)
                          / a_m.norm(dim=-1).clamp(min=1e-20))[free].median())
    k = len(CHECK)
    return am / k, ash / k, ale / k, 100 * esh / k, 100 * ele / k


def run(d):
    net = AE(d, args.hidden).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    t0 = time.time()
    bar = tqdm(range(args.steps), desc=f"d={d}", ncols=90)
    for it in bar:
        s = TRAIN[torch.randint(len(TRAIN), (1,)).item()]
        x, F = s[0].to(dev).float(), s[1].to(dev).float()
        xh, Fh = net.decode(net.encode(x, F))
        loss = ((xh - x) / XS).norm(dim=-1).mean() \
            + args.w_F * ((Fh - F) / FS).norm(dim=-1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step(); sch.step()
        if it % 50 == 0:
            bar.set_postfix(loss=f"{float(loss):.4f}")

    with torch.no_grad():
        ex = ef = 0.0
        for x_c, F_c in CHECK:
            x, F = x_c.to(dev).float(), F_c.to(dev).float()
            xh, Fh = net.decode(net.encode(x, F))
            ex += float((xh - x).norm(dim=-1).mean())
            dd = (Fh - F).norm(dim=-1)
            ss = (F - EYE9).norm(dim=-1).clamp(min=1e-20)
            ef += float((dd / ss).median())
        k = len(CHECK)
    if args.save:
        torch.save({"d": d, "hidden": args.hidden, "sd": net.state_dict(),
                    "XS": XS, "FS": FS}, f"{args.save}_d{d}.pt")
    return ex / k, 100 * ef / k, time.time() - t0, net


bx, bf = baseline()
print(f"\n{'scheme':28} {'|x err|':>10} {'F err':>9}")
print(f"{'anchors + shape matching':28} {bx:10.3e} {bf:8.2f}%   <- the wall to beat")
NETS = {}
for d in [int(t) for t in args.dims.split(",")]:
    ex, ef, dt, net = run(d)
    NETS[d] = net
    print(f"{'learned code, d=' + str(d):28} {ex:10.3e} {ef:8.2f}%   ({dt:.0f}s)")

print(f"\n=== what that F is worth: anchor acceleration ===")
print(f"{'F from':28} {'|a| median':>12} {'err vs MPM F':>14}")
for d, net in NETS.items():
    am, ash, ale, esh, ele = physics(net)
    if d == list(NETS)[0]:
        print(f"{'MPM own F (reference)':28} {am:12.1f} {'--':>14}")
        print(f"{'shape matching':28} {ash:12.1f} {esh:13.1f}%")
    print(f"{'learned code, d=' + str(d):28} {ale:12.1f} {ele:13.1f}%")

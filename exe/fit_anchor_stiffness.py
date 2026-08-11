"""Give every anchor its own effective stiffness, and fit the 512 of them to MPM.

With the material held at the config's values, tuning the discretisation
globally takes the gap to MPM from 21% to 5.2% -- but one impulse stays at 21%
whatever is done to it. A residual that survives every global setting is
mode-dependent, and a single number cannot fix a mode-dependent error.

Which is the expected shape of the problem. The effective stiffness of a coarse
element is not the material's: it depends on the local geometry, and ficus has
anchors whose eight neighbours lie almost on a line next to anchors whose
neighbours span properly. So the correction has to vary over the object.

The material stays exactly what the config says. What is fitted is a positive
per-anchor factor on top of it, interpolated to the Gaussians by the same
weights everything else uses:

    mu_i = mu_i(config) * sum_k w_ik s_[nn(i,k)]

Fitted by matching the force the anchor simulator produces at MPM's own
configurations against the acceleration MPM's trajectory implies -- one step at
a time, no backpropagation through a rollout. This project has repeatedly found
that fitting one step well does not mean the rollout follows, so the verdict is
a full rollout on impulses that were not fitted.

The gradient is not taken by autograd. The force is itself a gradient of the
energy, so differentiating a coefficient it depends on is a second-order pass
through a polar decomposition -- and near-rigid regions have F^T F close to the
identity, where the eigenvalue derivative divides by the gap between equal
eigenvalues. Measured: the first-order force already comes back NaN. The energy
is linear in the per-anchor coefficient, so the gradient is available in closed
form instead:

    E(s)   = sum_i c_i(s) e_i(q),      c = W s
    f      = -sum_i c_i de_i/dq
    dL/ds_n = -sum_i w_in (u . de_i/dq),   u_m = (2/m_m)(f_m/m_m - a_m)

and the bracket is a directional derivative of the per-Gaussian energy, which is
two evaluations of it. Three kernel calls an iteration, no autograd anywhere.
"""
from __future__ import annotations

import argparse
import os
import sys

_lib = os.path.join(os.path.dirname(__file__), "..", "lib")
sys.path.insert(0, _lib)

import torch
from tqdm import tqdm

from anchorflow import scene_setup, anchor_mpm
from anchorflow.streams import rand_rot

ap = argparse.ArgumentParser()
ap.add_argument("--ply", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--dreamphysics", default="/workspace/DreamPhysics")
ap.add_argument("--frames", type=int, default=60)
ap.add_argument("--dt_mult", type=int, default=40)
ap.add_argument("--n_anchors", type=int, default=512)
ap.add_argument("--K", type=int, default=8)
ap.add_argument("--n_grid", type=int, default=100)
ap.add_argument("--grid_lim", type=float, default=2.0)
ap.add_argument("--eig_floor", type=float, default=0.02,
                 help="left at a small non-zero value: the floor decides where deformation "
                      "is forbidden for lack of data, which no stiffness factor can imitate, "
                      "and setting it to zero made 2048 anchors diverge")
ap.add_argument("--n_fit", type=int, default=3, help="impulses fitted on")
ap.add_argument("--n_check", type=int, default=3, help="impulses checked on")
ap.add_argument("--iters", type=int, default=400)
ap.add_argument("--lr", type=float, default=0.02)
ap.add_argument("--rot_fallback", type=int, default=1,
                 help="rotate the unspanned directions rather than freeze them; this alone "
                      "takes the gap to MPM from 21% to 5.5%, and the fit starts from there")
ap.add_argument("--out", default=None)
args = ap.parse_args()

sys.path.insert(0, args.dreamphysics)
import warp as wp
from mpm_solver_warp.mpm_solver_warp import MPM_Simulator_WARP

dev = "cuda"
torch.manual_seed(0)
wp.init()

sc = scene_setup.build(args.ply, args.config, args.n_anchors, args.K,
                        n_grid=args.n_grid, grid_lim=args.grid_lim, device=dev,
                        frozen_weights=True, eig_floor=args.eig_floor,
                        rot_fallback=bool(args.rot_fallback))
AC, fixed, keep = sc.anchor_canonical, sc.fixed_mask, sc.keep
cfg, sim = sc.cfg, sc.sim
mat = torch.nonzero(keep, as_tuple=False).squeeze(-1)
pos_m, vol_m = sc.pos[mat].contiguous(), sc.volume[mat].contiguous()
base_force = None
for bc in cfg.get("boundary_conditions", []):
    if bc["type"] == "particle_impulse":
        base_force = torch.tensor(bc["force"], device=dev)
MU0, LAM0 = sc.mu.clone(), sc.lam.clone()
W0 = sim._canonical_weights()
dt = args.dt_mult * sc.sub_dt
print(f"[setup] {sc.M} anchors, eig floor {args.eig_floor}, material from the config")


def mpm_run(force):
    solver = MPM_Simulator_WARP(10)
    solver.load_initial_data_from_torch(
        pos_m, vol_m, torch.zeros((mat.shape[0], 6), device=dev),
        n_grid=args.n_grid, grid_lim=args.grid_lim)
    mp = {k: cfg[k] for k in ("E", "nu", "density", "material") if k in cfg}
    mp.update({"n_grid": args.n_grid, "grid_lim": args.grid_lim,
               "g": cfg.get("g", [0, 0, 0]),
               "grid_v_damping_scale": cfg.get("grid_v_damping_scale", 1.0)})
    if "additional_material_params" in cfg:
        mp["additional_material_params"] = cfg["additional_material_params"]
    solver.set_parameters_dict(mp)
    solver.finalize_mu_lam()
    for bc in cfg.get("boundary_conditions", []):
        if bc["type"] == "cuboid":
            solver.set_velocity_on_cuboid(bc["point"], bc["size"], [0.0, 0.0, 0.0],
                                           start_time=0.0, end_time=999.0, reset=1)
    dv = sc.impulse_dv(force)
    v0 = (W0[mat].unsqueeze(-1) * dv[sim.nn_idx[mat]]).sum(1).contiguous()
    solver.import_particle_v_from_torch(v0)
    out = [pos_m.clone()]
    for _ in range(args.frames):
        for _ in range(args.dt_mult):
            solver.p2g2p(None, sc.sub_dt, device=dev)
        out.append(solver.export_particle_x_to_torch().clone())
    return torch.stack(out)


# MPM particles -> anchor positions, the same reduction the projection floor used
wm = W0[mat]
den = torch.zeros(sc.M, device=dev).index_add_(
    0, sim.nn_idx[mat].reshape(-1), wm.reshape(-1)).clamp(min=1e-12)


def to_anchors(X):
    num = torch.zeros(X.shape[0], sc.M, 3, device=dev)
    for t in range(X.shape[0]):
        num[t] = torch.zeros(sc.M, 3, device=dev).index_add_(
            0, sim.nn_idx[mat].reshape(-1),
            (wm.unsqueeze(-1) * X[t].unsqueeze(1)).reshape(-1, 3))
    p = num / den.view(1, -1, 1)
    return torch.where(fixed.view(1, -1, 1), AC.unsqueeze(0), p)


import anchorstep


def c_of(s):
    """the per-Gaussian factor a per-anchor one implies"""
    return (W0 * s[sim.nn_idx]).sum(-1).clamp(min=1e-4)


def force_at(p, c):
    """elastic force at p, with the config material scaled by c"""
    f, _, _, _ = anchorstep.fused_energy_force(
        sim.gaussian_canonical, sc.pos, p, AC, sim.nn_idx, sc.volume,
        sim.radius, MU0 * c, LAM0 * c, eig_floor_frac=sim.eig_floor,
        w_in=sim.frozen_w, rot_fallback=sim.rot_fallback)
    return f


def energy_per_gaussian(p):
    """e_i(q) at the config material -- the c-independent factor"""
    _, _, _, psi = anchorstep.fused_energy_force(
        sim.gaussian_canonical, sc.pos, p, AC, sim.nn_idx, sc.volume,
        sim.radius, MU0, LAM0, eig_floor_frac=sim.eig_floor,
        w_in=sim.frozen_w, rot_fallback=sim.rot_fallback)
    return sc.volume * psi


gen = torch.Generator(device=dev); gen.manual_seed(7777)


def impulses(n, g):
    out = [base_force]
    for _ in range(n - 1):
        k = 0.5 * (4.0 ** torch.rand(1, device=dev, generator=g).item())
        out.append((rand_rot(g, dev) @ base_force) * k)
    return out


FIT_F = impulses(args.n_fit, gen)
CHK_F = impulses(args.n_check + 1, torch.Generator(device=dev).manual_seed(20260811))[1:]

P, A = [], []
for f in tqdm(FIT_F, desc="MPM (fit)", ncols=90):
    X = mpm_run(f)
    pa = to_anchors(X)
    # the acceleration MPM's own trajectory implies at each anchor
    P.append(pa[1:-1])
    A.append((pa[2:] - 2 * pa[1:-1] + pa[:-2]) / dt ** 2)
P = torch.cat(P); A = torch.cat(A)
mv = ~fixed
print(f"[fit] {P.shape[0]} configurations, |a| mean {A[:, mv].norm(dim=-1).mean():.2f}")

S = torch.ones(sc.M, device=dev)
opt = torch.optim.Adam([S], lr=args.lr)
S.requires_grad_(False)
scale = A[:, mv].norm(dim=-1).mean().clamp(min=1e-9)
nmv = int(mv.sum())
bar = tqdm(range(args.iters), desc="fit", ncols=90)
for it in bar:
    i = torch.randint(P.shape[0], (4,), device=dev)
    c = c_of(S)
    g_acc, total = torch.zeros_like(S), 0.0
    for j in i.tolist():
        p = P[j]
        r = force_at(p, c) / sc.mass.unsqueeze(-1) - A[j]
        r = torch.where(mv.unsqueeze(-1), r, torch.zeros_like(r))
        total += ((r / scale) ** 2).sum().item() / (3 * nmv * i.shape[0])
        u = (2.0 / (3 * nmv * i.shape[0])) * r / (sc.mass.unsqueeze(-1) * scale ** 2)
        # directional derivative of each Gaussian's own energy along u
        eps = 1e-4 * p[mv].norm(dim=-1).mean() / u.norm(dim=-1).max().clamp(min=1e-20)
        h = (energy_per_gaussian(p + eps * u) - energy_per_gaussian(p - eps * u)) / (2 * eps)
        g_acc -= torch.zeros(sc.M, device=dev).index_add_(
            0, sim.nn_idx.reshape(-1), (W0 * h.unsqueeze(-1)).reshape(-1))
    S.grad = g_acc
    opt.step()
    with torch.no_grad():
        S.clamp_(0.02, 50.0)
    if it % 25 == 0:
        bar.set_postfix(loss=f"{total:.4f}", s=f"{S.min().item():.2f}-{S.max().item():.2f}")
S = S.detach()
print(f"\n[fitted] per-anchor factor: min {S.min():.3f}, median {S.median():.3f}, "
      f"max {S.max():.3f}")
if args.out:
    torch.save({"s": S.cpu(), "eig_floor": args.eig_floor,
                 "n_anchors": args.n_anchors, "K": args.K}, args.out)
    print(f"[fitted] saved to {args.out}")


def rollout_err(force, s_or_none):
    old_mu, old_lam = sc.mu, sc.lam
    if s_or_none is not None:
        c_ = c_of(s_or_none)
        sc.mu, sc.lam = MU0 * c_, LAM0 * c_
    with torch.no_grad():
        p, v, gp = AC.clone(), sc.initial_velocity(force), sc.pos.clone()
        out = [gp[mat].clone()]
        for _ in range(args.frames):
            p, v, gp = sc.explicit_step(p, v, gp, args.dt_mult)
            if not torch.isfinite(p).all():
                sc.mu, sc.lam = old_mu, old_lam
                return None, None
            out.append(gp[mat].clone())
    sc.mu, sc.lam = old_mu, old_lam
    return torch.stack(out), None


def score(Aa, M):
    span = (M - M[0]).norm(dim=-1).max().clamp(min=1e-12)
    return (100 * ((Aa - M).norm(dim=-1).mean(-1) / span).mean().item(),
            100 * ((Aa - Aa[0]).norm(dim=-1).max() / span).item())


print(f"\n{'impulse':>10} {'uniform':>10} {'per-anchor':>12} {'motion vs MPM':>15}")
for tag, forces in (("fitted", FIT_F), ("held out", CHK_F)):
    for i, f in enumerate(forces):
        M = mpm_run(f)
        a0, _ = rollout_err(f, None)
        a1, _ = rollout_err(f, S)
        e0 = score(a0, M)[0] if a0 is not None else float("nan")
        e1, m1 = score(a1, M) if a1 is not None else (float("nan"), float("nan"))
        print(f"  {tag[:7]:>7} {i}  {e0:9.2f}% {e1:11.2f}% {m1:14.0f}%")
print(f"\n[note] 512 parameters can always be made to fit the trajectories they were "
      f"fitted on.\n       The held-out rows are the result.")

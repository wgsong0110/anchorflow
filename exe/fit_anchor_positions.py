"""Move the anchors so that 512 of them represent MPM's motion as well as possible.

The anchors are placed by sampling the material cloud, with nothing in the
choice about how well they will carry the deformation. Two things are known
about the current placement: reducing MPM's particle trajectory to anchors and
skinning it back loses 2.4%, and how ill-conditioned a Gaussian's neighbourhood
is has no relationship to where the simulator disagrees with MPM (correlation
-0.04). So this is not expected to fix the dynamics -- but 2.4% is a floor on
everything downstream, and it is a floor set by where the anchors happen to sit.

The objective is that reconstruction, which is differentiable in the anchor
positions and does not touch the polar decomposition that makes the force
non-differentiable. The scatter matrix's own eigendecomposition is fine here:
its eigenvalue ratio is 0.32 at the median and never below 0.011, so the
derivative has no near-degenerate gap to divide by.

Connectivity stays fixed. Recomputing which anchors own which Gaussian would
make the objective discontinuous, so the anchors move while keeping the
Gaussians they started with.

The reconstruction alone is not enough to optimise. It rewards packing anchors
into wherever the deformation is complicated, and unopposed it cut the nearest
anchor distance from 0.041 to 0.004 while halving the error -- at which point
the explicit step is far past its stability limit and the simulator diverges by
frame 8. So the anchors are also held apart: a hinge below the spacing they
started with, which costs nothing until they close in.
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
from anchorflow.streams import rand_rot
from eigen3x3 import eigh3x3

ap = argparse.ArgumentParser()
ap.add_argument("--ply", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--dreamphysics", default="/workspace/DreamPhysics")
ap.add_argument("--frames", type=int, default=60)
ap.add_argument("--keep_every", type=int, default=6, help="frames kept from each trajectory")
ap.add_argument("--dt_mult", type=int, default=40)
ap.add_argument("--n_anchors", type=int, default=512)
ap.add_argument("--K", type=int, default=8)
ap.add_argument("--n_grid", type=int, default=100)
ap.add_argument("--grid_lim", type=float, default=2.0)
ap.add_argument("--n_fit", type=int, default=3)
ap.add_argument("--n_check", type=int, default=3)
ap.add_argument("--iters", type=int, default=400)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--eig_floor", type=float, default=0.2)
ap.add_argument("--min_sep", type=float, default=1.0,
                 help="separation to hold, as a fraction of the closest pair in the original "
                      "placement. The explicit step limit scales with anchor spacing, so an "
                      "unconstrained fit buys reconstruction with divergence.")
ap.add_argument("--w_sep", type=float, default=1.0)
ap.add_argument("--sep_K", type=int, default=12, help="anchor neighbours the hinge watches")
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
                        frozen_weights=True, eig_floor=args.eig_floor, rot_fallback=True)
AC0, fixed, sim, cfg = sc.anchor_canonical.clone(), sc.fixed_mask, sc.sim, sc.cfg
mat = torch.nonzero(sc.keep, as_tuple=False).squeeze(-1)
pos_m, vol_m = sc.pos[mat].contiguous(), sc.volume[mat].contiguous()
idx = sim.nn_idx[mat]                       # [Nm,K] fixed connectivity
Xc = sim.gaussian_canonical[mat]            # [Nm,3] canonical Gaussian positions
R2 = 2.0 * sim.radius ** 2
base_force = None
for bc in cfg.get("boundary_conditions", []):
    if bc["type"] == "particle_impulse":
        base_force = torch.tensor(bc["force"], device=dev)
print(f"[setup] {sc.M} anchors ({int(fixed.sum())} pinned), {mat.shape[0]} material Gaussians")


def mpm_run(force):
    s = MPM_Simulator_WARP(10)
    s.load_initial_data_from_torch(pos_m, vol_m, torch.zeros((mat.shape[0], 6), device=dev),
                                   n_grid=args.n_grid, grid_lim=args.grid_lim)
    mp = {k: cfg[k] for k in ("E", "nu", "density", "material") if k in cfg}
    mp.update({"n_grid": args.n_grid, "grid_lim": args.grid_lim,
               "g": cfg.get("g", [0, 0, 0]),
               "grid_v_damping_scale": cfg.get("grid_v_damping_scale", 1.0)})
    if "additional_material_params" in cfg:
        mp["additional_material_params"] = cfg["additional_material_params"]
    s.set_parameters_dict(mp); s.finalize_mu_lam()
    for bc in cfg.get("boundary_conditions", []):
        if bc["type"] == "cuboid":
            s.set_velocity_on_cuboid(bc["point"], bc["size"], [0., 0., 0.], 0.0, 999.0, 1)
    W = sim._canonical_weights()
    dv = sc.impulse_dv(force)
    s.import_particle_v_from_torch((W[mat].unsqueeze(-1) * dv[idx]).sum(1).contiguous())
    out = []
    for t in range(args.frames):
        for _ in range(args.dt_mult):
            s.p2g2p(None, sc.sub_dt, device=dev)
        if (t + 1) % args.keep_every == 0:
            out.append(s.export_particle_x_to_torch().clone())
    return torch.stack(out)


def reconstruct(A, X):
    """MPM particles -> anchors -> Gaussians, with A as the rest configuration.

    Everything here is differentiable in A. The scatter matrix's eigendecomposition
    is well separated, and the polar decomposition -- the part that is not
    differentiable near rigid motion -- belongs to the energy and is not used.
    """
    nbr_rest = A[idx]                                        # [Nm,K,3]
    d2 = ((Xc.unsqueeze(1) - nbr_rest) ** 2).sum(-1)
    w = torch.exp(-d2 / R2) + 1e-8
    w = w / w.sum(-1, keepdim=True)                          # [Nm,K]

    # particles -> anchors, mass-free weighted average
    num = torch.zeros(A.shape[0], 3, device=dev, dtype=A.dtype).index_add_(
        0, idx.reshape(-1), (w.unsqueeze(-1) * X.unsqueeze(1)).reshape(-1, 3))
    den = torch.zeros(A.shape[0], device=dev, dtype=A.dtype).index_add_(
        0, idx.reshape(-1), w.reshape(-1)).clamp(min=1e-12)
    P = num / den.unsqueeze(-1)
    P = torch.where(fixed.unsqueeze(-1), A, P)

    # anchors -> Gaussians, the same shape-matching fit the simulator uses
    nbr_cur = P[idx]
    w_ = w.unsqueeze(-1)
    rc = (w_ * nbr_rest).sum(1)
    cc = (w_ * nbr_cur).sum(1)
    q = nbr_rest - rc.unsqueeze(1)
    p = nbr_cur - cc.unsqueeze(1)
    Am = torch.einsum("nk,nki,nkj->nij", w, p, q)
    Bm = torch.einsum("nk,nki,nkj->nij", w, q, q)
    ev, evec = eigh3x3(Bm)
    lmax = ev[..., -1:].clamp(min=1e-12)
    well = ev > args.eig_floor * lmax
    Av = Am @ evec
    Fv = torch.where(well.unsqueeze(-2), Av / ev.clamp(min=1e-12).unsqueeze(-2), evec)
    F = Fv @ evec.transpose(-1, -2)
    return cc + torch.einsum("nij,nj->ni", F, Xc - rc)


# who each anchor is held apart from, fixed like the Gaussian connectivity so
# the penalty is smooth. A wider list than the anchors can plausibly reach, so
# a pair that closes in is already being watched before it gets close.
with torch.no_grad():
    D0 = torch.cdist(AC0, AC0)
    D0.fill_diagonal_(float("inf"))
    sep_idx = D0.topk(args.sep_K, dim=1, largest=False).indices
    SEP = args.min_sep * D0.min().item()
print(f"[spacing] closest pair {D0.min():.5f}, holding anchors {SEP:.5f} apart")


def sep_penalty(A):
    d = (A.unsqueeze(1) - A[sep_idx]).norm(dim=-1)
    return (SEP - d).clamp(min=0).pow(2).sum(1).mean()


gen = torch.Generator(device=dev); gen.manual_seed(7777)


def impulses(n, g):
    out = [base_force]
    for _ in range(n - 1):
        k = 0.5 * (4.0 ** torch.rand(1, device=dev, generator=g).item())
        out.append((rand_rot(g, dev) @ base_force) * k)
    return out


FIT = [mpm_run(f) for f in tqdm(impulses(args.n_fit, gen), desc="MPM (fit)", ncols=90)]
CHK = [mpm_run(f) for f in tqdm(
    impulses(args.n_check + 1, torch.Generator(device=dev).manual_seed(20260811))[1:],
    desc="MPM (check)", ncols=90)]
FITX = torch.cat(FIT); CHKX = torch.cat(CHK)
span = torch.cat([x - pos_m for x in FIT]).norm(dim=-1).max()
print(f"[data] {FITX.shape[0]} fit frames, {CHKX.shape[0]} check frames, "
      f"peak displacement {span:.5f}")


def err(A, X):
    with torch.no_grad():
        e = sum((reconstruct(A, X[j]) - X[j]).norm(dim=-1).mean().item()
                for j in range(X.shape[0]))
        return 100 * e / X.shape[0] / span.item()


A = AC0.clone().requires_grad_(True)
opt = torch.optim.Adam([A], lr=args.lr)
print(f"\n[before] fit {err(A, FITX):.2f}%   held out {err(A, CHKX):.2f}%")
bar = tqdm(range(args.iters), desc="place", ncols=90)
for it in bar:
    j = torch.randint(FITX.shape[0], (2,), device=dev)
    rec = sum(((reconstruct(A, FITX[k]) - FITX[k]) ** 2).sum(-1).mean() for k in j.tolist()) / 2
    loss = rec + args.w_sep * sep_penalty(A)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    with torch.no_grad():
        A.grad[fixed] = 0            # pinned anchors define the boundary; leave them
    opt.step()
    if it % 40 == 0:
        with torch.no_grad():
            dmin = torch.cdist(A, A).fill_diagonal_(float("inf")).min().item()
        bar.set_postfix(rec=f"{rec.item():.3e}", closest=f"{dmin:.4f}",
                        moved=f"{(A - AC0).norm(dim=-1).max().item():.4f}")
A = A.detach()
print(f"[after ] fit {err(A, FITX):.2f}%   held out {err(A, CHKX):.2f}%")
print(f"[moved ] max {(A - AC0).norm(dim=-1).max():.4f}, median "
      f"{(A - AC0).norm(dim=-1).median():.4f}  (anchor spacing {sim.radius:.4f})")
print(f"[spacing] closest pair {torch.cdist(A, A).fill_diagonal_(float('inf')).min():.5f} "
      f"(was {D0.min():.5f}); the explicit step limit scales with this")
if args.out:
    torch.save({"anchor_canonical": A.cpu(), "n_anchors": args.n_anchors, "K": args.K},
                args.out)
    print(f"[saved ] {args.out}")

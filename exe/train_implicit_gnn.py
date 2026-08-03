"""Train a GNN to predict the implicit step's displacement increment du in one
shot, replacing i-PhysGaussian's Newton-GMRES iteration, by minimizing the
within-step momentum residual (as an incremental potential -- see
lib/anchorflow/implicit.py for why the two are the same thing here).

Setup:
  * anchors carry (position, velocity, acceleration); the GNN sees v, a and the
    step size and outputs a dimensionless per-anchor correction to the
    displacement those imply: du = predictor + |predictor| * net (see
    predict_du). The decoder is zero-init so training starts exactly at
    du = predictor, i.e. the pure inertial coast an explicit step would take,
    and the first gradient it sees is the elastic force -- the right
    inductive bias.
  * loss is the incremental potential L(du), whose gradient is exactly the
    negative momentum residual -R(du).
  * the reported metric is ||R|| relative to ||R(du=predictor)||: below 1 means
    the network's correction is genuinely closer to momentum balance than the
    explicit prediction it started from.

Training states come from a real physics rollout of the same scene, so the
network is asked to take ONE big step from states the simulator actually visits.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

_lib = os.path.join(os.path.dirname(__file__), "..", "lib")
sys.path.insert(0, _lib)

import torch
import torch.nn as nn
from tqdm import tqdm

from anchorflow.anchors import AnchorSet
from anchorflow.anchor_mpm import AnchorElasticSim, lame_from_E_nu
from anchorflow.dynamics import mlp, MPNNLayer
from anchorflow import graph as G
from anchorflow.implicit import (incremental_potential, newmark_predictor,
                                  newmark_velocity, newton_reference, DuCorrector,
                                  predict_du)

ap = argparse.ArgumentParser()
ap.add_argument("--ply", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--n_anchors", type=int, default=512)
ap.add_argument("--K", type=int, default=8)
ap.add_argument("--k_graph", type=int, default=8)
ap.add_argument("--hidden", type=int, default=128)
ap.add_argument("--mp_steps", type=int, default=4)
ap.add_argument("--iters", type=int, default=2000)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--n_states", type=int, default=32, help="physics snapshots to train from")
ap.add_argument("--state_substeps", type=int, default=200, help="substeps between snapshots")
ap.add_argument("--dt_big", type=float, default=4e-3, help="the step the GNN must take in one shot")
ap.add_argument("--beta", type=float, default=0.25)
ap.add_argument("--gamma", type=float, default=0.5)
ap.add_argument("--out", default=None)
args = ap.parse_args()

from scene.gaussian_model import GaussianModel

dev = "cuda"
cfg = json.load(open(args.config))
torch.manual_seed(0)

# ---- scene -> MPM space (same convention as render_physgaussian_faithful) ----
g = GaussianModel(3, fea_dim=0)
g.load_ply(args.ply)
xyz = g.get_xyz.detach().clone(); op = g.get_opacity.detach().clone()
pos_w = xyz[op[:, 0] > cfg["opacity_threshold"]]
pmin, pmax = pos_w.min(0).values, pos_w.max(0).values
scale = 1.0 / (pmax - pmin).max()
POS = ((pos_w - (pmin + pmax) / 2) * scale + torch.tensor([1., 1., 1.], device=dev)).contiguous()
N = POS.shape[0]

ng, grid_lim = 100, 2.0
dx = grid_lim / ng
vi = (POS / dx).long().clamp(0, ng - 1)
flat = (vi[:, 0] * ng + vi[:, 1]) * ng + vi[:, 2]
cnt = torch.zeros(ng ** 3, device=dev).index_add_(0, flat, torch.ones(N, device=dev))
VOL = ((dx ** 3) / cnt[flat]).contiguous()

E_p = torch.full((N,), float(cfg["E"]), device=dev)
nu_p = torch.full((N,), float(cfg["nu"]), device=dev)
dens_p = torch.full((N,), float(cfg["density"]), device=dev)
for reg in cfg.get("additional_material_params", []):
    c = torch.tensor(reg["point"], device=dev); s = torch.tensor(reg["size"], device=dev)
    inside = ((POS - c).abs() <= s).all(-1)
    E_p[inside] = reg["E"]; nu_p[inside] = reg["nu"]; dens_p[inside] = reg["density"]
mass_p = dens_p * VOL
MU, LAM = lame_from_E_nu(E_p, nu_p)

anchor_set, _ = AnchorSet.from_gaussians(POS, node_num=args.n_anchors, latent_dim=0, e_dim=0, K=args.K)
AC = anchor_set.canonical.clone().contiguous()
M = AC.shape[0]
sim = AnchorElasticSim(POS, AC, K=args.K)
w0 = sim._weights(POS, AC)
MASS = torch.zeros(M, device=dev).index_add_(
    0, sim.nn_idx.reshape(-1), (mass_p.unsqueeze(-1) * w0).reshape(-1)).clamp(min=1e-12)

fixed_mask = torch.zeros(M, dtype=torch.bool, device=dev)
f_ext_const = None
for bc in cfg.get("boundary_conditions", []):
    if bc["type"] == "cuboid":
        c = torch.tensor(bc["point"], device=dev); s = torch.tensor(bc["size"], device=dev)
        fixed_mask |= ((AC - c).abs() <= s).all(-1)
gravity = torch.tensor(cfg["g"], dtype=torch.float32, device=dev)
print(f"[setup] N={N} M={M} pinned={int(fixed_mask.sum())} dt_big={args.dt_big} "
      f"(explicit substep in config = {cfg['substep_dt']})")

edge_index = G.knn_graph(AC, k=args.k_graph)


net = DuCorrector(args.hidden, args.mp_steps).to(dev)
opt = torch.optim.Adam(net.parameters(), lr=args.lr)
print(f"[setup] GNN params: {sum(p.numel() for p in net.parameters())/1e3:.1f}k")

# ---- collect physics states (p, v, a) the simulator actually visits ----
print("[states] running explicit physics to collect training states...")
states = []
p_c, v_c = AC.clone(), torch.zeros(M, 3, device=dev)
a_c = torch.zeros(M, 3, device=dev)
gpp = POS.clone()
sub_dt = float(cfg["substep_dt"])
# kick it into motion exactly like the config's particle_impulse does
for bc in cfg.get("boundary_conditions", []):
    if bc["type"] == "particle_impulse":
        force = torch.tensor(bc["force"], device=dev)
        wsum = torch.zeros(M, device=dev).index_add_(0, sim.nn_idx.reshape(-1), w0.reshape(-1))
        v_c = v_c + (wsum.unsqueeze(-1) * force.unsqueeze(0) * sub_dt) / MASS.unsqueeze(-1)
        v_c[fixed_mask] = 0
with torch.enable_grad():
    for i in tqdm(range(args.n_states * args.state_substeps), desc="collect", ncols=90):
        p_prev_v = v_c.clone()
        p_c, v_c, gpp, _F = sim.step(p_c, v_c, MASS, gpp, VOL, MU, LAM, sub_dt,
                                      gravity=(gravity if gravity.abs().sum() > 0 else None),
                                      damping=float(cfg.get("grid_v_damping_scale", 1.0)),
                                      fixed_mask=fixed_mask)
        a_c = (v_c - p_prev_v) / sub_dt
        if torch.isnan(p_c).any():
            print(f"[states] NaN at substep {i}; keeping {len(states)} states")
            break
        if (i + 1) % args.state_substeps == 0:
            states.append((p_c.clone(), v_c.clone(), a_c.clone(), gpp.clone()))
print(f"[states] collected {len(states)}")
assert len(states) >= 4, "not enough states"

# ---- train ----
hist = []
pbar = tqdm(range(1, args.iters + 1), desc="train", ncols=110)
for it in pbar:
    p_n, v_n, a_n, gp = states[torch.randint(len(states), (1,)).item()]
    du, pred0 = predict_du(net, p_n, v_n, a_n, args.dt_big, edge_index,
                            args.beta, fixed_mask)

    L, R = incremental_potential(du, sim, p_n, gp, VOL, MU, LAM, MASS,
                                  v_n, a_n, args.dt_big, None, args.beta, fixed_mask)
    opt.zero_grad(set_to_none=True)
    L.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    opt.step()

    if it % 25 == 0 or it == 1:
        with torch.no_grad():
            _, R0 = incremental_potential(
                pred0, sim, p_n, gp, VOL, MU, LAM, MASS, v_n, a_n, args.dt_big, None,
                args.beta, fixed_mask)
            rel = (R.norm() / R0.norm().clamp(min=1e-20)).item()
        hist.append((it, float(L), R.norm().item(), rel))
        pbar.set_postfix(L=f"{float(L):.4e}", Rrel=f"{rel:.4f}")

print("\n[result] iter      L            ||R||        ||R||/||R_explicit||")
for it, L, rn, rel in hist[::max(1, len(hist)//20)]:
    print(f"  {it:6d}  {L: .6e}  {rn: .6e}   {rel:.4f}")

# ---- compare one-shot GNN vs iterated solve on held-out states ----
print("\n[eval] one-shot GNN vs LBFGS-iterated solve of the same potential")
print(f"  {'state':>6} {'R_explicit':>12} {'R_gnn':>12} {'R_iter':>12} {'gnn/expl':>9} {'iter/expl':>9}")
for si in range(0, len(states), max(1, len(states)//5)):
    p_n, v_n, a_n, gp = states[si]
    with torch.no_grad():
        du_g, pred0 = predict_du(net, p_n, v_n, a_n, args.dt_big, edge_index,
                                  args.beta, fixed_mask)
        _, R0 = incremental_potential(pred0, sim, p_n, gp, VOL, MU, LAM, MASS, v_n, a_n,
                                       args.dt_big, None, args.beta, fixed_mask)
        _, Rg = incremental_potential(du_g, sim, p_n, gp, VOL, MU, LAM, MASS, v_n, a_n,
                                       args.dt_big, None, args.beta, fixed_mask)
    du_i = newton_reference(sim, p_n, gp, VOL, MU, LAM, MASS, v_n, a_n, args.dt_big,
                             None, args.beta, fixed_mask, iters=25)
    with torch.no_grad():
        _, Ri = incremental_potential(du_i, sim, p_n, gp, VOL, MU, LAM, MASS, v_n, a_n,
                                       args.dt_big, None, args.beta, fixed_mask)
    n0, ng_, ni = R0.norm().item(), Rg.norm().item(), Ri.norm().item()
    print(f"  {si:>6} {n0:12.4e} {ng_:12.4e} {ni:12.4e} {ng_/n0:9.4f} {ni/n0:9.4f}")

if args.out:
    torch.save({"model": net.state_dict(), "args": vars(args), "hist": hist}, args.out)
    print(f"[save] {args.out}")

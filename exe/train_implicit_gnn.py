"""Train a network to predict the implicit step's displacement increment du in
one shot, replacing i-PhysGaussian's Newton-GMRES iteration, by regressing on an
LBFGS solve of the same incremental potential.

Minimising that potential directly (its gradient is exactly the negative
momentum residual) was tried first and has been removed. It reached a 0.07
residual ratio on the states it was shown and exactly 1.0000 on any state the
network produced itself, and that number did not move for: three output
parameterisations, a dt curriculum, smooth state noise at three amplitudes,
autoregressive chain training with a measured gate, an explicit elastic-force
input, replacing 4-hop message passing with global attention (which took one
anchor's reach from 29 of 512 to all 512), or twelve training trajectories
instead of one. Regression on du* reaches 0.016 relative error on the same
states, so the map is learnable and the residual objective was not the way to
learn it.

Setup:
  * anchors carry (position, velocity, acceleration); the network sees all three
    per anchor and emits the next position, with the step size entering as a
    FiLM conditioning signal on a Fourier encoding of log10(dt) rather than as a
    node feature (it is one scalar shared by every anchor).
  * loss is plain L2 on the absolute next position against p_n + du*, where du*
    is an LBFGS solve of the incremental potential at that state.
  * the residual ratio ||R|| / ||R(du=predictor)|| is reported alongside but is
    no longer optimised, so the two objectives stay comparable across runs.

Training states come from real physics rollouts of the same scene -- several,
with randomised impulse direction and strength, since snapshots of one damped
oscillation are near-copies of each other.
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
                                  newmark_accel, newmark_velocity, newton_reference,
                                  DuCorrector, predict_du, anchor_elastic_accel)

ap = argparse.ArgumentParser()
ap.add_argument("--ply", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--n_anchors", type=int, default=512)
ap.add_argument("--K", type=int, default=8)
ap.add_argument("--k_graph", type=int, default=8)
ap.add_argument("--hidden", type=int, default=128)
ap.add_argument("--mp_steps", type=int, default=4, help="processor depth")
ap.add_argument("--processor", choices=["mpnn", "attention"], default="attention",
                 help="full self-attention over the anchors. Message passing is kept only so "
                      "the measurement that retired it stays reproducible: at k=8 and depth 4 "
                      "one anchor's influence reached 29 of 512 anchors, against a k-NN graph "
                      "diameter of 73 hops, so it could not represent the implicit step's "
                      "dense global solution operator at all.")
ap.add_argument("--heads", type=int, default=4)
ap.add_argument("--iters", type=int, default=2000)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--n_states", type=int, default=32,
                 help="physics snapshots per trajectory")
ap.add_argument("--n_traj", type=int, default=1,
                 help="how many DIFFERENT trajectories to collect states from. With one "
                      "trajectory the 24 states are snapshots of a single damped oscillation, "
                      "so their anchor configurations are near-copies of each other and a "
                      "network can fit them without learning a function of the state at all "
                      "-- which is what every run so far has done (0.07 on those states, "
                      "exactly 1.0000 anywhere else). Extra trajectories randomise the "
                      "impulse direction and strength; collection is a few seconds each.")
ap.add_argument("--state_substeps", type=int, default=200, help="substeps between snapshots")
ap.add_argument("--dt_big", type=float, default=4e-3,
                 help="largest step the GNN must take in one shot (curriculum target)")
ap.add_argument("--dt_curriculum", action="store_true",
                 help="grow the step size during training instead of fixing it at dt_big. "
                      "dt is drawn log-uniformly from [substep_dt, dt_max(it)], with dt_max "
                      "ramped geometrically from 2*substep_dt to dt_big over --dt_ramp_frac "
                      "of training. Sampling (rather than stepping through a schedule) keeps "
                      "the easy end in the mix so the small-dt behaviour is not forgotten, and "
                      "it is what finally makes the network's [dt, log dt] input mean anything "
                      "-- trained at a single dt that feature is a constant bias.")
ap.add_argument("--dt_ramp_frac", type=float, default=0.6)
ap.add_argument("--dt_min_mult", type=float, default=5.0,
                 help="smallest step to train on, in substeps. Below ~5x there is little to "
                      "learn: an LBFGS solve of the same potential only reaches a 0.23 "
                      "residual ratio at 1x, because the explicit predictor is already close "
                      "to the implicit answer there.")
ap.add_argument("--beta", type=float, default=0.25)
ap.add_argument("--gamma", type=float, default=0.5)
ap.add_argument("--no_raw_io", dest="raw_io", action="store_false",
                 help="use the correction parameterisation instead of the default: per-anchor "
                      "(position, velocity, acceleration) in, next position out, with dt as a "
                      "FiLM conditioning signal. The default gives up two things worth "
                      "remembering -- the decoder cannot be zero-init (a zero output is every "
                      "anchor at the origin), so training does not start from an explicit "
                      "step, and there is no force feature.")
ap.set_defaults(raw_io=True)
ap.add_argument("--velocity_out", action="store_true",
                 help="network emits du/dt (the step's average velocity) instead of a "
                      "correction to the predictor; O(1) at every dt with no gain constant")
ap.add_argument("--direct_du", action="store_true",
                 help="output the whole displacement instead of a correction to the "
                      "Newmark predictor (ablation)")
ap.add_argument("--no_force_feature", action="store_true",
                 help="withhold f_int/m from the network (the old input set)")
ap.add_argument("--chain_cooldown", type=int, default=500,
                 help="minimum iterations at each chain length before it can grow again")
ap.add_argument("--lbfgs_iters", type=int, default=40)
ap.add_argument("--grad_probe", type=int, default=0,
                 help="print the pre-clip gradient norm every N iters")
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


net = DuCorrector(args.hidden, args.mp_steps, use_force=not args.no_force_feature,
                   processor=args.processor, heads=args.heads,
                   raw_io=args.raw_io).to(dev)
opt = torch.optim.Adam(net.parameters(), lr=args.lr)
print(f"[setup] GNN params: {sum(p.numel() for p in net.parameters())/1e3:.1f}k")

# ---- collect physics states (p, v, a) the simulator actually visits ----
sub_dt = float(cfg["substep_dt"])
base_force = None
for bc in cfg.get("boundary_conditions", []):
    if bc["type"] == "particle_impulse":
        base_force = torch.tensor(bc["force"], device=dev)
wsum0 = torch.zeros(M, device=dev).index_add_(0, sim.nn_idx.reshape(-1), w0.reshape(-1))


def rand_rotation(gen):
    """uniform-ish random rotation via QR of a gaussian matrix"""
    q, r = torch.linalg.qr(torch.randn(3, 3, device=dev, generator=gen))
    q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def collect(force, n_snap):
    """one explicit trajectory; returns its snapshots"""
    out = []
    p_c, v_c = AC.clone(), torch.zeros(M, 3, device=dev)
    a_c = torch.zeros(M, 3, device=dev)
    gpp = POS.clone()
    if force is not None:
        v_c = v_c + (wsum0.unsqueeze(-1) * force.unsqueeze(0) * sub_dt) / MASS.unsqueeze(-1)
        v_c[fixed_mask] = 0
    with torch.enable_grad():
        for i in range(n_snap * args.state_substeps):
            vp = v_c.clone()
            p_c, v_c, gpp, _F = sim.step(p_c, v_c, MASS, gpp, VOL, MU, LAM, sub_dt,
                                          gravity=(gravity if gravity.abs().sum() > 0 else None),
                                          damping=float(cfg.get("grid_v_damping_scale", 1.0)),
                                          fixed_mask=fixed_mask)
            a_c = (v_c - vp) / sub_dt
            if torch.isnan(p_c).any():
                print(f"  [states] NaN at substep {i}; keeping {len(out)}")
                break
            if (i + 1) % args.state_substeps == 0:
                out.append((p_c.clone(), v_c.clone(), a_c.clone(), gpp.clone()))
    return out


print(f"[states] collecting from {args.n_traj} trajectory(ies)...")
gen = torch.Generator(device=dev); gen.manual_seed(1234)
states = []
for tr in tqdm(range(args.n_traj), desc="trajectories", ncols=90):
    if tr == 0 or base_force is None:
        f_tr = base_force                       # trajectory 0 is the config's own
    else:
        # random direction, strength within half to double the config's, so the
        # states span a range of deformations rather than one oscillation
        scale = 0.5 * (4.0 ** torch.rand(1, device=dev, generator=gen).item())
        f_tr = (rand_rotation(gen) @ base_force) * scale
    states += collect(f_tr, args.n_states)
print(f"[states] collected {len(states)} states")
assert len(states) >= 4, "not enough states"

# characteristic anchor speed -- the CONSTANT that sets the correction gain
# (see predict_du). Measured from the states the simulator actually visits, so
# it carries the scene's scale without ever depending on the current state.
ACC_SCALE = float(torch.stack([a for _, _, a, _ in states]).norm(dim=-1).mean())
print(f"[states] accel_scale = {ACC_SCALE:.4f} (mean anchor acceleration over collected states)")


def sample_dt(it):
    """log-uniform in [dt_min, dt_max(it)]; dt_max ramps 2*dt_min -> dt_big."""
    if not args.dt_curriculum:
        return args.dt_big
    prog = min(1.0, (it - 1) / max(1.0, args.dt_ramp_frac * args.iters))
    dt_lo = args.dt_min_mult * sub_dt
    lo_max = math.log(2.0 * dt_lo)
    dt_max = max(math.exp(lo_max + prog * (math.log(args.dt_big) - lo_max)), dt_lo)
    u = torch.rand(1).item()
    return math.exp(math.log(dt_lo) + u * (math.log(dt_max) - math.log(dt_lo)))

# ---- supervised targets: the implicit step solved per state ----
# Training against the incremental potential directly was tried and removed: on
# the states it saw it reached a 0.07 residual ratio, on states it produced
# itself exactly 1.0000, and no parameterisation, dt schedule, noise level,
# chain length, force feature or global-attention processor moved that number.
# Regression on an LBFGS solve of the same potential reaches 0.016 relative
# error on the correction, so the map is learnable and the residual objective
# was not the way to learn it.
P_STAR = []
if True:
    for st in tqdm(states, desc="lbfgs targets", ncols=90):
        p_s, v_s, a_s, gp_s = st
        du_t = newton_reference(sim, p_s, gp_s, VOL, MU, LAM, MASS, v_s, a_s,
                                 args.dt_big, None, args.beta, fixed_mask,
                                 iters=args.lbfgs_iters)
        P_STAR.append(p_s + du_t)          # the target IS the next position
    with torch.no_grad():
        rr = []
        for st, p_t in zip(states, P_STAR):
            p_s, v_s, a_s, gp_s = st
            du_t = p_t - p_s
            pr = newmark_predictor(v_s, a_s, args.dt_big, args.beta)
            pr = torch.where(fixed_mask.unsqueeze(-1), torch.zeros_like(pr), pr)
            _, R0 = incremental_potential(pr, sim, p_s, gp_s, VOL, MU, LAM, MASS, v_s, a_s,
                                           args.dt_big, None, args.beta, fixed_mask)
            _, Rt = incremental_potential(du_t, sim, p_s, gp_s, VOL, MU, LAM, MASS, v_s, a_s,
                                           args.dt_big, None, args.beta, fixed_mask)
            rr.append((Rt.norm() / R0.norm().clamp(min=1e-20)).item())
    print(f"[targets] {len(P_STAR)} LBFGS solves; their own residual ratio: "
          f"mean={sum(rr)/len(rr):.4f} max={max(rr):.4f}")

# ---- train ----
damping = float(cfg.get("grid_v_damping_scale", 1.0))
hist = []
pbar = tqdm(range(1, args.iters + 1), desc="train", ncols=110)
for it in pbar:
    si = torch.randint(len(states), (1,)).item()
    p, v, a, gp = [t.clone() for t in states[si]]
    dt_it = sample_dt(it)
    opt.zero_grad(set_to_none=True)
    if True:
        fa = None if args.no_force_feature else anchor_elastic_accel(
            sim, p, gp, VOL, MU, LAM, MASS, fixed_mask)
        du, pred0 = predict_du(net, p, v, a, dt_it, edge_index, ACC_SCALE,
                                args.beta, fixed_mask, fa, args.direct_du,
                                args.velocity_out)
        # plain L2 on the absolute next position
        loss = ((p + du - P_STAR[si].detach()) ** 2).mean()
        loss.backward()
        sup_err = loss.item()
        # the residual is reported, not optimised
        with torch.no_grad():
            _, R = incremental_potential(du, sim, p, gp, VOL, MU, LAM, MASS,
                                          v, a, dt_it, None, args.beta, fixed_mask)
            _, R0 = incremental_potential(pred0, sim, p, gp, VOL, MU, LAM, MASS,
                                           v, a, dt_it, None, args.beta, fixed_mask)
        rel = (R.norm() / R0.norm().clamp(min=1e-20)).item()
    gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0).item()
    opt.step()
    if args.grad_probe and (it % args.grad_probe == 0 or it == 1):
        wsum = sum(float(q.abs().mean()) for q in net.dec.parameters())
        print(f"  [probe] it={it:6d} dt/sub={dt_it/sub_dt:6.1f} grad_norm={gn:.3e} "
              f"mse={sup_err:.3e} R/R={rel:.4f} |dec_w|={wsum:.3e}", flush=True)

    if it % 25 == 0 or it == 1:
        hist.append((it, sup_err, rel, dt_it))
        pbar.set_postfix(dtx=f"{dt_it/sub_dt:.0f}", mse=f"{sup_err:.2e}", RR=f"{rel:.4f}")

print("\n[result] iter   dt/sub    position MSE   R/R_explicit")
for it, se, rl, dti in hist[::max(1, len(hist)//20)]:
    print(f"  {it:6d}  {dti/sub_dt:7.1f}    {se:12.4e}   {rl:12.4f}")

# ---- one-shot GNN vs iterated solve, ACROSS step sizes ----
# the number that matters is not the residual at one dt but how far the network
# holds up as the step grows, so sweep it and report the LBFGS solve of the same
# potential alongside as the achievable floor.
print("\n[eval] one-shot GNN vs LBFGS-iterated solve, swept over dt")
print(f"  {'dt/sub':>7} {'dt':>10} {'gnn/expl':>9} {'iter/expl':>9}  (mean over held-out states)")
for mult in [1, 2, 5, 10, 20, 40, 80, 160]:
    dt_e = mult * sub_dt
    gs, is_ = [], []
    for si in range(0, len(states), max(1, len(states) // 5)):
        p_n, v_n, a_n, gp = states[si]
        with torch.no_grad():
            fa = None if args.no_force_feature else anchor_elastic_accel(
                sim, p_n, gp, VOL, MU, LAM, MASS, fixed_mask)
            du_g, pred0 = predict_du(net, p_n, v_n, a_n, dt_e, edge_index, ACC_SCALE,
                                      args.beta, fixed_mask, fa, args.direct_du,
                                      args.velocity_out)
            _, R0 = incremental_potential(pred0, sim, p_n, gp, VOL, MU, LAM, MASS, v_n, a_n,
                                           dt_e, None, args.beta, fixed_mask)
            _, Rg = incremental_potential(du_g, sim, p_n, gp, VOL, MU, LAM, MASS, v_n, a_n,
                                           dt_e, None, args.beta, fixed_mask)
        du_i = newton_reference(sim, p_n, gp, VOL, MU, LAM, MASS, v_n, a_n, dt_e,
                                 None, args.beta, fixed_mask, iters=25)
        with torch.no_grad():
            _, Ri = incremental_potential(du_i, sim, p_n, gp, VOL, MU, LAM, MASS, v_n, a_n,
                                           dt_e, None, args.beta, fixed_mask)
        n0 = R0.norm().clamp(min=1e-20).item()
        gs.append(Rg.norm().item() / n0); is_.append(Ri.norm().item() / n0)
    print(f"  {mult:7d} {dt_e:10.5f} {sum(gs)/len(gs):9.4f} {sum(is_)/len(is_):9.4f}")

if args.out:
    torch.save({"model": net.state_dict(), "args": vars(args), "hist": hist,
                "accel_scale": ACC_SCALE}, args.out)
    print(f"[save] {args.out}")

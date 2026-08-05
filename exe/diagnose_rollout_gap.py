"""Why does a network that reaches a 0.07 residual ratio on simulator states
contribute exactly nothing (ratio 1.0000) the moment it drives its own rollout?

A ratio that pins to 1.0000 to four decimals is not a network that is trying and
failing -- it is a network whose output is not reaching du at all. So compare,
side by side and starting from the SAME initial condition, what the network
actually sees and emits in the two settings:

  * training: states from the explicit simulator, whose acceleration is the real
    f/m of the previous substep;
  * rollout: a chain seeded with a=0, whose acceleration afterwards is whatever
    Newmark eq. (8) reads back out of the du the network itself produced.

If the input distributions differ by orders of magnitude the network is simply
being asked a question it never saw, and no amount of noise or chaining in the
current setup would fix it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_lib = os.path.join(os.path.dirname(__file__), "..", "lib")
sys.path.insert(0, _lib)

import torch
from tqdm import tqdm

from anchorflow.anchors import AnchorSet
from anchorflow.anchor_mpm import AnchorElasticSim, lame_from_E_nu
from anchorflow.implicit import (incremental_potential, newmark_predictor, newmark_accel,
                                  newmark_velocity, DuCorrector, predict_du,
                                  anchor_elastic_accel, newton_reference)
from anchorflow import graph as G

ap = argparse.ArgumentParser()
ap.add_argument("--ply", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--ckpt", required=True)
ap.add_argument("--steps", type=int, default=12)
args = ap.parse_args()

from scene.gaussian_model import GaussianModel

dev = "cuda"
torch.set_grad_enabled(False)
cfg = json.load(open(args.config))
ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
targs = ck["args"]
ACC = ck["accel_scale"]
beta, gamma = targs["beta"], targs["gamma"]
sub_dt = float(cfg["substep_dt"])
dt_big = targs["dt_big"]
nsub = int(round(dt_big / sub_dt))

g = GaussianModel(3, fea_dim=0)
g.load_ply(args.ply)
xyz = g.get_xyz.detach().clone(); op = g.get_opacity.detach().clone()
keep = op[:, 0] > cfg["opacity_threshold"]
xw = xyz[keep]
pmin, pmax = xw.min(0).values, xw.max(0).values
mid = (pmin + pmax) / 2; sc = 1.0 / (pmax - pmin).max()
POS = ((xyz - mid) * sc + torch.tensor([1., 1., 1.], device=dev)).contiguous()
PM = POS[keep].contiguous()
N = POS.shape[0]

ng = 100; dx = 2.0 / ng
vi = (PM / dx).long().clamp(0, ng - 1)
flat = (vi[:, 0] * ng + vi[:, 1]) * ng + vi[:, 2]
cnt = torch.zeros(ng ** 3, device=dev).index_add_(0, flat, torch.ones(PM.shape[0], device=dev))
VOL = torch.zeros(N, device=dev); VOL[keep] = (dx ** 3) / cnt[flat]; VOL = VOL.contiguous()
E_p = torch.full((N,), float(cfg["E"]), device=dev)
nu_p = torch.full((N,), float(cfg["nu"]), device=dev)
d_p = torch.full((N,), float(cfg["density"]), device=dev)
for reg in cfg.get("additional_material_params", []):
    c = torch.tensor(reg["point"], device=dev); s = torch.tensor(reg["size"], device=dev)
    ins = ((POS - c).abs() <= s).all(-1)
    E_p[ins] = reg["E"]; nu_p[ins] = reg["nu"]; d_p[ins] = reg["density"]
MU, LAM = lame_from_E_nu(E_p, nu_p)
mass_p = d_p * VOL

aset, _ = AnchorSet.from_gaussians(PM, node_num=targs["n_anchors"], latent_dim=0, e_dim=0, K=targs["K"])
AC = aset.canonical.clone().contiguous(); M = AC.shape[0]
rad = AnchorElasticSim(PM, AC, K=targs["K"]).radius
sim = AnchorElasticSim(POS, AC, K=targs["K"], radius=rad)
w0 = sim._weights(POS, AC)
MASS = torch.zeros(M, device=dev).index_add_(
    0, sim.nn_idx.reshape(-1), (mass_p.unsqueeze(-1) * w0).reshape(-1)).clamp(min=1e-12)
fixed = torch.zeros(M, dtype=torch.bool, device=dev)
for bc in cfg.get("boundary_conditions", []):
    if bc["type"] == "cuboid":
        c = torch.tensor(bc["point"], device=dev); s = torch.tensor(bc["size"], device=dev)
        fixed |= ((AC - c).abs() <= s).all(-1)
gravity = torch.tensor(cfg["g"], dtype=torch.float32, device=dev)
edge_index = G.knn_graph(AC, k=targs["k_graph"])
damping = float(cfg.get("grid_v_damping_scale", 1.0))

net = DuCorrector(targs["hidden"], targs["mp_steps"],
                   use_force=not targs.get("no_force_feature", True),
                   processor=targs.get("processor", "mpnn"),
                   heads=targs.get("heads", 4),
                   raw_io=targs.get("raw_io", True)).to(dev)
net.load_state_dict(ck["model"]); net.eval()


def impulse():
    v = torch.zeros(M, 3, device=dev)
    for bc in cfg.get("boundary_conditions", []):
        if bc["type"] == "particle_impulse":
            f = torch.tensor(bc["force"], device=dev)
            ws = torch.zeros(M, device=dev).index_add_(
                0, sim.nn_idx.reshape(-1), (w0 * keep.unsqueeze(-1)).reshape(-1))
            v = v + (ws.unsqueeze(-1) * f.unsqueeze(0) * sub_dt) / MASS.unsqueeze(-1)
    v[fixed] = 0
    return v


def true_du_gap(p, v, a, gp):
    """how far the ACTUAL implicit solution sits from the Newmark predictor.

    This is what decides whether outputting a correction to the predictor is the
    right parameterisation: if the true du is a few percent away, the predictor
    is a good baseline and the network only has to supply the small part; if it
    is several times away, the residual form is asking the network to operate
    far outside the range it ever learned."""
    du = newton_reference(sim, p, gp, VOL, MU, LAM, MASS, v, a, dt_big,
                           None, beta, fixed, iters=40)
    pred = newmark_predictor(v, a, dt_big, beta)
    pred = torch.where(fixed.unsqueeze(-1), torch.zeros_like(pred), pred)
    return (du - pred).norm() / pred.norm().clamp(min=1e-20)


def report(tag, p, v, a, gp):
    fa = anchor_elastic_accel(sim, p, gp, VOL, MU, LAM, MASS, fixed) if net.use_force else None
    du, pred = predict_du(net, p, v, a, dt_big, edge_index, ACC, beta, fixed, fa,
                           targs.get("direct_du", False))
    corr = du - pred
    with torch.enable_grad():
        gap = true_du_gap(p, v, a, gp)
    _, R = incremental_potential(du, sim, p, gp, VOL, MU, LAM, MASS, v, a, dt_big,
                                  None, beta, fixed)
    _, R0 = incremental_potential(pred, sim, p, gp, VOL, MU, LAM, MASS, v, a, dt_big,
                                   None, beta, fixed)
    print(f"  {tag:22s} |v|={v.norm(dim=-1).mean():9.4f} |a|={a.norm(dim=-1).mean():10.2f} "
          f"|pred|={pred.norm(dim=-1).mean():9.3e} |corr|={corr.norm(dim=-1).mean():9.3e} "
          f"corr/pred={corr.norm()/pred.norm().clamp(min=1e-20):7.4f} "
          f"TRUE/pred={gap:7.4f} "
          f"R/R0={R.norm()/R0.norm().clamp(min=1e-20):7.4f}")


print(f"[cfg] dt_big={dt_big} ({nsub}x), accel_scale={ACC:.2f}, gain=beta*dt^2*A="
      f"{beta*dt_big*dt_big*ACC:.4e}")

print("\n--- A. states the TRAINER samples (explicit simulator, a = real f/m) ---")
p_s, v_s = AC.clone(), impulse()
a_s = torch.zeros(M, 3, device=dev); gp_s = POS.clone()
with torch.enable_grad():
    for i in tqdm(range(args.steps * nsub), desc="explicit", ncols=80, leave=False):
        vp = v_s.clone()
        p_s, v_s, gp_s, _ = sim.step(p_s, v_s, MASS, gp_s, VOL, MU, LAM, sub_dt,
                                      gravity=(gravity if gravity.abs().sum() > 0 else None),
                                      damping=damping, fixed_mask=fixed)
        a_s = (v_s - vp) / sub_dt
        if (i + 1) % (nsub * max(1, args.steps // 6)) == 0:
            with torch.no_grad():
                report(f"explicit t={(i+1)*sub_dt:.3f}s", p_s, v_s, a_s, gp_s)

print("\n--- B. states the ROLLOUT produces (a from Newmark eq. 8, seeded a=0) ---")
p, v = AC.clone(), impulse()
gp = POS.clone()
a = anchor_elastic_accel(sim, p, gp, VOL, MU, LAM, MASS, fixed)
for f in range(args.steps):
    if f % max(1, args.steps // 6) == 0:
        report(f"rollout frame {f}", p, v, a, gp)
    fa = anchor_elastic_accel(sim, p, gp, VOL, MU, LAM, MASS, fixed) if net.use_force else None
    du, _ = predict_du(net, p, v, a, dt_big, edge_index, ACC, beta, fixed, fa,
                        targs.get("direct_du", False))
    a_next = newmark_accel(du, v, a, dt_big, beta)
    v = damping * newmark_velocity(a_next, v, a, dt_big, gamma)
    p = p + du; a = a_next
    v = torch.where(fixed.unsqueeze(-1), torch.zeros_like(v), v)
    p = torch.where(fixed.unsqueeze(-1), AC, p)
    a = torch.where(fixed.unsqueeze(-1), torch.zeros_like(a), a)
    _, _, gp, _ = sim.step(p, torch.zeros_like(v), MASS, gp, VOL, MU, LAM, 0.0,
                            gravity=None, damping=1.0, fixed_mask=fixed)

"""Is one implicit step actually cheaper than the explicit substeps it replaces?

The whole premise of predicting the implicit step is that iterating is too
expensive. That was assumed, never measured. This times, for the same physical
step size:

  * N explicit substeps (what the implicit step has to beat), and
  * one LBFGS solve of the incremental potential at a range of iteration counts,

and reports the residual each leaves behind, so the accuracy bought per
millisecond is visible. If a 5-iteration solve is both faster than the explicit
substeps and accurate enough to roll out, there is nothing left for a network to
amortise.
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
from anchorflow.implicit import (incremental_potential, newmark_predictor,
                                  newton_reference)

ap = argparse.ArgumentParser()
ap.add_argument("--ply", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--n_anchors", type=int, default=512)
ap.add_argument("--K", type=int, default=8)
ap.add_argument("--dt_mult", type=int, default=40)
ap.add_argument("--warm_substeps", type=int, default=800)
ap.add_argument("--repeat", type=int, default=10)
args = ap.parse_args()

from scene.gaussian_model import GaussianModel

dev = "cuda"
torch.set_grad_enabled(False)
cfg = json.load(open(args.config))
sub_dt = float(cfg["substep_dt"])
dt_big = args.dt_mult * sub_dt

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

aset, _ = AnchorSet.from_gaussians(PM, node_num=args.n_anchors, latent_dim=0, e_dim=0, K=args.K)
AC = aset.canonical.clone().contiguous(); M = AC.shape[0]
rad = AnchorElasticSim(PM, AC, K=args.K).radius
sim = AnchorElasticSim(POS, AC, K=args.K, radius=rad)
w0 = sim._weights(POS, AC)
MASS = torch.zeros(M, device=dev).index_add_(
    0, sim.nn_idx.reshape(-1), ((d_p * VOL).unsqueeze(-1) * w0).reshape(-1)).clamp(min=1e-12)
fixed = torch.zeros(M, dtype=torch.bool, device=dev)
for bc in cfg.get("boundary_conditions", []):
    if bc["type"] == "cuboid":
        c = torch.tensor(bc["point"], device=dev); s = torch.tensor(bc["size"], device=dev)
        fixed |= ((AC - c).abs() <= s).all(-1)
gravity = torch.tensor(cfg["g"], dtype=torch.float32, device=dev)
damping = float(cfg.get("grid_v_damping_scale", 1.0))

# warm the scene into a deformed state so the timings are not measured at rest
p, v = AC.clone(), torch.zeros(M, 3, device=dev)
for bc in cfg.get("boundary_conditions", []):
    if bc["type"] == "particle_impulse":
        f = torch.tensor(bc["force"], device=dev)
        ws = torch.zeros(M, device=dev).index_add_(
            0, sim.nn_idx.reshape(-1), (w0 * keep.unsqueeze(-1)).reshape(-1))
        v = v + (ws.unsqueeze(-1) * f.unsqueeze(0) * sub_dt) / MASS.unsqueeze(-1)
v[fixed] = 0
gp = POS.clone(); a = torch.zeros(M, 3, device=dev)
with torch.enable_grad():
    for _ in tqdm(range(args.warm_substeps), desc="warm-up", ncols=80, leave=False):
        vp = v.clone()
        p, v, gp, _ = sim.step(p, v, MASS, gp, VOL, MU, LAM, sub_dt,
                                gravity=(gravity if gravity.abs().sum() > 0 else None),
                                damping=damping, fixed_mask=fixed)
        a = (v - vp) / sub_dt

print(f"[setup] N={N} M={M} dt={dt_big} ({args.dt_mult} substeps of {sub_dt})")


def timeit(fn):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(args.repeat):
        fn()
    e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1) / args.repeat


def explicit_block():
    pp, vv, gg = p.clone(), v.clone(), gp.clone()
    with torch.enable_grad():
        for _ in range(args.dt_mult):
            pp, vv, gg, _ = sim.step(pp, vv, MASS, gg, VOL, MU, LAM, sub_dt,
                                      gravity=(gravity if gravity.abs().sum() > 0 else None),
                                      damping=damping, fixed_mask=fixed)
    return pp


ms_exp = timeit(explicit_block)
pred = newmark_predictor(v, a, dt_big, 0.25)
pred = torch.where(fixed.unsqueeze(-1), torch.zeros_like(pred), pred)
_, R0 = incremental_potential(pred, sim, p, gp, VOL, MU, LAM, MASS, v, a, dt_big,
                               None, 0.25, fixed)
n0 = R0.norm().clamp(min=1e-20)
print(f"\n{'method':>22} {'ms/step':>9} {'vs explicit':>12} {'R/R_explicit':>13}")
print(f"{f'{args.dt_mult} explicit substeps':>22} {ms_exp:9.3f} {1.0:12.2f} {'-':>13}")

for k in (1, 2, 3, 5, 10, 20, 40):
    def solve(k=k):
        with torch.enable_grad():
            return newton_reference(sim, p, gp, VOL, MU, LAM, MASS, v, a, dt_big,
                                     None, 0.25, fixed, iters=k)
    ms = timeit(solve)
    du = solve()
    _, R = incremental_potential(du, sim, p, gp, VOL, MU, LAM, MASS, v, a, dt_big,
                                  None, 0.25, fixed)
    print(f"{f'LBFGS {k} iters':>22} {ms:9.3f} {ms_exp/ms:11.2f}x {(R.norm()/n0).item():13.5f}")

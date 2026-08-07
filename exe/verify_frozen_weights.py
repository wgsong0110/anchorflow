"""Does the kernel's new w_in path compute what it claims?

Two things have to hold before any result built on frozen weights means
anything:

  * passing the weights the kernel would have built itself must reproduce the
    kernel's own answer exactly -- otherwise w_in is not just "supply the
    weights", it is a second, subtly different physics;
  * with canonical weights supplied, the kernel must agree with the torch
    reference given the same weights, so the frozen path is verified the same
    way the lagged one was.

Then the property the whole change exists for: the same anchor configuration
reached by two different histories has to produce the same elastic
acceleration. Under the lagged weights it does not, and that is what makes the
target a learned stepper is asked to fit not a function of its inputs.
"""
from __future__ import annotations

import argparse
import os
import sys

_lib = os.path.join(os.path.dirname(__file__), "..", "lib")
sys.path.insert(0, _lib)

import torch

from anchorflow import scene_setup, anchor_mpm
import anchorstep

ap = argparse.ArgumentParser()
ap.add_argument("--ply", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--n_anchors", type=int, default=512)
ap.add_argument("--K", type=int, default=8)
ap.add_argument("--warm", type=int, default=10)
args = ap.parse_args()

dev = "cuda"
torch.set_grad_enabled(False)
assert anchorstep.HAVE_CUDA, "kernel not built"
sc = scene_setup.build(args.ply, args.config, args.n_anchors, args.K, device=dev)
sim, AC = sc.sim, sc.anchor_canonical


def rel(a, b):
    return ((a - b).norm() / b.norm().clamp(min=1e-20)).item()


# a deformed configuration to test on
p, v = AC.clone(), sc.initial_velocity()
gp = sc.pos.clone()
for _ in range(args.warm):
    p, v, gp = sc.explicit_step(p, v, gp, 40)

print(f"[setup] N={sc.N} M={sc.M}, {args.warm} coarse steps of deformation")

# --- 1. w_in reproduces the kernel's own weights exactly ---------------------
w_lag = sim._weights(gp, p)
f0, pos0, F0, psi0 = anchorstep.fused_energy_force(
    sim.gaussian_canonical, gp, p, AC, sim.nn_idx, sc.volume, sim.radius, sc.mu, sc.lam)
f1, pos1, F1, psi1 = anchorstep.fused_energy_force(
    sim.gaussian_canonical, gp, p, AC, sim.nn_idx, sc.volume, sim.radius, sc.mu, sc.lam,
    w_in=w_lag)
print(f"\n[1] w_in fed the kernel's own weights -- must be a no-op")
for name, a, b in (("force", f1, f0), ("gaussian pos", pos1, pos0), ("F", F1, F0), ("psi", psi1, psi0)):
    print(f"    {name:>14} max abs {(a - b).abs().max():.3e}   relative {rel(a, b):.3e}")

# --- 2. frozen kernel vs torch reference, same weights ----------------------
W0 = sim._canonical_weights()
f_k, pos_k, F_k, _ = anchorstep.fused_energy_force(
    sim.gaussian_canonical, gp, p, AC, sim.nn_idx, sc.volume, sim.radius, sc.mu, sc.lam,
    w_in=W0)
orig = anchor_mpm.AnchorElasticSim._weights
anchor_mpm.AnchorElasticSim._weights = lambda self, gpp, ap_: W0
with torch.enable_grad():
    ap_g = p.detach().clone().requires_grad_(True)
    E, F_t, pos_t = sim.elastic_energy(ap_g, gp, sc.volume, sc.mu, sc.lam)
    (g,) = torch.autograd.grad(E, ap_g)
    f_t = -g
anchor_mpm.AnchorElasticSim._weights = orig
print(f"\n[2] frozen kernel vs torch reference, same canonical weights")
for name, a, b in (("force", f_k, f_t), ("gaussian pos", pos_k, pos_t), ("F", F_k, F_t)):
    print(f"    {name:>14} max abs {(a - b).abs().max():.3e}   relative {rel(a, b):.3e}")

# --- 3. the property the change exists for ---------------------------------
print(f"\n[3] same anchors, two histories -- is the acceleration the same?")
for frozen in (False, True):
    sc.sim.freeze_weights(frozen)
    gpA = sc.skin(p, gp)                      # cloud carried from the simulation
    gpB = sc.skin(p, sc.pos.clone())          # cloud reached from canonical
    aA = sc.elastic_accel(p, gpA)
    aB = sc.elastic_accel(p, gpB)
    tag = "frozen  " if frozen else "lagged  "
    print(f"    {tag} cloud differs by {(gpA - gpB).abs().max():.3e}, "
          f"acceleration by {rel(aA, aB):.3e}  (|a| = {aB.norm(dim=-1).mean():.2f})")
sc.sim.freeze_weights(False)

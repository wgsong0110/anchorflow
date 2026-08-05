"""Is the coarse-step displacement even a function of (p, v, a)?

A network with 2.9M parameters cannot fit 60 training pairs -- it stalls at a
0.16 relative step error whether it has 0.7M parameters or 2.9M, on one
trajectory or on forty. That is not a capacity or a data problem, so the next
suspect is the target not being determined by the inputs at all.

The simulator's state is larger than what the network sees. The RBF weights that
build every Gaussian's deformation gradient are computed against the PREVIOUS
step's Gaussian positions, and that lagged cloud is not a function of the current
anchor positions -- it carries its own history. Two runs can arrive at the same
(p, v, a) with different clouds and then diverge.

This measures exactly that: take one state, pair it with clouds of differing
provenance, integrate the same explicit substeps, and compare the resulting
displacements. If they differ by as much as the network's error, the network is
being asked to predict something its inputs do not determine.
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

ap = argparse.ArgumentParser()
ap.add_argument("--ply", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--n_anchors", type=int, default=512)
ap.add_argument("--K", type=int, default=8)
ap.add_argument("--dt_mult", type=int, default=40)
ap.add_argument("--warm", type=int, default=10, help="coarse steps before probing")
ap.add_argument("--probes", type=int, default=6)
args = ap.parse_args()

dev = "cuda"
torch.manual_seed(0)
sc = scene_setup.build(args.ply, args.config, args.n_anchors, args.K, device=dev)
AC, fixed = sc.anchor_canonical, sc.fixed_mask
print(f"[setup] M={sc.M} N={sc.N} coarse step = {args.dt_mult} substeps")

# a state partway along the config's own trajectory, with its true cloud
p, v = AC.clone(), sc.initial_velocity()
gp = sc.pos.clone()
for _ in tqdm(range(args.warm), desc="warm", ncols=80, leave=False):
    p, v, gp = sc.explicit_step(p, v, gp, args.dt_mult)
p0, v0, gp_true = p.clone(), v.clone(), gp.clone()


def advance(p, v, gp):
    p2, _, _ = sc.explicit_step(p.clone(), v.clone(), gp.clone(), args.dt_mult)
    return p2 - p


du_true = advance(p0, v0, gp_true)
scale = du_true.norm(dim=-1).mean()
print(f"[state] after {args.warm} coarse steps: |du| mean = {scale:.6f}")

# clouds that a network with the same (p, v, a) could equally have been handed
variants = {}
variants["skinned from p (what the rollout uses)"] = sc.skin(p0, sc.pos.clone())
variants["skinned, canonical seed"] = sc.skin(p0, sc.pos.clone())
lag = gp_true.clone()
for k in (1, 5, 20):
    # a cloud that lags the anchors by k substeps: the same anchor state reached
    # with a slightly different history
    pp, vv, gg = p0.clone(), v0.clone(), gp_true.clone()
    for _ in range(k):
        pp, vv, gg = sc.explicit_step(pp, vv, gg, 1)
    variants[f"cloud from {k} substeps ahead"] = gg

print(f"\n{'cloud':>40} {'||du - du_true|| / |du|':>24}")
print(f"{'the simulator                    ':>40} {0.0:24.5f}")
for name, g in variants.items():
    du = advance(p0, v0, g)
    rel = ((du - du_true).norm() / du_true.norm()).item()
    print(f"{name:>40} {rel:24.5f}")

# and the other direction: how much does the cloud itself differ?
print(f"\n{'cloud':>40} {'mean Gaussian offset':>22} {'as % of |du|':>14}")
for name, g in variants.items():
    d = (g - gp_true).norm(dim=-1).mean()
    print(f"{name:>40} {d:22.6f} {100 * d / scale:13.2f}%")

# finally: repeatability. same inputs, same cloud, run twice -- this is the floor
# any of the above has to be read against.
d1, d2 = advance(p0, v0, gp_true), advance(p0, v0, gp_true)
print(f"\n[floor] same cloud twice: {((d1 - d2).norm() / du_true.norm()).item():.3e} "
      f"(nondeterminism in the kernel, if any)")

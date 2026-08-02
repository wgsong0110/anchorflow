"""Fast physics-only parameter sweep for the anchor-elastodynamics ficus sim
(no rendering) -- find a configuration that produces NATURAL elastic motion:
an impulse should make branches swing AND return (oscillation), not coast
monotonically into a NaN blowup (the v8 failure: free anchors rigid-body
coasting, strain confined to a thin boundary layer at the pinned pot).

For each config: run the sim, record max free-anchor displacement every
`probe_every` substeps, then classify:
  - 'NAN @ step'            : exploded
  - 'CREEP'                 : displacement monotonically increasing at the end
  - 'OSCILLATES (n peaks)'  : displacement went up and came back down >= once
The oscillating, non-NaN config with visible amplitude is what we render.

Uses the fused CUDA step (lib/anchorstep) when built -- the whole point is
this sweep is only tractable with it.
"""
from __future__ import annotations

import os
import sys

_lib = os.path.join(os.path.dirname(__file__), "..", "lib")
sys.path.insert(0, _lib)

import torch
from tqdm import tqdm

from anchorflow.anchors import AnchorSet
from anchorflow.anchor_mpm import AnchorElasticSim, lame_from_E_nu

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--ply", required=True)
ap.add_argument("--n_anchors", type=int, default=512)
ap.add_argument("--K", type=int, default=8)
ap.add_argument("--up_axis", type=int, default=2)
ap.add_argument("--pot_height_frac", type=float, default=0.18)
ap.add_argument("--density", type=float, default=200.0)
ap.add_argument("--voxel_grid", type=int, default=64)
args = ap.parse_args()

from scene.gaussian_model import GaussianModel

dev = "cuda"
torch.set_grad_enabled(False)
g = GaussianModel(3, fea_dim=0)
g.load_ply(args.ply)
gaussian_canonical = g.get_xyz.detach().clone().contiguous()
N = gaussian_canonical.shape[0]
anchor_set, _ = AnchorSet.from_gaussians(gaussian_canonical, node_num=args.n_anchors,
                                          latent_dim=0, e_dim=0, K=args.K)
anchor_canonical = anchor_set.canonical.clone().contiguous()
M = anchor_canonical.shape[0]
sim = AnchorElasticSim(gaussian_canonical, anchor_canonical, K=args.K)

bbox_min = gaussian_canonical.min(0).values
bbox_max = gaussian_canonical.max(0).values
G = args.voxel_grid
span = (bbox_max - bbox_min).max() + 1e-6
voxel_dx = span / G
vidx = ((gaussian_canonical - bbox_min) / voxel_dx).long().clamp(0, G - 1)
flat = (vidx[:, 0] * G + vidx[:, 1]) * G + vidx[:, 2]
counts = torch.zeros(G * G * G, device=dev).index_add_(0, flat, torch.ones(N, device=dev))
gaussian_volume = ((voxel_dx ** 3) / counts[flat]).contiguous()
particle_mass = args.density * gaussian_volume
w0 = sim._weights(gaussian_canonical, anchor_canonical)
anchor_mass = torch.zeros(M, device=dev).index_add_(
    0, sim.nn_idx.reshape(-1), (particle_mass.unsqueeze(-1) * w0).reshape(-1)).clamp(min=1e-8)

up = gaussian_canonical[:, args.up_axis]
thr = up.min() + args.pot_height_frac * (up.max() - up.min())
fixed_mask = anchor_canonical[:, args.up_axis] < thr
print(f"[sweep] N={N} M={M} pinned={fixed_mask.sum().item()}")

def run(E, nu, dt, steps, impulse, damping, probe_every=200):
    mu, lam = lame_from_E_nu(torch.tensor(E, device=dev), torch.tensor(nu, device=dev))
    ap_ = anchor_canonical.clone()
    av = torch.zeros(M, 3, device=dev)
    av[:, 0] = impulse
    gpp = gaussian_canonical.clone()
    trace = []
    for s in range(1, steps + 1):
        ap_, av, gpp, F = sim.step(ap_, av, anchor_mass, gpp, gaussian_volume,
                                    float(mu), float(lam), dt, damping=damping,
                                    fixed_mask=fixed_mask)
        if torch.isnan(ap_).any():
            return f"NAN @ {s}", trace
        if s % probe_every == 0:
            d = (ap_[~fixed_mask] - anchor_canonical[~fixed_mask]).norm(dim=-1).max().item()
            trace.append(d)
    # classify: count local maxima in the displacement trace
    peaks = sum(1 for i in range(1, len(trace) - 1)
                if trace[i] > trace[i - 1] and trace[i] > trace[i + 1])
    tail_rising = len(trace) >= 2 and trace[-1] > trace[-2]
    amp = max(trace) if trace else 0.0
    if peaks >= 1:
        cls = f"OSCILLATES ({peaks} peaks, amp={amp:.3f}, end={trace[-1]:.3f})"
    elif tail_rising:
        cls = f"CREEP (end={trace[-1]:.3f}, still rising)"
    else:
        cls = f"SETTLED (amp={amp:.3f}, end={trace[-1]:.3f})"
    return cls, trace

CONFIGS = [
    # (label, E, nu, dt, steps, impulse_x, damping)
    ("A: paper-ish  E=0.1 dt=2e-5", 0.1, 0.4, 2e-5, 15000, -1.0, 0.9999),
    ("B: softer     E=0.01 dt=5e-5", 0.01, 0.4, 5e-5, 10000, -1.0, 0.9999),
    ("C: stiffer    E=1.0 dt=1e-5", 1.0, 0.4, 1e-5, 20000, -1.0, 0.9999),
    ("D: E=0.1 small-dt long", 0.1, 0.4, 1e-5, 30000, -1.0, 0.9999),
    ("E: E=0.1 gentle impulse", 0.1, 0.4, 2e-5, 15000, -0.3, 0.9999),
    ("F: E=0.3 dt=1e-5", 0.3, 0.4, 1e-5, 25000, -0.5, 0.9999),
]

results = []
for label, E, nu, dt, steps, imp, damp in tqdm(CONFIGS, desc="sweep", ncols=100):
    cls, trace = run(E, nu, dt, steps, imp, damp)
    results.append((label, cls, trace))
    short = " ".join(f"{t:.3f}" for t in trace[:12])
    print(f"\n[{label}] -> {cls}\n  trace: {short}{' ...' if len(trace)>12 else ''}")

print("\n=== SUMMARY ===")
for label, cls, _ in results:
    print(f"  {label:32s} {cls}")

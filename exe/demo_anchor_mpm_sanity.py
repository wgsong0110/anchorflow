"""Sanity checks for lib/anchorflow/anchor_mpm.py's grid-free anchor
elastodynamics, before trusting it on a real scene:

1. Pure rigid rotation+translation of all anchors -> F should be EXACTLY the
   rotation (up to numerical precision) at every Gaussian, and elastic energy
   should be ~0 (no spurious strain from rigid motion -- this is the core
   correctness property the whole shape-matching design relies on).
2. A simple uniform stretch -> F should recover the stretch factor, energy
   should be > 0 and increase with stretch magnitude.
3. A short multi-step rollout (gravity + Fixed Corotated elasticity, random
   synthetic anchor/Gaussian cloud) -> just checks nothing NaNs/explodes and
   that turning up stiffness (higher E) visibly resists deformation more.

Synthetic random point clouds (no real scene data needed) -- this is a math/
plumbing check, not a quality evaluation.
"""
from __future__ import annotations

import os
import sys

_lib = os.path.join(os.path.dirname(__file__), "..", "lib")
sys.path.insert(0, _lib)

import torch

from anchorflow.anchor_mpm import AnchorElasticSim, lame_from_E_nu, _polar_decompose

torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"

M, N, K = 64, 500, 4
anchor_canonical = torch.randn(M, 3, device=dev)
gaussian_canonical = torch.randn(N, 3, device=dev) * 1.2

sim = AnchorElasticSim(gaussian_canonical, anchor_canonical, K=K)
mu, lam = lame_from_E_nu(torch.tensor(0.1, device=dev), torch.tensor(0.4, device=dev))
gaussian_volume = torch.full((N,), 1.0 / N, device=dev)

print("=" * 70)
print("[test 1] pure rigid rotation + translation")
print("=" * 70)
theta = 0.7
axis = torch.tensor([0.3, 1.0, -0.2], device=dev)
axis = axis / axis.norm()
K_ = torch.tensor([[0, -axis[2], axis[1]],
                    [axis[2], 0, -axis[0]],
                    [-axis[1], axis[0], 0]], device=dev)
R_true = torch.eye(3, device=dev) + torch.sin(torch.tensor(theta)) * K_ + \
         (1 - torch.cos(torch.tensor(theta))) * (K_ @ K_)
d_true = torch.tensor([0.5, -0.3, 0.8], device=dev)

anchor_rotated = (R_true @ anchor_canonical.T).T + d_true
# weights compare gaussian_pos_prev against CURRENT anchor positions -- for
# this to be rotation-invariant (as the module's design assumes for a real
# rollout, where gaussian_pos_prev is itself the correctly-evolved previous
# state) both sides must be transformed consistently. Using the untouched
# REST Gaussian position here (mismatched against rotated anchors) would
# test something the module never claims to handle -- a real rollout's
# gaussian_pos_prev always lags the SAME transform by exactly one step, so
# feed the (already known, since this is a synthetic single-shot rigid test)
# correctly-rotated Gaussian position instead.
expected_gaussian = (R_true @ gaussian_canonical.T).T + d_true
gaussian_pos_prev = expected_gaussian.clone()

with torch.no_grad():
    w = sim._weights(gaussian_pos_prev, anchor_rotated)
    F, gaussian_pos = sim._shape_match(anchor_rotated, w)

F_err = (F - R_true.unsqueeze(0)).abs().max().item()
pos_err = (gaussian_pos - expected_gaussian).abs().max().item()
print(f"  max|F - R_true| = {F_err:.3e}  (should be ~0)")
print(f"  max|gaussian_pos - expected| = {pos_err:.3e}  (should be ~0)")

from anchorflow.anchor_mpm import fixed_corotated_energy_density
psi = fixed_corotated_energy_density(F, mu, lam)
E_rigid = (gaussian_volume * psi).sum().item()
print(f"  elastic energy under pure rigid motion = {E_rigid:.3e}  (should be ~0)")

R_est, S_est = _polar_decompose(F)
print(f"  polar-decompose sanity: max|R_est - R_true| = {(R_est - R_true.unsqueeze(0)).abs().max().item():.3e}, "
      f"max|S_est - I| = {(S_est - torch.eye(3, device=dev)).abs().max().item():.3e}")

print()
print("=" * 70)
print("[test 2] uniform stretch (scale anchors by s, no rotation)")
print("=" * 70)
for s in [1.0, 1.1, 1.5, 2.0]:
    anchor_stretched = anchor_canonical * s
    with torch.no_grad():
        w = sim._weights(gaussian_pos_prev, anchor_stretched)
        F, _ = sim._shape_match(anchor_stretched, w)
        psi = fixed_corotated_energy_density(F, mu, lam)
        E = (gaussian_volume * psi).sum().item()
    # F should be close to s*I for an interior Gaussian far from the anchor cloud's
    # centroid-driven boundary effects -- check the MEAN diagonal instead of
    # exact equality since boundary Gaussians (few neighbors span the full
    # scale) will be noisier.
    mean_diag = F.diagonal(dim1=-2, dim2=-1).mean().item()
    print(f"  scale={s:.2f}: mean(diag F)={mean_diag:.4f} (expect ~{s:.2f})  energy={E:.4f}")

print()
print("=" * 70)
print("[test 3] short rollout: gravity + Fixed Corotated, check stability")
print("=" * 70)
for E_stiff, label in [(0.001, "soft"), (0.05, "stiff")]:
    mu_r, lam_r = lame_from_E_nu(torch.tensor(E_stiff, device=dev), torch.tensor(0.4, device=dev))
    anchor_pos = anchor_canonical.clone()
    anchor_vel = torch.zeros(M, 3, device=dev)
    anchor_mass = torch.full((M,), 1.0, device=dev)
    gaussian_pos_prev = gaussian_canonical.clone()
    gravity = torch.tensor([0.0, -0.5, 0.0], device=dev)
    dt = 1e-4
    nan_hit = False
    for step_i in range(50):
        anchor_pos, anchor_vel, gaussian_pos_prev, F = sim.step(
            anchor_pos, anchor_vel, anchor_mass, gaussian_pos_prev,
            gaussian_volume, mu_r, lam_r, dt, gravity=gravity, damping=0.999)
        if torch.isnan(anchor_pos).any() or torch.isnan(anchor_vel).any():
            nan_hit = True
            break
    disp = (anchor_pos - anchor_canonical).norm(dim=-1).mean().item()
    print(f"  {label} (E={E_stiff}): 50 steps, NaN={nan_hit}, mean anchor displacement={disp:.4f}")

print()
print("[done] all tests ran without crashing.")

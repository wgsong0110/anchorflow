"""Parity check: lib/anchorstep fused CUDA step vs the torch reference path in
lib/anchorflow/anchor_mpm.py (AnchorElasticSim.elastic_energy + autograd).

Checks, on real-scale random data (N=8000 Gaussians, M=512 anchors, K=8,
positions at ficus-like magnitudes):
  1. forward parity: psi, F, skinned gaussian_pos vs the torch path
  2. force parity: -dE/danchor vs torch.autograd.grad through elastic_energy
     (expect close but NOT identical -- the fused path freezes weights w.r.t.
     differentiation while the torch path differentiates through them; the
     check quantifies that gap on top of fp differences)
  3. speed: fused step vs torch step, forward+force, x100 reps
"""
from __future__ import annotations

import os
import sys
import time

_lib = os.path.join(os.path.dirname(__file__), "..", "lib")
sys.path.insert(0, _lib)

import torch

from anchorflow.anchor_mpm import AnchorElasticSim
from anchorstep import fused_energy_force, HAVE_CUDA

print(f"[verify] anchorstep HAVE_CUDA={HAVE_CUDA}")
assert HAVE_CUDA

torch.manual_seed(0)
dev = "cuda"
N, M, K = 8000, 512, 8
gaussian_canonical = (torch.randn(N, 3, device=dev) * 0.8).contiguous()
anchor_canonical = (torch.randn(M, 3, device=dev) * 0.8).contiguous()
sim = AnchorElasticSim(gaussian_canonical, anchor_canonical, K=K)
volume = torch.full((N,), 1.0 / N, device=dev)
mu, lam = 3.6e5, 1.4e6

# a mildly deformed state (so F != I and forces are nonzero)
anchor_pos = (anchor_canonical + 0.03 * torch.randn(M, 3, device=dev)).contiguous()
gaussian_pos_prev = gaussian_canonical.clone()

# ---- torch reference ----
ap_ref = anchor_pos.detach().requires_grad_(True)
E_ref, F_ref, pos_ref = sim.elastic_energy(ap_ref, gaussian_pos_prev, volume, mu, lam)
(g_ref,) = torch.autograd.grad(E_ref, ap_ref)
f_ref = -g_ref

from anchorflow.anchor_mpm import fixed_corotated_energy_density
with torch.no_grad():
    w_ref = sim._weights(gaussian_pos_prev, anchor_pos)
    F_ref2, _ = sim._shape_match(anchor_pos, w_ref)
    psi_ref = fixed_corotated_energy_density(F_ref2, mu, lam)

# ---- fused ----
f_fused, pos_fused, F_fused, psi_fused = fused_energy_force(
    gaussian_canonical, gaussian_pos_prev, anchor_pos, anchor_canonical,
    sim.nn_idx, volume, sim.radius, mu, lam)

F_err = (F_fused - F_ref2).abs().max().item()
pos_err = (pos_fused - pos_ref).abs().max().item()
psi_rel = ((psi_fused - psi_ref).abs() / psi_ref.abs().clamp(min=1e-3)).max().item()
E_fused = (volume * psi_fused).sum().item()
print(f"[verify] F err max         = {F_err:.3e}")
print(f"[verify] pos err max       = {pos_err:.3e}")
print(f"[verify] psi rel err max   = {psi_rel:.3e}")
print(f"[verify] E: fused={E_fused:.6e} ref={E_ref.item():.6e} rel={(abs(E_fused-E_ref.item())/abs(E_ref.item())):.3e}")

cos = torch.nn.functional.cosine_similarity(f_fused.flatten(), f_ref.flatten(), dim=0).item()
rel = ((f_fused - f_ref).norm() / f_ref.norm()).item()
print(f"[verify] force: cos={cos:.6f} rel_l2={rel:.3e}  "
      f"(expect cos~1; rel gap includes the frozen-weights approximation)")

# ---- speed ----
torch.cuda.synchronize(); t0 = time.time()
for _ in range(100):
    ap = anchor_pos.detach().requires_grad_(True)
    E, _, _ = sim.elastic_energy(ap, gaussian_pos_prev, volume, mu, lam)
    torch.autograd.grad(E, ap)
torch.cuda.synchronize(); t_torch = (time.time() - t0) / 100

torch.cuda.synchronize(); t0 = time.time()
for _ in range(100):
    fused_energy_force(gaussian_canonical, gaussian_pos_prev, anchor_pos,
                        anchor_canonical, sim.nn_idx, volume, sim.radius, mu, lam)
torch.cuda.synchronize(); t_fused = (time.time() - t0) / 100

print(f"[verify] speed: torch={t_torch*1000:.2f}ms/step  fused={t_fused*1000:.2f}ms/step  "
      f"speedup={t_torch/t_fused:.1f}x")

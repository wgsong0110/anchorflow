"""Quick A/B timing microbenchmark for anchor_mpm's per-step cost, at a
scale matching a real scene (thousands of Gaussians, hundreds of anchors) --
isolates the physics step cost from rendering/IO to see the eigh3x3 CUDA
kernel's actual speedup over torch.linalg.eigh.
"""
import os
import sys
import time

_lib = os.path.join(os.path.dirname(__file__), "..", "lib")
sys.path.insert(0, _lib)

import torch

from anchorflow.anchor_mpm import AnchorElasticSim, lame_from_E_nu

torch.manual_seed(0)
dev = "cuda"
N, M, K = 8000, 512, 8
gaussian_canonical = torch.randn(N, 3, device=dev) * 1.5
anchor_canonical = torch.randn(M, 3, device=dev)

sim = AnchorElasticSim(gaussian_canonical, anchor_canonical, K=K)
mu, lam = lame_from_E_nu(torch.tensor(0.1, device=dev), torch.tensor(0.4, device=dev))
gaussian_volume = torch.full((N,), 1.0 / N, device=dev)
gravity = torch.tensor([0.0, 0.0, -2.0], device=dev)

anchor_pos = anchor_canonical.clone()
anchor_vel = torch.zeros(M, 3, device=dev)
anchor_mass = torch.full((M,), 1.0, device=dev)
gaussian_pos_prev = gaussian_canonical.clone()

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 200
dt = 1e-5   # tiny dt just to avoid NaN derailing the timing loop

torch.cuda.synchronize()
t0 = time.time()
with torch.enable_grad():
    for _ in range(STEPS):
        anchor_pos, anchor_vel, gaussian_pos_prev, F = sim.step(
            anchor_pos, anchor_vel, anchor_mass, gaussian_pos_prev,
            gaussian_volume, mu, lam, dt, gravity=gravity, damping=0.999)
torch.cuda.synchronize()
elapsed = time.time() - t0
print(f"N={N} M={M} K={K} steps={STEPS}: {elapsed:.3f}s total, {elapsed/STEPS*1000:.2f}ms/step, "
      f"nan={torch.isnan(anchor_pos).any().item()}")

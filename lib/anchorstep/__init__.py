"""Fused anchor-elastodynamics step (CUDA): the entire per-substep hot path of
lib/anchorflow/anchor_mpm.py -- RBF weights, shape-matching F (with the same
eigh + identity-fallback as the torch reference), Fixed Corotated energy, and
ANALYTIC anchor forces (closed-form PK1 stress, no autograd graph) -- in two
kernels. See anchorstep_cuda.cu's header comment for the math.

NOT YET NUMERICALLY VERIFIED against the torch reference path -- run
exe/verify_anchorstep.py on a GPU instance (forward psi/F/pos parity + force
parity vs autograd.grad through anchor_mpm.elastic_energy) before trusting.
Falls back to None-export if unbuilt; callers must check HAVE_CUDA.
"""

import torch

try:
    from ._C import forward as _fwd, backward as _bwd
    HAVE_CUDA = True
except Exception:
    HAVE_CUDA = False


def fused_energy_force(gaussian_canonical, gaussian_pos_prev, anchor_pos,
                        anchor_rest, nn_idx, volume, radius, mu, lam,
                        eig_floor_frac=0.2):
    """One fused physics evaluation. All tensors float32 CUDA.

    mu/lam may be scalars OR per-particle [N] tensors: PhysGaussian scenes set
    different material per region (ficus makes the leaf canopy 10x softer than
    the trunk, which is what produces its 'stiff stem, fluttering leaves'
    motion -- collapsing that to one volume-averaged value loses the effect).

    Returns (f_elastic [M,3] = -dE/danchor, gaussian_pos [N,3], F [N,3,3],
    psi [N]). Weights are frozen w.r.t. differentiation (the torch reference
    differentiates through them too, but they're built from LAGGED gaussian
    positions and act as a per-step discretization choice -- verify script
    quantifies the difference)."""
    N = gaussian_canonical.shape[0]
    if not torch.is_tensor(mu):
        mu = torch.full((N,), float(mu), device=gaussian_canonical.device, dtype=torch.float32)
    if not torch.is_tensor(lam):
        lam = torch.full((N,), float(lam), device=gaussian_canonical.device, dtype=torch.float32)
    mu = mu.contiguous().float(); lam = lam.contiguous().float()
    w, Binv, qbar, F, pos, psi, R = _fwd(
        gaussian_canonical, gaussian_pos_prev, anchor_pos, anchor_rest,
        nn_idx, volume, mu, lam, float(radius), float(eig_floor_frac))
    grad = _bwd(anchor_rest, nn_idx, volume, w, Binv, qbar, F, R,
                mu, lam, anchor_rest.shape[0])
    return -grad, pos, F.view(-1, 3, 3), psi

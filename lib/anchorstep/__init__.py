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
try:
    from ._C import backward_gather as _bwd_gather
    HAVE_GATHER = True
except Exception:
    HAVE_GATHER = False

_CSR_CACHE = {}


def build_csr(nn_idx, M):
    """anchor -> (gaussian, slot) CSR for the contention-free backward.

    Connectivity is fixed at construction, so this is built once and cached by
    (tensor identity, M). Needed because the atomic-scatter backward is
    contention-bound exactly in the sparse-anchor regime this method targets:
    measured 0.383 ms at M=512 vs 0.160 ms at M=4096 for identical arithmetic
    (4.1M atomicAdds onto only M*3 addresses)."""
    key = (nn_idx.data_ptr(), int(M), tuple(nn_idx.shape))
    hit = _CSR_CACHE.get(key)
    if hit is not None:
        return hit
    N, K = nn_idx.shape
    flat = nn_idx.reshape(-1).to(torch.int64)                      # [N*K]
    order = torch.argsort(flat)                                    # group by anchor
    sorted_anchor = flat[order]
    gid = (order // K).to(torch.int32).contiguous()
    slot = (order % K).to(torch.int32).contiguous()
    counts = torch.bincount(sorted_anchor, minlength=int(M))
    off = torch.zeros(int(M) + 1, dtype=torch.int32, device=nn_idx.device)
    off[1:] = counts.cumsum(0).to(torch.int32)
    out = (off.contiguous(), gid, slot)
    _CSR_CACHE[key] = out
    return out


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
    w, Binv, qbar, F, pos, psi, R, G, c = _fwd(
        gaussian_canonical, gaussian_pos_prev, anchor_pos, anchor_rest,
        nn_idx, volume, mu, lam, float(radius), float(eig_floor_frac))
    M = anchor_rest.shape[0]
    if HAVE_GATHER:
        off, gid, slot = build_csr(nn_idx, M)
        grad = _bwd_gather(anchor_rest, volume, w, G, c, off, gid, slot,
                            nn_idx.shape[1], M)
    else:
        grad = _bwd(anchor_rest, nn_idx, volume, w, Binv, qbar, F, R, mu, lam, M)
    return -grad, pos, F.view(-1, 3, 3), psi

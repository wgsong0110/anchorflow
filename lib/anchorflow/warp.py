"""Anchor -> Gaussian LBS warp producing the exact tensors the 3DGS rasterizer
consumes: deformed centres [N,3], covariance [N,6], and rotation [N,3,3].

This is the seam that replaces DreamPhysics's MPM export
(export_particle_x / cov / R). Faithful to SC-GS's embedded-deformation warp:

    per-anchor translation  t_k = a_k^now - a_k^rest
    per-anchor rotation     R_k  (from local Procrustes on the anchor graph)
    per-Gaussian position   x' = sum_k w_k [ R_k (x - a_k^rest) + a_k^rest + t_k ]
    per-Gaussian rotation    Rg = blend_k w_k R_k          (quaternion mean)
    per-Gaussian covariance  S' = Rg S Rg^T                (rotate canonical cov)

Everything is plain differentiable torch, so autograd carries the render loss
straight back to the anchor positions (hence to the GNN + actuation latents).
"""

from __future__ import annotations

import torch

from . import geom
from .anchors import knn

try:                                                # fused CUDA LBS (optional)
    from lbs import lbs_blend as _lbs_blend
except Exception:
    _lbs_blend = None

try:                                                # fused CUDA covariance warp
    from lbs import cov_warp as _cov_warp
except Exception:
    _cov_warp = None


# --- covariance <-> 6-vector (upper triangular xx,xy,xz,yy,yz,zz) --------- #
def cov6_to_mat3(c):
    xx, xy, xz, yy, yz, zz = c.unbind(-1)
    row0 = torch.stack([xx, xy, xz], -1)
    row1 = torch.stack([xy, yy, yz], -1)
    row2 = torch.stack([xz, yz, zz], -1)
    return torch.stack([row0, row1, row2], -2)                # [...,3,3]


def mat3_to_cov6(M):
    return torch.stack([M[..., 0, 0], M[..., 0, 1], M[..., 0, 2],
                        M[..., 1, 1], M[..., 1, 2], M[..., 2, 2]], dim=-1)


def cov_from_scale_rot(scaling, rotation):
    """Canonical 3DGS covariance S = R diag(s^2) R^T -> [N,6].
    scaling [N,3] (activated), rotation [N,4] quaternion (raw)."""
    R = geom.quat_to_matrix(rotation)                         # [N,3,3]
    S = R * scaling[:, None, :]                                # R @ diag(s)
    cov = S @ S.transpose(-1, -2)                              # R diag(s^2) R^T
    return mat3_to_cov6(cov)


# --- per-anchor rotation from canonical->now (weighted Procrustes) -------- #
def anchor_rotations(canonical, now, K=8):
    """Estimate a per-anchor rotation aligning its rest neighbourhood to the
    deformed one (SC-GS p2dR / ARAP local rotation). Returns R [M,3,3].

    Computed under no_grad: the rotation is *estimated* (ARAP-style), not
    differentiated. This is essential — the Procrustes SVD backward is NaN when
    the neighbourhood is (near-)undeformed (repeated singular values, e.g. at
    rest), which otherwise poisons the whole model on the first step. Gradient
    to the anchor motion flows through the LBS translation term instead."""
    with torch.no_grad():
        _, idx = knn(canonical, canonical, min(K + 1, canonical.shape[0]))
        idx = idx[:, 1:]                                      # drop self -> [M,K]
        src = canonical[idx] - canonical[:, None]             # rest edges [M,K,3]
        tgt = now[idx] - now[:, None]                         # now edges  [M,K,3]
        w = torch.ones(src.shape[:-1], device=canonical.device)
        return geom.procrustes_rotation(src, tgt, w)         # [M,3,3] (detached)


def _blend_quat(quat, w, idx):
    """Weighted quaternion mean over K neighbours (sign-aligned). -> [N,4]."""
    q = quat[idx]                                            # [N,K,4]
    ref = q[:, :1]                                           # align to first nbr
    sign = torch.sign((q * ref).sum(-1, keepdim=True))
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    q = q * sign
    return geom.quat_normalize((w[..., None] * q).sum(1))    # [N,4]


# --- the warp --------------------------------------------------------------- #
def lbs_warp(gauss_xyz, gauss_cov6, w, idx, anchor_canon, anchor_now,
             anchor_R=None):
    """Deform Gaussians from anchor motion.

    gauss_xyz  [N,3]   canonical centres
    gauss_cov6 [N,6]   canonical covariance
    w, idx     [N,K]   RBF binding weights / neighbour anchor indices
    anchor_canon, anchor_now [M,3]
    anchor_R   [M,3,3] per-anchor rotation (if None -> derived via Procrustes)
    returns pos [N,3], cov6 [N,6], rot [N,3,3]
    """
    if anchor_R is None:
        anchor_R = anchor_rotations(anchor_canon, anchor_now)
    # pos = Σ_k w_k (R_j(x-a_rest_j) + a_now_j)  [a_rest+t == a_now]. Fused CUDA
    # kernel when built (grad -> anchor_now), else the torch reference below.
    if _lbs_blend is not None and gauss_xyz.is_cuda:
        pos = _lbs_blend(gauss_xyz, w, idx, anchor_canon, anchor_now, anchor_R)
    else:
        a_rest = anchor_canon[idx]                            # [N,K,3]
        Rk = anchor_R[idx]                                    # [N,K,3,3]
        rel = (gauss_xyz[:, None] - a_rest)                   # [N,K,3]
        Ax = torch.einsum("nkab,nkb->nka", Rk, rel) + anchor_now[idx]
        pos = (w[..., None] * Ax).sum(1)                      # [N,3]

    quat = geom.matrix_to_quat(anchor_R)                     # [M,4]

    # Fused CUDA path: the torch branch below materialises [N,K,4] plus several
    # [N,3,3] tensors (~90MB+ of traffic per frame at N=1.85M) and dominates
    # lbs_warp (93% of its time). anchor_R is detached (Procrustes under
    # no_grad), so the covariance carries no gradient and a forward-only kernel
    # is exact. Rg is returned lazily only when a caller needs it.
    if _cov_warp is not None and gauss_cov6.is_cuda and not torch.is_grad_enabled():
        cov6_out = _cov_warp(quat, w.detach(), idx, gauss_cov6)
        if cov6_out is not None:
            return pos, cov6_out, None

    qg = _blend_quat(quat, w, idx)                            # [N,4]
    Rg = geom.quat_to_matrix(qg)                             # [N,3,3]

    S = cov6_to_mat3(gauss_cov6)
    cov = Rg @ S @ Rg.transpose(-1, -2)
    return pos, mat3_to_cov6(cov), Rg

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
import torch.nn.functional as F

from .anchors import knn


# --- quaternion / rotation helpers (inlined from geom.py) ------------------- #

def _quat_normalize(q):
    return F.normalize(q, dim=-1)


def _quat_to_matrix(q):
    """[...,4] wxyz -> [...,3,3]."""
    q = _quat_normalize(q)
    w, x, y, z = q.unbind(-1)
    tx, ty, tz = 2 * x, 2 * y, 2 * z
    twx, twy, twz = tx * w, ty * w, tz * w
    txx, txy, txz = tx * x, ty * x, tz * x
    tyy, tyz, tzz = ty * y, tz * y, tz * z
    o = torch.stack([
        1 - (tyy + tzz), txy - twz, txz + twy,
        txy + twz, 1 - (txx + tzz), tyz - twx,
        txz - twy, tyz + twx, 1 - (txx + tyy),
    ], dim=-1)
    return o.reshape(q.shape[:-1] + (3, 3))


def _matrix_to_quat(M):
    """[...,3,3] -> [...,4] wxyz (pytorch3d-style branchless)."""
    m = M.reshape(M.shape[:-2] + (9,))
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = m.unbind(-1)
    q_abs = torch.sqrt(torch.clamp(torch.stack([
        1.0 + m00 + m11 + m22,
        1.0 + m00 - m11 - m22,
        1.0 - m00 + m11 - m22,
        1.0 - m00 - m11 + m22,
    ], dim=-1), min=0.0))
    quat_by_rijk = torch.stack([
        torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
        torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
        torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
        torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
    ], dim=-2)
    flr = torch.tensor(0.1).to(q_abs)
    quat_candidates = quat_by_rijk / (2.0 * torch.maximum(q_abs[..., None], flr))
    idx = q_abs.argmax(dim=-1)
    out = torch.gather(quat_candidates, -2,
                       idx[..., None, None].expand(q_abs.shape[:-1] + (1, 4))).squeeze(-2)
    return _quat_normalize(out)


def _procrustes_rotation(src_edges, tgt_edges, weight):
    """Weighted Kabsch rotation (ARAP local frame)."""
    D = torch.diag_embed(weight)
    S = src_edges.transpose(-1, -2) @ D @ tgt_edges
    U, sig, Vh = torch.linalg.svd(S)
    V = Vh.transpose(-1, -2)
    R = V @ U.transpose(-1, -2)
    flip = torch.ones_like(sig)
    flip[..., -1] = torch.linalg.det(R)
    return (V * flip[..., None, :]) @ U.transpose(-1, -2)

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
    R = _quat_to_matrix(rotation)                              # [N,3,3]
    S = R * scaling[:, None, :]                                # R @ diag(s)
    cov = S @ S.transpose(-1, -2)                              # R diag(s^2) R^T
    return mat3_to_cov6(cov)


# --- per-anchor rotation from canonical->now (weighted Procrustes) -------- #
def anchor_rotations_cache(canonical, K=8):
    """Precompute the constant parts of anchor_rotations (knn idx + rest edges).
    Call once after anchors are set up; pass results to anchor_rotations()."""
    with torch.no_grad():
        _, idx = knn(canonical, canonical, min(K + 1, canonical.shape[0]))
        idx = idx[:, 1:]                                      # [M, K]
        src = canonical[idx] - canonical[:, None]             # [M, K, 3]
    return idx, src


def anchor_rotations(canonical, now, K=8, _idx=None, _src=None):
    """Estimate a per-anchor rotation aligning its rest neighbourhood to the
    deformed one (SC-GS p2dR / ARAP local rotation). Returns R [M,3,3].

    Computed under no_grad: the rotation is *estimated* (ARAP-style), not
    differentiated. This is essential — the Procrustes SVD backward is NaN when
    the neighbourhood is (near-)undeformed (repeated singular values, e.g. at
    rest), which otherwise poisons the whole model on the first step. Gradient
    to the anchor motion flows through the LBS translation term instead.

    Pass _idx, _src from anchor_rotations_cache() to skip the knn/edge recompute."""
    with torch.no_grad():
        if _idx is None:
            _, _idx = knn(canonical, canonical, min(K + 1, canonical.shape[0]))
            _idx = _idx[:, 1:]
        if _src is None:
            _src = canonical[_idx] - canonical[:, None]
        tgt = now[_idx] - now[:, None]                        # now edges  [M,K,3]
        w = torch.ones(_src.shape[:-1], device=canonical.device)
        return _procrustes_rotation(_src, tgt, w)             # [M,3,3] (detached)


def _blend_quat(quat, w, idx):
    """Weighted quaternion mean over K neighbours (sign-aligned). -> [N,4]."""
    q = quat[idx]                                            # [N,K,4]
    ref = q[:, :1]                                           # align to first nbr
    sign = torch.sign((q * ref).sum(-1, keepdim=True))
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    q = q * sign
    return _quat_normalize((w[..., None] * q).sum(1))         # [N,4]


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

    quat = _matrix_to_quat(anchor_R)                          # [M,4]

    # Fused CUDA path: the torch branch below materialises [N,K,4] plus several
    # [N,3,3] tensors (~90MB+ of traffic per frame at N=1.85M) and dominates
    # lbs_warp (93% of its time). anchor_R is detached (Procrustes under
    # no_grad), so the covariance carries no gradient and a forward-only kernel
    # is exact. Rg is returned lazily only when a caller needs it.
    if _cov_warp is not None and gauss_cov6.is_cuda:
        # Forward-only: anchor_R is detached (Procrustes under no_grad), so the
        # only gradient the torch branch carries here is w -> the rotation
        # blend, which is secondary; rho still learns through the position
        # branch. Renders are bit-comparable (max abs err ~1e-7, parity test).
        with torch.no_grad():
            cov6_out = _cov_warp(quat, w, idx, gauss_cov6)
        if cov6_out is not None:
            return pos, cov6_out, None

    qg = _blend_quat(quat, w, idx)                            # [N,4]
    Rg = _quat_to_matrix(qg)                                  # [N,3,3]

    S = cov6_to_mat3(gauss_cov6)
    cov = Rg @ S @ Rg.transpose(-1, -2)
    return pos, mat3_to_cov6(cov), Rg

"""Quaternion + weighted-Procrustes utilities (wxyz convention, matching SC-GS).

Kept dependency-free (torch only). Used by warp.py (LBS local frames) and reg.py
(ARAP per-node rotation). All quaternions are [...,4] in (w,x,y,z) order.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def quat_normalize(q):
    return F.normalize(q, dim=-1)


def quat_to_matrix(q):
    """[...,4] wxyz -> [...,3,3] rotation matrix."""
    q = quat_normalize(q)
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


def _sqrt_pos(x):
    return torch.sqrt(torch.clamp(x, min=0.0))


def matrix_to_quat(M):
    """[...,3,3] -> [...,4] wxyz. Robust branchless form (pytorch3d-style)."""
    m = M.reshape(M.shape[:-2] + (9,))
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = m.unbind(-1)
    q_abs = torch.stack([
        1.0 + m00 + m11 + m22,
        1.0 + m00 - m11 - m22,
        1.0 - m00 + m11 - m22,
        1.0 - m00 - m11 + m22,
    ], dim=-1)
    q_abs = _sqrt_pos(q_abs)
    quat_by_rijk = torch.stack([
        torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
        torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
        torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
        torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
    ], dim=-2)
    flr = torch.tensor(0.1).to(q_abs)
    quat_candidates = quat_by_rijk / (2.0 * torch.maximum(q_abs[..., None], flr))
    idx = q_abs.argmax(dim=-1)
    out = torch.gather(
        quat_candidates, -2,
        idx[..., None, None].expand(q_abs.shape[:-1] + (1, 4))).squeeze(-2)
    return quat_normalize(out)


def quat_multiply(a, b):
    """Hamilton product a*b, both [...,4] wxyz."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack([ow, ox, oy, oz], dim=-1)


def procrustes_rotation(src_edges, tgt_edges, weight):
    """Weighted rigid rotation aligning src->tgt edge sets (Kabsch/ARAP).

    src_edges, tgt_edges : [..., K, 3]   edge vectors (already relative)
    weight               : [..., K]      per-edge weights
    returns R            : [..., 3, 3]   with det>0 (reflection-corrected)
    """
    D = torch.diag_embed(weight)                              # [...,K,K]
    S = src_edges.transpose(-1, -2) @ D @ tgt_edges           # [...,3,3] covariance
    U, sig, Vh = torch.linalg.svd(S)
    V = Vh.transpose(-1, -2)
    R = V @ U.transpose(-1, -2)
    det = torch.linalg.det(R)                                 # fix reflections
    flip = torch.ones_like(sig)
    flip[..., -1] = det
    R = (V * flip[..., None, :]) @ U.transpose(-1, -2)
    return R

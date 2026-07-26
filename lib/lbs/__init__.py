"""Fused anchor-LBS position blend (CUDA), parity-matched to the torch reference
in anchorflow.warp.lbs_warp. Falls back to torch if the extension isn't built.

    pos[n] = sum_k w[n,k] · ( R[j] (x[n] - a_rest[j]) + a_now[j] ),  j = idx[n,k]

Only a_now is differentiable (R is computed under no_grad in the reference), so the
backward is a weighted scatter-add: grad_a_now[j] += sum_{n,k: idx==j} w[n,k]·grad_pos[n].
"""

import torch

try:
    from ._C import forward as _fwd, backward as _bwd
    _HAVE_CUDA = True
except Exception:
    _HAVE_CUDA = False

try:
    from ._C import cov_warp as _cov_warp_cuda
    _HAVE_COV_CUDA = True
except Exception:
    _HAVE_COV_CUDA = False

try:
    from ._C import rot_batch_forward as _rot_batch_fwd, rot_batch_backward as _rot_batch_bwd
    _HAVE_ROT_BATCH_CUDA = True
except Exception:
    _HAVE_ROT_BATCH_CUDA = False


class _LBSBlend(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, idx, a_rest, a_now, R):
        ctx.save_for_backward(w, idx.long())
        ctx.M = a_rest.shape[0]
        return _fwd(x.contiguous(), w.contiguous(), idx.contiguous().long(),
                    a_rest.contiguous(), a_now.contiguous(), R.contiguous())

    @staticmethod
    def backward(ctx, grad_out):
        w, idx = ctx.saved_tensors
        grad_a_now = _bwd(grad_out.contiguous(), w, idx, ctx.M)
        return None, None, None, None, grad_a_now, None    # only a_now needs grad


def _lbs_blend_torch(x, w, idx, a_rest, a_now, R):
    Ax = torch.einsum("nkab,nkb->nka", R[idx], x[:, None] - a_rest[idx]) + a_now[idx]
    return (w[..., None] * Ax).sum(1)


def lbs_blend(x, w, idx, a_rest, a_now, R):
    """CUDA fused LBS position blend (falls back to torch). grad flows to a_now."""
    if _HAVE_CUDA and x.is_cuda:
        return _LBSBlend.apply(x, w, idx, a_rest, a_now, R)
    return _lbs_blend_torch(x, w, idx, a_rest, a_now, R)


class _LBSRotBatch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, idx, a_canon, a_now, a_drot):
        idx = idx.contiguous().long()
        w = w.contiguous(); a_canon = a_canon.contiguous()
        a_now = a_now.contiguous(); a_drot = a_drot.contiguous()
        ctx.save_for_backward(w, idx, a_canon, a_now, a_drot)
        out_xyz, out_rot = _rot_batch_fwd(w, idx, a_canon, a_now, a_drot)
        return out_xyz, out_rot

    @staticmethod
    def backward(ctx, grad_out_xyz, grad_out_rot):
        w, idx, a_canon, a_now, a_drot = ctx.saved_tensors
        grad_out_xyz = grad_out_xyz.contiguous()
        grad_out_rot = grad_out_rot.contiguous()
        grad_w, grad_a_now, grad_a_drot = _rot_batch_bwd(
            grad_out_xyz, grad_out_rot, w, idx, a_canon, a_now, a_drot)
        return grad_w, None, None, grad_a_now, grad_a_drot


def _lbs_blend_rot_batched_torch(w, idx, a_canon, a_now, a_drot):
    d_anchor = a_now - a_canon[None]
    d_xyz = (w[None, ..., None] * d_anchor[:, idx]).sum(2)
    d_rot = (w[None, ..., None] * a_drot[:, idx]).sum(2)
    return d_xyz, d_rot


def lbs_blend_rot_batched(w, idx, a_canon, a_now, a_drot):
    """Fused batched anchor-LBS displacement + residual-rotation blend
    (CUDA, falls back to torch). One call handles all B sampled frames for
    all N Gaussians -- replaces a python for-loop of per-frame gathers.
    Grad flows to w, a_now, a_drot (a_canon is a fixed buffer, idx is int).

    w, idx: [N, K]   a_canon: [M, 3]
    a_now: [B, M, 3]   a_drot: [B, M, 4]
    returns d_xyz [B, N, 3], d_rot [B, N, 4]"""
    if _HAVE_ROT_BATCH_CUDA and a_now.is_cuda:
        return _LBSRotBatch.apply(w, idx, a_canon, a_now, a_drot)
    return _lbs_blend_rot_batched_torch(w, idx, a_canon, a_now, a_drot)


def cov_warp(quat, w, idx, cov6):
    """Fused covariance warp: cov6' = R(q̄) · cov6 · R(q̄)^T with q̄ the
    weighted, sign-aligned quaternion mean over the K bound anchors.

    Forward only — anchor rotations are Procrustes-estimated under no_grad in
    the reference, so nothing here needs a gradient. Falls back to torch.
    """
    if _HAVE_COV_CUDA and cov6.is_cuda:
        return _cov_warp_cuda(quat.contiguous(), w.contiguous(),
                              idx.contiguous().long(), cov6.contiguous())
    return None

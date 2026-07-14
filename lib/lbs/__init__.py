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

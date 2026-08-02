"""Fused batched closed-form symmetric 3x3 eigendecomposition (CUDA), a drop-in
differentiable replacement for `torch.linalg.eigh` on a batch of [N,3,3]
symmetric matrices -- built for anchor_mpm's shape-matching B-matrix solve
(lib/anchorflow/anchor_mpm.py), where eigh is called once per Gaussian per
physics substep and a general LAPACK-backed eigh has too much per-call
overhead for many tiny matrices.

NOT YET NUMERICALLY VERIFIED against torch.linalg.eigh -- run
exe/verify_eigen3x3.py on a GPU instance and confirm forward+backward parity
before wiring eigh3x3() into anchor_mpm.py's hot path. Falls back to
torch.linalg.eigh if the extension isn't built (always correct, just slow).
"""

import torch

try:
    from ._C import forward as _fwd, backward as _bwd
    _HAVE_CUDA = True
except Exception:
    _HAVE_CUDA = False


class _Eigh3x3(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B):
        eigval, eigvec = _fwd(B.contiguous())
        ctx.save_for_backward(eigval, eigvec)
        return eigval, eigvec

    @staticmethod
    def backward(ctx, grad_eigval, grad_eigvec):
        eigval, eigvec = ctx.saved_tensors
        grad_eigval = grad_eigval if grad_eigval is not None else torch.zeros_like(eigval)
        grad_eigvec = grad_eigvec if grad_eigvec is not None else torch.zeros_like(eigvec)
        grad_B = _bwd(grad_eigval.contiguous(), grad_eigvec.contiguous(), eigval, eigvec)
        return grad_B


def eigh3x3(B):
    """B: [N,3,3] symmetric -> (eigval [N,3] ascending, eigvec [N,3,3], columns
    are eigenvectors) -- CUDA fused closed-form solver (falls back to
    torch.linalg.eigh, which this is meant to match numerically)."""
    if _HAVE_CUDA and B.is_cuda and B.dtype == torch.float32:
        return _Eigh3x3.apply(B)
    return torch.linalg.eigh(B)

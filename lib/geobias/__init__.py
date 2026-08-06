"""Fused relative-geometry attention bias (CUDA).

The eager version of this 292-parameter layer was 35% of a training iteration
because it is applied to every ordered pair of anchors and PyTorch materialises
the [B, M, M, 32] hidden activation -- 268 MB at B=8, M=512, written in forward
and read again in backward. The kernel keeps the chain in registers and writes
only the [B, 4, M, M] result. Falls back to the eager path when unbuilt; see
geobias_cuda.cu for the math and exe/verify_geobias.py for the parity check.
"""
import torch

try:
    from ._C import forward as _fwd, backward as _bwd
    HAVE_CUDA = True
except Exception:
    HAVE_CUDA = False


class _FusedGeoBias(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pos, w1, b1, w2, b2, inv_s):
        out = _fwd(pos, w1, b1, w2, b2, inv_s)
        ctx.save_for_backward(pos, w1, b1, w2, b2)
        ctx.inv_s = inv_s
        return out

    @staticmethod
    def backward(ctx, gout):
        pos, w1, b1, w2, b2 = ctx.saved_tensors
        g = _bwd(pos, w1, b1, w2, b2, gout.contiguous(), ctx.inv_s)
        return g[0], g[1], g[2], g[3], g[4], None


def fused_geo_bias(pos, w1, b1, w2, b2, inv_scale):
    """pos [B,M,3] -> bias [B,4,M,M]; w1 [32,4], b1 [32], w2 [4,32], b2 [4]."""
    return _FusedGeoBias.apply(pos, w1, b1, w2, b2, float(inv_scale))

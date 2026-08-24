"""Fused CUDA path for the sparse anchor discretisation (anchor_sparse.py).

lib/anchorstep serves the original set -- fixed K neighbours, isotropic weights,
one material stiffness -- and none of those hold once the discretisation is
fitted: anchors gain an orientation and three extents, membership becomes
whatever falls inside G(x) > c, and each anchor carries its own stiffness. That
left the fitted simulator on the torch path, which materialises a [P,3,3]
tensor of outer products twice per substep and scatters it, at 2.6x the pairs
the fixed-neighbour version had.

FORWARD ONLY. The fit needs gradients and keeps the torch path; this is for
running a simulator whose parameters are already fixed, which is every rollout,
every evaluation and every frame the student ever produces.

Callers must check HAVE_CUDA -- the module exports it False rather than failing
to import when the extension is not built.
"""

import torch

try:
    from ._C import skin as _skin, force as _force, deform as _deform
    from ._C import deform_fwd as _deform_fwd, deform_bwd as _deform_bwd
    from ._C import gather_fwd as _gather_fwd, gather_bwd as _gather_bwd
    from ._C import deform_bwd_rs as _deform_bwd_rs
    from ._C import moment_fwd as _moment_fwd, moment_bwd as _moment_bwd
    HAVE_CUDA = True
except Exception:
    HAVE_CUDA = False

_CSR_CACHE = {}


def build_csr(pair_g, pair_a, N, M):
    """(row_off, pair_a32, pair_g32, acsr_off, acsr_pair) for one pair list.

    Rebuilt only when refresh() changes the pairs, so it is keyed on the tensors
    themselves. Row offsets are free: refresh() ends with argsort(g * M + a), so
    the list already arrives grouped by Gaussian and in a stable order within a
    group. The anchor-major list is a real sort, and it is what lets the force
    gather run without atomics -- a few hundred anchors receiving millions of
    contributions is the worst case for an atomic scatter.
    """
    key = (pair_g.data_ptr(), pair_a.data_ptr(), int(pair_g.shape[0]), int(N), int(M))
    hit = _CSR_CACHE.get(key)
    if hit is not None:
        return hit

    dev = pair_g.device
    # the kernels index with int, so the pair list has to fit in one; at the
    # sizes this runs at (millions) it does, and a silent wrap would be a wrong
    # force rather than a crash
    P = int(pair_g.shape[0])
    if P > 2 ** 31 - 1:
        raise RuntimeError(f"{P} pairs exceeds the int32 indexing the kernel uses")

    row_off = torch.zeros(int(N) + 1, dtype=torch.int32, device=dev)
    row_off[1:] = torch.bincount(pair_g, minlength=int(N)).cumsum(0).to(torch.int32)

    order = torch.argsort(pair_a)
    acsr_off = torch.zeros(int(M) + 1, dtype=torch.int32, device=dev)
    acsr_off[1:] = torch.bincount(pair_a, minlength=int(M)).cumsum(0).to(torch.int32)

    out = (row_off.contiguous(),
           pair_a.to(torch.int32).contiguous(),
           pair_g.to(torch.int32).contiguous(),
           acsr_off.contiguous(),
           order.to(torch.int32).contiguous())
    _CSR_CACHE[key] = out
    return out


def _f32(t):
    return t.detach().contiguous().float()


def skin(p, csr, w, q, Binv, blocked, Xc, rc):
    """anchor positions -> Gaussian positions. Returns (x [N,3], F [N,3,3])."""
    row_off, pair_a, _pair_g, _ao, _ap = csr
    return _skin(_f32(p), row_off, pair_a, _f32(w), _f32(q),
                 _f32(Binv).reshape(-1, 9), _f32(blocked).reshape(-1, 9),
                 _f32(Xc), _f32(rc))


def force(p, csr, w, q, Binv, blocked, vol, mu, lam, M,
          polar_iters=6, polar_ridge=1e-6):
    """anchor positions -> elastic anchor forces [M,3].

    mu and lam must already carry the per-anchor stiffness multiplier blended
    onto Gaussians. That blend is a function of the weights, which are constant
    between refreshes, so it belongs on the python side and is done once.
    """
    row_off, pair_a, pair_g, acsr_off, acsr_pair = csr
    return _force(_f32(p), row_off, pair_a, pair_g, _f32(w), _f32(q),
                  _f32(Binv).reshape(-1, 9), _f32(blocked).reshape(-1, 9),
                  _f32(vol), _f32(mu), _f32(lam), acsr_off, acsr_pair,
                  int(M), int(polar_iters), float(polar_ridge))


def deform(p, csr, w, q, Binv, blocked):
    """shape-matching F [N,3,3] and the deformed centroid [N,3]."""
    row_off, pair_a, _pair_g, _ao, _ap = csr
    return _deform(_f32(p), row_off, pair_a, _f32(w), _f32(q),
                   _f32(Binv).reshape(-1, 9), _f32(blocked).reshape(-1, 9))


# ---- differentiable, for the fit -------------------------------------------
#
# The forward kernels above are no use to the fit, which differentiates through
# everything: one substep costs 121 ms on the torch path against the kernel's
# 2.15, and nearly all of that is the [P,3,3] tensor of outer products being
# built, scattered, and built again in reverse -- 129 MB per direction per
# substep at 3.6M pairs.
#
# Only the PAIR-level reductions are fused here. The per-Gaussian 3x3 work --
# polar factor, determinant, inverse -- stays in autograd, where it is 171k tiny
# problems rather than 3.6M and where the delicate derivatives live. The two
# meet at A and at P@Binv, so the split is clean.


class _Deform(torch.autograd.Function):
    """(p, w, q, RS) -> (weighted centroid, scatter of w [(p-cc) q^T + RS])

    RS is R_a S_a for oriented anchors and an empty tensor otherwise. Its
    gradient is one more anchor-major gather, which is why it belongs here
    rather than in a torch expression over [P,3,3].
    """

    @staticmethod
    def forward(ctx, p, w, q, RS, row_off, pair_a, pair_g, acsr_off, acsr_pair, N, M):
        rs = RS.contiguous() if RS is not None and RS.numel() else \
            torch.empty(0, device=p.device, dtype=p.dtype)
        cc, A, S = _deform_fwd(p.contiguous(), row_off, pair_a,
                                w.contiguous(), q.contiguous(), rs, int(N))
        ctx.save_for_backward(p, w, q, cc, S, pair_a, pair_g, acsr_off, acsr_pair)
        ctx.M = int(M)
        ctx.has_rs = rs.numel() > 0
        return cc, A

    @staticmethod
    def backward(ctx, gcc, gA):
        p, w, q, cc, S, pair_a, pair_g, acsr_off, acsr_pair = ctx.saved_tensors
        gA = gA.contiguous()
        gp, gw, gq = _deform_bwd(gcc.contiguous(), gA, p, w, q, cc, S,
                                  pair_a, pair_g, acsr_off, acsr_pair, ctx.M)
        gRS = _deform_bwd_rs(gA, w, pair_g, acsr_off, acsr_pair, ctx.M) \
            if ctx.has_rs else None
        return gp, gw, gq, gRS, None, None, None, None, None, None, None


class _Gather(torch.autograd.Function):
    """(P@Binv, w, q) -> anchor forces"""

    @staticmethod
    def forward(ctx, PB, w, q, vol, row_off, pair_a, pair_g, acsr_off, acsr_pair, N, M):
        f = _gather_fwd(PB.contiguous(), q.contiguous(), w.contiguous(),
                         vol.contiguous(), pair_g, acsr_off, acsr_pair, int(M))
        ctx.save_for_backward(PB, w, q, vol, row_off, pair_a, pair_g)
        ctx.N = int(N)
        return f

    @staticmethod
    def backward(ctx, gf):
        PB, w, q, vol, row_off, pair_a, pair_g = ctx.saved_tensors
        gPB, gw, gq = _gather_bwd(gf.contiguous(), PB, q, w, vol,
                                   row_off, pair_a, pair_g, ctx.N)
        return gPB, gw, gq, None, None, None, None, None, None, None, None


class _Moment(torch.autograd.Function):
    """P@Binv -> the per-anchor stress moment the torque is built from"""

    @staticmethod
    def forward(ctx, PB, w, vol, row_off, pair_a, pair_g, acsr_off, acsr_pair, N, M):
        Mo = _moment_fwd(PB.contiguous(), w.contiguous(), vol.contiguous(),
                          pair_g, acsr_off, acsr_pair, int(M))
        ctx.save_for_backward(w, vol, row_off, pair_a)
        ctx.N = int(N)
        return Mo

    @staticmethod
    def backward(ctx, gM):
        w, vol, row_off, pair_a = ctx.saved_tensors
        gPB = _moment_bwd(gM.contiguous(), w, vol, row_off, pair_a, ctx.N)
        return gPB, None, None, None, None, None, None, None, None, None


def deform_diff(p, w, q, csr, N, M, RS=None):
    """differentiable (cc, A). Returns None if the extension is not built."""
    row_off, pair_a, pair_g, acsr_off, acsr_pair = csr
    return _Deform.apply(p, w, q, RS, row_off, pair_a, pair_g, acsr_off,
                         acsr_pair, N, M)


def moment_diff(PB, w, vol, csr, N, M):
    """differentiable per-anchor stress moment"""
    row_off, pair_a, pair_g, acsr_off, acsr_pair = csr
    return _Moment.apply(PB, w, vol, row_off, pair_a, pair_g, acsr_off,
                          acsr_pair, N, M)


def gather_diff(PB, w, q, vol, csr, N, M):
    """differentiable anchor forces from P@Binv"""
    row_off, pair_a, pair_g, acsr_off, acsr_pair = csr
    return _Gather.apply(PB, w, q, vol, row_off, pair_a, pair_g,
                          acsr_off, acsr_pair, N, M)

"""셀 다항식 에너지를 **커널 한 번**으로 계산한다 (Triton).

한 사면체의 기여는 그 네 꼭짓점 변위 12 자유도의 2차식이다:

    E_t = 1/2 sum_ij M_ij (U_i . U_j) + sum_i (g_i - f_i) . U_i
          + 1/2 sum_ijab K[i,a,j,b] U[i,a] U[j,b]

파이토치로 쓰면 작은 einsum 여러 개가 각각 커널로 떠서, 입자가 적은 씬에서는
연산량이 아니라 **발사 지연**이 비용을 지배한다. 여기서는 사면체 하나를 프로그램
하나가 맡아 순전파 한 번, 역전파 한 번으로 끝낸다.
"""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except Exception:                                    # pragma: no cover
    HAVE_TRITON = False

if HAVE_TRITON:

    @triton.jit
    def _fwd_kernel(DPN, ROWS, M, FG, K, E, NT, BLOCK: tl.constexpr):
        t = tl.program_id(0)
        if t < NT:
            acc = 0.0
            for i in range(4):
                ri = tl.load(ROWS + t * 4 + i)
                for a in range(3):
                    ua = tl.load(DPN + ri * 3 + a)
                    # 1 차항 (g - f)
                    acc += tl.load(FG + (t * 4 + i) * 3 + a) * ua
                    # 관성 2 차항
                    for j in range(4):
                        rj = tl.load(ROWS + t * 4 + j)
                        ub = tl.load(DPN + rj * 3 + a)
                        acc += 0.5 * tl.load(M + (t * 4 + i) * 4 + j) * ua * ub
                    # 탄성 2 차항
                    for j in range(4):
                        rj = tl.load(ROWS + t * 4 + j)
                        for b in range(3):
                            ub = tl.load(DPN + rj * 3 + b)
                            kk = tl.load(K + (((t * 4 + i) * 3 + a) * 4 + j)
                                         * 3 + b)
                            acc += 0.5 * kk * ua * ub
            tl.store(E + t, acc)

    @triton.jit
    def _bwd_kernel(DPN, ROWS, M, FG, K, GOUT, DGRAD, NT, BLOCK: tl.constexpr):
        t = tl.program_id(0)
        if t < NT:
            go = tl.load(GOUT)
            for i in range(4):
                ri = tl.load(ROWS + t * 4 + i)
                for a in range(3):
                    d = tl.load(FG + (t * 4 + i) * 3 + a)
                    for j in range(4):
                        rj = tl.load(ROWS + t * 4 + j)
                        ub = tl.load(DPN + rj * 3 + a)
                        d += tl.load(M + (t * 4 + i) * 4 + j) * ub
                        for b in range(3):
                            ubb = tl.load(DPN + rj * 3 + b)
                            kk = tl.load(K + (((t * 4 + i) * 3 + a) * 4 + j)
                                         * 3 + b)
                            d += kk * ubb
                    tl.atomic_add(DGRAD + ri * 3 + a, go * d)


if HAVE_TRITON:

    @triton.jit
    def _matvec_kernel(U, ROWS, M, K, OUT, NT, BLOCK: tl.constexpr):
        """A u 를 셀 조립으로 (행렬을 만들지 않는다). 커널 한 번."""
        t = tl.program_id(0)
        if t < NT:
            for i in range(4):
                ri = tl.load(ROWS + t * 4 + i)
                for a in range(3):
                    d = 0.0
                    for j in range(4):
                        rj = tl.load(ROWS + t * 4 + j)
                        d += tl.load(M + (t * 4 + i) * 4 + j) * tl.load(
                            U + rj * 3 + a)
                        for b in range(3):
                            kk = tl.load(K + (((t * 4 + i) * 3 + a) * 4 + j)
                                         * 3 + b)
                            d += kk * tl.load(U + rj * 3 + b)
                    tl.atomic_add(OUT + ri * 3 + a, d)

    @triton.jit
    def _scatter_kernel(V, ROWS, OUT, NT, BLOCK: tl.constexpr):
        t = tl.program_id(0)
        if t < NT:
            for i in range(4):
                ri = tl.load(ROWS + t * 4 + i)
                for a in range(3):
                    tl.atomic_add(OUT + ri * 3 + a,
                                  tl.load(V + (t * 4 + i) * 3 + a))


def matvec(u, rows, M, K):
    """A u (관성+탄성 2차형식의 행렬-벡터 곱). 커널 한 번."""
    out = torch.zeros_like(u)
    _matvec_kernel[(rows.shape[0],)](u, rows, M, K, out, rows.shape[0],
                                     BLOCK=1)
    return out


def scatter(v, rows, M_nodes):
    """셀별 [T,4,3] 값을 노드로 더한다. 커널 한 번."""
    out = torch.zeros(M_nodes, 3, device=v.device, dtype=v.dtype)
    _scatter_kernel[(rows.shape[0],)](v.contiguous(), rows, out,
                                      rows.shape[0], BLOCK=1)
    return out


def solve_cg(rows, M, K, FG, n_nodes, free, u0=None, iters=200, tol=1e-12):
    """1/2 u^T A u + FG . u 를 최소화한다 (고정 노드는 그대로 둔다).

    A u 는 커널 한 번, 나머지는 벡터 연산 몇 개뿐이라 반복당 발사 수가
    파이토치 자동미분 경로(수천)에서 몇 개로 떨어진다.
    """
    rows32 = rows.to(torch.int32).contiguous()
    b = -scatter(FG, rows32, n_nodes)
    u = torch.zeros(n_nodes, 3, device=M.device, dtype=M.dtype) \
        if u0 is None else u0.clone()
    fm = free.reshape(-1, 1).to(u.dtype)
    r = (b - matvec(u, rows32, M, K)) * fm
    p = r.clone()
    rs = float((r * r).sum())
    for _ in range(int(iters)):
        if rs < tol:
            break
        Ap = matvec(p, rows32, M, K) * fm
        pAp = float((p * Ap).sum())
        if pAp <= 0:
            break
        al = rs / pAp
        u = u + al * p
        r = r - al * Ap
        rs2 = float((r * r).sum())
        p = r + (rs2 / max(rs, 1e-30)) * p
        rs = rs2
    return u


class CellPolyEnergy(torch.autograd.Function):
    """E = sum_t E_t. dpn [M,3] 에 대해 미분 가능."""

    @staticmethod
    def forward(ctx, dpn, rows, M, FG, K):
        nt = rows.shape[0]
        e = torch.empty(nt, device=dpn.device, dtype=dpn.dtype)
        _fwd_kernel[(nt,)](dpn, rows, M, FG, K, e, nt, BLOCK=1)
        ctx.save_for_backward(dpn, rows, M, FG, K)
        return e.sum()

    @staticmethod
    def backward(ctx, go):
        dpn, rows, M, FG, K = ctx.saved_tensors
        gd = torch.zeros_like(dpn)
        nt = rows.shape[0]
        _bwd_kernel[(nt,)](dpn, rows, M, FG, K, go.reshape(1).contiguous(),
                           gd, nt, BLOCK=1)
        return gd, None, None, None, None


def energy(dpn, rows, M, FG, K):
    """커널 한 번으로 에너지. Triton 이 없으면 파이토치로 되돌린다."""
    if HAVE_TRITON and dpn.is_cuda:
        return CellPolyEnergy.apply(dpn, rows, M.contiguous(),
                                    FG.contiguous(), K.contiguous())
    U = dpn[rows]
    return (0.5 * torch.einsum("tij,tia,tja->t", M, U, U)
            + torch.einsum("tia,tia->t", FG, U)
            + 0.5 * torch.einsum("tiajb,tia,tjb->t", K, U, U)).sum()

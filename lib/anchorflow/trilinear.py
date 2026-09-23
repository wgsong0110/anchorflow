"""가우시안 <-> 격자 전달을 trilinear 로만 한다 (MPM 의 P2G/G2P 와 같은 구조).

가우시안은 자기가 들어 있는 육면체의 **꼭짓점 8개**에만 기여하고, 이동량도 같은
8개의 변위를 보간해 받는다. 이웃 탐색이 없고, 집계와 스키닝이 **같은 가중치**를 쓴다.

학습되는 반경·온도가 없으므로 변형 사상의 세밀함은 격자 해상도가 정한다.
"""
from __future__ import annotations

import torch

# 육면체 꼭짓점 오프셋 (순서 고정)
_C = torch.tensor([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
                   [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])


def corners(x, lo, h, n):
    """가우시안마다 (꼭짓점 평탄인덱스 [N,8], trilinear 가중치 [N,8])."""
    dev = x.device
    c8 = _C.to(dev)
    u = (x - lo) / h
    i0 = torch.floor(u).long()
    i0 = torch.stack([i0[:, d].clamp(0, int(n[d]) - 2) for d in range(3)], -1)
    f = (u - i0.to(u.dtype)).clamp(0, 1)                      # [N,3]
    idx = i0.unsqueeze(1) + c8.unsqueeze(0)                   # [N,8,3]
    w = torch.ones(x.shape[0], 8, device=dev, dtype=x.dtype)
    for d in range(3):
        fd = f[:, d].unsqueeze(1)
        cd = c8[:, d].to(x.dtype).unsqueeze(0)
        w = w * (cd * fd + (1.0 - cd) * (1.0 - fd))
    flat = (idx[:, :, 0] * int(n[1]) + idx[:, :, 1]) * int(n[2]) + idx[:, :, 2]
    return flat, w


def active(flat):
    """쓰이는 격자점만 남긴다 -> (행번호 [N,8], 평탄인덱스 [M])."""
    uniq, inv = torch.unique(flat.reshape(-1), return_inverse=True)
    return inv.reshape(flat.shape), uniq


def rows_for(flat, uniq):
    """고정된 활성 집합 uniq 에 대해 평탄인덱스를 행 번호로 옮긴다."""
    import torch as _t
    r = _t.searchsorted(uniq, flat.reshape(-1)).clamp(max=uniq.numel() - 1)
    return r.reshape(flat.shape)


def unflatten(uniq, n, lo, h, dtype):
    nz, ny = int(n[2]), int(n[1])
    cz = uniq % nz
    cy = (uniq // nz) % ny
    cx = uniq // (ny * nz)
    co = torch.stack([cx, cy, cz], -1)
    return co, (co.to(dtype) + 0.5 * 0.0) * h + lo      # 격자점은 칸 모서리다


def p2g(rows, w, vals, M):
    """가우시안 값 [N,F] 를 trilinear 가중으로 격자점에 누적 -> [M,F]."""
    N, K = rows.shape
    out = torch.zeros(M, vals.shape[-1], device=vals.device, dtype=vals.dtype)
    src = (w.unsqueeze(-1) * vals.unsqueeze(1)).reshape(N * K, -1)
    return out.index_add_(0, rows.reshape(-1), src)


def g2p(rows, w, grid_vals):
    """격자점 값 [M,F] 를 가우시안으로 보간 -> [N,F]."""
    return (w.unsqueeze(-1) * grid_vals[rows]).sum(1)


def _inv3(A):
    import torch as _t
    d = _t.linalg.det(A)
    return _t.linalg.inv(A + 1e-9 * _t.eye(3, device=A.device)), d


def tri_feats(x, v, X, m, rows, w, M, pa, h):
    """격자점마다 통계를 trilinear 가중으로 쌓는다.

    aggregate() 와 같은 항목을 내되 가중치가 (질량 x trilinear) 다.
    """
    import torch as _t
    dev = x.device
    wm = (w * m.unsqueeze(1))                                  # [N,8]
    ones = _t.ones_like(wm)

    def acc(vals):                                             # vals [N,8,F]
        F = vals.shape[-1]
        out = _t.zeros(M, F, device=dev, dtype=vals.dtype)
        return out.index_add_(0, rows.reshape(-1), vals.reshape(-1, F))

    wmf = wm.unsqueeze(-1)
    g1 = acc(_t.cat([wmf, wmf * x.unsqueeze(1), wmf * X.unsqueeze(1),
                     wmf * v.unsqueeze(1), ones.unsqueeze(-1)], -1))
    Wa = g1[:, 0].clamp(min=1e-12)
    Wi = Wa.unsqueeze(-1)
    cx, cX, cv = g1[:, 1:4] / Wi, g1[:, 4:7] / Wi, g1[:, 7:10] / Wi
    cnt = g1[:, 10:11]

    cxg, cXg, cvg = cx[rows], cX[rows], cv[rows]               # [N,8,3]
    dx = x.unsqueeze(1) - cxg
    dX = X.unsqueeze(1) - cXg
    dv = v.unsqueeze(1) - cvg
    K = rows.shape[1]
    ww = wm.reshape(-1, K, 1, 1)
    g2 = acc(_t.cat([
        (ww * (dx.unsqueeze(-1) * dx.unsqueeze(-2))).reshape(-1, K, 9),
        wmf * _t.cross(dx, dv, dim=-1),
        (ww * (dx.unsqueeze(-1) * dX.unsqueeze(-2))).reshape(-1, K, 9),
        (ww * (dX.unsqueeze(-1) * dX.unsqueeze(-2))).reshape(-1, K, 9)], -1))
    S = g2[:, :9].reshape(M, 3, 3) / Wa.reshape(M, 1, 1)
    iu = _t.triu_indices(3, 3, device=dev)
    S6 = S[:, iu[0], iu[1]] / (h * h)
    L = g2[:, 9:12]
    tr = S.diagonal(dim1=-2, dim2=-1).sum(-1).reshape(M, 1, 1)
    I3 = _t.eye(3, device=dev)
    Ii, _ = _inv3((tr * I3 - S) * Wa.reshape(M, 1, 1) + 1e-8 * I3)
    om = (Ii @ L.unsqueeze(-1)).squeeze(-1)
    A = g2[:, 12:21].reshape(M, 3, 3)
    B = g2[:, 21:30].reshape(M, 3, 3)
    Bi, _ = _inv3(B + (1e-6 * h * h) * I3)
    Fa = (A @ Bi).clamp(-20.0, 20.0)
    detF = _t.linalg.det(Fa).reshape(M, 1)
    return _t.cat([
        _t.log(Wa).reshape(M, 1), _t.log1p(cnt), (cx - cX) / h, (cx - pa) / h,
        cv, S6, om, Fa.reshape(M, 9),
        _t.sign(detF) * _t.log(detF.abs().clamp(min=1e-6))], -1)


def cell_index(x, lo, h, n):
    """가우시안이 속한 **셀 하나**의 인덱스. (rows [N,1], w [N,1]=1, 셀격자 크기)"""
    import torch as _t
    ci = _t.floor((x - lo) / h).long()
    nc = [max(int(n[d]) - 1, 1) for d in range(3)]
    ci = _t.stack([ci[:, d].clamp(0, nc[d] - 1) for d in range(3)], -1)
    flat = (ci[:, 0] * nc[1] + ci[:, 1]) * nc[2] + ci[:, 2]
    return flat.unsqueeze(1), _t.ones(x.shape[0], 1, device=x.device,
                                      dtype=x.dtype), nc

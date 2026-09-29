"""입자 -> **셀 특징** 집계 (망 입력 쪽 전용).

격자-입자 **전달**(격자점 변위를 가우시안에 입히는 것)은 이 파일에 없다 --
trilinear 전달은 셀이 뒤집혀 겹칠 수 있어 제거됐고, 전달은 Kuhn 사면체
barycentric(sitreg_warp.bary_g2p) 하나뿐이다. 여기 남은 것은 입자 물리량을
셀로 모으는 가중 평균이라 기하를 변형하지 않으며 접힘과 무관하다.
"""
from __future__ import annotations

import torch

# 육면체 꼭짓점 오프셋 (순서 고정)
_C = torch.tensor([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
                   [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])


def _inv3(A):
    """3x3 역행렬. 정규화를 **행렬 크기에 비례**시킨다.

    빈 칸이나 입자가 한둘뿐인 셀에서는 관성텐서가 특이해지는데, 절대 상수
    1e-9 로는 질량 스케일(1e-6 수준)에 비해 너무 작아 역행렬이 터진다
    (전 조합 학습에서 실제로 LinAlgError 로 죽었다). 대각합에 비례한 능선을
    더하고, 그래도 특이하면 유사역행렬로 물러선다.
    """
    import torch as _t
    I3 = _t.eye(3, device=A.device, dtype=A.dtype)
    d = _t.linalg.det(A)
    scale = A.diagonal(dim1=-2, dim2=-1).abs().sum(-1, keepdim=True
                                                   ).unsqueeze(-1) / 3.0
    # 능선이 너무 작으면(빈 셀) 역행렬이 1e20 급으로 커져 뒤에서 float32 가
    # 넘친다 -- 실제로 전 조합 학습이 NaN 으로 죽었다. 하한을 두고, 질량이
    # 사실상 0 인 셀은 역행렬을 0 으로 둔다 (그 셀은 어차피 기여가 없다).
    # 능선을 **절대로 0 에 가깝게 두지 않는다**. 예전에는 빈 셀을 where 로
    # 가렸는데, 그러면 순전파는 멀쩡해도 역전파에서 NaN 이 그대로 새어 나온다
    # (0 * NaN = NaN). 전체 평균 크기를 바닥으로 깔아 항상 잘 정의되게 한다.
    # 바닥을 **가장 큰 셀 기준**으로 잡는다. 평균으로 잡으면 얇은 형상(lego 처럼
    # 판이 얇은 경우)에서 거의 빈 셀의 역행렬이 여전히 1e6 배로 커져 float32 가
    # 넘친다.
    floor = scale.max().clamp_min(1e-12) * 1e-6
    Ar = A + (1e-6 * scale + floor) * I3
    try:
        return _t.linalg.inv(Ar), d
    except Exception:
        return _t.linalg.pinv(Ar), d


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

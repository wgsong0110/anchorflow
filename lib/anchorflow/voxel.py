"""복셀 다운샘플링으로 앵커를 매 프레임 새로 뽑는다.

앞선 구성은 앵커를 t=0 에 FPS 로 고르고 모델이 낸 변위로만 옮겼고, 매 프레임
어느 가우시안이 어느 앵커에 속하는지를 kNN 으로 다시 찾았다. 비용이 여기 몰려
있었다 (입자 138 만 기준: FPS 74 ms, kNN 5 ms, 집계 13 ms).

복셀로 뽑으면 그 셋이 하나로 접힌다.

  앵커 선정   현재 위치를 한 변 H 인 격자에 넣고, 점유된 칸마다 앵커 하나.
              정렬·유일화 한 번이면 끝이고 FPS 가 필요 없다.
  소속        가우시안의 앵커는 자기 칸이다. 이웃까지 보려면 3x3x3 만 보면
              되므로 후보가 512 개에서 27 개로 준다 -- 탐색이 아니라 색인이다.
  집계        같은 유일화가 내주는 역색인으로 흩뿌리면 되고, 짝이 N x k 가 아니라
              N 개다 (16 배 적다).

그리고 앵커가 매 프레임 재질을 따라 다시 놓이므로, 조각이 떨어져 나가도 앵커가
따라간다 -- 고정 앵커를 변위로만 옮길 때 생기던 "앵커가 재질에서 이탈하면 돌아올
길이 없다" 가 사라진다.

대가는 불연속이다. 가우시안이 칸 경계를 넘는 순간 소속이 바뀌므로 변형장이 떨릴
수 있다. 앵커 위치를 칸 중심이 아니라 **그 칸 가우시안들의 질량중심**으로 두어
칸 안에서는 매끄럽게 움직이게 했지만, 경계에서의 전환 자체는 남는다 -- 실제로
문제가 되는지는 exe/verify_voxel.py 가 잰다.
"""
from __future__ import annotations

import torch

try:
    import deformcuda as _dc
    _HAVE_DC = _dc.HAVE_CUDA
except Exception:
    _dc, _HAVE_DC = None, False


class VoxelAnchors:
    """한 프레임의 복셀 앵커와 그 통계.

    keys  [M] 정렬된 유일 복셀 키   pos [M,3] 앵커 위치(질량중심)
    inv   [N] 각 가우시안의 앵커 번호
    """

    __slots__ = ("keys", "pos", "inv", "moments", "lo", "cell", "D1", "D2", "M")


def build(x, X, v, m, cell, lo=None):
    """현재 배치에서 복셀 앵커를 뽑고, 같은 패스에서 통계까지 낸다.

    x [N,3] 현재 위치, X [N,3] 정준 위치, v [N,3] 속도, m [N] 질량.
    -> VoxelAnchors
    """
    dev = x.device
    if lo is None:
        lo = x.min(0).values - cell
    vi = ((x - lo) / cell).floor().long()
    D1 = int(vi[:, 1].max()) + 2
    D2 = int(vi[:, 2].max()) + 2
    key = (vi[:, 0] * D1 + vi[:, 1]) * D2 + vi[:, 2]
    keys, inv = torch.unique(key, sorted=True, return_inverse=True)
    M = keys.numel()

    # 집계: 짝이 아니라 가우시안당 하나라 흩뿌리기가 16 배 적다. 1 차 모멘트를
    # 먼저 내고(앵커 위치가 거기서 나온다), 그 중심 기준으로 2 차를 낸다.
    mu = m.unsqueeze(-1)
    g1 = torch.zeros(M, 11, device=dev).index_add_(
        0, inv, torch.cat([mu, mu * x, mu * X, mu * v,
                           torch.ones_like(mu)], -1))
    W = g1[:, 0].clamp(min=1e-12)
    Wi = W.unsqueeze(-1)
    cx, cX, cv = g1[:, 1:4] / Wi, g1[:, 4:7] / Wi, g1[:, 7:10] / Wi
    dx, dX, dv = x - cx[inv], X - cX[inv], v - cv[inv]
    g2 = torch.zeros(M, 30, device=dev).index_add_(
        0, inv, torch.cat([
            (mu.reshape(-1, 1, 1) * (dx.unsqueeze(-1) * dx.unsqueeze(-2))
             ).reshape(-1, 9),
            mu * torch.cross(dx, dv, dim=-1),
            (mu.reshape(-1, 1, 1) * (dx.unsqueeze(-1) * dX.unsqueeze(-2))
             ).reshape(-1, 9),
            (mu.reshape(-1, 1, 1) * (dX.unsqueeze(-1) * dX.unsqueeze(-2))
             ).reshape(-1, 9)], -1))

    a = VoxelAnchors()
    a.keys, a.inv, a.M = keys, inv, M
    a.pos = cx.contiguous()
    a.lo, a.cell, a.D1, a.D2 = lo, float(cell), D1, D2
    a.moments = (W, cx, cX, cv, g1[:, 10:11], g2)
    return a


def neighbors(x, va, k, radius=1):
    """가우시안마다 이웃 복셀의 앵커 중 가까운 k 개. 빈 자리는 -1.

    후보가 (2r+1)^3 개뿐이라 전역 탐색이 아니다. 커널이 없으면 파이토치로
    같은 것을 계산한다 -- 값이 같아야 하므로 검증에 쓴다.
    """
    if _HAVE_DC and x.is_cuda and x.dtype == torch.float32:
        return _dc.voxel_knn(x, va.pos, va.keys, va.D1, va.D2,
                             float(va.lo[0]), float(va.lo[1]), float(va.lo[2]),
                             va.cell, int(k), int(radius))
    dev = x.device
    vi = ((x - va.lo) / va.cell).floor().long()
    r = torch.arange(-radius, radius + 1, device=dev)
    off = torch.stack(torch.meshgrid(r, r, r, indexing="ij"), -1).reshape(-1, 3)
    nb = vi.unsqueeze(1) + off.unsqueeze(0)                      # [N,C,3]
    q = (nb[..., 0] * va.D1 + nb[..., 1]) * va.D2 + nb[..., 2]
    pos_in = torch.searchsorted(va.keys, q.reshape(-1)).clamp(max=va.M - 1)
    hit = va.keys[pos_in].reshape(q.shape) == q
    aid = torch.where(hit, pos_in.reshape(q.shape), torch.full_like(q, -1))
    d = torch.where(hit, (x.unsqueeze(1) - va.pos[aid.clamp(min=0)]).norm(dim=-1),
                    torch.full(q.shape, float("inf"), device=dev))
    kk = min(k, d.shape[1])
    dv, di = d.topk(kk, dim=1, largest=False)
    gi = torch.gather(aid, 1, di)
    gi = torch.where(torch.isfinite(dv), gi, torch.full_like(gi, -1))
    dv = torch.where(torch.isfinite(dv), dv, torch.zeros_like(dv))
    if kk < k:
        pad = k - kk
        gi = torch.cat([gi, torch.full((gi.shape[0], pad), -1, device=dev,
                                       dtype=gi.dtype)], 1)
        dv = torch.cat([dv, torch.zeros(dv.shape[0], pad, device=dev)], 1)
    return gi, dv

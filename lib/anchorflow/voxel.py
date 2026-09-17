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

    __slots__ = ("keys", "pos", "inv", "moments", "lo", "cell", "D1", "D2", "M",
                 "offs", "stride", "L")


def _bspline_w(t):
    """2 차 B-스플라인 가중. t 는 노드 기준 거리(칸 단위), |t| <= 1.5 에서만 0 이 아니다.

    MPM 이 입자를 격자에 뿌릴 때 쓰는 바로 그 가중이다. 하드 배정과 달리 경계에서
    0 으로 매끄럽게 죽으므로, 입자가 칸을 넘어가도 기여가 **연속**으로 옮겨간다.
    """
    a = t.abs()
    return torch.where(a < 0.5, 0.75 - a * a,
                       torch.where(a < 1.5, 0.5 * (1.5 - a) ** 2,
                                   torch.zeros_like(a)))


def build(x, X, v, m, cell, lo=None, soft=False, offsets=None):
    """현재 배치에서 복셀 앵커를 뽑고, 같은 패스에서 통계까지 낸다.

    x [N,3] 현재 위치, X [N,3] 정준 위치, v [N,3] 속도, m [N] 질량.
    soft   True 면 자기 칸 하나가 아니라 이웃 3x3x3 에 B-스플라인 가중으로 뿌린다.
           앵커 위치가 가중 질량중심이 되어 입자가 칸을 넘어도 **정확히 연속**이다.
           흩뿌리기가 27 배 늘지만, 불연속을 근본에서 없앤다.
    offsets [L,3] 격자를 칸의 몇 분의 몇만큼 옮길지. 여러 개를 주면 **한 번에**
           처리한다 -- 키에 격자 번호를 실어 하나의 정렬된 배열에 담으므로,
           입자 좌표를 한 번만 읽고 커널도 한 번만 뜬다. 파이썬으로 L 번 돌면
           그만큼 배로 든다.
    -> VoxelAnchors
    """
    dev = x.device
    if lo is None:
        # 원점은 **공간에 고정**해야 한다. 프레임마다 바운딩박스에서 새로 잡으면
        # 물체가 떨어지는 것만으로 모든 복셀 키가 바뀌어, 소속이 실제로 변한 것과
        # 격자가 따라 움직인 것을 구별할 수 없다 (그렇게 재서 70% 가 나왔다).
        lo = x.min(0).values - cell
    if offsets is None:
        offs = torch.zeros(1, 3, device=dev, dtype=x.dtype)
    else:
        offs = torch.as_tensor(offsets, device=dev, dtype=x.dtype).reshape(-1, 3)
    L = offs.shape[0]
    # 격자별 칸 번호. 범위는 모든 격자를 덮도록 한 번에 잡는다.
    vi0 = ((x - lo) / cell).floor().long()
    D1 = int(vi0[:, 1].max()) + 3
    D2 = int(vi0[:, 2].max()) + 3
    stride = (int(vi0[:, 0].max()) + 3) * D1 * D2
    vi = vi0
    use_hash = _HAVE_DC and x.is_cuda and x.dtype == torch.float32
    if use_hash:
        # 키를 파이토치에서 만들지도, 정렬하지도 않는다 -- 커널이 해시로 O(N) 에
        # 번호를 매긴다. L 이 커질수록 이득이 커진다.
        key = None
    else:
        ks = []
        for l in range(L):
            vl = ((x - lo - offs[l] * cell) / cell).floor().long()
            ks.append(l * stride + (vl[:, 0] * D1 + vl[:, 1]) * D2 + vl[:, 2])
        key = torch.cat(ks) if L > 1 else ks[0]
    keys, inv = torch.unique(key, sorted=True, return_inverse=True)
    M = keys.numel()

    # 집계: 짝이 아니라 가우시안당 하나라 흩뿌리기가 16 배 적다. 1 차 모멘트를
    # 먼저 내고(앵커 위치가 거기서 나온다), 그 중심 기준으로 2 차를 낸다.
    # 앵커 집합은 **점유 복셀**이다. 부드러운 배정에서도 앵커를 새로 만들지 않고
    # 이웃 중 점유된 칸에만 가중을 뿌린다 -- 가중이 0 으로 죽는 자리에는 어차피
    # 기여가 없다. 이 규칙 덕분에 unique 를 짝(N x 27)이 아니라 입자(N) 위에서
    # 한 번만 돌면 된다.
    if use_hash:
        keys = _dc.voxel_hash(x, offs, D1, D2, stride,
                              float(lo[0]), float(lo[1]), float(lo[2]),
                              float(cell), max(1024, 4 * L * x.shape[0] // 8))
    else:
        keys = torch.unique(key, sorted=True)
    M = keys.numel()
    if _HAVE_DC and x.is_cuda and x.dtype == torch.float32:
        g1, g2, g3, cx, cX, cv = _dc.voxel_moments(
            x, X, v, m, keys, offs, D1, D2, stride,
            float(lo[0]), float(lo[1]), float(lo[2]), float(cell), soft)
        W = g1[:, 0].clamp(min=1e-12)
        cnt = g1[:, 10:11]
        g2 = torch.cat([g2, g3], -1)
        inv = None
    else:
        inv = torch.searchsorted(keys, key)
        if soft:
            r = torch.arange(-1, 2, device=dev)
            off = torch.stack(torch.meshgrid(r, r, r, indexing="ij"),
                              -1).reshape(-1, 3)
            u = (x - lo) / cell - 0.5
            nb = vi.unsqueeze(1) + off.unsqueeze(0)
            wt = _bspline_w(u.unsqueeze(1) - nb.to(x.dtype)).prod(-1)
            qk = (nb[..., 0] * D1 + nb[..., 1]) * D2 + nb[..., 2]
            pin = torch.searchsorted(keys, qk.reshape(-1)).clamp(max=M - 1)
            live = (keys[pin].reshape(qk.shape) == qk) & (wt > 1e-9)
            gi = torch.nonzero(live, as_tuple=True)[0]
            pair_inv = pin.reshape(qk.shape)[live]
            mw = (m[gi] * wt[live]).unsqueeze(-1)
            src_x, src_X, src_v = x[gi], X[gi], v[gi]
        else:
            pair_inv, mw = inv, m.unsqueeze(-1)
            src_x, src_X, src_v = x, X, v
        g1 = torch.zeros(M, 11, device=dev).index_add_(
            0, pair_inv, torch.cat([mw, mw * src_x, mw * src_X, mw * src_v,
                                    torch.ones_like(mw)], -1))
        W = g1[:, 0].clamp(min=1e-12)
        Wi = W.unsqueeze(-1)
        cx, cX, cv = g1[:, 1:4] / Wi, g1[:, 4:7] / Wi, g1[:, 7:10] / Wi
        cnt = g1[:, 10:11]
        dx, dX, dv = (src_x - cx[pair_inv], src_X - cX[pair_inv],
                      src_v - cv[pair_inv])
        g2 = torch.zeros(M, 30, device=dev).index_add_(
            0, pair_inv, torch.cat([
                (mw.reshape(-1, 1, 1) * (dx.unsqueeze(-1) * dx.unsqueeze(-2))
                 ).reshape(-1, 9),
                mw * torch.cross(dx, dv, dim=-1),
                (mw.reshape(-1, 1, 1) * (dx.unsqueeze(-1) * dX.unsqueeze(-2))
                 ).reshape(-1, 9),
                (mw.reshape(-1, 1, 1) * (dX.unsqueeze(-1) * dX.unsqueeze(-2))
                 ).reshape(-1, 9)], -1))

    a = VoxelAnchors()
    a.keys, a.inv, a.M = keys, inv, M
    a.pos = cx.contiguous()
    a.lo, a.cell, a.D1, a.D2 = lo, float(cell), D1, D2
    a.offs, a.stride, a.L = offs, stride, L
    a.moments = (W, cx, cX, cv, cnt, g2)
    return a


def neighbors(x, va, k, radius=1):
    """가우시안마다 이웃 복셀의 앵커 중 가까운 k 개. 빈 자리는 -1.

    후보가 (2r+1)^3 개뿐이라 전역 탐색이 아니다. 커널이 없으면 파이토치로
    같은 것을 계산한다 -- 값이 같아야 하므로 검증에 쓴다.
    """
    if _HAVE_DC and x.is_cuda and x.dtype == torch.float32:
        return _dc.voxel_knn(x, va.pos, va.keys, va.offs, va.D1, va.D2,
                             va.stride,
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

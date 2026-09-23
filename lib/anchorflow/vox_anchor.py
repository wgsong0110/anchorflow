"""복셀 중심을 앵커로 쓰고, 가우시안마다 kNN 앵커를 **탐색 없이** 구한다.

앵커가 규칙 격자 위에 있으므로 가우시안이 속한 칸은 나눗셈 한 번으로 나오고,
그 주변 칸 중심까지의 거리는 닫힌 형태다. 따라서 KD 트리나 전수 거리 계산이
필요 없고 가우시안당 상수 시간이다.

앵커는 **모든 칸**이다 (빈 칸 포함). 점유 칸만 쓰면 후보마다 점유 여부를 조회해야
하고 경계에서 k 개를 못 채우는 일이 생긴다.
"""
from __future__ import annotations

import torch


def grid_for(x, n_anchors, pad=1e-4):
    """가우시안 구름을 덮는 격자를 만든다. 칸 수가 n_anchors 에 가깝도록 크기를 정한다."""
    lo = x.min(0).values - pad
    hi = x.max(0).values + pad
    ext = (hi - lo).clamp(min=1e-6)
    h = float((ext.prod() / max(n_anchors, 1)) ** (1.0 / 3.0))
    n = torch.clamp((ext / h).ceil().long(), min=1)
    return lo, h, n


def centers(lo, h, n):
    """[nx*ny*nz, 3] 칸 중심. 인덱스는 (ix*ny + iy)*nz + iz 순."""
    dev = lo.device
    ax = [(torch.arange(int(n[d]), device=dev, dtype=lo.dtype) + 0.5) * h + lo[d]
          for d in range(3)]
    g = torch.stack(torch.meshgrid(ax[0], ax[1], ax[2], indexing="ij"), -1)
    return g.reshape(-1, 3)


def knn(x, lo, h, n, k, rad=1):
    """탐색 없이 kNN 앵커 인덱스 [N,k] 를 돌려준다.

    rad 는 후보 이웃 반경이다. (2*rad+1)^3 개 후보에서 거리로 k 개를 고른다.
    k=16 이면 rad=1 (27 개) 로 충분하다.
    """
    dev = x.device
    ci = torch.floor((x - lo) / h).long()
    ci = torch.stack([ci[:, d].clamp(0, int(n[d]) - 1) for d in range(3)], -1)
    off = torch.arange(-rad, rad + 1, device=dev)
    o = torch.stack(torch.meshgrid(off, off, off, indexing="ij"), -1).reshape(-1, 3)
    cand = ci.unsqueeze(1) + o.unsqueeze(0)                      # [N,C,3]
    for d in range(3):                                           # 격자 밖은 접어 넣는다
        cand[:, :, d] = cand[:, :, d].clamp(0, int(n[d]) - 1)
    cc = (cand.to(x.dtype) + 0.5) * h + lo                       # 후보 중심
    d2 = ((x.unsqueeze(1) - cc) ** 2).sum(-1)                    # [N,C]
    kk = min(k, d2.shape[1])
    sel = d2.topk(kk, dim=1, largest=False).indices              # [N,k]
    pick = torch.gather(cand, 1, sel.unsqueeze(-1).expand(-1, -1, 3))
    flat = (pick[:, :, 0] * int(n[1]) + pick[:, :, 1]) * int(n[2]) + pick[:, :, 2]
    if kk < k:                                                   # 모자라면 가장 가까운 것으로 채운다
        flat = torch.cat([flat, flat[:, :1].expand(-1, k - kk)], 1)
    return flat


def occupied(x, lo, h, n):
    """가우시안이 들어 있는 칸만 돌려준다 -> (coords [M,3] long, centers [M,3], flat->행 사상)."""
    import torch as _t
    ci = _t.floor((x - lo) / h).long()
    ci = _t.stack([ci[:, d].clamp(0, int(n[d]) - 1) for d in range(3)], -1)
    flat = (ci[:, 0] * int(n[1]) + ci[:, 1]) * int(n[2]) + ci[:, 2]
    uniq, inv = _t.unique(flat, return_inverse=True)
    cz = uniq % int(n[2])
    cy = (uniq // int(n[2])) % int(n[1])
    cx = uniq // (int(n[1]) * int(n[2]))
    coords = _t.stack([cx, cy, cz], -1)
    cen = (coords.to(x.dtype) + 0.5) * h + lo
    return coords, cen, uniq


def knn_occ(x, lo, h, n, k, uniq, rad=2):
    """점유 칸만 앵커일 때의 kNN. 후보를 (2rad+1)^3 에서 뽑고 점유된 것만 남긴다."""
    import torch as _t
    dev = x.device
    ci = _t.floor((x - lo) / h).long()
    ci = _t.stack([ci[:, d].clamp(0, int(n[d]) - 1) for d in range(3)], -1)
    off = _t.arange(-rad, rad + 1, device=dev)
    o = _t.stack(_t.meshgrid(off, off, off, indexing="ij"), -1).reshape(-1, 3)
    cand = ci.unsqueeze(1) + o.unsqueeze(0)
    for d in range(3):
        cand[:, :, d] = cand[:, :, d].clamp(0, int(n[d]) - 1)
    cf = (cand[:, :, 0] * int(n[1]) + cand[:, :, 1]) * int(n[2]) + cand[:, :, 2]
    row = _t.searchsorted(uniq, cf.reshape(-1)).clamp(max=uniq.numel() - 1)
    ok = uniq[row] == cf.reshape(-1)
    row = row.reshape(cf.shape)
    ok = ok.reshape(cf.shape)
    cc = (cand.to(x.dtype) + 0.5) * h + lo
    d2 = ((x.unsqueeze(1) - cc) ** 2).sum(-1)
    d2 = _t.where(ok, d2, _t.full_like(d2, float("inf")))
    kk = min(k, d2.shape[1])
    sel = d2.topk(kk, dim=1, largest=False).indices
    out = _t.gather(row, 1, sel)
    # 비어 있던 자리는 가장 가까운 점유 칸으로 채운다
    bad = ~_t.gather(ok, 1, sel)
    out = _t.where(bad, out[:, :1].expand_as(out), out)
    if kk < k:
        out = _t.cat([out, out[:, :1].expand(-1, k - kk)], 1)
    return out


def knn_union(x, lo, h, n, k, rad=1):
    """전체 칸을 앵커로 두되, **어떤 가우시안의 kNN 에 뽑힌 칸만** 활성으로 남긴다.

    값이 0 인 칸을 연산에서 빼는 것뿐이라 "전체 복셀이 앵커" 라는 성질은 그대로다.
    돌려주는 것: (idx [N,k] 활성 행번호, coords [M,3], centers [M,3])
    """
    import torch as _t
    flat = knn(x, lo, h, n, k, rad=rad)                    # [N,k] 전역 칸 번호
    uniq, inv = _t.unique(flat.reshape(-1), return_inverse=True)
    idx = inv.reshape(flat.shape)
    nz = int(n[2])
    ny = int(n[1])
    cz = uniq % nz
    cy = (uniq // nz) % ny
    cx = uniq // (ny * nz)
    coords = _t.stack([cx, cy, cz], -1)
    cen = (coords.to(x.dtype) + 0.5) * h + lo
    return idx, coords, cen

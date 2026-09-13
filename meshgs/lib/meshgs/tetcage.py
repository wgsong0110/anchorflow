"""3DGS -> 사면체 케이지 -> 가우시안 결속 (VR-GS 계열 파이프라인의 핵심).

VR-GS 도 UniMGS 도 2606.00444 도 코드를 공개하지 않아, 세 논문이 공통으로 쓰는
구조만 직접 만든다:

  1. 가우시안 중심에서 점유 격자를 만들고 marching cubes 로 표면을 뽑는다.
  2. 그 표면 안을 사면체로 채운다 (여기서는 격자 -> BCC 사면체 분할).
  3. 가우시안을 자기가 들어 있는 사면체에 **무게중심 좌표로 결속**한다.
     이후 사면체 정점이 움직이면 가우시안 중심은 무게중심 보간으로,
     공분산은 그 사면체의 변형 구배 F 로 따라간다 (Sigma' = F Sigma F^T).

위상은 고정이다 -- 리메싱이 없다. 이것이 소성 시험의 핵심 조건이다.
"""
from __future__ import annotations

import numpy as np
import torch


def occupancy(xyz, res=64, pad=2):
    """가우시안 중심에서 점유 격자. (occ [R,R,R], 원점, 격자간격)"""
    lo = xyz.min(0).values
    hi = xyz.max(0).values
    ext = (hi - lo).max()
    h = float(ext) / (res - 2 * pad)
    org = lo - pad * h
    idx = ((xyz - org) / h).long().clamp(0, res - 1)
    occ = torch.zeros(res, res, res, dtype=torch.bool, device=xyz.device)
    occ[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return occ, org, h


def dilate_fill(occ, iters=1):
    """얇은 껍질을 메워 속이 빈 사면체가 안 생기게 한다."""
    o = occ.clone()
    for _ in range(iters):
        p = torch.nn.functional.max_pool3d(
            o.float()[None, None], kernel_size=3, stride=1, padding=1)[0, 0]
        o = p > 0
    return o


def cube_to_tets(i, j, k):
    """격자 셀 하나를 사면체 6 개로 (정점 인덱스는 (i,j,k) 오프셋)."""
    c = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
         (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
    tets = [(0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6),
            (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6)]
    return c, tets


def build_tet_mesh(occ, org, h):
    """점유 격자 -> 사면체 메시 (정점 [V,3], 사면체 [T,4]). 위상 고정."""
    dev = occ.device
    R = occ.shape[0]
    cell = occ[:-1, :-1, :-1] & occ[1:, :-1, :-1] & occ[1:, 1:, :-1] \
        & occ[:-1, 1:, :-1] & occ[:-1, :-1, 1:] & occ[1:, :-1, 1:] \
        & occ[1:, 1:, 1:] & occ[:-1, 1:, 1:]
    ci = torch.nonzero(cell)                                    # [C,3]
    corners, tets = cube_to_tets(0, 0, 0)
    off = torch.tensor(corners, device=dev)                     # [8,3]
    vid = (ci[:, None, :] + off[None]).reshape(-1, 3)           # [C*8,3]
    key = (vid[:, 0] * R + vid[:, 1]) * R + vid[:, 2]
    uk, inv = torch.unique(key, return_inverse=True)
    V = torch.stack([uk // (R * R), (uk // R) % R, uk % R], -1).float()
    V = V * h + org
    corner_idx = inv.reshape(-1, 8)                             # [C,8]
    T = torch.stack([corner_idx[:, list(t)] for t in tets], 1).reshape(-1, 4)
    return V, T


def _bary(P, A, B, C, D):
    """점 P 의 사면체 (A,B,C,D) 무게중심 좌표 [N,4]."""
    T = torch.stack([A - D, B - D, C - D], -1)                  # [N,3,3]
    w = torch.linalg.solve(T, (P - D).unsqueeze(-1)).squeeze(-1)
    return torch.cat([w, 1.0 - w.sum(-1, keepdim=True)], -1)


def bind_gaussians(xyz, V, T, chunk=200_000):
    """각 가우시안을 담고 있는 사면체와 무게중심 좌표를 찾는다.

    셀 격자 구조를 쓰지 않고 가장 가까운 사면체 중심 k 개만 검사한다 --
    사면체가 격자 기반이라 후보가 국소적이다.
    """
    cen = V[T].mean(1)                                          # [T,3]
    from scipy.spatial import cKDTree
    tree = cKDTree(cen.detach().cpu().numpy())
    K = 16
    _, cand = tree.query(xyz.detach().cpu().numpy(), k=K)
    cand = torch.from_numpy(np.atleast_2d(cand)).long().to(xyz.device)
    N = xyz.shape[0]
    tid = torch.full((N,), -1, dtype=torch.long, device=xyz.device)
    bw = torch.zeros(N, 4, device=xyz.device)
    todo = torch.arange(N, device=xyz.device)
    for k in range(K):
        if todo.numel() == 0:
            break
        t = cand[todo, k]
        v = V[T[t]]                                             # [n,4,3]
        w = _bary(xyz[todo], v[:, 0], v[:, 1], v[:, 2], v[:, 3])
        ok = (w >= -1e-6).all(-1)
        sel = todo[ok]
        tid[sel] = t[ok]
        bw[sel] = w[ok]
        todo = todo[~ok]
    if todo.numel():                                            # 밖으로 샌 점은 최근접
        t = cand[todo, 0]
        v = V[T[t]]
        tid[todo] = t
        bw[todo] = _bary(xyz[todo], v[:, 0], v[:, 1], v[:, 2], v[:, 3])
    return tid, bw


def skin(V, T, tid, bw):
    """사면체 정점 -> 가우시안 중심."""
    return (V[T[tid]] * bw.unsqueeze(-1)).sum(1)


def tet_F(V0, V, T):
    """사면체별 변형 구배 F = Ds Dm^{-1} [T,3,3] 와 rest 역행렬."""
    def _D(P):
        return torch.stack([P[:, 0] - P[:, 3], P[:, 1] - P[:, 3],
                            P[:, 2] - P[:, 3]], -1)
    Dm = _D(V0[T])
    return _D(V[T]) @ torch.linalg.inv(Dm), Dm

"""사면체 복합체: **몸대각선을 z 로 세운 고정 Kuhn 분할**.

발판 격자는 정육면체(일반적으로는 능면체)이고, 모든 셀이 **같은 몸대각선
(1,1,1)** 을 공유 대각선으로 쓰는 Kuhn 6-사면체 분할을 한다. 예전에는 셀을
소큐브 8 개로 자르고 대각선을 옥탄트마다 몸중심 쪽으로 꼬아 큐브 대칭 48 개에
불변으로 만들었는데, 지금은 **일부러 방향 의존적으로** 대각선 하나를 z 에
세운다.

그 대각선 방향에서 내려다보면 격자는 xy 에 평행한 **정삼각형 층**이 되고
**ABCABC** 로 쌓인다. 1 층과 4 층의 같은 사영 위치를 잇는 것이 바로 그
몸대각선이다 (세 층 간격). 면내 간격 h 와 층 간격 hz 를 따로 주므로, 같은
조합 구조가 단순입방(정육면체를 [111] 로 본 것)부터 조밀 쌓기까지 덮는다 --
쌓기끼리의 차이는 이 두 간격에만 들어간다.

격자 기저 (열이 a_i):

    a_i = (r*u_i, hz),   r = h/sqrt(3),   u_i 는 120 도 간격 단위벡터
    -> |a_i - a_j| = h (면내 최근접),  sum_i a_i = (0, 0, 3*hz) (몸대각선)
    hz = h/sqrt(6) 이면 정육면체다.

노드는 격자 정수점 **한 종류**다 (예전의 2 배 격자 27 종은 소큐브 분할 때문에
필요했던 것이라 함께 사라졌다).

입자는 자기가 든 사면체의 4 꼭짓점 barycentric 으로 움직인다. 사면체 안이
아핀이라 grad_Phi 가 사면체별 **상수**이고 닫힌 형식이며, det 도 스칼라 하나다.
"""
import math
from collections import namedtuple

import torch

__all__ = ["Lat", "lattice", "grid_for_nodes", "node_pos", "locate", "g2p",
           "g2p_jac", "g2p_pre", "g2p_jac_pre", "tet_det", "tet_id",
           "active_nodes", "edges_of",
           "EDGE_OFFSETS", "N_EDGE_CLASS", "HZ_CUBE"]

# 정육면체가 되는 층간격 비 (hz = HZ_CUBE * h)
HZ_CUBE = 1.0 / math.sqrt(6.0)

# A: 열이 격자 기저, Ai: 역행렬, s: 대표 길이(면내 간격 h)
Lat = namedtuple("Lat", "A Ai s hz")


def lattice(h, hz, device=None, dtype=torch.float32):
    """면내 간격 h, 층 간격 hz 의 능면체 격자."""
    r = float(h) / math.sqrt(3.0)
    c, s3 = 0.5, math.sqrt(3.0) / 2.0
    u = ((1.0, 0.0), (-c, s3), (-c, -s3))
    A = torch.tensor([[r * u[i][0] for i in range(3)],
                      [r * u[i][1] for i in range(3)],
                      [float(hz)] * 3], device=device, dtype=dtype)
    return Lat(A, torch.linalg.inv(A), float(h), float(hz))


def grid_for_nodes(x, n_nodes, margin=0.05, hz_ratio=HZ_CUBE, off=None):
    """점들을 덮는 격자를 잡는다 -> (lo, lat, nn).

    lo 는 격자 정수 원점의 **월드 위치**, nn 은 축별 정수 좌표 개수다.

    격자는 **이산화 선택**이지 물리량이 아니다. x 가 그래프에 있을 때 (K>1
    언롤의 둘째 서브스텝부터) 바운딩박스를 그대로 쓰면 격자 원점이 미분가능해져
    극단 입자 하나가 격자 전체의 기울기를 받는다 -- 그래서 끊는다.
    """
    xd = x.detach()
    lo_ = xd.min(0).values
    hi_ = xd.max(0).values
    ext = (hi_ - lo_).max().clamp_min(1e-12)
    pad = margin * ext
    # n_nodes 는 정수일 필요가 없다 -- 셀 부피를 고정한 채 층 간격만 바꾸려면
    # h 를 연속으로 잡아야 한다 (h ∝ (V/hz_ratio)^(1/3)).
    h = float((ext + 2 * pad) / max(float(n_nodes), 2.0))
    lat = lattice(h, hz_ratio * h, device=x.device, dtype=x.dtype)
    # 패딩된 상자의 8 꼭짓점을 격자 좌표로 보내 정수 범위를 잡는다
    b0, b1 = lo_ - pad, hi_ + pad
    cor = torch.stack([torch.stack([b0[0] if (m & 1) else b1[0],
                                    b0[1] if (m & 2) else b1[1],
                                    b0[2] if (m & 4) else b1[2]])
                       for m in range(8)], 0)                    # [8,3]
    y = cor @ lat.Ai.T
    imin = torch.floor(y.min(0).values).long() - 1
    imax = torch.ceil(y.max(0).values).long() + 1
    nn = (imax - imin + 1).clamp_min(2)
    lo = (lat.A @ imin.to(lat.A.dtype))
    if off is not None:
        # 격자 원점을 셀의 **분수만큼** 민다. 같은 문제를 다른 이산화로 푸는
        # 것이라 빈 셀·고아 셀 패턴이 달라진다 (앙상블용). 민 만큼 덮는 범위가
        # 줄지 않게 정수 칸을 하나 늘린다.
        _o = torch.as_tensor(off, device=lo.device, dtype=lat.A.dtype)
        lo = lo - lat.A @ _o
        nn = nn + 1
    return lo, lat, nn


def node_pos(lo, lat, nn, uniq):
    """압축 노드 색인 -> 월드 위치 [M,3]."""
    nnl = [int(nn[k]) for k in range(3)]
    k2 = uniq % nnl[2]
    k1 = (uniq // nnl[2]) % nnl[1]
    k0 = uniq // (nnl[1] * nnl[2])
    ijk = torch.stack([k0, k1, k2], -1).to(lat.A.dtype)          # [M,3]
    return ijk @ lat.A.T + lo


def locate(q, lo, lat, nn):
    """점 -> (사면체 4 꼭짓점 평탄색인 [N,4], barycentric [N,4], 보조).

    셀 안 좌표 f 를 내림차순 정렬하면 어느 사면체인지가 정해지고(축 순열),
    정렬값의 차가 그대로 barycentric 이다. 대각선이 항상 +(1,1,1) 이라
    예전의 옥탄트 미러링·걸음 부호가 없다.

    보조 = (ci [N,3] 셀 정수좌표, rank [N,3] 축->순위, perm [N,3] 순위->축)
    """
    nnl = [int(nn[k]) for k in range(3)]
    y = (q - lo) @ lat.Ai.T                             # 격자 좌표
    ci = y.floor().long()
    ci = torch.stack([ci[:, k].clamp(0, nnl[k] - 2) for k in range(3)], -1)
    f = (y - ci).clamp(0.0, 1.0)
    perm = torch.argsort(f.detach(), dim=1, descending=True)
    rank = torch.argsort(perm, dim=1)
    sv = f.gather(1, perm)                              # s0 >= s1 >= s2
    lam = torch.stack([1.0 - sv[:, 0], sv[:, 0] - sv[:, 1],
                       sv[:, 1] - sv[:, 2], sv[:, 2]], -1)
    # 사슬 꼭짓점: v0 = ci, 한 걸음마다 +e_{perm[r]}, v3 = ci + (1,1,1)
    eye = torch.eye(3, device=q.device, dtype=torch.long)
    dirs = eye[perm]                                    # [N,3,3]
    verts = torch.cat([torch.zeros_like(dirs[:, :1]),
                       dirs.cumsum(1)], 1) + ci.unsqueeze(1)     # [N,4,3]
    idx = ((verts[..., 0] * nnl[1] + verts[..., 1]) * nnl[2]
           + verts[..., 2])
    return idx, lam, (ci, rank, perm)


def g2p(q, lo, lat, nn, dp):
    """노드 변위를 입자로: u(q) = sum_i lam_i dp_{v_i} (사면체 4 꼭짓점)."""
    idx, lam, _ = locate(q, lo, lat, nn)
    return (lam.unsqueeze(-1) * dp[idx]).sum(1)


def g2p_jac(q, lo, lat, nn, dp):
    """값과 grad_x u [N,3,3] -- 사면체별 상수, 닫힌 형식.

    사슬을 따라 u 는 아핀이고 r 번째 걸음이 격자축 perm[r] 이므로
        du/dy_{perm[r]} = dp_{r+1} - dp_r,
    그리고 y = Ai (x - lo) 이므로 du/dx = (du/dy) Ai.
    """
    idx, lam, (ci, rank, perm) = locate(q, lo, lat, nn)
    dpc = dp[idx]                                       # [N,4,3]
    u = (lam.unsqueeze(-1) * dpc).sum(1)
    d = (dpc[:, 1:] - dpc[:, :-1])                      # [N,3(r),3(i)]
    d = d.transpose(1, 2)                               # [N,3(i),3(r)]
    Dy = d.gather(2, rank.unsqueeze(1).expand(-1, 3, -1))    # 격자축 순서로
    G = Dy @ lat.Ai.to(Dy.dtype)                        # [N,3,3] grad_x u
    return u, G


def g2p_pre(rows, lam, dp_act):
    """`locate` 를 이미 한 경우의 전달. u = sum_i lam_i dp_{rows_i}.

    rows 는 **압축 노드** 색인(active_nodes 의 출력)이고 dp_act 는 그 노드들의
    변위다. locate 는 tau 에 무관하므로 jvp 안에서 다시 돌 이유가 없고, Mtot
    크기 zeros + index_copy 도 필요 없다.
    """
    return (lam.unsqueeze(-1) * dp_act[rows]).sum(1)


def g2p_jac_pre(rows, lam, aux, lat, dp_act):
    """`locate` 를 이미 한 경우의 값·grad_x u. g2p_jac 과 같은 식이다."""
    _ci, rank, _perm = aux
    dpc = dp_act[rows]                                  # [N,4,3]
    u = (lam.unsqueeze(-1) * dpc).sum(1)
    d = (dpc[:, 1:] - dpc[:, :-1]).transpose(1, 2)      # [N,3(i),3(r)]
    Dy = d.gather(2, rank.unsqueeze(1).expand(-1, 3, -1))
    return u, Dy @ lat.Ai.to(Dy.dtype)


def tet_det(G):
    """det(I + grad u) [N] -- 사면체별 스칼라."""
    I3 = torch.eye(3, device=G.device, dtype=G.dtype)
    return torch.linalg.det(I3 + G)


def tet_id(lo, lat, nn, aux):
    """사면체 유일 ID [N] = (셀, 축순열 6). 중복 제거용."""
    ci, rank, perm = aux
    nnl = [int(nn[k]) for k in range(3)]
    cell = (ci[:, 0] * nnl[1] + ci[:, 1]) * nnl[2] + ci[:, 2]
    p = perm[:, 0] * 3 + perm[:, 1]        # 앞 둘이면 순열이 정해진다
    return cell * 9 + p


def active_nodes(idx):
    """쓰이는 노드만 -> (압축 색인 [N,4], 원본 평탄색인 [M])."""
    uniq, inv = torch.unique(idx.reshape(-1), return_inverse=True)
    return inv.reshape(idx.shape), uniq


# 고정 대각선 Kuhn 분할이 쓰는 변은 세 종류뿐이다 (방향까지 구분해 14 가지):
#   격자축   e1, e2, e3                  -> 한 층 넘는다
#   면대각   e1+e2, e1+e3, e2+e3         -> 두 층
#   몸대각   e1+e2+e3                    -> 세 층 (z 에 평행, 공유 대각선)
# (1,-1,0) 같은 나머지 면대각은 이 분할에 나타나지 않는다 -- 그래서 **같은 층
# 안에는 변이 없고** 모든 변이 z 로 층을 넘는다.
_POS_OFFSETS = [(1, 0, 0), (0, 1, 0), (0, 0, 1),
                (1, 1, 0), (1, 0, 1), (0, 1, 1),
                (1, 1, 1)]
EDGE_OFFSETS = _POS_OFFSETS + [(-i, -j, -k) for (i, j, k) in _POS_OFFSETS]
N_EDGE_CLASS = len(EDGE_OFFSETS)                        # 14
_OFF2CLS = {o: c for c, o in enumerate(EDGE_OFFSETS)}


def edges_of(idx_rows, uniq, nn):
    """활성 사면체의 변 -> (src [E], dst [E], 클래스 [E]).

    양방향 모두 담는다 (오프셋이 반대인 서로 다른 클래스로 들어간다).
    """
    nnl = [int(nn[k]) for k in range(3)]
    z = uniq % nnl[2]
    y = (uniq // nnl[2]) % nnl[1]
    xx = uniq // (nnl[1] * nnl[2])
    pos = torch.stack([xx, y, z], -1)                   # [M,3] 정수 격자좌표
    # **사면체를 먼저 중복 제거한다.** idx_rows 는 입자마다 한 행이라 251001 개
    # 입자면 변 쌍이 300 만인데, 서로 다른 사면체는 수만 개뿐이다. 먼저 줄이지
    # 않으면 매 스텝 300 만 쌍을 정렬한다 (실측 셀집계의 대부분이 이것이었다).
    tets = torch.unique(idx_rows, dim=0)                # [T,4]
    pair = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    a = torch.cat([tets[:, i] for i, _ in pair])
    b = torch.cat([tets[:, j] for _, j in pair])
    src = torch.cat([a, b])
    dst = torch.cat([b, a])
    e = torch.unique(torch.stack([src, dst], -1), dim=0)
    src, dst = e[:, 0], e[:, 1]
    off = pos[dst] - pos[src]                           # [E,3]
    # 오프셋 -> 클래스. 표에 없는 값은 나오지 않아야 한다.
    key = (off[:, 0] + 1) * 9 + (off[:, 1] + 1) * 3 + (off[:, 2] + 1)
    tbl = torch.full((27,), -1, device=off.device, dtype=torch.long)
    for o, c in _OFF2CLS.items():
        tbl[(o[0] + 1) * 9 + (o[1] + 1) * 3 + (o[2] + 1)] = c
    cls = tbl[key]
    return src, dst, cls

"""사면체 복합체: 이제 **사면체가 셀**이다.

육면체 격자는 복합체를 만드는 발판으로만 쓴다. 발판 셀 하나를 소큐브 8개로
자르고, 각 소큐브를 "부모 코너 -> 몸중심" 을 주대각선으로 하는 Kuhn 6-사면체
로 자른다 (셀당 48 사면체). 8 개의 대각선이 전부 몸중심으로 수렴하므로 배치가
큐브 대칭군 48 개 전체에 불변이다 -- 단일 Kuhn 의 특권 대각 방향이 없다.

**노드는 27 종 좌표 전부**(코너 8 + 변중점 12 + 면중심 6 + 몸중심 1)이고 각각
독립 자유도다. 예전처럼 코너 평균으로 유도하지 않는다. 좌표를 **2 배 격자**
(간격 h/2) 의 정수점으로 보면 27 종이 전부 유일하게 색인되고, 이웃 셀과 공유
되는 노드도 저절로 하나로 합쳐진다.

  발판 셀 간격 h,  노드 간격 h/2.
  사용자는 노드 간격만 정한다 (--n_nodes) -- 발판 격자는 내부 구현이다.

입자는 자기가 든 사면체의 4 꼭짓점 barycentric 으로 움직인다. 사면체 안이
아핀이라 grad_Phi 가 사면체별 **상수**이고 닫힌 형식이며, det 도 스칼라 하나다.
"""
import torch

__all__ = ["grid_for_nodes", "locate", "g2p", "g2p_jac", "tet_det",
           "active_nodes", "edges_of", "EDGE_OFFSETS", "N_EDGE_CLASS"]


def grid_for_nodes(x, n_nodes, margin=0.05):
    """점들을 덮는 **2 배 격자**를 잡는다 -> (lo, hn, nn) .

    hn 은 노드 간격, nn 은 축별 노드 개수. 발판 셀 간격은 2*hn 이고 노드
    색인이 짝수면 코너, 홀수 성분이 있으면 변중점/면중심/몸중심이다.
    사면체가 발판 셀 안에서 닫히려면 축별 노드 수가 **홀수**여야 한다.
    """
    lo_ = x.min(0).values
    hi_ = x.max(0).values
    ext = (hi_ - lo_).max().clamp_min(1e-12)
    pad = margin * ext
    lo_ = lo_ - pad
    hn = float((ext + 2 * pad) / max(int(n_nodes), 2))
    nn = []
    for k in range(3):
        c = int(torch.ceil((hi_[k] + pad - lo_[k]) / hn).item()) + 1
        c = max(c, 3)
        if c % 2 == 0:            # 발판 셀(2칸) 로 딱 떨어지게 홀수로
            c += 1
        nn.append(c)
    return lo_, hn, torch.tensor(nn, device=x.device)


def locate(q, lo, hn, nn):
    """점 -> (사면체 4 꼭짓점 평탄색인 [N,4], barycentric [N,4], 보조).

    보조 = (base [N,3] 발판 셀 원점(2배격자 색인), oct [N,3] 소큐브 비트,
            rank [N,3] 축->순위) -- 야코비안·사면체 ID 에 쓴다.
    """
    nnl = [int(nn[k]) for k in range(3)]
    ncell = [(c - 1) // 2 for c in nnl]                 # 발판 셀 수
    t = (q - lo) / (2.0 * hn)                           # 발판 셀 좌표
    ci = t.floor().long()
    ci = torch.stack([ci[:, k].clamp(0, ncell[k] - 1) for k in range(3)], -1)
    f = (t - ci).clamp(0.0, 1.0)                        # 셀 안 [0,1]^3
    oc = (f >= 0.5)                                     # 소큐브 옥탄트
    # 소큐브 안 좌표를 "자기 코너에서 몸중심 쪽" 으로 미러링해 [0,1]^3 로
    m = torch.where(oc, 2.0 * (1.0 - f), 2.0 * f)
    perm = torch.argsort(m.detach(), dim=1, descending=True)
    rank = torch.argsort(perm, dim=1)
    sv = m.gather(1, perm)                              # s1 >= s2 >= s3
    lam = torch.stack([1.0 - sv[:, 0], sv[:, 0] - sv[:, 1],
                       sv[:, 1] - sv[:, 2], sv[:, 2]], -1)
    # 사슬 꼭짓점 (2배 격자 색인): v0 = 자기 코너, 한 걸음마다 몸중심 쪽으로 +-1
    # 발판 셀은 2 배 격자에서 [2ci, 2ci+2] 를 차지한다 -- 옥탄트 1 쪽 "자기
    # 코너" 는 2ci+2 이지 2ci+1(몸중심) 이 아니다.
    base2 = 2 * ci + 2 * oc.long()                      # 자기 코너의 노드 색인
    step = torch.where(oc, -1, 1)                       # 몸중심 방향
    eye = torch.eye(3, device=q.device, dtype=torch.long)
    dirs = eye[perm] * step.gather(1, perm).unsqueeze(-1)
    verts = torch.cat([torch.zeros_like(dirs[:, :1]),
                       dirs.cumsum(1)], 1) + base2.unsqueeze(1)   # [N,4,3]
    idx = ((verts[..., 0] * nnl[1] + verts[..., 1]) * nnl[2]
           + verts[..., 2])
    return idx, lam, (base2, oc, rank, perm, step)


def g2p(q, lo, hn, nn, dp):
    """노드 변위를 입자로: u(q) = sum_i lam_i dp_{v_i} (사면체 4 꼭짓점)."""
    idx, lam, _ = locate(q, lo, hn, nn)
    return (lam.unsqueeze(-1) * dp[idx]).sum(1)


def g2p_jac(q, lo, hn, nn, dp):
    """값과 grad u [N,3,3] -- 사면체별 상수, 닫힌 형식.

    사슬을 따라 u 는 아핀이고, r 번째 걸음의 방향이 축 perm[r] 이므로
        du/d(축 perm[r]) = (dp_{r+1} - dp_r) * step_r / hn.
    """
    idx, lam, (base2, oc, rank, perm, step) = locate(q, lo, hn, nn)
    dpc = dp[idx]                                       # [N,4,3]
    u = (lam.unsqueeze(-1) * dpc).sum(1)
    sp = step.gather(1, perm).to(u.dtype)               # [N,3] 걸음 방향
    d = (dpc[:, 1:] - dpc[:, :-1]) * sp.unsqueeze(-1) / hn   # [N,3(r),3(i)]
    d = d.transpose(1, 2)                               # [N,3(i),3(r)]
    G = d.gather(2, rank.unsqueeze(1).expand(-1, 3, -1))     # 축 순서로
    return u, G


def tet_det(G):
    """det(I + grad u) [N] -- 사면체별 스칼라."""
    I3 = torch.eye(3, device=G.device, dtype=G.dtype)
    return torch.linalg.det(I3 + G)


def tet_id(lo, hn, nn, aux):
    """사면체 유일 ID [N] = (발판셀, 옥탄트 8, 축순열 6). 중복 제거용."""
    base2, oc, rank, perm, step = aux
    nnl = [int(nn[k]) for k in range(3)]
    cell = ((base2[:, 0] // 2 * nnl[1] + base2[:, 1] // 2) * nnl[2]
            + base2[:, 2] // 2)
    o = (oc[:, 0].long() * 2 + oc[:, 1].long()) * 2 + oc[:, 2].long()
    p = (perm[:, 0] * 3 + perm[:, 1])            # 앞 둘이면 순열이 정해진다
    return (cell * 8 + o) * 9 + p


def active_nodes(idx):
    """쓰이는 노드만 -> (압축 색인 [N,4], 원본 평탄색인 [M])."""
    uniq, inv = torch.unique(idx.reshape(-1), return_inverse=True)
    return inv.reshape(idx.shape), uniq


# 2 배 격자에서 사면체 변이 가질 수 있는 상대 오프셋: {-1,0,1}^3 의 0 아닌 것
# 전부 (축 6 + 면대각 12 + 체대각 8 = 26). **방향까지 구분**해 종류를 나눈다 --
# +x 와 -x 는 다른 간선이다.
EDGE_OFFSETS = [(i, j, k)
                for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)
                if (i, j, k) != (0, 0, 0)]
N_EDGE_CLASS = len(EDGE_OFFSETS)                        # 26


def edges_of(idx_rows, uniq, nn):
    """활성 사면체의 변 -> (src [E], dst [E], 클래스 [E]).

    양방향 모두 담는다 (오프셋이 반대인 서로 다른 클래스로 들어간다).
    """
    nnl = [int(nn[k]) for k in range(3)]
    # 압축 노드의 3D 좌표 (2 배 격자 색인)
    z = uniq % nnl[2]
    y = (uniq // nnl[2]) % nnl[1]
    xx = uniq // (nnl[1] * nnl[2])
    pos = torch.stack([xx, y, z], -1)                   # [M,3]
    pair = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    a = torch.cat([idx_rows[:, i] for i, _ in pair])
    b = torch.cat([idx_rows[:, j] for _, j in pair])
    src = torch.cat([a, b])
    dst = torch.cat([b, a])
    e = torch.stack([src, dst], -1)
    e = torch.unique(e, dim=0)
    src, dst = e[:, 0], e[:, 1]
    off = pos[dst] - pos[src]                           # [E,3] in {-1,0,1}
    cls = ((off[:, 0] + 1) * 3 + (off[:, 1] + 1)) * 3 + (off[:, 2] + 1)
    cls = torch.where(cls > 13, cls - 1, cls)           # (0,0,0)=13 을 뺀다
    return src, dst, cls

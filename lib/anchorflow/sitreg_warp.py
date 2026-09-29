"""접힘 없는 격자점 워프: 상한 지키는 미소 변형의 합성.

격자-입자 전달은 **Kuhn 사면체 barycentric 하나만** 쓴다 (2026-09-29 결정 --
학습 반경 skin, B-스플라인, trilinear 는 전부 제거됐다: trilinear 는 셀이
뒤집혀 겹칠 수 있다). 접힘 방지는 다음 논증으로 성립한다:

    변위장 u 가 Lipschitz 상수 L < 1 이면 x + u(x) 는 단사다.
    ‖(x+u(x)) − (y+u(y))‖ ≥ (1−L)‖x−y‖ > 0     (매끄러움이 필요 없다)

PL barycentric 보간은 ∇u 의 열이 사슬 꼭짓점 변위의 차분/h 라
‖∇u‖₂ ≤ 2√3·max‖dp‖/h 이고, 따라서

    max‖dp_a‖ < h / (2√3) ≈ 0.2887 h   이면 L < 1.

또한 ‖∇u‖<1 이면 det(I+∇u) > 0 (a.e.) 이라 방향도 보존된다. 한 번에 큰 변형을
내면 상한에 걸리므로, 상한을 지키는 미소 변형을 K 번 **합성**한다 -- 단사의
합성은 단사다. (이 합성 아이디어는 SITReg, Honkamaa & Marttinen MELBA 2024
에서 왔다. 그쪽의 3차 B-스플라인 상한·역변환 층은 더 이상 쓰지 않는다.)

상한을 지키게 하는 방법은 hard clamp 가 아니라 매끄러운 squash 다:
    c = m * tanh(dp / m)
이면 |c| < m 이 항상 성립하고 dp=0 근처에서 항등이라 기울기가 살아 있다.
"""
import math

import torch

__all__ = ["WARP_BOUND", "squash_to_bound", "BoundedWarp",
           "bary_g2p", "bary_g2p_jac"]

# 성분별 상한이라 ‖dp‖ ≤ √3·m 까지 갈 수 있어, 성분 상한은 h/(2·3) 로 잡아야
# ‖dp‖ < h/(2√3) 이 보장된다. 약간의 여유를 두고 1/6 보다 조금 작게 둔다.
WARP_BOUND = 1.0 / (2.0 * 3.0) * 0.98               # ≈ 0.163 (간격 1 기준)


def squash_to_bound(dp, bound):
    """|c| < bound 를 매끄럽게 강제한다 (hard clamp 가 아니다)."""
    return bound * torch.tanh(dp / bound)


class BoundedWarp:
    """제어점 변위를 접힘 없는 변형으로 바꾼다.

    사용: w = BoundedWarp(bound, K); x2 = w.apply(x, dp, warp_fn)
    warp_fn(q, c) 는 제어점 변위 c 로 점 q 를 옮기는 함수 (barycentric 전달).
    """

    def __init__(self, bound, n_compose=1):
        self.bound = float(bound)
        self.K = max(int(n_compose), 1)

    def apply(self, x, dp, warp_fn):
        """K 번 합성. 각 단계는 dp/K 를 상한 안으로 squash 해 쓴다."""
        c = squash_to_bound(dp / self.K, self.bound)
        q = x
        for _ in range(self.K):
            q = warp_fn(q, c)
        return q

    def apply_multi(self, x, dps, warp_fn):
        """단계마다 다른 제어점을 쓰는 판본 (다중 해상도에 가깝다)."""
        q = x
        for d in dps:
            q = warp_fn(q, squash_to_bound(d, self.bound))
        return q

    def apply_jac(self, x, dp, jac_fn):
        """K 합성의 값과 **해석적** 야코비안을 함께. jac_fn(q, c) -> (u, ∇u).

        연쇄법칙 그대로다: J = Π_k (I + ∇u(q_k)). 자동미분(역전파 3회)보다
        싸고, no_grad 아래(롤아웃)에서도 돈다.
        """
        c = squash_to_bound(dp / self.K, self.bound)
        q, J = x, None
        I3 = torch.eye(3, device=x.device, dtype=x.dtype)
        for _ in range(self.K):
            u, G = jac_fn(q, c)
            q = q + u
            S = I3 + G
            J = S if J is None else S @ J
        return q, J


# ---------------------------------------------------------------------------
# 방향 비의존 사면체 분할 위의 barycentric 전달 -- 유일한 격자-입자 전달
#
# 셀을 8개의 소큐브로 자르고, 각 소큐브를 **부모 코너 -> 몸중심** 을 주대각선
# 으로 하는 Kuhn 6-사면체로 자른다 (셀당 48 사면체). 8개의 대각선이 전부
# 몸중심으로 수렴하므로 이 배치는 큐브 대칭군 48개 전체에 불변이다 -- 단일
# Kuhn 분할의 특권 대각 방향이 사라진다. 이웃 셀과의 공유 소면에서도 양쪽 다
# "셀 꼭짓점 <-> 면중심" 대각선이 유도되어 면 정합이 성립한다 (C0).
#
# 가상 꼭짓점(변중점·면중심·몸중심)의 값은 저장하지 않고 **주변 셀 꼭짓점의
# 평균으로 유도**한다: 사슬 꼭짓점 값 A_r 은 코너 집합 S_0={s} ⊂ S_1 ⊂ S_2
# ⊂ S_3=전체 8개 의 평균이라, 입자 하나가 모으는 격자점은 여전히 자기 칸의
# 8꼭짓점뿐이고 가중치만 다르다:
#     w_b = σ_{R(b)},  σ_R = Σ_{k≥R} λ_k / 2^k,
#     R(b) = (b 가 s 와 다른 축들 중 순열 순위의 최댓값)+? -- 코드 참조.
# 배정·가중치는 여전히 탐색 없이 나온다 (옥탄트 판정 + 미러 + argsort).
#
# ∇u 의 각 열은 이웃 코너 차분들의 **평균**/h 이라 상한 상수는 단일 Kuhn 과
# 동일하게 성분 h/6 (WARP_BOUND) 이고, 아핀 장 정확 재현도 유지된다.
# ---------------------------------------------------------------------------

# 8 코너 비트 (i,j,k) -- 평탄색인 규약 (ix*ny+iy)*nz+iz 과 같은 순서
_B8 = [(b >> 2 & 1, b >> 1 & 1, b & 1) for b in range(8)]


def _sym_locate(q, lo, h, n3):
    """점 -> (셀 8꼭짓점 평탄색인 [N,8], 대칭 가중치 [N,8], 미분 재료).

    미분 재료 = (s [N,3] 옥탄트 비트, rank [N,3] 축->순위, tv [N,3] 정렬좌표).
    """
    n3l = [int(n3[k]) for k in range(3)]
    t = (q - lo) / h
    base = t.floor().long()
    base = torch.stack([base[:, k].clamp(0, n3l[k] - 2) for k in range(3)], -1)
    f = (t - base).clamp(0.0, 1.0)
    s = (f >= 0.5)                                     # [N,3] 소큐브 옥탄트
    m = torch.where(s, 2.0 * (1.0 - f), 2.0 * f)       # 0=자기 코너, 1=몸중심
    perm = torch.argsort(m.detach(), dim=1, descending=True)
    rank = torch.argsort(perm, dim=1)                  # 축 -> 순위 (0,1,2)
    tv = m.gather(1, perm)                             # t1 >= t2 >= t3
    # 접미합 가중 σ_R = Σ_{k≥R} λ_k/2^k  (λ = (1-t1, t1-t2, t2-t3, t3))
    s3 = tv[:, 2] / 8.0
    s2 = tv[:, 1] / 4.0 - s3
    s1 = tv[:, 0] / 2.0 - tv[:, 1] / 4.0 - s3
    s0 = 1.0 - tv[:, 0] / 2.0 - tv[:, 1] / 4.0 - s3
    sig = torch.stack([s0, s1, s2, s3], -1)            # [N,4]
    sl = s.long()
    idxs, ws = [], []
    for b in _B8:
        diff = torch.stack([(sl[:, d] != b[d]).long() for d in range(3)], -1)
        R = (diff * (rank + 1)).amax(-1)               # [N] 0..3
        ws.append(sig.gather(1, R.unsqueeze(1)).squeeze(1))
        idxs.append(((base[:, 0] + b[0]) * n3l[1] + base[:, 1] + b[1])
                    * n3l[2] + base[:, 2] + b[2])
    return torch.stack(idxs, -1), torch.stack(ws, -1), (s, rank, tv)


def bary_g2p(q, lo, h, n3, dp):
    """대칭 barycentric 전달의 값: u(q) = Σ_b w_b(q) dp_b (자기 칸 8꼭짓점)."""
    idx, w, _ = _sym_locate(q, lo, h, n3)
    return (w.unsqueeze(-1) * dp[idx]).sum(1)


def bary_g2p_jac(q, lo, h, n3, dp):
    """대칭 barycentric 전달의 값과 ∇u [N,3,3] (소사면체별 상수, 닫힌 형식).

    ∂σ_R/∂t_r = (r==R ? +1 : r>R ? -1 : 0)/2^r 이고 ∂t_r/∂q_ax 는 순위가
    r-1 인 축에서만 ±2/h (미러 부호) 이므로 전부 표로 조립된다.
    """
    idx, w, (s, rank, tv) = _sym_locate(q, lo, h, n3)
    dpc = dp[idx]                                      # [N,8,3]
    u = (w.unsqueeze(-1) * dpc).sum(1)
    # torch.where(bool, 파이썬실수, 파이썬실수) 는 **기본 dtype(float32)** 로
    # 떨어진다 -- 2/h 가 float32 에서 안 떨어지는 h 면 상수 배율 오차(실측
    # 6.0e-9)가 야코비안 전체에 실린다. dtype 을 박아 만든다.
    _c2 = torch.as_tensor(2.0 / h, device=q.device, dtype=u.dtype)
    sgn = torch.where(s, -_c2, _c2)                    # [N,3] dm/df /h
    r_ax = rank + 1                                    # [N,3] 축의 t 색인 1..3
    half = (2.0 ** (-r_ax.to(u.dtype)))                # 1/2^r
    sl = s.long()
    G = None
    for bi, b in enumerate(_B8):
        diff = torch.stack([(sl[:, d] != b[d]).long() for d in range(3)], -1)
        R = (diff * (rank + 1)).amax(-1, keepdim=True)  # [N,1]
        coef = torch.where(r_ax < R, torch.zeros_like(half),
                           torch.where(r_ax == R, half, -half))
        gw = sgn * coef                                # [N,3] = ∇w_b
        g = dpc[:, bi].unsqueeze(-1) * gw.unsqueeze(1)  # [N,3(i),3(j)]
        G = g if G is None else G + g
    return u, G

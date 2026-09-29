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
# Kuhn 사면체 분할 위의 barycentric 전달 -- 유일한 격자-입자 전달
#
# 각 큐브를 주대각선을 공유하는 사면체 6개로 자른다 (축 순열 하나당 하나,
# f_{π1} ≥ f_{π2} ≥ f_{π3} 영역). 꼭짓점은 큐브 꼭짓점 그대로이고 이웃 큐브와
# 면이 정합이라 전체가 simplicial complex 다. 점의 사면체 배정과 barycentric
# 가중치는 탐색 없이 나온다: 상대좌표를 내림차순 정렬하면 순열이 곧 사면체,
# 정렬값의 차분 (1-s1, s1-s2, s2-s3, s3) 이 곧 꼭짓점 4개의 가중치다.
#
# PL 이라 ∇u 가 사면체별 **상수**이고 열이 사슬 꼭짓점 변위의 차분이다:
#     ∇u 의 π_r 축 열 = (dp_{r+1} - dp_r) / h
# 그래서 성분 상한 h/6 이면 Lipschitz < 1 이 성립한다 --
# WARP_BOUND 를 쓴다.
# ---------------------------------------------------------------------------

def _kuhn_locate(q, lo, h, n3):
    """점 -> (꼭짓점 평탄색인 [N,4], barycentric [N,4], 축 순열 [N,3])."""
    n3l = [int(n3[k]) for k in range(3)]
    t = (q - lo) / h
    base = t.floor().long()
    base = torch.stack([base[:, k].clamp(0, n3l[k] - 2) for k in range(3)], -1)
    f = (t - base).clamp(0.0, 1.0)
    perm = torch.argsort(f.detach(), dim=1, descending=True)   # [N,3] 축 순열
    sv = f.gather(1, perm)                                     # s1 >= s2 >= s3
    lam = torch.stack([1.0 - sv[:, 0], sv[:, 0] - sv[:, 1],
                       sv[:, 1] - sv[:, 2], sv[:, 2]], -1)     # [N,4]
    # 사슬 꼭짓점: v0 = base, v_{r+1} = v_r + e_{perm_r}
    eye = torch.eye(3, device=q.device, dtype=torch.long)
    steps = eye[perm]                                          # [N,3,3]
    verts = torch.cat([torch.zeros_like(steps[:, :1]),
                       steps.cumsum(1)], 1) + base.unsqueeze(1)  # [N,4,3]
    idx = ((verts[..., 0] * n3l[1] + verts[..., 1]) * n3l[2]
           + verts[..., 2])                                    # [N,4]
    return idx, lam, perm


def bary_g2p(q, lo, h, n3, dp):
    """barycentric 전달의 값: u(q) = Σ_i λ_i dp_{v_i}."""
    idx, lam, _ = _kuhn_locate(q, lo, h, n3)
    return (lam.unsqueeze(-1) * dp[idx]).sum(1)


def bary_g2p_jac(q, lo, h, n3, dp):
    """barycentric 전달의 값과 ∇u [N,3,3] (사면체별 상수, 닫힌 형식)."""
    idx, lam, perm = _kuhn_locate(q, lo, h, n3)
    dpc = dp[idx]                                              # [N,4,3]
    u = (lam.unsqueeze(-1) * dpc).sum(1)
    diffs = (dpc[:, 1:] - dpc[:, :-1]).transpose(1, 2) / h     # [N,3(i),3(r)]
    inv = torch.argsort(perm, dim=1)                           # 축 -> 순위
    G = diffs.gather(2, inv.unsqueeze(1).expand(-1, 3, -1))    # [N,3(i),3(j)]
    return u, G

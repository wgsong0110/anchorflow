"""접힘 없는 격자점 워프: 상한 지키는 미소 변형의 합성 (trilinear 전달).

격자-입자 전달 가중치는 **trilinear 하나만** 쓴다 (2026-09-29 결정 -- 학습
반경 skin 과 B-스플라인 전달은 제거됐다). 접힘 방지는 다음 논증으로 성립한다:

    변위장 u 가 Lipschitz 상수 L < 1 이면 x + u(x) 는 단사다.
    ‖(x+u(x)) − (y+u(y))‖ ≥ (1−L)‖x−y‖ > 0     (매끄러움이 필요 없다)

trilinear 보간 u(x) = Σ_a w_a(x) dp_a 는 ‖∂u/∂x_i‖ ≤ 2·max‖dp‖/h 이므로
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

__all__ = ["TRILINEAR_BOUND", "squash_to_bound", "BoundedWarp"]

# 성분별 상한이라 ‖dp‖ ≤ √3·m 까지 갈 수 있어, 성분 상한은 h/(2·3) 로 잡아야
# ‖dp‖ < h/(2√3) 이 보장된다. 약간의 여유를 두고 1/6 보다 조금 작게 둔다.
TRILINEAR_BOUND = 1.0 / (2.0 * 3.0) * 0.98          # ≈ 0.163 (간격 1 기준)


def squash_to_bound(dp, bound):
    """|c| < bound 를 매끄럽게 강제한다 (hard clamp 가 아니다)."""
    return bound * torch.tanh(dp / bound)


class BoundedWarp:
    """제어점 변위를 접힘 없는 변형으로 바꾼다.

    사용: w = BoundedWarp(bound, K); x2 = w.apply(x, dp, warp_fn)
    warp_fn(q, c) 는 제어점 변위 c 로 점 q 를 옮기는 함수 (trilinear 전달).
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

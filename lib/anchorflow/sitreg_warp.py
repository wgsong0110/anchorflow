"""SITReg 식 **위상 보존** 변형: 제어점 상한 + 미소 변형 합성.

출처: Honkamaa & Marttinen, "SITReg: Multi-resolution architecture for symmetric,
inverse consistent, and topology preserving image registration" (MELBA 2024),
https://github.com/honkamj/SITReg

우리가 쓰는 것은 두 조각이다.

1) **제어점 상한** -- 3차 B-spline 으로 만든 변형이 접히지 않으려면 제어점 변위의
   절대값이 어떤 상한보다 작아야 한다 (그쪽 `compute_max_control_point_value`).
   상한은 업샘플 배수와 차원만으로 정해지는 상수라 한 번 구해 두면 된다.
2) **합성** -- 한 번에 큰 변형을 내면 상한에 걸리므로, 상한을 지키는 미소 변형을
   K 번 합성한다. 미분동형사상의 합성은 미분동형사상이므로 위상이 보존된다.

역변환 층(고정점 반복)은 **쓰지 않는다** -- 그것은 대칭성·역일관성을 위한 것이고
우리는 접힘 방지만 필요하다. 그래서 추가 비용이 합성 횟수에 선형이다.

상한을 지키게 하는 방법은 hard clamp 가 아니라 매끄러운 squash 다:
    c = m * tanh(dp / m)
이면 |c| < m 이 항상 성립하고 dp=0 근처에서 항등이라 기울기가 살아 있다.
"""
import torch

__all__ = ["max_control_point_value", "squash_to_bound", "SITRegWarp"]

# 3 차원, 업샘플 배수별 상한 (그쪽 compute_max_control_point_value 로 구한 값을
# 캐시한다. 값은 제어점 간격 1 기준이므로 실제로는 간격을 곱해 쓴다).
_BOUND_CACHE = {}


def max_control_point_value(factors, dtype=torch.float32, device="cpu",
                            sitreg_path=None):
    """제어점 상한. SITReg 구현을 그대로 불러 계산하고 캐시한다."""
    key = (tuple(int(f) for f in factors), str(dtype))
    if key in _BOUND_CACHE:
        return _BOUND_CACHE[key]
    import os
    import sys
    p = sitreg_path or os.environ.get(
        "AF_SITREG", "/home/dkta/work/SITReg/src")
    if p not in sys.path:
        sys.path.insert(0, p)
    from algorithm.cubic_b_spline_control_point_upper_bound import (
        compute_max_control_point_value)
    v = float(compute_max_control_point_value(
        [int(f) for f in factors], dtype=dtype, device=torch.device("cpu")))
    _BOUND_CACHE[key] = v
    return v


def squash_to_bound(dp, bound):
    """|c| < bound 를 매끄럽게 강제한다 (hard clamp 가 아니다)."""
    return bound * torch.tanh(dp / bound)


class SITRegWarp:
    """제어점 변위를 위상 보존 변형으로 바꾼다.

    사용: w = SITRegWarp(bound, K); x2 = w.apply(x, dp, warp_fn)
    warp_fn(q, c) 는 제어점 변위 c 로 점 q 를 옮기는 함수 (B-spline 전달).
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

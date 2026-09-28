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

__all__ = ["max_control_point_value", "squash_to_bound", "SITRegWarp",
           "cubic_bspline_g2p"]

# AF_SITREG 저장소가 없는 환경에서 쓰는 [4,4,4] 상한의 근사값 (실측 0.398*간격)
FALLBACK_BOUND_444 = 0.398

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


_OFF64 = {}


def _offs(device):
    """4^3 스텐실 오프셋을 장치별로 한 번만 만든다."""
    key = str(device)
    if key not in _OFF64:
        r = torch.arange(4, device=device) - 1
        _OFF64[key] = torch.stack(
            torch.meshgrid(r, r, r, indexing="ij"), -1).reshape(-1, 3)
    return _OFF64[key]


def cubic_bspline_g2p(q, lo, h, n3, dp):
    """3 차 B-spline 으로 격자점 변위 dp 를 점 q 로 옮긴다 (64 스텐실, 벡터화).

    SITReg 의 제어점 상한은 이 전달(3차)에 대해 성립한다 -- 2차 전달에 상한을
    걸면 보장이 없다. 격자는 (lo, h, n3=격자점 수), dp 는 격자점 변위 [N노드,3]
    평탄 순서 (ix*ny + iy)*nz + iz.
    """
    t = (q - lo) / h
    base = (t - 0.5).floor()
    f = t - base
    base = base.long()
    w = torch.stack([(1 - f) ** 3 / 6,
                     (3 * f ** 3 - 6 * f ** 2 + 4) / 6,
                     (-3 * f ** 3 + 3 * f ** 2 + 3 * f + 1) / 6,
                     f ** 3 / 6], -1)                      # [N,3,4]
    off = _offs(q.device)                                  # [64,3]
    n3l = [int(n3[k]) for k in range(3)]
    idx = base.unsqueeze(1) + off.unsqueeze(0)             # [N,64,3]
    idx = torch.stack([idx[..., k].clamp(0, n3l[k] - 1) for k in range(3)], -1)
    fl = (idx[..., 0] * n3l[1] + idx[..., 1]) * n3l[2] + idx[..., 2]
    i0, i1, i2 = off[:, 0] + 1, off[:, 1] + 1, off[:, 2] + 1
    ww = w[:, 0][:, i0] * w[:, 1][:, i1] * w[:, 2][:, i2]  # [N,64]
    return torch.einsum("nk,nkd->nd", ww, dp[fl])

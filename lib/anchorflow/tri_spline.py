"""셀 내부 상대좌표의 단조 유리이차 스플라인(RQS) 재배열.

Knothe--Rosenblatt 식 좌표별 단조 맵의 **축분리** 판본이다. 각 축의
T_i: [0,1] -> [0,1] 을 단조 유리이차 스플라인 (Durkan et al., "Neural Spline
Flows", NeurIPS 2019) 으로 두면
  - 셀 안에서 단사가 자명하다 (겹침·꼬임 불가, det > 0 구조적)
  - 기울기를 아무리 가파르게 해도 단사성이 안 깨져 준불연속 재배열을 담는다
  - 조건부(삼각 의존)를 빼서 비용이 가장 싸다 -- 빈 탐색 + 원소별 유리식뿐
셀 경계는 끝점이 0,1 로 고정되므로 면을 가로지르는 **법선** 성분은 구조적으로
연속이다. 접선 성분은 이웃 셀 파라미터가 다르면 불연속이 될 수 있고, 이는
cont_penalty 로 "되도록" 이어 붙인다 (구조 강제가 아니라 벌점 -- 손상이 필요한
곳은 벌점을 이기고 갈라질 수 있다).

파라미터는 원시(무제약) 값이다: 축마다 (폭 K, 높이 K, 매듭 기울기 K+1) 이라
P = 3*(3K+1). 전부 0 이면 **정확히 항등**이다 (softmax 균등 폭/높이,
softplus 이동으로 기울기 1). 단조성은 softmax/softplus 로 구조적으로 강제하고
hard clamp 는 쓰지 않는다.
"""
import math

import torch
import torch.nn.functional as Fn

__all__ = ["n_params", "rqs", "remap", "cont_penalty"]

_SP1 = math.log(math.e - 1.0)          # softplus(_SP1) = 1


def n_params(bins):
    """축 3 개 합친 셀당 파라미터 수."""
    return 3 * (3 * int(bins) + 1)


def rqs(u, theta, bins, min_bin=1e-3, min_d=1e-3, return_deriv=False):
    """단조 유리이차 스플라인. u [..., A] in [0,1], theta [..., A, 3K+1].

    축이 서로 독립이라 마지막 차원 규약만 지키면 임의 배치 모양에서 돈다.
    반환은 u 와 같은 모양, 각 성분이 [0,1] 안의 단조 상이다.
    return_deriv 면 (T(u), dT/du) 를 준다 -- 도함수도 닫힌 형식이다 (NSF Eq.5):
        T' = s² (d₁ξ² + 2sξ(1−ξ) + d₀(1−ξ)²) / den²
    """
    K = int(bins)
    w = Fn.softmax(theta[..., :K], -1) * (1 - K * min_bin) + min_bin
    hg = Fn.softmax(theta[..., K:2 * K], -1) * (1 - K * min_bin) + min_bin
    dv = Fn.softplus(theta[..., 2 * K:] + _SP1) + min_d
    cw = Fn.pad(torch.cumsum(w, -1), (1, 0))       # [...,K+1], 0..1
    ch = Fn.pad(torch.cumsum(hg, -1), (1, 0))
    uc = u.clamp(0.0, 1.0)
    # 빈 색인은 파라미터에 대해 조각별 상수라 미분이 0 이다. detach 해 두면
    # 값은 그대로이고 순방향 AD(searchsorted 에 탄젠트를 못 붙인다)도 통과한다.
    k = (torch.searchsorted(cw.detach().contiguous(),
                            uc.detach().unsqueeze(-1).contiguous())
         - 1).clamp(0, K - 1)

    def g(t, i):
        return t.gather(-1, i).squeeze(-1)

    x0, x1 = g(cw, k), g(cw, k + 1)
    y0, y1 = g(ch, k), g(ch, k + 1)
    d0, d1 = g(dv, k), g(dv, k + 1)
    wb = (x1 - x0).clamp_min(1e-12)
    hb = y1 - y0
    s = hb / wb
    xi = ((uc - x0) / wb).clamp(0.0, 1.0)
    om = xi * (1 - xi)
    num = hb * (s * xi * xi + d0 * om)
    den = (s + (d0 + d1 - 2 * s) * om).clamp_min(1e-12)
    out = y0 + num / den
    if not return_deriv:
        return out
    dv = s * s * (d1 * xi * xi + 2 * s * om + d0 * (1 - xi) ** 2) / (den * den)
    return out, dv


def remap(x, lo, h, n3, theta_cells, bins, min_bin=1e-3, min_d=1e-3,
          return_jac=False):
    """점들을 자기 셀 안에서 재배열한 기준 위치로 옮긴다.

    x [N,3], 격자 (lo, h, n3=격자점 수), theta_cells [Ncell, P] 는 셀 평탄
    순서 (ix*cy + iy)*cz + iz. 반환 x_ref = lo + (셀색인 + T(u)) * h.
    T 가 [0,1]^3 -> [0,1]^3 이라 점이 자기 셀을 못 벗어난다 -- 셀들이 서로소
    이므로 전역 단사가 유지된다.
    """
    nc = [max(int(n3[k]) - 1, 1) for k in range(3)]
    t = (x - lo) / h
    ci = t.floor().long()
    ci = torch.stack([ci[:, k].clamp(0, nc[k] - 1) for k in range(3)], -1)
    ur = t - ci                                    # 격자 안이면 [0,1)
    flat = (ci[:, 0] * nc[1] + ci[:, 1]) * nc[2] + ci[:, 2]
    th = theta_cells[flat].reshape(x.shape[0], 3, -1)
    ut, dv = rqs(ur.clamp(0.0, 1.0), th, bins, min_bin=min_bin, min_d=min_d,
                 return_deriv=True)
    # 격자 **밖**(셀 색인을 clamp 한 점)은 항등으로 둔다. 예전처럼 u 를 clamp 만
    # 하면 그 성분의 기울기가 0 이 되어 야코비안에 0 행이 생기고, F 가 특이해져
    # Psi 의 log det 가 터진다 (실측: det grad Phi 최소값이 정확히 0 이었다).
    inside = (ur >= 0.0) & (ur <= 1.0)
    ut = torch.where(inside, ut, ur)
    out = lo + (ci.to(x.dtype) + ut) * h
    if not return_jac:
        return out
    # 축분리라 야코비안이 대각이고, h·(1/h) 가 상쇄돼 성분이 곧 스플라인
    # 기울기다. 격자 밖 항등 구간은 기울기 1.
    return out, torch.where(inside, dv, torch.ones_like(dv))


def cont_penalty(theta_grid):
    """이웃 셀 파라미터 차의 제곱 평균. theta_grid [..., cx, cy, cz, P].

    원시 파라미터에서 잰다 -- 항등 근방에서 활성화 후 차이와 같은 차수이고
    가장 싸다. 법선 성분은 끝점 고정으로 이미 연속이라, 이 벌점은 접선
    성분의 면 불일치를 "되도록" 줄이는 역할이다.
    """
    p, n = 0.0, 0
    for ax in (-4, -3, -2):
        d = theta_grid.diff(dim=ax)
        if d.numel():                  # 축 방향 셀이 1 개면 이웃이 없다
            p = p + d.pow(2).mean()
            n += 1
    return p / max(n, 1)

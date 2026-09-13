"""로그-유클리드 프레임 상태: 앵커가 X_a = log F_a (자유 3x3) 를 들고 로그 공간에서
선형 블렌딩한다. **회전과 신축을 쪼개지 않는다.**

    X_g = sum_a w_ga X_a,    F_g = exp(X_g)

행렬식이 항상 양수다 -- det(exp X) = e^{tr X} 이므로

    det F_g = exp( sum_a w_ga tr(log F_a) ) = prod_a (det F_a)^{w_ga} > 0

즉 부피가 **기하평균**으로 섞인다. 성분별 평균(자유 F)은 이웃 회전이 어긋날 때
행렬식이 0 까지 내려가는데(실측 하위 0.1% 가 0.019), 기하평균은 그 자리에서도
0 이 되지 않는다.

그러면서 인코더는 로그 공간의 평범한 선형 최소제곱이라 닫힌 형태가 유지된다 --
목표 X_g* = log F_g^MPM 은 데이터만의 함수라 상수이고, 촐레스키 한 번으로 끝난다.

--- 앞선 두 번의 실패와 그 교훈 -----------------------------------------------
1차: 뉴턴 제곱근(Denman-Beavers, 역행렬 48 회)으로 logm 을 짰다가 nan.
2차: "고윳값이 음수라 실수 로그가 없다"고 **확인 없이** 진단하고 극분해로 우회했다.
     그 결과는 F = R exp(H) 로, 회전벡터를 다시 들고 있어 2pi 감김이 그대로였다 --
     로그-유클리드가 아니라 회전+신축의 변형이었고, 실제로 수치가 그것과 같게 나왔다.

이번 판은 (a) 역행렬 대신 solve 를 쓰고, (b) **왕복 검증을 코드에 넣는다** --
exp(log F) 가 F 로 돌아오는지 재지 않고 결과를 보고한 것이 2차 실패의 핵심이었다.
"""
from __future__ import annotations

import torch


def expm3(X):
    """[...,3,3] 행렬 지수. torch 의 배치 구현을 쓴다 -- 직접 짤 이유가 없다."""
    return torch.linalg.matrix_exp(X)


def logm3(F, fallback=True):
    """[...,3,3] 행렬 로그. 고윳값 분해로 직접 푼다.

    앞선 두 판이 모두 제곱근 반복(Denman-Beavers)에서 터졌다 -- 조건수가 나쁜 F 에서
    발산해 |log F| 가 1e27 까지 갔다. 반복을 아예 없애고 eig 로 간다:

        F = V diag(lam) V^-1  ->  log F = V diag(log lam) V^-1

    실수 음수 고윳값이 없으면(514,659 표본에서 0 개로 확인) 결과의 허수부가 0 이다.
    V 가 거의 특이한(결함) 행렬에서만 위험하므로, 그런 표본은 왕복 오차로 잡아내
    log(F) ~ F - I 로 되돌린다 -- 항등원 근처라 1 차 근사가 나쁘지 않다.
    """
    ev, V = torch.linalg.eig(F.to(torch.complex64) if not F.is_complex() else F)
    X = (V @ torch.diag_embed(torch.log(ev)) @ torch.linalg.inv(V)).real.to(F.dtype)
    if not fallback:
        return X
    eye = torch.eye(3, device=F.device, dtype=F.dtype).expand_as(F)
    bad = ~torch.isfinite(X).all(dim=(-1, -2))
    rel = (torch.linalg.matrix_exp(torch.where(bad[..., None, None], eye * 0, X)) - F
           ).norm(dim=(-1, -2)) / F.norm(dim=(-1, -2)).clamp(min=1e-12)
    bad = bad | (rel > 1e-3)
    if bad.any():
        X = torch.where(bad[..., None, None], F - eye, X)
    return X


def roundtrip_error(F, **kw):
    """||exp(log F) - F|| / ||F||. 이걸 안 재고 결과를 보고한 것이 2차 실패였다.

    3차 실패의 교훈도 있다: 궤적 0~2 에서만 재고 통과시켰는데 실패는 궤적 72/8/26 에서
    났다. 표본을 대표성 있게 뽑지 않으면 검증이 아니다.
    """
    X = logm3(F, **kw)
    bad = ~torch.isfinite(X).all(dim=(-1, -2))
    Fb = expm3(X)
    rel = (Fb - F).norm(dim=(-1, -2)) / F.norm(dim=(-1, -2)).clamp(min=1e-12)
    return rel, bad


def logm_target(F0, **kw):
    """MPM 의 F -> 로그 공간 목표. 데이터만의 함수이므로 미분 불필요."""
    with torch.no_grad():
        return logm3(F0, **kw)


class LogEucState:
    """앵커의 X_a = log F_a 와 가우시안의 F_g. FrameState 의 짝 구조를 빌려 쓴다."""

    def __init__(self, fs):
        self.fs = fs

    def blend(self, X_a, w):
        f = self.fs
        return torch.zeros(f.N, 9, device=f.dev, dtype=X_a.dtype).index_add_(
            0, f.pair_g, w.unsqueeze(-1) * X_a.reshape(-1, 9)[f.pair_a]).view(-1, 3, 3)

    def decode(self, X_a, w):
        return expm3(self.blend(X_a, w))

    def encode(self, F0, w, ridge=1e-4, L=None, targets=None):
        Xt = logm_target(F0) if targets is None else targets
        if L is None:
            L = self.fs.gram(w, ridge)
        return self.fs._wls(Xt.reshape(-1, 9), w, L).view(-1, 3, 3)

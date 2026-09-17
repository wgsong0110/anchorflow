"""변형 모델 한 프레임의 두 병목을 융합한 CUDA 커널.

둘 다 연산이 아니라 **중간 텐서의 전역 메모리 왕복**이 비용이었다. 실측
(입자 138 만, 앵커 512, k=16):

  kNN        행렬곱 6.2 ms + topk 33 ms.  [N,512] 점수판 1.4 GB 를 만들었다가
             topk 가 다시 훑는다. 앵커를 공유 메모리에 올리고 고른 k 개를
             레지스터에 유지하면 점수판이 아예 없다.
  스키닝+J   24 ms.  p[idx], dp[idx], dvec 같은 [N,k,3] 을 여러 번 만든다.
             한 입자를 한 스레드가 맡으면 전부 레지스터에 남는다.

빌드되지 않은 환경에서는 파이토치 경로로 되돌아간다 -- 값이 같아야 하므로
exe/verify_deformcuda.py 가 둘을 대조한다.
"""
import torch

try:
    from . import _C
    HAVE_CUDA = True
except Exception:
    _C = None
    HAVE_CUDA = False


def knn(x, p, k):
    """[N,3], [M,3] -> (idx [N,k] long, dist [N,k])"""
    return tuple(_C.knn(x.contiguous(), p.contiguous(), int(k)))


def skin_jacobian(x, p, dp, log_r, log_t, idx, h, tau_min=1e-4):
    """스키닝 결과와 그 야코비안. -> (out [N,3], J [N,3,3])"""
    return tuple(_C.skin_jacobian(x.contiguous(), p.contiguous(),
                                  dp.contiguous(), log_r.contiguous(),
                                  log_t.contiguous(), idx.contiguous(),
                                  float(h), float(tau_min)))


def aggregate_moments(x, X, v, m, idx, M):
    """앵커별 질량 가중 모멘트. -> (g1 [M,11], g2 [M,12], g3 [M,18])

    g1 = (질량, 질량x위치, 질량x정준위치, 질량x속도, 개수)
    g2 = (2차 모멘트 S 9, 각운동량 L 3)
    g3 = (교차 모멘트 A 9, 정준 2차 모멘트 B 9)

    파이토치 쪽 aggregate 가 index_add_ 로 하는 것과 같은 합이다 -- 이후의
    나눗셈·역행렬·행렬식은 앵커 수가 작아 파이토치에 그대로 둔다.
    """
    return tuple(_C.aggregate_moments(x.contiguous(), X.contiguous(),
                                      v.contiguous(), m.contiguous(),
                                      idx.contiguous(), int(M)))


def fps(x, M, first=0):
    """가장 먼 점 표본추출. -> idx [M] long"""
    return _C.fps(x.contiguous(), int(M), int(first))

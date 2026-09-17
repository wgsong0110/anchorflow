"""융합 CUDA 커널이 파이토치 경로와 **같은 값**을 내는지 대조하고, 속도를 잰다.

빠른 것을 재기 전에 같은지부터 본다. 두 커널 모두 파이토치 쪽 정의(kNN 은
정확한 k 최근접, 스키닝/야코비안은 lib/anchorflow/deform.py 의 식)를 그대로
옮긴 것이므로, 차이는 부동소수점 오차 수준이어야 한다.

kNN 은 거리가 같은 앵커가 있으면 색인이 갈릴 수 있으므로, 색인이 아니라
**뽑힌 거리 집합**과 그로부터 나온 결과로 비교한다.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

_lib = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib")
sys.path.insert(0, _lib)

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--N", type=int, default=1381100)
ap.add_argument("--M", type=int, default=512)
ap.add_argument("--k", type=int, nargs="+", default=[16, 8])
ap.add_argument("--reps", type=int, default=20)
ap.add_argument("--out", default=None)
a = ap.parse_args()

dev = "cuda"
torch.manual_seed(0)
import deformcuda                                               # noqa: E402
from anchorflow.deform import dense_knn, skin_with_jacobian     # noqa: E402

if not deformcuda.HAVE_CUDA:
    raise SystemExit("deformcuda 가 빌드되지 않았다 (_C 없음)")

x = torch.rand(a.N, 3, device=dev)
p = torch.rand(a.M, 3, device=dev)
dp = torch.randn(a.M, 3, device=dev) * 0.01
log_r = torch.full((a.M,), -3.0, device=dev) + 0.3 * torch.randn(a.M, device=dev)
log_t = 0.3 * torch.randn(a.M, device=dev)
h = 0.05


def timeit(fn, n):
    for _ in range(4):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e3


rep = {}
for k in a.k:
    i_t, d_t = dense_knn(x, p, k, half=True)
    i_c, d_c = deformcuda.knn(x, p, k)
    # 거리 집합으로 비교한다 (동률이면 색인이 갈릴 수 있다)
    e_d = float((d_t.sort(1).values - d_c.sort(1).values).abs().max())
    same = float((i_t.sort(1).values == i_c.sort(1).values).all(1).float().mean())
    o_t, _, J_t = skin_with_jacobian(x, p, dp, log_r, log_t, i_c, h)
    o_c, J_c = deformcuda.skin_jacobian(x, p, dp, log_r, log_t, i_c, h)
    e_o = float((o_t - o_c).abs().max() / o_t.abs().max())
    e_J = float((J_t - J_c).abs().max() / J_t.abs().max())
    t_kt = timeit(lambda: dense_knn(x, p, k, half=True), a.reps)
    t_kc = timeit(lambda: deformcuda.knn(x, p, k), a.reps)
    t_st = timeit(lambda: skin_with_jacobian(x, p, dp, log_r, log_t, i_c, h),
                  a.reps)
    t_sc = timeit(lambda: deformcuda.skin_jacobian(x, p, dp, log_r, log_t, i_c,
                                                   h), a.reps)
    rep[k] = dict(dist_err=e_d, idx_same=same, out_err=e_o, J_err=e_J,
                  knn_torch=t_kt, knn_cuda=t_kc, skinj_torch=t_st,
                  skinj_cuda=t_sc)
    print(f"\n[k={k}]  거리 최대차 {e_d:.2e}  색인 일치 {100*same:.2f}%  "
          f"위치 상대차 {e_o:.2e}  J 상대차 {e_J:.2e}", flush=True)
    print(f"  kNN         파이토치 {t_kt:7.2f} ms -> 커널 {t_kc:6.2f} ms "
          f"({t_kt/max(t_kc,1e-9):5.1f}배)", flush=True)
    print(f"  스키닝+J    파이토치 {t_st:7.2f} ms -> 커널 {t_sc:6.2f} ms "
          f"({t_st/max(t_sc,1e-9):5.1f}배)", flush=True)
    print(f"  두 단계 합   {t_kt+t_st:7.2f} ms -> {t_kc+t_sc:6.2f} ms", flush=True)

if a.out:
    os.makedirs(a.out, exist_ok=True)
    import json
    json.dump({str(k): v for k, v in rep.items()},
              open(os.path.join(a.out, "deformcuda_verify.json"), "w"), indent=1)
print("VERIFY_OK")

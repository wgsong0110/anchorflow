"""집계 커널 두 판을 같은 입력에서 재고, 값이 같은지 본다.

프레임 전체 21 ms 중 집계가 10.9 ms 로 절반이었다. 1 판은 스레드가 (입자,이웃)
짝 하나를 맡아 같은 입자의 위치·속도를 K 번 다시 읽고, 2 차 모멘트를 두 판으로
나눠 돈다. 2 판은 입자 하나를 맡고 두 판을 합쳤다.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

_lib = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib")
sys.path.insert(0, _lib)

import torch                                                      # noqa: E402

import deformcuda as dc                                           # noqa: E402
from anchorflow.deform import anchor_knn, fps                     # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=1381100)
ap.add_argument("--m", type=int, default=512)
ap.add_argument("--k", type=int, default=16)
ap.add_argument("--reps", type=int, default=30)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

dev = "cuda"
torch.manual_seed(a.seed)
x = torch.rand(a.n, 3, device=dev)
X = x + 0.01 * torch.randn_like(x)
v = 0.1 * torch.randn_like(x)
m = torch.rand(a.n, device=dev) + 0.5
p = x[fps(x, a.m)]
idx, _ = anchor_knn(x, p, a.k)
print(f"[설정] 입자 {a.n}, 앵커 {a.m}, 이웃 {a.k}, 짝 {a.n * a.k / 1e6:.1f}M",
      flush=True)


def timeit(fn, n=None):
    n = n or a.reps
    for _ in range(3):
        fn()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1e3


o1 = dc.aggregate_moments(x, X, v, m, idx, a.m)
o2 = dc.aggregate_moments2(x, X, v, m, idx, a.m)
o3 = dc.aggregate_moments2(x, X, v, m, idx, a.m, merge=1)
worst = 0.0
for i, (a1, a2) in enumerate(zip(o1, o2)):
    d = (a1 - a2).abs()
    r = (d / a1.abs().clamp(min=1e-6)).max().item()
    worst = max(worst, r)
    print(f"  g{i + 1}  최대 절대차 {d.max():.3e}  상대차 {r:.3e}", flush=True)
print(f"[정확성] 최대 상대차 {worst:.3e} "
      f"({'통과' if worst < 2e-3 else '실패 -- 합이 다르다'})", flush=True)

t1 = timeit(lambda: dc.aggregate_moments(x, X, v, m, idx, a.m))
t2 = timeit(lambda: dc.aggregate_moments2(x, X, v, m, idx, a.m))
t3 = timeit(lambda: dc.aggregate_moments2(x, X, v, m, idx, a.m, merge=1))
print(f"[속도] 짝마다 {t1:.2f} ms | 입자마다 {t2:.2f} ms ({t1 / t2:.2f} 배) | "
      f"입자마다+합침 {t3:.2f} ms ({t1 / t3:.2f} 배)", flush=True)
print("AGGBENCH_OK", flush=True)

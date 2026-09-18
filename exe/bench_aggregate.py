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
ap.add_argument("--coherent", action="store_true",
                help="입자를 공간순으로 정렬한다. 실제 가우시안 파일이 그렇게 "
                     "담겨 있고, 그러면 이웃 입자들이 같은 앵커를 골라 원자합 "
                     "경합 양상이 완전히 달라진다 -- 난수 구름으로만 재면 속는다")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

dev = "cuda"
torch.manual_seed(a.seed)
x = torch.rand(a.n, 3, device=dev)
X = x + 0.01 * torch.randn_like(x)
v = 0.1 * torch.randn_like(x)
m = torch.rand(a.n, device=dev) + 0.5
if a.coherent:
    c = (x * 64).long().clamp(0, 63)
    key = (c[:, 0] * 64 + c[:, 1]) * 64 + c[:, 2]
    o = key.argsort()
    x, X, v, m = x[o].contiguous(), X[o].contiguous(), v[o].contiguous(), m[o].contiguous()
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
# 성분마다 나누면 0 에 가까운 성분에서 상대차가 터진다. 2200 만 항을 float32 로
# 더한 것이라 덧셈 순서만 달라도 그렇게 되므로, 그 판의 **최대 크기**로 나눈다.
worst = 0.0
for i, (a1, a2) in enumerate(zip(o1, o2)):
    d = (a1 - a2).abs().max().item()
    r = d / max(a1.abs().max().item(), 1e-12)
    worst = max(worst, r)
    print(f"  g{i + 1}  최대 절대차 {d:.3e}  최대치 대비 {r:.3e}", flush=True)
print(f"[정확성] 최대치 대비 최대 어긋남 {worst:.3e} "
      f"({'통과' if worst < 1e-5 else '실패 -- 합이 다르다'})", flush=True)

t1 = timeit(lambda: dc.aggregate_moments(x, X, v, m, idx, a.m))
t2 = timeit(lambda: dc.aggregate_moments2(x, X, v, m, idx, a.m))
t3 = timeit(lambda: dc.aggregate_moments2(x, X, v, m, idx, a.m, merge=1))
print(f"[속도] 짝마다 {t1:.2f} ms | 입자마다 {t2:.2f} ms ({t1 / t2:.2f} 배) | "
      f"입자마다+합침 {t3:.2f} ms ({t1 / t3:.2f} 배)", flush=True)
# 부분표본: 집계는 앵커별 평균이라 일부만 써도 요약이 거의 같다. 값이 바뀌므로
# 공짜는 아니고, 얼마나 싸지는지와 얼마나 달라지는지를 같이 본다.
for frac in (0.25, 0.1):
    n = int(a.n * frac)
    g = torch.randperm(a.n, device=dev)[:n]
    xs, Xs, vs, ms, ids = x[g], X[g], v[g], m[g], idx[g]
    ts = timeit(lambda: dc.aggregate_moments2(xs, Xs, vs, ms, ids, a.m))
    os_ = dc.aggregate_moments2(xs, Xs, vs, ms, ids, a.m)
    # 1 차 모멘트는 질량합이라 표본 비율만큼 작아진다 -- 평균끼리 견준다
    c_full = o2[0][:, 1:4] / o2[0][:, 0:1].clamp(min=1e-12)
    c_sub = os_[0][:, 1:4] / os_[0][:, 0:1].clamp(min=1e-12)
    rel = ((c_full - c_sub).norm(dim=-1) / c_full.norm(dim=-1).clamp(min=1e-9))
    print(f"[부분표본 {frac:.0%}] {ts:.2f} ms ({t2 / ts:.1f} 배), "
          f"앵커 무게중심 상대차 중앙 {rel.median():.2e} 최대 {rel.max():.2e}",
          flush=True)

print("AGGBENCH_OK", flush=True)

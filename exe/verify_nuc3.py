"""닫힌형 핵 노름이 torch.linalg.svdvals 와 같은 값·같은 기울기를 주는지 본다.

거의 등방인 행렬(고유값이 겹치는 자리)이 삼각함수 해의 약점이라 일부러 섞는다.
"""
from __future__ import annotations

import os
import sys
import time

_lib = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib")
sys.path.insert(0, _lib)

import torch                                                      # noqa: E402

from anchorflow.deform import _nuc3                               # noqa: E402

dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
N = 20000
cases = {
    "일반": torch.randn(N, 3, 3, device=dev),
    "거의 등방": torch.eye(3, device=dev).expand(N, 3, 3)
                 + 1e-6 * torch.randn(N, 3, 3, device=dev),
    "특이": torch.randn(N, 3, 3, device=dev) * torch.tensor(
        [1.0, 1.0, 1e-7], device=dev),
    "아주 작은": 1e-9 * torch.randn(N, 3, 3, device=dev),
}
ok = True
for name, M in cases.items():
    ref = torch.linalg.svdvals(M).clamp_min(1e-8).sum(-1)
    got = _nuc3(M, 1e-8)
    rel = ((got - ref).abs() / ref.abs().clamp_min(1e-12)).max().item()
    ok &= rel < 1e-4
    print(f"  {name:10s} 최대 상대차 {rel:.3e}", flush=True)

M = torch.randn(N, 3, 3, device=dev, requires_grad=True)
g1 = torch.autograd.grad(torch.linalg.svdvals(M).clamp_min(1e-8).sum(), M)[0]
g2 = torch.autograd.grad(_nuc3(M, 1e-8).sum(), M)[0]
gr = ((g1 - g2).abs().max() / g1.abs().max()).item()
ok &= gr < 1e-4
print(f"  기울기 최대 상대차 {gr:.3e}", flush=True)


def timeit(fn, n=30):
    for _ in range(3):
        fn()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1e3


A = torch.randn(N, 3, 3, device=dev)
t1 = timeit(lambda: torch.linalg.svdvals(A).clamp_min(1e-8).sum(-1))
t2 = timeit(lambda: _nuc3(A, 1e-8))
print(f"[속도] svdvals {t1:.2f} ms -> 닫힌형 {t2:.2f} ms ({t1 / t2:.1f} 배)",
      flush=True)
print("NUC3_" + ("OK" if ok else "FAIL"), flush=True)

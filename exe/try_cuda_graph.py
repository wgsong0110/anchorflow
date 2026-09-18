"""어텐션을 CUDA 그래프로 잡을 수 있는지, 잡으면 얼마나 빨라지는지만 따로 본다.

앵커 512 개짜리 신경망은 연산이 아니라 커널 실행 횟수가 비용이라(2.6 ms) 그래프로
접으면 크게 준다. 다만 **잡기에 실패하면 그 프로세스의 CUDA 문맥이 망가져** 뒤의
측정이 전부 죽으므로, 프레임 벤치와 같은 프로세스에서 시도하지 않는다.

SDPA 의 flash/mem-efficient 백엔드가 잡히지 않는 경우가 있어 백엔드별로 해 본다.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

_lib = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib")
sys.path.insert(0, _lib)

import torch                                                      # noqa: E402

from anchorflow.deform import DeformNet                           # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--m", type=int, default=512)
ap.add_argument("--n_feat", type=int, default=41)
ap.add_argument("--hidden", type=int, default=128)
ap.add_argument("--depth", type=int, default=4)
ap.add_argument("--heads", type=int, default=4)
ap.add_argument("--backend", default="auto",
                choices=("auto", "math", "flash", "mem"))
ap.add_argument("--reps", type=int, default=200)
a = ap.parse_args()

dev = "cuda"
torch.set_grad_enabled(False)
net = DeformNet(n_feat=a.n_feat, hidden=a.hidden, depth=a.depth, heads=a.heads,
                scale=0.03, h=0.06, ext=1.62).to(dev).eval()
p = torch.rand(a.m, 3, device=dev)
f = torch.randn(a.m, a.n_feat, device=dev)
DT = 0.01

if a.backend != "auto":
    torch.backends.cuda.enable_flash_sdp(a.backend == "flash")
    torch.backends.cuda.enable_mem_efficient_sdp(a.backend == "mem")
    torch.backends.cuda.enable_math_sdp(a.backend == "math")


def timeit(fn, n):
    for _ in range(10):
        fn()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1e3


t_plain = timeit(lambda: net(p, f, DT), a.reps)
print(f"[{a.backend}] 그냥 부르기 {t_plain:.3f} ms", flush=True)

ref = [z.clone() for z in net(p, f, DT)]
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        net(p, f, DT)
torch.cuda.current_stream().wait_stream(s)
torch.cuda.synchronize()
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    out = net(p, f, DT)
torch.cuda.synchronize()
g.replay(); torch.cuda.synchronize()
err = max(float((o - r).abs().max()) for o, r in zip(out, ref))
t_graph = timeit(g.replay, a.reps)
print(f"[{a.backend}] 그래프 재생 {t_graph:.3f} ms ({t_plain / t_graph:.2f} 배), "
      f"출력 최대 어긋남 {err:.3e}", flush=True)
print("GRAPH_OK", flush=True)

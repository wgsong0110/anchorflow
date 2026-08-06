"""Where does a training iteration actually go?

732k parameters over 512 anchors at batch 8 should be a few milliseconds, and
the observed rate is ~43 ms an iteration. This times the pieces separately so
the answer is measured rather than guessed: the pairwise attention bias builds
a [B, M, M, hidden] tensor, which is 134 MB at these sizes and is the first
thing to suspect.
"""
from __future__ import annotations

import argparse
import os
import sys

_lib = os.path.join(os.path.dirname(__file__), "..", "lib")
sys.path.insert(0, _lib)

import torch

from anchorflow.nextstate import NextStep

ap = argparse.ArgumentParser()
ap.add_argument("--M", type=int, default=512)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--hidden", type=int, default=128)
ap.add_argument("--depth", type=int, default=4)
ap.add_argument("--bias_hidden", type=int, default=32)
ap.add_argument("--repeat", type=int, default=30)
args = ap.parse_args()

dev = "cuda"
torch.manual_seed(0)
B, M = args.batch, args.M
p = torch.randn(B, M, 3, device=dev)
v = torch.randn(B, M, 3, device=dev)
a = torch.randn(B, M, 3, device=dev)
tgt = torch.randn(B, M, 3, device=dev) * 1e-3
net = NextStep(args.hidden, args.depth, 4, 1e-3, 1.0, 1.0).to(dev)
opt = torch.optim.Adam(net.parameters(), lr=1e-3, fused=True)
print(f"[setup] {torch.cuda.get_device_name(0)}  B={B} M={M} hidden={args.hidden} "
      f"depth={args.depth}  params={sum(q.numel() for q in net.parameters())/1e3:.0f}k")


def timeit(fn, n=None):
    n = n or args.repeat
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(n):
        fn()
    e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1) / n


def bias_only():
    with torch.no_grad():
        net.bias(p)


def fwd():
    with torch.no_grad():
        net(p, v, a, 4e-3)


def fwd_bwd():
    du = net(p, v, a, 4e-3)
    ((du - tgt) ** 2).mean().backward()


def full():
    opt.zero_grad(set_to_none=True)
    du = net(p, v, a, 4e-3)
    loss = ((du - tgt) ** 2).mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    opt.step()


ms_b = timeit(bias_only)
ms_f = timeit(fwd)
ms_fb = timeit(fwd_bwd)
ms_a = timeit(full)
print(f"\n{'pairwise geometry bias':>28} {ms_b:8.2f} ms   {100*ms_b/ms_a:5.1f}%")
print(f"{'forward (incl. bias)':>28} {ms_f:8.2f} ms   {100*ms_f/ms_a:5.1f}%")
print(f"{'forward + backward':>28} {ms_fb:8.2f} ms   {100*ms_fb/ms_a:5.1f}%")
print(f"{'full iteration':>28} {ms_a:8.2f} ms   -> {1000/ms_a:.1f} it/s")

big = B * M * M * args.bias_hidden * 4 / 1e6
print(f"\n[note] the bias MLP's hidden activation is [B,M,M,{args.bias_hidden}] = {big:.0f} MB, "
      f"kept for backward")
print(f"[note] attention itself is [B,4,M,M] = {B*4*M*M*4/1e6:.0f} MB per layer")

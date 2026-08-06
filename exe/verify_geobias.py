"""Does the fused geometry bias compute the same function as the eager one?

The kernel exists only to move memory less, so it has to agree with the layer it
replaces -- forward, the gradient to the anchor positions, and the gradients to
all 292 parameters. Association order differs from cuBLAS's, so agreement is to
float32 rounding rather than bitwise; anything beyond that is a bug.
"""
from __future__ import annotations

import argparse
import os
import sys

_lib = os.path.join(os.path.dirname(__file__), "..", "lib")
sys.path.insert(0, _lib)

import torch

from anchorflow.nextstate import GeoAttentionBias
import geobias

ap = argparse.ArgumentParser()
ap.add_argument("--B", type=int, default=8)
ap.add_argument("--M", type=int, default=512)
ap.add_argument("--repeat", type=int, default=20)
args = ap.parse_args()

dev = "cuda"
torch.manual_seed(0)
print(f"[setup] geobias HAVE_CUDA={geobias.HAVE_CUDA}  B={args.B} M={args.M}")
assert geobias.HAVE_CUDA, "kernel not built"

layer = GeoAttentionBias(4).to(dev)
p = (torch.randn(args.B, args.M, 3, device=dev) * 0.3).requires_grad_(True)

with torch.no_grad():
    rel0 = p.unsqueeze(2) - p.unsqueeze(1)
    inv_s = 1.0 / rel0.norm(dim=-1).mean().clamp(min=1e-12)

lin = [m for m in layer.mlp.modules() if isinstance(m, torch.nn.Linear)]
gout = torch.randn(args.B, 4, args.M, args.M, device=dev)

# eager
out_e = layer._eager(p, inv_s)
gs_e = torch.autograd.grad(out_e, [p] + [q for m in lin for q in (m.weight, m.bias)],
                            grad_outputs=gout, retain_graph=False)

# fused
p2 = p.detach().clone().requires_grad_(True)
ws = [lin[0].weight.detach().clone().requires_grad_(True),
      lin[0].bias.detach().clone().requires_grad_(True),
      lin[1].weight.detach().clone().requires_grad_(True),
      lin[1].bias.detach().clone().requires_grad_(True)]
out_f = geobias.fused_geo_bias(p2, ws[0], ws[1], ws[2], ws[3], float(inv_s))
gs_f = torch.autograd.grad(out_f, [p2] + ws, grad_outputs=gout)


def rel(a, b):
    return ((a - b).norm() / b.norm().clamp(min=1e-20)).item()


print(f"\n{'quantity':>22} {'max abs diff':>14} {'relative':>11}")
print(f"{'forward':>22} {(out_f - out_e).abs().max():14.3e} {rel(out_f, out_e):11.3e}")
names = ["d/d positions", "d/d W1", "d/d b1", "d/d W2", "d/d b2"]
for n, a, b in zip(names, gs_f, gs_e):
    print(f"{n:>22} {(a - b).abs().max():14.3e} {rel(a, b):11.3e}")


def timeit(fn):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(args.repeat):
        fn()
    e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1) / args.repeat


def eager_fb():
    o = layer._eager(p, inv_s)
    torch.autograd.grad(o, [p] + [q for m in lin for q in (m.weight, m.bias)],
                         grad_outputs=gout)


def fused_fb():
    o = geobias.fused_geo_bias(p2, ws[0], ws[1], ws[2], ws[3], float(inv_s))
    torch.autograd.grad(o, [p2] + ws, grad_outputs=gout)


me, mf = timeit(eager_fb), timeit(fused_fb)
print(f"\n[time] forward+backward   eager {me:7.3f} ms   fused {mf:7.3f} ms   "
      f"-> {me / mf:.2f}x")
print(f"[memory] eager keeps a [B,M,M,32] activation = "
      f"{args.B * args.M * args.M * 32 * 4 / 1e6:.0f} MB; the kernel writes only "
      f"[B,4,M,M] = {args.B * 4 * args.M * args.M * 4 / 1e6:.0f} MB")

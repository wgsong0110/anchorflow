"""Check the two processors run, start at zero, and differ where it matters:
how far one anchor's influence reaches in a single forward pass.

The implicit step's solution operator is dense -- every anchor's displacement
depends on every other's -- so a processor that cannot propagate information
across the whole anchor set in one pass cannot represent it, no matter how it
is trained. This measures that directly: perturb one anchor's input velocity
and count how many outputs move.
"""
from __future__ import annotations

import argparse
import os
import sys

_lib = os.path.join(os.path.dirname(__file__), "..", "lib")
sys.path.insert(0, _lib)

import torch

from anchorflow.implicit import DuCorrector
from anchorflow import graph as G

ap = argparse.ArgumentParser()
ap.add_argument("--M", type=int, default=512)
ap.add_argument("--hidden", type=int, default=128)
ap.add_argument("--depth", type=int, default=4)
ap.add_argument("--k", type=int, default=8)
args = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
M = args.M
# a branch-like cloud rather than a blob: the failure being probed is reach
# along a slender structure, which is what ficus is
t = torch.linspace(0, 1, M, device=dev).unsqueeze(-1)
p = torch.cat([0.15 * torch.sin(6 * t), t, 0.15 * torch.cos(6 * t)], -1)
p = p + 0.01 * torch.randn(M, 3, device=dev)
v = torch.randn(M, 3, device=dev) * 0.5
a = torch.randn(M, 3, device=dev) * 100
f = torch.randn(M, 3, device=dev) * 100
ei = G.knn_graph(p, k=args.k)

print(f"[setup] M={M} hidden={args.hidden} depth={args.depth} k={args.k} dev={dev}")
print(f"{'processor':>10} {'out':>12} {'zero-init':>10} {'params':>9} "
      f"{'fwd ms':>8} {'reached':>12} {'median |d|':>11}")

for proc in ("mpnn", "attention"):
    net = DuCorrector(args.hidden, args.depth, use_force=True,
                       processor=proc, heads=4).to(dev)
    with torch.no_grad():
        out = net(p, v, a, 4e-3, ei, f)
    zero = out.abs().max().item()
    nparam = sum(q.numel() for q in net.parameters())

    if dev == "cuda":
        for _ in range(3):
            net(p, v, a, 4e-3, ei, f)
        torch.cuda.synchronize()
        ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
        ev0.record()
        for _ in range(50):
            with torch.no_grad():
                net(p, v, a, 4e-3, ei, f)
        ev1.record(); torch.cuda.synchronize()
        ms = ev0.elapsed_time(ev1) / 50
    else:
        ms = float("nan")

    # the decoder is zero-init, so un-zero it before probing reach
    last = [m for m in net.dec.modules() if isinstance(m, torch.nn.Linear)][-1]
    torch.nn.init.normal_(last.weight, std=0.1)
    with torch.no_grad():
        base = net(p, v, a, 4e-3, ei, f)
        v2 = v.clone(); v2[0] += 1.0
        pert = net(p, v2, a, 4e-3, ei, f)
    d = (pert - base).norm(dim=-1)
    reached = int((d > 1e-6).sum())
    print(f"{proc:>10} {str(tuple(out.shape)):>12} {zero:10.2e} {nparam/1e3:8.1f}k "
          f"{ms:8.3f} {reached:>7}/{M} {d.median().item():11.2e}")

# how many hops the k-NN graph actually needs to cross this cloud, i.e. what
# depth a message-passing processor would need before it could even see it
src, dst = ei
reach = torch.zeros(M, dtype=torch.bool, device=dev); reach[0] = True
for hop in range(1, 64):
    nxt = reach.clone()
    nxt[dst[reach[src]]] = True
    if nxt.sum() == reach.sum():
        break
    reach = nxt
    if reach.all():
        print(f"\n[graph] k-NN graph diameter from anchor 0: {hop} hops "
              f"(message passing needs at least this depth to couple the ends)")
        break
else:
    print(f"\n[graph] anchor 0 reaches only {int(reach.sum())}/{M} within 64 hops")

"""Structural checks on the DuCorrector variants -- no training, no physics.

Three things are worth knowing before spending GPU hours on any of them:

  * REACH. The implicit step's solution operator is a dense global inverse:
    pinning the pot changes the branch tips within the same step. A processor
    that cannot propagate one anchor's influence to all the others in a single
    forward pass cannot represent it however it is trained. Perturb one anchor's
    input and count how many outputs move.
  * dt CONDITIONING. FiLM is zero-init, so at the start gamma = 1, beta = 0 and
    dt does nothing -- that is intended, but it has to be checked that the path
    exists at all, i.e. that a non-trivial FiLM does change the output.
  * COST. Attention is O(M^2 d) against message passing's O(M k d^2); at M = 512
    the arithmetic says attention should be the cheaper of the two, which is
    worth confirming on the actual device rather than on paper.
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
# a slender branch-like cloud, not a blob: the reach that matters is along a
# thin structure, which is what ficus is
t = torch.linspace(0, 1, M, device=dev).unsqueeze(-1)
p = torch.cat([0.15 * torch.sin(6 * t), t, 0.15 * torch.cos(6 * t)], -1)
p = p + 0.01 * torch.randn(M, 3, device=dev)
v = torch.randn(M, 3, device=dev) * 0.5
a = torch.randn(M, 3, device=dev)
f = torch.randn(M, 3, device=dev)
ei = G.knn_graph(p, k=args.k)

# how many hops the k-NN graph needs to cross this cloud -- the depth a
# message-passing processor would need before it could even see end to end
src, dst = ei
reach = torch.zeros(M, dtype=torch.bool, device=dev); reach[0] = True
diameter = None
for hop in range(1, 256):
    nxt = reach.clone()
    nxt[dst[reach[src]]] = True
    if bool(nxt.all()):
        diameter = hop
        break
    if int(nxt.sum()) == int(reach.sum()):
        break
    reach = nxt
print(f"[setup] M={M} hidden={args.hidden} depth={args.depth} k={args.k} dev={dev}")
print(f"[graph] k-NN hops needed to reach every anchor from anchor 0: {diameter} "
      f"(message passing at depth {args.depth} covers {args.depth})")

CASES = [("mpnn", False), ("attention", False), ("attention", True)]
print(f"\n{'processor':>10} {'raw_io':>7} {'out':>11} {'init|out|':>10} {'params':>9} "
      f"{'fwd ms':>8} {'reached':>11} {'dt effect':>10}")

for proc, raw in CASES:
    net = DuCorrector(args.hidden, args.depth, use_force=True,
                       processor=proc, heads=4, raw_io=raw).to(dev)
    with torch.no_grad():
        out = net(p, v, a, 4e-3, ei, f)
    nparam = sum(q.numel() for q in net.parameters())

    # FiLM must start as the identity
    g, b = net.film(4e-3, dev)
    assert torch.allclose(g, torch.ones_like(g)) and torch.allclose(b, torch.zeros_like(b)), \
        "FiLM is not identity at init"

    if dev == "cuda":
        for _ in range(5):
            net(p, v, a, 4e-3, ei, f)
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        with torch.no_grad():
            for _ in range(50):
                net(p, v, a, 4e-3, ei, f)
        e1.record(); torch.cuda.synchronize()
        ms = e0.elapsed_time(e1) / 50
    else:
        ms = float("nan")

    # un-zero the decoder (and the FiLM head) so the probes see a live network
    for mod in (net.dec, net.film.mlp):
        last = [m for m in mod.modules() if isinstance(m, torch.nn.Linear)][-1]
        torch.nn.init.normal_(last.weight, std=0.1)
        torch.nn.init.normal_(last.bias, std=0.1)

    with torch.no_grad():
        base = net(p, v, a, 4e-3, ei, f)
        v2 = v.clone(); v2[0] += 1.0
        pert = net(p, v2, a, 4e-3, ei, f)
        slow = net(p, v, a, 1e-4, ei, f)
    reached = int(((pert - base).norm(dim=-1) > 1e-6).sum())
    dt_eff = ((slow - base).norm() / base.norm().clamp(min=1e-20)).item()

    print(f"{proc:>10} {str(raw):>7} {str(tuple(out.shape)):>11} {out.abs().max():10.2e} "
          f"{nparam/1e3:8.1f}k {ms:8.3f} {reached:>7}/{M} {dt_eff:10.4f}")

print("\n[note] init|out| is the output magnitude at initialisation: zero for the "
      "correction parameterisations (training starts at an explicit step), nonzero "
      "for raw_io, whose decoder cannot be zero-init.")

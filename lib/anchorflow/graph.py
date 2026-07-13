"""Anchor graph construction.

Given anchor centres [N, 3] we build an edge list [2, E] (directed, both
directions stored so message passing is symmetric).  Convention throughout the
package:

    edge_index[0] = src  (sender  / neighbour j)
    edge_index[1] = dst  (receiver/ node       i)

so a message computed for column e flows j -> i and is aggregated at dst=i.

Two builders, both dependency-free (only torch).  Anchor counts here are in the
tens/hundreds, so the O(N^2) distance matrix is entirely fine and keeps the code
transparent.  Swap in PyG's ``torch_geometric.nn.radius_graph`` / ``knn_graph``
unchanged if you later need large-N sampling.
"""

from __future__ import annotations

import torch


def _pairwise_dist(pos: torch.Tensor) -> torch.Tensor:
    return torch.cdist(pos, pos)                       # [N, N]


def knn_graph(pos: torch.Tensor, k: int, loop: bool = False) -> torch.Tensor:
    """k nearest neighbours of every node.  Returns edge_index [2, E].

    Edges are made symmetric (if j is a neighbour of i we add both i<-j and
    j<-i) and de-duplicated, so E <= 2*N*k.
    """
    N = pos.shape[0]
    k = min(k, N - 1) if not loop else min(k, N)
    d = _pairwise_dist(pos)
    if not loop:
        d.fill_diagonal_(float("inf"))
    nn = d.topk(k, largest=False).indices              # [N, k] neighbour ids
    dst = torch.arange(N, device=pos.device).repeat_interleave(k)
    src = nn.reshape(-1)
    edge = torch.stack([src, dst], dim=0)              # j -> i
    return _symmetrize(edge, N, loop)


def radius_graph(pos: torch.Tensor, r: float, max_neighbors: int = 32,
                 loop: bool = False) -> torch.Tensor:
    """All neighbours within radius r (capped at max_neighbors closest)."""
    N = pos.shape[0]
    d = _pairwise_dist(pos)
    if not loop:
        d.fill_diagonal_(float("inf"))
    within = d <= r                                    # [N, N] bool
    # cap degree: keep the max_neighbors closest among the in-radius set
    masked = torch.where(within, d, torch.full_like(d, float("inf")))
    kk = min(max_neighbors, N)
    vals, idx = masked.topk(kk, largest=False)         # [N, kk]
    keep = torch.isfinite(vals)
    dst = torch.arange(N, device=pos.device)[:, None].expand(-1, kk)[keep]
    src = idx[keep]
    if src.numel() == 0:                               # degenerate: fall back to knn
        return knn_graph(pos, k=min(6, N - 1), loop=loop)
    edge = torch.stack([src, dst], dim=0)
    return _symmetrize(edge, N, loop)


def _symmetrize(edge: torch.Tensor, N: int, loop: bool) -> torch.Tensor:
    both = torch.cat([edge, edge.flip(0)], dim=1)      # add reverse edges
    if not loop:
        both = both[:, both[0] != both[1]]
    key = both[0] * N + both[1]                        # unique (src,dst) pairs
    key, order = torch.unique(key, return_inverse=False, sorted=True), None
    src = key // N
    dst = key % N
    return torch.stack([src, dst], dim=0).long()


def edge_features(pos: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Relative displacement + distance for every edge -> [E, 4].

    Uses only *relative* geometry so the model is translation-invariant.
    """
    src, dst = edge_index
    rel = pos[src] - pos[dst]                          # j - i  (points to sender)
    dist = rel.norm(dim=-1, keepdim=True)
    return torch.cat([rel, dist], dim=-1)

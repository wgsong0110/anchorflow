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


def build_hierarchical_graph(
    anchor_xyz: torch.Tensor,   # [M, 3]
    anchor_obj: torch.Tensor,   # [M] long, object IDs
    k_intra: int = 12,
    k_inter: int = 4,
) -> tuple:
    """Build intra-object KNN + inter-object boundary edges (both bidirectional).

    Returns:
        intra_edge_index [2, E_intra]
        intra_rest       [E_intra]   canonical distances
        inter_edge_index [2, E_inter]
        inter_rest       [E_inter]   canonical distances
    """
    M     = anchor_xyz.shape[0]
    n_obj = int(anchor_obj.max().item()) + 1
    dev   = anchor_xyz.device

    intra_pairs: list[tuple] = []
    inter_pairs: list[tuple] = []

    for oi in range(n_obj):
        idx_i = (anchor_obj == oi).nonzero(as_tuple=True)[0]
        if len(idx_i) < 2:
            continue
        xyz_i = anchor_xyz[idx_i]

        # ── intra-object KNN ──────────────────────────────────────────── #
        k = min(k_intra, len(idx_i) - 1)
        d = torch.cdist(xyz_i, xyz_i)
        d.fill_diagonal_(float("inf"))
        nbrs = d.topk(k, largest=False).indices       # [len_i, k]
        for li in range(len(idx_i)):
            for lj in nbrs[li].tolist():
                intra_pairs.append((int(idx_i[li]), int(idx_i[lj])))

        # ── inter-object boundary edges ───────────────────────────────── #
        for oj in range(n_obj):
            if oi >= oj:          # avoid double-counting (symmetrize later)
                continue
            idx_j = (anchor_obj == oj).nonzero(as_tuple=True)[0]
            if len(idx_j) == 0:
                continue
            xyz_j = anchor_xyz[idx_j]
            d_cross = torch.cdist(xyz_i, xyz_j)       # [len_i, len_j]

            # Only boundary anchors: bottom 30% closest distance to the other object
            min_dist = d_cross.min(dim=1).values
            thresh   = min_dist.quantile(0.30)
            border   = (min_dist <= thresh).nonzero(as_tuple=True)[0]

            k2 = min(k_inter, len(idx_j))
            nbrs_j = d_cross[border].topk(k2, largest=False).indices  # [|border|, k2]
            for bi, li in enumerate(border.tolist()):
                for lj in nbrs_j[bi].tolist():
                    inter_pairs.append((int(idx_i[li]), int(idx_j[lj])))

    def _to_edge(pairs):
        if not pairs:
            return (torch.zeros(2, 0, dtype=torch.long, device=dev),
                    torch.zeros(0, device=dev))
        t    = torch.tensor(pairs, dtype=torch.long, device=dev).T  # [2, E]
        both = torch.cat([t, t.flip(0)], dim=1)
        key, _ = torch.unique(both[0] * M + both[1], return_inverse=True)
        src, dst = key // M, key % M
        edge = torch.stack([src, dst], dim=0)
        rest = (anchor_xyz[src] - anchor_xyz[dst]).norm(dim=-1)
        return edge, rest

    intra_edge, intra_rest = _to_edge(intra_pairs)
    inter_edge, inter_rest = _to_edge(inter_pairs)

    print(f"[graph] intra edges={intra_edge.shape[1]}  inter edges={inter_edge.shape[1]}", flush=True)
    return intra_edge, intra_rest, inter_edge, inter_rest


def edge_features(pos: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Relative displacement + distance for every edge -> [E, 4].

    Uses only *relative* geometry so the model is translation-invariant.
    """
    src, dst = edge_index
    rel = pos[src] - pos[dst]                          # j - i  (points to sender)
    dist = rel.norm(dim=-1, keepdim=True)
    return torch.cat([rel, dist], dim=-1)

"""Motion regularizers on the anchor trajectory (SC-GS-faithful).

These constrain the GNN's rollout to a physically plausible, spatially coherent
manifold — compensating for the absence of MPM's physics prior (the trade we
accepted for a per-scene learned simulator). All operate directly on the rollout
node positions [T, M, 3] (no re-sampling of a time-conditioned network needed).

    arap    as-rigid-as-possible: each anchor's neighbourhood should move by a
            rigid rotation; penalise the residual stretch. (main regularizer)
    elastic temporal variance of edge lengths (cheap ARAP surrogate).
    acc     2nd-order finite-difference acceleration magnitude (smoothness).
"""

from __future__ import annotations

import torch

from . import geom
from .anchors import knn


def connectivity(nodes, K=10):
    """kNN graph on rest anchors -> (idx [M,K], w [M,K]) with exp(-d2/mean) weights."""
    K = min(K, nodes.shape[0] - 1)
    dist2, idx = knn(nodes, nodes, K + 1)
    dist2, idx = dist2[:, 1:], idx[:, 1:]                     # drop self
    w = torch.exp(-dist2 / (dist2.mean() + 1e-9))
    w = w / w.sum(dim=-1, keepdim=True)
    return idx, w


def arap_loss(node_seq, idx, w):
    """node_seq [T,M,3]. Sum_t Sum_i Sum_j w_ij || e_ij^t - R_i e_ij^0 ||^2,
    with per-anchor R_i from weighted Procrustes (fixed, no_grad — ARAP scheme)."""
    rest = node_seq[0]
    src = rest[idx] - rest[:, None]                           # [M,K,3] rest edges
    T = node_seq.shape[0]
    err = node_seq.new_zeros(())
    for t in range(1, T):
        tgt = node_seq[t][idx] - node_seq[t][:, None]         # [M,K,3]
        with torch.no_grad():
            R = geom.procrustes_rotation(src, tgt, w)         # [M,3,3]
        rigid = torch.einsum("mab,mkb->mka", R, src)
        err = err + (w * (tgt - rigid).pow(2).sum(-1)).sum()
    return err / max(1, T - 1)


def elastic_loss(node_seq, idx):
    """Temporal variance of edge lengths (self-normalised). node_seq [T,M,3]."""
    edge = (node_seq[:, idx] - node_seq[:, :, None]).norm(dim=-1)   # [T,M,K]
    var = edge.var(dim=0)                                     # [M,K]
    return (var / (var.detach() + 1e-5)).mean()


def acc_loss(node_seq):
    """2nd-order finite-difference acceleration magnitude. node_seq [T,M,3]."""
    acc = node_seq[2:] + node_seq[:-2] - 2 * node_seq[1:-1]   # [T-2,M,3]
    return acc.norm(dim=-1).mean()


def total(node_seq, idx, w, lambdas=(1.0, 0.1, 0.01)):
    la, le, lc = lambdas
    return la * arap_loss(node_seq, idx, w) \
        + le * elastic_loss(node_seq, idx) \
        + lc * acc_loss(node_seq)

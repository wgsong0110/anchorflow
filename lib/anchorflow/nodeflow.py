"""NodeFlow: GNN-based Gaussian deformation from semantic anchor nodes.

Architecture (From Tokens to Nodes + DreamPhysics MDS):
  1. Anchor nodes: placed by tokens_to_nodes (semantic-guided) or FPS fallback
  2. GNN: takes z0 [K, z0_dim] (initial state from bank) + canonical node positions
     → node displacements at time t via time-conditioned decoder
  3. LBS: node displacements → Gaussian displacements via RBF weights
  4. t=0 always returns zero displacement (canonical = plausible SVD conditioning frame)
  5. z0_bank [B, K, z0_dim] lives in the training script; each entry = one initial
     velocity/state; all entries are jointly optimized with GNN weights via MDS.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .anchors import fps


# ── GraphSAGE layer ──────────────────────────────────────────────────────────

class GNNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim * 2, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        K, D = h.shape
        agg = torch.zeros(K, D, device=h.device, dtype=h.dtype)
        cnt = torch.zeros(K, 1, device=h.device, dtype=h.dtype)
        agg.index_add_(0, dst, h[src])
        cnt.index_add_(0, dst, torch.ones(src.shape[0], 1, device=h.device, dtype=h.dtype))
        agg = agg / cnt.clamp(min=1.0)
        h_out = self.mlp(torch.cat([h, agg], dim=-1))
        res = h if h.shape == h_out.shape else h_out
        return self.norm(h_out + res)


# ── NodeFlow ─────────────────────────────────────────────────────────────────

class NodeFlow(nn.Module):
    """
    Args:
        canonical_xyz   [G,3]  Gaussian positions (fixed)
        node_positions  [K,3]  pre-computed semantic nodes; None → FPS
        n_nodes         K      used only when node_positions is None
        n_frames        T      video length
        hidden          H      GNN hidden dim
        n_gnn_layers    L      message-passing rounds
        k_node          kN     k-NN for node message passing
        k_gauss         kG     k-NN nodes per Gaussian for LBS
        z0_dim          D      initial-state dimensionality
        arap_k               neighbours for ARAP regularisation
    """

    def __init__(
        self,
        canonical_xyz: torch.Tensor,
        node_positions: torch.Tensor | None = None,
        n_nodes: int = 256,
        n_frames: int = 25,
        hidden: int = 128,
        n_gnn_layers: int = 4,
        k_node: int = 8,
        k_gauss: int = 4,
        z0_dim: int = 32,
        arap_k: int = 6,
    ):
        super().__init__()
        G = canonical_xyz.shape[0]

        if node_positions is not None:
            K = node_positions.shape[0]
            node_pos = node_positions.detach().clone()
        else:
            K = min(n_nodes, G)
            node_pos = canonical_xyz[fps(canonical_xyz, K)].detach().clone()

        self.register_buffer("canonical_xyz", canonical_xyz.detach().clone())
        self.register_buffer("node_pos", node_pos)
        self.n_nodes   = K
        self.n_frames  = n_frames
        self.z0_dim    = z0_dim

        # node-node kNN graph
        from .graph import knn_graph
        self.register_buffer("node_edge",  knn_graph(node_pos, k=min(k_node, K - 1)))
        self.register_buffer("arap_edge",  knn_graph(node_pos, k=min(arap_k, K - 1)))

        # Gaussian-to-node RBF binding (fixed at canonical, SC-GS style)
        kG = min(k_gauss, K)
        d = torch.cdist(canonical_xyz, node_pos)
        dist, gn_idx = d.topk(kG, largest=False, dim=-1)
        scale = dist[:, 0:1].clamp(min=1e-6)
        w = torch.softmax(-dist / scale, dim=-1)
        self.register_buffer("gauss_node_idx", gn_idx)    # [G, kG]
        self.register_buffer("gauss_node_w",   w)          # [G, kG]

        # ── GNN: (node_pos ∥ z0) → hidden features ──────────────────────────
        self.node_encoder = nn.Sequential(
            nn.Linear(3 + z0_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
        )
        self.gnn_layers = nn.ModuleList(
            [GNNLayer(hidden, hidden) for _ in range(n_gnn_layers)]
        )
        # Time-conditioned displacement decoder
        # Input: h [K,H] ∥ t_norm [K,1] (sin+cos fourier for time)
        t_feat = 8   # fourier features for time
        self.decoder = nn.Sequential(
            nn.Linear(hidden + t_feat * 2, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden // 2, 3),
        )
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

        # Frequencies for Fourier time embedding
        freqs = 2.0 ** torch.arange(t_feat).float()
        self.register_buffer("t_freqs", freqs)

    # ── time embedding ───────────────────────────────────────────────────────
    def _t_emb(self, t_norm: float, K: int) -> torch.Tensor:
        """Fourier positional encoding for scalar time. Returns [K, 2*t_feat]."""
        ang = t_norm * self.t_freqs * math.pi   # [t_feat]
        emb = torch.cat([ang.sin(), ang.cos()])  # [2*t_feat]
        return emb.unsqueeze(0).expand(K, -1)    # [K, 2*t_feat]

    # ── encode: GNN pass (t-independent; call once per step) ────────────────
    def encode(self, z0: torch.Tensor) -> torch.Tensor:
        """z0 [K, z0_dim] → node features h [K, H]. Call once per step."""
        h = self.node_encoder(torch.cat([self.node_pos, z0], dim=-1))
        for layer in self.gnn_layers:
            h = layer(h, self.node_edge)
        return h

    # ── decode: time-conditioned displacement from cached h ──────────────────
    def decode(self, h: torch.Tensor, t: float) -> torch.Tensor:
        """h [K, H], t float → [G, 3] Gaussian displacement."""
        G   = self.canonical_xyz.shape[0]
        dev = self.node_pos.device
        if t <= 0.0:
            return torch.zeros(G, 3, device=dev)
        K = self.n_nodes
        t_norm    = t / max(self.n_frames - 1, 1)
        t_emb     = self._t_emb(t_norm, K)
        node_disp = self.decoder(torch.cat([h, t_emb.to(h.dtype)], dim=-1))  # [K, 3]
        nd_nb      = node_disp[self.gauss_node_idx]                           # [G, kG, 3]
        return (self.gauss_node_w.unsqueeze(-1) * nd_nb).sum(1)               # [G, 3]

    # ── forward: kept for compatibility (rollout / arap) ─────────────────────
    def forward(self, z0: torch.Tensor, t: float) -> torch.Tensor:
        """
        z0  : [K, z0_dim]  initial state sampled from z0_bank
        t   : float ∈ [0, T-1]; t=0 → zero displacement (canonical)
        Returns [G, 3] Gaussian displacement.
        """
        return self.decode(self.encode(z0), t)

    # ── regularisation ───────────────────────────────────────────────────────
    def arap_loss(self, h: torch.Tensor, t: float) -> torch.Tensor:
        """h [K, H] already encoded; returns ARAP loss at time t."""
        if t <= 0:
            return torch.tensor(0.0, device=self.node_pos.device)
        t_norm    = t / max(self.n_frames - 1, 1)
        t_emb     = self._t_emb(t_norm, self.n_nodes)
        node_disp = self.decoder(torch.cat([h, t_emb.to(h.dtype)], dim=-1))
        node_now  = self.node_pos + node_disp
        src, dst  = self.arap_edge
        d_rest = (self.node_pos[src] - self.node_pos[dst]).norm(dim=-1)
        d_now  = (node_now[src]      - node_now[dst]     ).norm(dim=-1)
        return ((d_now - d_rest) ** 2).mean()

    def smooth_z0_loss(self, z0_bank: torch.Tensor) -> torch.Tensor:
        """Penalise large magnitudes in z0_bank to keep initial states near canonical."""
        return (z0_bank ** 2).mean()

"""NodeFlow: GNN-based Gaussian deformation from sparse control nodes.

From Tokens to Nodes-inspired architecture:
  1. K control nodes via FPS on canonical Gaussian positions (fixed)
  2. Shared node trajectory: [K, T-1, 3] learnable displacement per frame
     (t=0 is implicitly zero → canonical = rest pose)
  3. Per-view initial state: [V, K, 3] learnable offset for each generated video
     node_disp(v, t) = init_offset[v] + trajectory(t)
  4. GNN: K nodes → G Gaussians via message passing + weighted aggregation
     node_encoder → L x GraphSAGE layers → Gaussian aggregation → decoder → [G,3]

The init_offset makes each video's initial frame independently trainable while the
trajectory (motion from t=0) is shared and physical.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .anchors import fps


# ---------------------------------------------------------------------------
# GNN layer (GraphSAGE-style, dependency-free)
# ---------------------------------------------------------------------------
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
        """h [K,D], edge_index [2,E] (src->dst). Returns [K,D]."""
        src, dst = edge_index
        K, D = h.shape
        # mean-aggregate neighbour features
        agg = torch.zeros(K, D, device=h.device, dtype=h.dtype)
        cnt = torch.zeros(K, 1, device=h.device, dtype=h.dtype)
        agg.index_add_(0, dst, h[src])
        cnt.index_add_(0, dst, torch.ones(dst.shape[0], 1, device=h.device, dtype=h.dtype))
        agg = agg / cnt.clamp(min=1.0)
        h_out = self.mlp(torch.cat([h, agg], dim=-1))
        return self.norm(h_out + h if h.shape == h_out.shape else h_out)


# ---------------------------------------------------------------------------
# NodeFlow model
# ---------------------------------------------------------------------------
class NodeFlow(nn.Module):
    """
    Args:
        canonical_xyz  [G,3]  static Gaussian positions (buffer, no grad)
        n_nodes        K      control nodes (FPS-sampled, fixed)
        n_views        V      number of generated single-view videos
        n_frames       T      video length (frames per video)
        hidden         H      GNN hidden dimension
        n_gnn_layers   L      message-passing rounds
        k_node         kN     k-NN among control nodes for message passing
        k_gauss        kG     k-NN nodes per Gaussian for aggregation
    """

    def __init__(
        self,
        canonical_xyz: torch.Tensor,
        n_nodes: int = 256,
        n_views: int = 8,
        n_frames: int = 21,
        hidden: int = 64,
        n_gnn_layers: int = 3,
        k_node: int = 8,
        k_gauss: int = 4,
        arap_k: int = 6,
    ):
        super().__init__()
        G = canonical_xyz.shape[0]
        K = min(n_nodes, G)

        # --- fixed canonical and control nodes ---
        self.register_buffer("canonical_xyz", canonical_xyz.detach().clone())
        node_idx = fps(canonical_xyz, K)
        node_pos = canonical_xyz[node_idx].detach().clone()
        self.register_buffer("node_pos", node_pos)          # [K,3]

        # --- node-node kNN graph (for message passing) ---
        from .graph import knn_graph
        edge_index = knn_graph(node_pos, k=min(k_node, K - 1))
        self.register_buffer("node_edge", edge_index)       # [2,E]

        # --- kNN for ARAP regularization (kept small) ---
        arap_edge = knn_graph(node_pos, k=min(arap_k, K - 1))
        self.register_buffer("arap_edge", arap_edge)

        # --- Gaussian-to-node binding (precomputed, fixed) ---
        kG = min(k_gauss, K)
        d = torch.cdist(canonical_xyz, node_pos)                # [G,K]
        dist, gn_idx = d.topk(kG, largest=False, dim=-1)        # [G,kG]
        # temperature-softmax: tighter binding near nodes
        scale = dist[:, 0:1].clamp(min=1e-6)
        w = torch.softmax(-dist / scale, dim=-1)                # [G,kG]
        self.register_buffer("gauss_node_idx", gn_idx)          # [G,kG]
        self.register_buffer("gauss_node_w", w)                 # [G,kG]

        # --- learnable parameters ---
        # trajectory: t=0 is 0 (canonical), t=1..T-1 are learned
        self.node_traj = nn.Parameter(torch.zeros(K, n_frames - 1, 3))
        # per-view initial offset (physical initial state per video)
        self.init_offset = nn.Parameter(torch.zeros(n_views, K, 3))

        # --- GNN ---
        self.node_encoder = nn.Sequential(
            nn.Linear(3, hidden),
            nn.ReLU(inplace=True),
        )
        self.gnn_layers = nn.ModuleList(
            [GNNLayer(hidden, hidden) for _ in range(n_gnn_layers)]
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 3),
        )

        self.n_frames = n_frames
        self.n_views = n_views
        self.n_nodes = K

        # zero-init decoder to start from identity (no displacement)
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    # ------------------------------------------------------------------
    def get_node_disp(self, view_idx: int, t: float) -> torch.Tensor:
        """Node displacement for view `view_idx` at time step `t` ∈ [0, T-1].
        Returns [K, 3].  Trajectory interpolated linearly between frames.
        """
        T = self.n_frames
        dev = self.node_traj.device

        # trajectory part (shared across views): zero at t=0
        if t <= 0.0:
            delta_t = torch.zeros(self.n_nodes, 3, device=dev)
        elif t >= T - 1:
            delta_t = self.node_traj[:, -1, :]
        else:
            t_lo = int(t)
            alpha = t - t_lo
            t0 = self.node_traj[:, t_lo - 1, :] if t_lo > 0 else torch.zeros(self.n_nodes, 3, device=dev)
            t1 = self.node_traj[:, t_lo, :]
            delta_t = (1.0 - alpha) * t0 + alpha * t1

        delta_init = self.init_offset[view_idx]               # [K,3]
        return delta_init + delta_t                            # [K,3]

    # ------------------------------------------------------------------
    def forward(self, view_idx: int, t: float) -> torch.Tensor:
        """Returns [G, 3] Gaussian displacement from canonical."""
        node_disp = self.get_node_disp(view_idx, t)           # [K,3]

        h = self.node_encoder(node_disp)                       # [K,H]
        for layer in self.gnn_layers:
            h = layer(h, self.node_edge)

        # aggregate node features to Gaussians
        h_nb = h[self.gauss_node_idx]                          # [G,kG,H]
        h_agg = (self.gauss_node_w.unsqueeze(-1) * h_nb).sum(1)  # [G,H]
        return self.decoder(h_agg)                             # [G,3]

    # ------------------------------------------------------------------
    # Regularization losses
    # ------------------------------------------------------------------
    def arap_loss(self, view_idx: int, t: float) -> torch.Tensor:
        """As-rigid-as-possible: penalize changes in inter-node distances."""
        node_disp = self.get_node_disp(view_idx, t)
        node_now = self.node_pos + node_disp
        src, dst = self.arap_edge
        d_rest = (self.node_pos[src] - self.node_pos[dst]).norm(dim=-1)
        d_now  = (node_now[src]      - node_now[dst]     ).norm(dim=-1)
        return ((d_now - d_rest) ** 2).mean()

    def smooth_loss(self) -> torch.Tensor:
        """Penalise trajectory acceleration (second-order finite differences)."""
        if self.node_traj.shape[1] < 2:
            return self.node_traj.sum() * 0.0
        traj = torch.cat([
            torch.zeros(self.n_nodes, 1, 3, device=self.node_traj.device),
            self.node_traj,
        ], dim=1)                                              # [K, T, 3]
        vel = traj[:, 1:] - traj[:, :-1]                      # [K, T-1, 3]
        acc = vel[:, 1:] - vel[:, :-1]                        # [K, T-2, 3]
        return (acc ** 2).mean()

    def traj_magnitude_loss(self) -> torch.Tensor:
        """L2 on init_offset to keep per-view initial states near canonical."""
        return (self.init_offset ** 2).mean()

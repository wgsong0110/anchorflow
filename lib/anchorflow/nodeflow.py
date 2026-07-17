"""NodeFlow: physics-based Gaussian deformation via anchor node integration.

Architecture:
  z0   [K, 3]  — initial velocity per node (from z0_bank)
  h    [K, H]  — scene-structure features from GNN(canonical_pos)  [once per step]
  acc[t][K,3]  — acceleration predicted by accel_decoder(h, t_emb)
  vel[t]       = z0 + cumsum(acc[0..t-1])
  disp[t]      = cumsum(vel[1..t])          (displacement from canonical)

All time steps are vectorised via cumsum — no sequential Python loop.
t=0 → zero displacement (canonical = plausible SVD conditioning frame).
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn

from .anchors import fps

try:                                  # fused CUDA LBS (lib/lbs)
    from lbs import lbs_blend as _lbs_blend_cuda, _HAVE_CUDA as _LBS_CUDA
except Exception:
    _LBS_CUDA = False


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
        canonical_xyz   [G,3]  Gaussian positions (fixed buffer)
        node_positions  [K,3]  pre-computed semantic nodes (None → FPS)
        n_nodes         K      used only when node_positions is None
        n_frames        T      video length (number of time steps)
        hidden          H      GNN hidden dim
        n_gnn_layers    L      message-passing rounds
        k_node          kN     k-NN for node graph
        k_gauss         kG     k-NN nodes per Gaussian for LBS
        arap_k               neighbours for ARAP regularisation
    Note: z0 is always [K, 3] (initial velocity), not a latent vector.
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
        arap_k: int = 6,
        dt: float = 1.0,
        # z0_dim kept for API compat but ignored (z0 is always [K,3])
        z0_dim: int = 3,
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
        self.n_nodes  = K
        self.n_frames = n_frames
        self.dt       = dt
        self.z0_dim   = 3   # always 3 (velocity vector)

        # node-node kNN graph
        from .graph import knn_graph
        self.register_buffer("node_edge", knn_graph(node_pos, k=min(k_node, K - 1)))
        self.register_buffer("arap_edge", knn_graph(node_pos, k=min(arap_k, K - 1)))

        # Gaussian-to-node RBF binding (fixed)
        kG = min(k_gauss, K)
        d = torch.cdist(canonical_xyz, node_pos)
        dist, gn_idx = d.topk(kG, largest=False, dim=-1)
        scale = dist[:, 0:1].clamp(min=1e-6)
        w = torch.softmax(-dist / scale, dim=-1)
        self.register_buffer("gauss_node_idx", gn_idx)   # [G, kG]
        self.register_buffer("gauss_node_w",   w)         # [G, kG]

        # ── GNN: canonical_pos → scene features h [K, H] ────────────────────
        # z0 is NOT fed to the GNN — it is the initial velocity condition applied
        # during integration. GNN encodes scene structure only.
        self.node_encoder = nn.Sequential(
            nn.Linear(3, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
        )
        self.gnn_layers = nn.ModuleList(
            [GNNLayer(hidden, hidden) for _ in range(n_gnn_layers)]
        )

        # ── Acceleration decoder: (h ∥ t_emb) → acc [K, 3] ─────────────────
        t_feat = 8
        self.accel_decoder = nn.Sequential(
            nn.Linear(hidden + t_feat * 2, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden // 2, 3),
        )
        # zero-init acceleration at start (no deformation before learning)
        nn.init.zeros_(self.accel_decoder[-1].weight)
        nn.init.zeros_(self.accel_decoder[-1].bias)

        freqs = 2.0 ** torch.arange(t_feat).float()
        self.register_buffer("t_freqs", freqs)

        # Constant operands that reduce the fused SC-GS LBS kernel to a pure
        # displacement blend: out[g] = sum_k w[g,k] * node_disp[idx[g,k]].
        # The kernel keeps no [G,kG,3] intermediate (~89MB/frame at G=1.85M);
        # its backward is the weighted scatter-add we need. Non-persistent so
        # they stay out of the checkpoint.
        self._use_lbs_cuda = _LBS_CUDA
        if _LBS_CUDA:
            self.register_buffer("_lbs_x0",    torch.zeros(G, 3), persistent=False)
            self.register_buffer("_lbs_arest", torch.zeros(K, 3), persistent=False)
            self.register_buffer("_lbs_eye",
                                 torch.eye(3).expand(K, 3, 3).contiguous(),
                                 persistent=False)

    # ── helpers ──────────────────────────────────────────────────────────────
    def _t_emb_batch(self, t_vals: torch.Tensor) -> torch.Tensor:
        """t_vals [T] (normalised 0-1) → [T, 2*t_feat]."""
        ang = t_vals.unsqueeze(1) * self.t_freqs.unsqueeze(0) * math.pi   # [T, t_feat]
        return torch.cat([ang.sin(), ang.cos()], dim=1)                     # [T, 2*t_feat]

    def _lbs(self, node_disp: torch.Tensor) -> torch.Tensor:
        """node_disp [K,3] or [T,K,3] → Gaussian disp [G,3] or [T,G,3]."""
        if node_disp.dim() == 2:
            nd_nb = node_disp[self.gauss_node_idx]              # [G,kG,3]
            return (self.gauss_node_w.unsqueeze(-1) * nd_nb).sum(1)   # [G,3]
        else:
            nd_nb = node_disp[:, self.gauss_node_idx, :]        # [T,G,kG,3]
            return (self.gauss_node_w.unsqueeze(-1) * nd_nb).sum(2)   # [T,G,3]

    # ── scene encoding (call once per step) ─────────────────────────────────
    def encode_scene(self) -> torch.Tensor:
        """Encode canonical node positions → h [K, H]. t-independent, z0-independent."""
        h = self.node_encoder(self.node_pos)
        for layer in self.gnn_layers:
            h = layer(h, self.node_edge)
        return h

    def lbs_frame(self, node_disp: torch.Tensor) -> torch.Tensor:
        """node_disp [K,3] → Gaussian disp [G,3]. Per-frame LBS.

        Use this instead of rollout() on large scenes: the batched form
        materialises [T,G,kG,3], which is ~2GB at G=1.85M/T=25.
        """
        if self._use_lbs_cuda and node_disp.is_cuda:
            return _lbs_blend_cuda(self._lbs_x0, self.gauss_node_w,
                                   self.gauss_node_idx, self._lbs_arest,
                                   node_disp, self._lbs_eye)
        nd_nb = node_disp[self.gauss_node_idx]                  # [G,kG,3]
        return (self.gauss_node_w.unsqueeze(-1) * nd_nb).sum(1)  # [G,3]

    # ── physics rollout (vectorised) ─────────────────────────────────────────
    def rollout_nodes(self, h: torch.Tensor, z0: torch.Tensor) -> torch.Tensor:
        """Node-space rollout → [T-1, K, 3]. No LBS, so memory stays O(T*K).

        Integration (symplectic Euler, step size dt):
            vel[t]  = z0 + dt * cumsum(acc)[t-1]
            disp[t] = dt * cumsum(vel)[t-1]
        """
        T   = self.n_frames
        K   = self.n_nodes
        dev = self.node_pos.device
        dt  = self.dt

        t_idx   = torch.arange(1, T, device=dev, dtype=torch.float32)
        t_norms = t_idx / max(T - 1, 1)
        t_emb   = self._t_emb_batch(t_norms)                      # [T-1, 2*t_feat]

        h_exp   = h.unsqueeze(0).expand(T - 1, -1, -1)            # [T-1, K, H]
        t_exp   = t_emb.unsqueeze(1).expand(-1, K, -1).to(h.dtype)
        acc     = self.accel_decoder(torch.cat([h_exp, t_exp], dim=-1))  # [T-1,K,3]

        vel = z0.unsqueeze(0) + dt * acc.cumsum(0)                # [T-1, K, 3]
        return dt * vel.cumsum(0)                                 # [T-1, K, 3]

    def rollout(self, h: torch.Tensor, z0: torch.Tensor) -> torch.Tensor:
        """
        h   : [K, H]   scene features from encode_scene()
        z0  : [K, 3]   initial velocity per node

        Returns [T-1, G, 3] Gaussian displacements for t = 1 .. T-1.
        (t=0 is canonical → zero displacement, not included)

        Integration (symplectic Euler, step size dt):
            vel[t] = z0 + dt * cumsum(acc)[t-1]
            disp[t] = dt * cumsum(vel)[t-1]       (positions relative to canonical)
        """
        T   = self.n_frames
        K   = self.n_nodes
        dev = self.node_pos.device
        dt  = self.dt

        # acceleration for t = 1 .. T-1
        t_idx   = torch.arange(1, T, device=dev, dtype=torch.float32)
        t_norms = t_idx / max(T - 1, 1)                              # [T-1]
        t_emb   = self._t_emb_batch(t_norms)                         # [T-1, 2*t_feat]

        h_exp   = h.unsqueeze(0).expand(T - 1, -1, -1)               # [T-1, K, H]
        t_exp   = t_emb.unsqueeze(1).expand(-1, K, -1).to(h.dtype)   # [T-1, K, 2*t_feat]
        acc     = self.accel_decoder(torch.cat([h_exp, t_exp], dim=-1))  # [T-1, K, 3]

        # velocity: vel[t] = z0 + dt * cumsum(acc)[t-1]
        acc_cs  = acc.cumsum(0)                                        # [T-1, K, 3]
        vel     = z0.unsqueeze(0) + dt * acc_cs                        # [T-1, K, 3]

        # displacement: disp[t] = dt * cumsum(vel)[t-1]
        node_disp = dt * vel.cumsum(0)                                 # [T-1, K, 3]

        return self._lbs(node_disp)                                    # [T-1, G, 3]

    # ── convenience: single-frame displacement ───────────────────────────────
    def forward(self, h: torch.Tensor, z0: torch.Tensor, t: int) -> torch.Tensor:
        """Returns [G, 3] displacement at frame t (0-indexed). t=0 → zeros."""
        if t <= 0:
            return torch.zeros(self.canonical_xyz.shape[0], 3, device=self.node_pos.device)
        disps = self.rollout(h, z0)   # [T-1, G, 3]
        return disps[min(t - 1, disps.shape[0] - 1)]

    # ── ARAP regularisation ──────────────────────────────────────────────────
    def arap_loss(self, h: torch.Tensor, z0: torch.Tensor, t: int) -> torch.Tensor:
        """ARAP rigidity loss on node positions at frame t."""
        if t <= 0:
            return torch.tensor(0.0, device=self.node_pos.device)
        T   = self.n_nodes
        dev = self.node_pos.device
        t_norm = t / max(self.n_frames - 1, 1)
        t_emb  = self._t_emb_batch(
            torch.tensor([t_norm], device=dev))           # [1, 2*t_feat]
        t_exp  = t_emb.expand(self.n_nodes, -1).to(h.dtype)
        acc_t  = self.accel_decoder(torch.cat([h, t_exp], dim=-1))    # [K,3] — acc at t
        # approximate: constant acc_t throughout, vel integrates linearly
        dt = self.dt
        node_disp = z0 * (dt * t) + acc_t * (dt * dt * t * (t + 1) / 2)
        node_now  = self.node_pos + node_disp
        src, dst  = self.arap_edge
        d_rest = (self.node_pos[src] - self.node_pos[dst]).norm(dim=-1)
        d_now  = (node_now[src]      - node_now[dst]     ).norm(dim=-1)
        return ((d_now - d_rest) ** 2).mean()

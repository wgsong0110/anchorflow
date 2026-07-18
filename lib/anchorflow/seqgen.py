"""Non-autoregressive sequence generation model for anchor node trajectories.

Given initial velocities for K (possibly sparse) conditioning nodes, generates
the full trajectory [T, N, 3] for all N anchor nodes in a single forward pass.

Architecture — Factored Spatial-Temporal Transformer:

    cond_vel [K, 3]  +  cond_node_ids [K]
        │
        ▼
    Conditioning tokens  [K, d]  =  MLP(node_embed[ids] ⊕ vel)
        │
        ▼  (cross-attention in each layer)
    Trajectory queries  [T, N, d]  =  time_embed[t] + node_embed[n]
        │
        ├─ Spatial self-attention:  queries[t, :, :] over N  (once per layer)
        ├─ Cross-attention:         queries attend to conditioning tokens
        └─ Temporal self-attention: queries[:, n, :] over T  (once per layer)
        │
        ▼
    Output MLP  →  Δpos [T, N, 3]
        │
        ▼
    traj = canon_pos[None] + Δpos       (absolute positions)

Training signal: MDS loss (SVDGuidance.mds_loss) backpropagates through
differentiable 3DGS rendering → LBS warp → traj → SeqGen weights.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── small helpers ─────────────────────────────────────────────────────────────

def mlp(dims: list[int], act=nn.GELU) -> nn.Sequential:
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


def sinusoidal_embed(positions: torch.Tensor, d: int) -> torch.Tensor:
    """positions: [*] float. Returns [*, d] sinusoidal positional encoding."""
    device = positions.device
    half = d // 2
    freq = torch.exp(
        torch.arange(half, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / (half - 1))
    )
    x = positions.float().unsqueeze(-1) * freq  # [*, half]
    return torch.cat([x.sin(), x.cos()], dim=-1)  # [*, d]


# ── per-layer building blocks ─────────────────────────────────────────────────

class PreNorm(nn.Module):
    def __init__(self, d: int, fn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.fn = fn

    def forward(self, x: torch.Tensor, **kw) -> torch.Tensor:
        return self.fn(self.norm(x), **kw)


class SelfAttn(nn.Module):
    def __init__(self, d: int, heads: int = 8):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.attn(x, x, x)
        return out


class CrossAttn(nn.Module):
    def __init__(self, d: int, heads: int = 8):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        out, _ = self.attn(q, kv, kv)
        return out


class FFN(nn.Module):
    def __init__(self, d: int, expand: int = 4):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, d * expand), nn.GELU(),
                                 nn.Linear(d * expand, d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SeqGenLayer(nn.Module):
    """One Spatial → Cross → Temporal → FFN layer."""

    def __init__(self, d: int, heads: int = 8):
        super().__init__()
        self.sp_attn = PreNorm(d, SelfAttn(d, heads))   # spatial (over N)
        self.cr_attn = PreNorm(d, CrossAttn(d, heads))  # cross (cond tokens)
        self.tm_attn = PreNorm(d, SelfAttn(d, heads))   # temporal (over T)
        self.ffn = PreNorm(d, FFN(d))

    def forward(self, q: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """q: [T, N, d]; cond: [K, d]. Returns [T, N, d]."""
        T, N, d = q.shape
        K = cond.shape[0]

        # Spatial: for each timestep, N nodes attend to each other.
        # MultiheadAttention(batch_first=True) sees [T, N, d]: T=batch, N=seq.
        q = q + self.sp_attn.fn(self.sp_attn.norm(q))    # [T, N, d]

        # Cross: all (t, n) queries attend to K conditioning tokens
        q_flat = q.reshape(T * N, d)
        q_flat = q_flat + self.cr_attn.fn(
            self.cr_attn.norm(q_flat).unsqueeze(0),      # [1, T*N, d]
            cond.unsqueeze(0)                             # [1, K, d]
        ).squeeze(0)
        q = q_flat.reshape(T, N, d)

        # Temporal: for each node, T timesteps attend to each other
        q_tm = q.permute(1, 0, 2)                         # [N, T, d]
        q_tm = q_tm + self.tm_attn.fn(self.tm_attn.norm(q_tm))
        q = q_tm.permute(1, 0, 2)                         # [T, N, d]

        # FFN
        q = q + self.ffn.fn(self.ffn.norm(q))
        return q


# ── main model ────────────────────────────────────────────────────────────────

class SeqGen(nn.Module):
    """Non-autoregressive anchor trajectory generator.

    Args:
        canon_pos   [N, 3]  Canonical anchor positions (fixed, used for node
                            position embedding and as residual base).
        n_frames    T       Number of output time steps.
        d_model             Transformer hidden dimension.
        n_layers            Number of SeqGenLayer blocks.
        n_heads             Attention heads.
        vel_in      3 or 6  Velocity input dim (3 = xyz vel, 6 = xyz + xyz pos).
    """

    def __init__(
        self,
        canon_pos: torch.Tensor,
        n_frames: int = 25,
        d_model: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        vel_in: int = 3,
    ):
        super().__init__()
        N = canon_pos.shape[0]
        self.N = N
        self.T = n_frames
        self.d = d_model

        self.register_buffer("canon_pos", canon_pos.float())

        # Node positional embedding: 3D position → d_model via MLP
        self.node_pos_embed = mlp([3, d_model, d_model])

        # Learnable time embeddings
        self.time_embed = nn.Embedding(n_frames, d_model)

        # Conditioning encoder: (node_pos_embed + velocity) → d_model token
        self.cond_encoder = mlp([d_model + vel_in, d_model, d_model])

        # Transformer layers
        self.layers = nn.ModuleList(
            [SeqGenLayer(d_model, n_heads) for _ in range(n_layers)]
        )

        # Output head: d_model → Δpos (3)
        self.out_head = mlp([d_model, d_model // 2, 3])
        # Zero-init output so the model starts near canonical (zero displacement)
        last = [m for m in self.out_head.modules() if isinstance(m, nn.Linear)][-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    # ── helpers ──────────────────────────────────────────────────────────────

    def _node_embeds(self) -> torch.Tensor:
        """[N, d] node embeddings from canonical positions."""
        return self.node_pos_embed(self.canon_pos)          # [N, d]

    def _time_embeds(self) -> torch.Tensor:
        """[T, d] time embeddings."""
        t_ids = torch.arange(self.T, device=self.canon_pos.device)
        return self.time_embed(t_ids)                       # [T, d]

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(
        self,
        cond_ids: torch.Tensor,   # [K] long — indices of conditioning nodes
        cond_vel: torch.Tensor,   # [K, 3] — initial velocities
    ) -> torch.Tensor:
        """Returns trajectory [T, N, 3] (absolute positions)."""
        node_emb = self._node_embeds()          # [N, d]
        time_emb = self._time_embeds()          # [T, d]

        # ── conditioning tokens ───────────────────────────────────────────
        cond_node_emb = node_emb[cond_ids]                  # [K, d]
        cond_feat = torch.cat([cond_node_emb, cond_vel], dim=-1)  # [K, d+3]
        cond_tokens = self.cond_encoder(cond_feat)           # [K, d]

        # ── trajectory queries ────────────────────────────────────────────
        # q[t, n] = time_embed[t] + node_embed[n]
        q = (time_emb[:, None, :] + node_emb[None, :, :])  # [T, N, d]

        # ── transformer ───────────────────────────────────────────────────
        for layer in self.layers:
            q = layer(q, cond_tokens)                       # [T, N, d]

        # ── output ────────────────────────────────────────────────────────
        delta = self.out_head(q)                            # [T, N, 3]
        traj = self.canon_pos[None] + delta                 # [T, N, 3]
        return traj

    def rollout(
        self,
        cond_ids: torch.Tensor,
        cond_vel: torch.Tensor,
        n_frames: int | None = None,
    ) -> torch.Tensor:
        """Alias for forward(); n_frames param for interface parity with GNS."""
        return self.forward(cond_ids, cond_vel)

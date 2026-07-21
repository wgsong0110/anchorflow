"""GNN physics simulator: GNN Encoder → GRU SSM → GNN Decoder.

Architecture per step t:
  Encoder : [x,v] → state_mlp → edge_mlp → scatter_add → enc_mlp → node_enc
            mean(node_enc) → pool_mlp → z^t   (global latent)
  SSM     : GRUCell(z^t, h^{t-1}) → h^t       (temporal context)
  Decoder : dec_mlp([node_enc, h^t broadcast]) → tanh * max_accel → a_gnn
  Physics : a = a_gnn + gravity + impulse(t=0, random subset)
                       − k_restore*(x − canonical)
            v = v*(1−damp) + dt*a;   x = x + dt*v

GRU is used as the SSM: same gating mechanism as Mamba but implemented in
PyTorch core (no CUDA extension needed). The key architectural insight is the
GNN encoder → global latent → temporal SSM → GNN decoder pipeline.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _mlp(in_d: int, hid: int, out_d: int, layers: int = 3,
         bias_last: bool = True) -> nn.Sequential:
    seq = [nn.Linear(in_d, hid), nn.LayerNorm(hid), nn.SiLU()]
    for _ in range(layers - 2):
        seq += [nn.Linear(hid, hid), nn.LayerNorm(hid), nn.SiLU()]
    seq.append(nn.Linear(hid, out_d, bias=bias_last))
    return nn.Sequential(*seq)


class GNNSim(nn.Module):
    def __init__(
        self,
        canonical: torch.Tensor,       # [M, 3]
        anchor_colors: torch.Tensor,   # [M, 3]
        edge_index: torch.Tensor,      # [2, E]
        rest_len: torch.Tensor,        # [E]
        T: int = 25,
        dt: float = 0.1,
        hidden_dim: int = 256,
        node_dim: int = 32,
        latent_dim: int = 256,
        gravity: float = 5.0,
        gravity_axis: int = 2,
        damping: float = 0.1,
        k_restore: float = 2.0,
        max_accel: float = 10.0,
        impulse_frac: float = 0.5,
        **kwargs,
    ):
        super().__init__()
        M = canonical.shape[0]
        self.M            = M
        self.T            = T
        self.dt           = dt
        self.gravity      = gravity
        self.gravity_axis = gravity_axis
        self.hidden_dim   = hidden_dim
        self.latent_dim   = latent_dim
        self.damping      = damping
        self.k_restore    = k_restore
        self.max_accel    = max_accel
        self.impulse_frac = impulse_frac

        self.register_buffer("canonical",     canonical.clone().float())
        self.register_buffer("anchor_colors", anchor_colors.float())
        self.register_buffer("edge_index",    edge_index)
        self.register_buffer("rest_len",      rest_len.float())

        self.node_emb = nn.Embedding(M, node_dim)
        STATIC = node_dim

        # ── Encoder ──────────────────────────────────────────────────────── #
        self.state_mlp = _mlp(6, hidden_dim, hidden_dim, bias_last=False)

        EDGE_IN = hidden_dim * 2 + STATIC * 2   # state_src + state_dst + static_src + static_dst
        self.edge_mlp = _mlp(EDGE_IN, hidden_dim, hidden_dim, bias_last=False)

        self.enc_mlp  = _mlp(hidden_dim * 2, hidden_dim, hidden_dim, bias_last=False)
        self.pool_mlp = _mlp(hidden_dim, hidden_dim, latent_dim, bias_last=False)

        # ── SSM (GRU) ────────────────────────────────────────────────────── #
        self.ssm = nn.GRUCell(latent_dim, latent_dim)

        # ── Decoder ──────────────────────────────────────────────────────── #
        self.dec_mlp = _mlp(hidden_dim + latent_dim, hidden_dim, 3, bias_last=False)

    @property
    def _static(self) -> torch.Tensor:
        idx = torch.arange(self.M, device=self.canonical.device)
        return self.node_emb(idx)                                # [M, node_dim]

    def _encode(self, x: torch.Tensor, v: torch.Tensor,
                static: torch.Tensor,
                src: torch.Tensor, dst: torch.Tensor):
        """Returns (node_enc [M, H], z [1, L])."""
        state   = self.state_mlp(torch.cat([x, v], dim=-1))        # [M, H]
        feat = torch.cat([
            state[src], state[dst],
            static[src], static[dst],
        ], dim=-1)
        msg = self.edge_mlp(feat)                                # [E, H]
        agg = torch.zeros(self.M, self.hidden_dim, device=x.device)
        agg.scatter_add_(0, dst[:, None].expand_as(msg), msg)
        deg = torch.zeros(self.M, device=x.device).scatter_add_(
            0, dst, torch.ones(dst.shape[0], device=x.device))
        agg = agg / deg.unsqueeze(1).clamp(min=1)                # mean aggregation
        node_enc = self.enc_mlp(torch.cat([agg, state], dim=-1))  # [M, H]
        z        = self.pool_mlp(node_enc.mean(dim=0, keepdim=True))  # [1, L]
        return node_enc, z

    def _decode(self, node_enc: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Returns a_gnn [M, 3]. h: [1, L]."""
        h_broad = h.expand(self.M, -1)                          # [M, L]
        return torch.tanh(
            self.dec_mlp(torch.cat([node_enc, h_broad], dim=-1))
        ) * self.max_accel

    def forward(self, f_ext: torch.Tensor, grad_steps: int = 5) -> torch.Tensor:
        """
        f_ext [M, 3]: per-node impulse at t=0 (zero for unselected nodes).
        grad_steps: ignored (full BPTT; kept for API compat).
        Returns traj [T, M, 3].
        """
        M   = self.M
        x   = self.canonical.clone()
        v   = torch.zeros_like(x)

        g_vec = torch.zeros(3, device=f_ext.device, dtype=f_ext.dtype)
        g_vec[self.gravity_axis] = -self.gravity

        static   = self._static
        src, dst = self.edge_index
        h        = torch.zeros(1, self.latent_dim, device=x.device, dtype=x.dtype)

        traj = [x.detach()]

        for t in range(self.T - 1):
            node_enc, z = self._encode(x, v, static, src, dst)
            h = self.ssm(z, h)                                   # GRU: [1, L]
            a_gnn = self._decode(node_enc, h)

            a = a_gnn + g_vec.unsqueeze(0)
            if t == 0:
                a = a + f_ext

            v = v * (1.0 - self.damping) + self.dt * a
            x = x + self.dt * v
            traj.append(x)

        return torch.stack(traj, dim=0)                          # [T, M, 3]

    @torch.no_grad()
    def forward_debug(self, f_ext: torch.Tensor):
        """f_ext [M, 3]. Returns (traj [T,M,3], accels [T,M,3])."""
        M   = self.M
        x   = self.canonical.clone()
        v   = torch.zeros_like(x)

        g_vec = torch.zeros(3, device=f_ext.device, dtype=f_ext.dtype)
        g_vec[self.gravity_axis] = -self.gravity

        static   = self._static
        src, dst = self.edge_index
        h        = torch.zeros(1, self.latent_dim, device=x.device, dtype=x.dtype)

        traj   = [x.clone()]
        accels = [torch.zeros(M, 3, device=x.device)]

        for t in range(self.T - 1):
            node_enc, z = self._encode(x, v, static, src, dst)
            h = self.ssm(z, h)
            a_gnn = self._decode(node_enc, h)
            accels.append(a_gnn.clone())                         # GNN pure output

            a = a_gnn + g_vec.unsqueeze(0)
            if t == 0:
                a = a + f_ext

            v = v * (1.0 - self.damping) + self.dt * a
            x = x + self.dt * v
            traj.append(x.clone())

        return torch.stack(traj, dim=0), torch.stack(accels, dim=0)

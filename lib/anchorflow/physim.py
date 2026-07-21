"""GNN physics simulator: GNN Encoder → per-node Selective SSM → GNN Decoder.

Architecture per step t:
  Encoder : PE(x,v) → state_mlp → edge_mlp([state,static]) → mean_pool
            → enc_mlp([agg, state]) → node_enc  [M, H]
  SSM     : SelectiveSSM(node_enc_i, h_i) → y_i, h_i_new   (per node)
  Decoder : dec_mlp(y) → tanh * max_accel → a_gnn  [M, 3]
  Physics : a = a_gnn + impulse(t=0)
            v = v*(1−damp) + dt*a;   x = x + dt*v

SelectiveSSM is a Mamba-style per-node SSM with CUDA kernel (lib/ssm).
Each node maintains an independent hidden state h_i [D, N], enabling
self-propelled dynamics where each part has its own internal state.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ssm import SelectiveSSM

PE_FREQS = 6  # Fourier PE frequencies; D → D*(1+2*PE_FREQS)


def _fourier_pe(x: torch.Tensor) -> torch.Tensor:
    """x [N, D] → [N, D*(1+2*PE_FREQS)]"""
    freqs = 2.0 ** torch.arange(PE_FREQS, device=x.device, dtype=x.dtype)
    args  = x.unsqueeze(-1) * freqs
    return torch.cat([x, args.sin().flatten(-2), args.cos().flatten(-2)], dim=-1)


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
        dt: float = 0.05,
        hidden_dim: int = 256,
        node_dim: int = 32,
        latent_dim: int = 256,
        d_state: int = 16,
        gravity: float = 0.0,
        gravity_axis: int = 2,
        damping: float = 0.2,
        k_restore: float = 0.0,
        max_accel: float = 5.0,
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
        self.damping      = damping
        self.max_accel    = max_accel
        self.impulse_frac = impulse_frac

        self.register_buffer("canonical",     canonical.clone().float())
        self.register_buffer("anchor_colors", anchor_colors.float())
        self.register_buffer("edge_index",    edge_index)
        self.register_buffer("rest_len",      rest_len.float())

        self.node_emb = nn.Embedding(M, node_dim)
        STATIC = node_dim

        PE_DIM = 3 * (1 + 2 * PE_FREQS)          # 3*(1+12) = 39

        # ── Encoder ──────────────────────────────────────────────────────── #
        self.state_mlp = _mlp(PE_DIM * 2, hidden_dim, hidden_dim, bias_last=False)

        EDGE_IN = hidden_dim * 2 + STATIC * 2
        self.edge_mlp = _mlp(EDGE_IN, hidden_dim, hidden_dim, bias_last=False)

        self.enc_mlp  = _mlp(hidden_dim * 2, hidden_dim, hidden_dim, bias_last=False)

        # ── Per-node Selective SSM ────────────────────────────────────────── #
        self.ssm = SelectiveSSM(d_model=hidden_dim, d_state=d_state)

        # ── Decoder ──────────────────────────────────────────────────────── #
        self.dec_mlp = _mlp(hidden_dim, hidden_dim, 3, bias_last=False)

    @property
    def _static(self) -> torch.Tensor:
        idx = torch.arange(self.M, device=self.canonical.device)
        return self.node_emb(idx)

    def _encode(self, x, v, static, src, dst):
        """Returns node_enc [M, H]."""
        state    = self.state_mlp(
            torch.cat([_fourier_pe(x), _fourier_pe(v)], dim=-1))   # [M, H]
        feat     = torch.cat([state[src], state[dst],
                               static[src], static[dst]], dim=-1)
        msg      = self.edge_mlp(feat)                               # [E, H]
        agg      = torch.zeros(self.M, self.hidden_dim, device=x.device)
        agg.scatter_add_(0, dst[:, None].expand_as(msg), msg)
        deg      = torch.zeros(self.M, device=x.device).scatter_add_(
            0, dst, torch.ones(dst.shape[0], device=x.device))
        agg      = agg / deg.unsqueeze(1).clamp(min=1)
        node_enc = self.enc_mlp(torch.cat([agg, state], dim=-1))
        return node_enc

    def forward(self, f_ext: torch.Tensor, grad_steps: int = 5) -> torch.Tensor:
        """
        f_ext [M, 3]: per-node impulse at t=0.
        Returns traj [T, M, 3].
        """
        x    = self.canonical.clone()
        v    = torch.zeros_like(x)

        g_vec = torch.zeros(3, device=f_ext.device, dtype=f_ext.dtype)
        g_vec[self.gravity_axis] = -self.gravity

        static   = self._static
        src, dst = self.edge_index
        h        = self.ssm.init_state(self.M, x.device, x.dtype)  # [M, D, N]

        traj = [x.detach()]

        for t in range(self.T - 1):
            node_enc       = self._encode(x, v, static, src, dst)
            y, h           = self.ssm(node_enc, h)                  # per-node SSM
            a_gnn          = torch.tanh(self.dec_mlp(y)) * self.max_accel
            a_gnn          = a_gnn - a_gnn.mean(dim=0, keepdim=True)

            a = a_gnn + g_vec.unsqueeze(0)
            if t == 0:
                a = a + f_ext

            v = v * (1.0 - self.damping) + self.dt * a
            x = x + self.dt * v
            traj.append(x)

        return torch.stack(traj, dim=0)

    @torch.no_grad()
    def forward_debug(self, f_ext: torch.Tensor):
        """Returns (traj [T,M,3], accels [T,M,3])."""
        x    = self.canonical.clone()
        v    = torch.zeros_like(x)

        g_vec = torch.zeros(3, device=f_ext.device, dtype=f_ext.dtype)
        g_vec[self.gravity_axis] = -self.gravity

        static   = self._static
        src, dst = self.edge_index
        h        = self.ssm.init_state(self.M, x.device, x.dtype)

        traj   = [x.clone()]
        accels = [torch.zeros(self.M, 3, device=x.device)]

        for t in range(self.T - 1):
            node_enc = self._encode(x, v, static, src, dst)
            y, h     = self.ssm(node_enc, h)
            a_gnn    = torch.tanh(self.dec_mlp(y)) * self.max_accel
            a_gnn    = a_gnn - a_gnn.mean(dim=0, keepdim=True)
            accels.append(a_gnn.clone())

            a = a_gnn + g_vec.unsqueeze(0)
            if t == 0:
                a = a + f_ext

            v = v * (1.0 - self.damping) + self.dt * a
            x = x + self.dt * v
            traj.append(x.clone())

        return torch.stack(traj, dim=0), torch.stack(accels, dim=0)

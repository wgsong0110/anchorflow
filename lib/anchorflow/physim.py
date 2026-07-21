"""GNN-based differentiable physics simulator (GNNSim).

Each anchor has:
  - static features  : canonical position x0, SH0 colour, per-anchor learnable embedding
  - dynamic state    : current position x, velocity v → embedded by state_mlp

Per step:
  1. state_mlp([x, v]) → state embedding [hidden]
  2. edge_mlp([state_i, state_j, static_i, static_j, rest_len]) → message [hidden]
  3. scatter_add → per-node aggregation
  4. node_mlp([agg, state, f_node]) → acceleration [3]
  5. Euler: v += dt*a,  x += dt*v

Graph: single spatial KNN (k_nn), no object-level distinction.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _mlp(in_d: int, hid: int, out_d: int, layers: int = 3) -> nn.Sequential:
    seq = [nn.Linear(in_d, hid), nn.SiLU()]
    for _ in range(layers - 2):
        seq += [nn.Linear(hid, hid), nn.SiLU()]
    seq.append(nn.Linear(hid, out_d))
    return nn.Sequential(*seq)


class GNNSim(nn.Module):
    def __init__(
        self,
        canonical: torch.Tensor,       # [M, 3]  anchor rest positions
        anchor_colors: torch.Tensor,   # [M, 3]  SH0 albedo (0-1)
        edge_index: torch.Tensor,      # [2, E]  KNN graph (both directions)
        rest_len: torch.Tensor,        # [E]     canonical edge lengths
        T: int = 25,
        dt: float = 0.1,
        hidden_dim: int = 256,
        node_dim: int = 32,            # per-anchor learnable embedding size
        gravity: float = 5.0,
        gravity_axis: int = 2,
        damping: float = 0.1,          # velocity damping per step: v *= (1 - damping)
        k_restore: float = 2.0,        # spring constant pulling back to canonical
        max_accel: float = 10.0,       # tanh soft-clamp on GNN acceleration output
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
        self.k_restore    = k_restore
        self.max_accel    = max_accel

        self.register_buffer("canonical",     canonical.clone().float())
        self.register_buffer("anchor_colors", anchor_colors.float())
        self.register_buffer("edge_index",    edge_index)
        self.register_buffer("rest_len",      rest_len.float())

        # per-anchor learnable embedding → encodes material/stiffness/mass
        self.node_emb = nn.Embedding(M, node_dim)

        STATIC   = node_dim                   # per-anchor learnable emb only

        # [x, v] (6) → hidden state embedding
        self.state_mlp = _mlp(6, hidden_dim, hidden_dim)

        # [state_i, state_j, rel_disp(3), dist(1), stretch(1), static_i, static_j, rest_len]
        EDGE_IN = hidden_dim * 2 + 5 + STATIC * 2 + 1
        self.edge_mlp = _mlp(EDGE_IN, hidden_dim, hidden_dim)

        # [agg, state] → internal acceleration [3]  (external forces added outside)
        NODE_IN = hidden_dim + hidden_dim
        self.node_mlp = _mlp(NODE_IN, hidden_dim, 3)

    @property
    def _static(self) -> torch.Tensor:
        idx = torch.arange(self.M, device=self.canonical.device)
        return self.node_emb(idx)                          # [M, node_dim]

    def forward(self, f_ext: torch.Tensor, grad_steps: int = 5) -> torch.Tensor:
        """
        f_ext      [3] : impulse at t=0 (wind gust direction × magnitude).
        grad_steps     : chunked BPTT — detach x,v every grad_steps steps.

        Returns trajectory [T, M, 3];  traj[0] == canonical (no grad).
        """
        M   = self.M
        x   = self.canonical.clone()
        v   = torch.zeros_like(x)

        g_vec = torch.zeros(3, device=f_ext.device, dtype=f_ext.dtype)
        g_vec[self.gravity_axis] = -self.gravity

        static = self._static                              # [M, 22]  fixed per forward
        src, dst = self.edge_index

        traj = [x.detach()]

        for t in range(self.T - 1):
            if t > 0 and t % grad_steps == 0:
                x = x.detach()
                v = v.detach()

            # ── state embedding ───────────────────────────────────────────── #
            state = self.state_mlp(torch.cat([x, v], dim=-1))  # [M, hidden]

            # ── edge messages ─────────────────────────────────────────────── #
            rel     = x[src] - x[dst]                            # [E, 3]
            dist    = rel.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # [E, 1]
            stretch = dist - self.rest_len[:, None]              # [E, 1]
            feat = torch.cat([
                state[src], state[dst],
                rel, dist, stretch,
                static[src], static[dst],
                self.rest_len[:, None],
            ], dim=-1)                                     # [E, EDGE_IN]
            msg  = self.edge_mlp(feat)                    # [E, hidden]
            agg  = torch.zeros(M, self.hidden_dim, device=x.device)
            agg.scatter_add_(0, dst[:, None].expand_as(msg), msg)

            # ── acceleration → Euler integration ─────────────────────────── #
            a = torch.tanh(self.node_mlp(torch.cat([agg, state], dim=-1))) * self.max_accel
            # external forces added outside GNN: gravity + impulse at t=0
            f_ext_t = g_vec.unsqueeze(0).expand(M, -1)
            if t == 0:
                f_ext_t = f_ext_t + f_ext.unsqueeze(0)
            a = a + f_ext_t
            a = a - self.k_restore * (x - self.canonical)               # restoring force
            v = v * (1.0 - self.damping) + self.dt * a                  # damped integration
            x = x + self.dt * v
            traj.append(x)

        return torch.stack(traj, dim=0)                   # [T, M, 3]

    @torch.no_grad()
    def forward_debug(self, f_ext: torch.Tensor):
        """Same as forward but also returns per-frame accelerations.

        Returns:
            traj   [T, M, 3]
            accels [T, M, 3]  — accel at frame 0 is zeros (canonical, no step)
        """
        M   = self.M
        x   = self.canonical.clone()
        v   = torch.zeros_like(x)

        g_vec = torch.zeros(3, device=f_ext.device, dtype=f_ext.dtype)
        g_vec[self.gravity_axis] = -self.gravity

        static = self._static
        src, dst = self.edge_index

        traj   = [x.clone()]
        accels = [torch.zeros(M, 3, device=f_ext.device)]  # frame 0 placeholder

        for t in range(self.T - 1):
            state = self.state_mlp(torch.cat([x, v], dim=-1))

            rel     = x[src] - x[dst]
            dist    = rel.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            stretch = dist - self.rest_len[:, None]
            feat = torch.cat([
                state[src], state[dst],
                rel, dist, stretch,
                static[src], static[dst],
                self.rest_len[:, None],
            ], dim=-1)
            msg = self.edge_mlp(feat)
            agg = torch.zeros(M, self.hidden_dim, device=x.device)
            agg.scatter_add_(0, dst[:, None].expand_as(msg), msg)

            a = self.node_mlp(torch.cat([agg, state], dim=-1))
            accels.append(a.clone())           # GNN pure output (before ext force + restoring)
            f_ext_t = g_vec.unsqueeze(0).expand(M, -1)
            if t == 0:
                f_ext_t = f_ext_t + f_ext.unsqueeze(0)
            a = a + f_ext_t
            a = a - self.k_restore * (x - self.canonical)

            v = v * (1.0 - self.damping) + self.dt * a
            x = x + self.dt * v
            traj.append(x.clone())

        return torch.stack(traj, dim=0), torch.stack(accels, dim=0)  # [T,M,3] each

"""GNN-based differentiable physics simulator (GNNSim).

Replaces the fixed Hookean SpringSim with learned message passing:

  - Intra-object edges : local spring/deformation dynamics within each object
  - Inter-object edges : boundary forces between objects (pot ↔ trunk ↔ leaf)
  - External force     : impulse at t=0 only (wind gust)
  - Gravity            : constant downward body force every step

At each time step the GNN computes a per-anchor acceleration, then Euler
integration updates velocity and position.  The graph topology is built once
from canonical positions + object IDs and reused across all T steps.

Learnable parameters
    obj_emb          [n_obj, 16]  object-type embedding
    intra_edge_mlp   MLP for intra-object message computation
    inter_edge_mlp   MLP for inter-object message computation
    node_mlp         MLP mapping aggregated messages → acceleration [3]
"""
from __future__ import annotations

import torch
import torch.nn as nn


# ── helpers ────────────────────────────────────────────────────────────────── #

def _mlp(in_d: int, hid: int, out_d: int, layers: int = 3) -> nn.Sequential:
    seq = [nn.Linear(in_d, hid), nn.SiLU()]
    for _ in range(layers - 2):
        seq += [nn.Linear(hid, hid), nn.SiLU()]
    seq.append(nn.Linear(hid, out_d))
    return nn.Sequential(*seq)


# ── GNNSim ─────────────────────────────────────────────────────────────────── #

class GNNSim(nn.Module):
    """GNN physics simulator with intra/inter-object hierarchical message passing."""

    def __init__(
        self,
        canonical: torch.Tensor,        # [M, 3]  anchor rest positions
        anchor_obj: torch.Tensor,        # [M]     long, 0-indexed object ID
        anchor_colors: torch.Tensor,     # [M, 3]  SH0 albedo (0-1)
        intra_edge_index: torch.Tensor,  # [2, E_intra]
        intra_rest: torch.Tensor,        # [E_intra]  canonical distances
        inter_edge_index: torch.Tensor,  # [2, E_inter]
        inter_rest: torch.Tensor,        # [E_inter]
        T: int = 14,
        dt: float = 0.04,
        hidden_dim: int = 128,
        gravity: float = 2.0,
        gravity_axis: int = 2,           # axis index that points 'up' (2 = Z)
    ):
        super().__init__()
        self.T   = T
        self.dt  = dt
        self.gravity      = gravity
        self.gravity_axis = gravity_axis
        self.hidden_dim   = hidden_dim

        n_obj = int(anchor_obj.max().item()) + 1
        self.n_obj = n_obj
        OBJ_DIM = 16

        self.register_buffer("canonical",         canonical.clone().float())
        self.register_buffer("anchor_obj",        anchor_obj)
        self.register_buffer("anchor_colors",     anchor_colors.float())
        self.register_buffer("intra_edge_index",  intra_edge_index)
        self.register_buffer("intra_rest",        intra_rest.float())
        self.register_buffer("inter_edge_index",  inter_edge_index)
        self.register_buffer("inter_rest",        inter_rest.float())

        # Object-type embedding
        self.obj_emb = nn.Embedding(n_obj, OBJ_DIM)

        # ── Edge MLPs ──────────────────────────────────────────────────────── #
        # Intra edge input:
        #   xi[3], xj[3], vi[3], vj[3],        — dynamic state
        #   disp_i[3], disp_j[3],               — displacement from canonical
        #   rest_len[1]                          — rest length
        #   = 19
        INTRA_IN = 3 + 3 + 3 + 3 + 3 + 3 + 1
        self.intra_edge_mlp = _mlp(INTRA_IN, hidden_dim, hidden_dim)

        # Inter edge input: same + object embeddings of both endpoints
        #   = 19 + OBJ_DIM + OBJ_DIM
        INTER_IN = INTRA_IN + OBJ_DIM + OBJ_DIM
        self.inter_edge_mlp = _mlp(INTER_IN, hidden_dim, hidden_dim)

        # ── Node MLP ───────────────────────────────────────────────────────── #
        # Input:
        #   intra_agg[hidden],  inter_agg[hidden],  — aggregated messages
        #   f_node[3],                              — external (gravity + impulse)
        #   x0[3], color[3], obj_emb[OBJ_DIM],     — static node features
        #   v[3]                                    — current velocity
        NODE_IN = hidden_dim + hidden_dim + 3 + 3 + 3 + OBJ_DIM + 3
        self.node_mlp = _mlp(NODE_IN, hidden_dim, 3)

    # ── internal helpers ───────────────────────────────────────────────────── #

    @property
    def _static(self) -> torch.Tensor:
        """[M, 3+3+OBJ_DIM] canonical pos + colour + obj embedding (cached)."""
        emb = self.obj_emb(self.anchor_obj)                       # [M, 16]
        return torch.cat([self.canonical, self.anchor_colors, emb], dim=-1)

    def _intra_agg(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        src, dst = self.intra_edge_index
        feat = torch.cat([
            x[src], x[dst],
            v[src], v[dst],
            x[src] - self.canonical[src],
            x[dst] - self.canonical[dst],
            self.intra_rest[:, None],
        ], dim=-1)                                                 # [E, 19]
        msg  = self.intra_edge_mlp(feat)                          # [E, d]
        agg  = torch.zeros(x.shape[0], self.hidden_dim, device=x.device)
        agg.scatter_add_(0, dst[:, None].expand_as(msg), msg)
        return agg                                                 # [M, d]

    def _inter_agg(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        if self.inter_edge_index.shape[1] == 0:
            return torch.zeros(x.shape[0], self.hidden_dim, device=x.device)
        src, dst = self.inter_edge_index
        obj_src = self.obj_emb(self.anchor_obj[src])              # [E, 16]
        obj_dst = self.obj_emb(self.anchor_obj[dst])              # [E, 16]
        feat = torch.cat([
            x[src], x[dst],
            v[src], v[dst],
            x[src] - self.canonical[src],
            x[dst] - self.canonical[dst],
            self.inter_rest[:, None],
            obj_src, obj_dst,
        ], dim=-1)                                                 # [E, 19+32]
        msg  = self.inter_edge_mlp(feat)                          # [E, d]
        agg  = torch.zeros(x.shape[0], self.hidden_dim, device=x.device)
        agg.scatter_add_(0, dst[:, None].expand_as(msg), msg)
        return agg                                                 # [M, d]

    # ── forward ───────────────────────────────────────────────────────────── #

    def forward(self, f_ext: torch.Tensor) -> torch.Tensor:
        """
        f_ext [3] : impulse applied at t=0 (wind gust direction × magnitude).
        Returns trajectory [T, M, 3] where traj[0] == canonical.
        """
        M  = self.canonical.shape[0]
        x  = self.canonical.clone()
        v  = torch.zeros_like(x)

        # Gravity: constant downward acceleration
        g_vec = torch.zeros(3, device=f_ext.device, dtype=f_ext.dtype)
        g_vec[self.gravity_axis] = -self.gravity

        static = self._static                                      # [M, 22]
        traj   = [x]

        for t in range(self.T - 1):
            ia = self._intra_agg(x, v)                            # [M, d]
            ra = self._inter_agg(x, v)                            # [M, d]

            # External force: gravity always, impulse only at t=0
            f_node = g_vec.unsqueeze(0).expand(M, -1).clone()
            if t == 0:
                f_node = f_node + f_ext.unsqueeze(0)

            node_in = torch.cat([ia, ra, f_node, static, v], dim=-1)
            a = self.node_mlp(node_in)                            # [M, 3]

            v = v + self.dt * a
            x = x + self.dt * v
            traj.append(x)

        return torch.stack(traj, dim=0)                           # [T, M, 3]

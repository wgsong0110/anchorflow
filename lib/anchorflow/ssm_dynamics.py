"""GNN (spatial) ⊗ per-node SSM (temporal) anchor dynamics — v2.

Replaces the Markov GNS decoder. Per anchor node i:
    physical state  (p_i, v_i, a_i)  — integrated explicitly; position is never
                                       decoded from the hidden state
    SSM hidden       h_i ∈ R^d        — SEPARATE recurrent memory that produces
                                       the acceleration (gait phase / momentum /
                                       actuation rhythm beyond a 2-frame window)

Per rollout step (dt is a hyperparameter, matched to the source video / MoSca):
    m_i  = GNN spatial message passing over the anchor graph
    u_i  = encode([v_i, m_i, e_i, z_i])          obs + spatial ctx + identity + control
    h_i  = SSM(h_i, u_i, dt)                       diagonal, bounded -> stable long rollout
    a_i  = tanh(decode(h_i)) * accel_scale         acceleration ONLY
    p_i' = p_i + v_i * dt                           explicit Euler (prev pos + prev vel)
    v_i' = v_i + a_i * dt

    h_i^0 = encode([e_i, z_i, init_vel_i, init_pos_i])

Per-anchor inputs:
    e_i  intrinsic identity (learned, FIXED across ICs — "what this anchor is")
    z_i  actuation/control  (varied per-IC by MDS to generalise the simulator)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dynamics import mlp, InteractionNetwork
from . import graph as G


class DiagonalSSM(nn.Module):
    """Per-node diagonal state-space recurrence (S4D-style leaky integrator).

        h^t = decay ⊙ h^{t-1} + (1 - decay) ⊙ (W u^t)
        decay = exp(-dt · softplus(rate))        per-channel, learnable

    decay ∈ (0,1) so the recurrence is bounded -> stable over long rollouts (the
    reason we use an SSM: stable extrapolation past the training/diffusion window)."""

    def __init__(self, dim):
        super().__init__()
        self.log_rate = nn.Parameter(torch.zeros(dim))
        self.in_proj = nn.Linear(dim, dim)

    def step(self, h, u, dt):
        decay = torch.exp(-dt * F.softplus(self.log_rate))      # (dim,) in (0,1)
        return decay * h + (1 - decay) * self.in_proj(u)


class SSMDynamics(nn.Module):
    def __init__(self, hidden=128, mp_steps=6, ssm_dim=128, e_dim=8, z_dim=8,
                 edge_in=4, accel_scale=0.1):
        super().__init__()
        self.accel_scale = accel_scale
        self.node_enc = mlp([3 + e_dim + z_dim, hidden, hidden])     # [v, e, z]
        self.edge_enc = mlp([edge_in, hidden, hidden])
        self.processor = nn.ModuleList(
            InteractionNetwork(hidden) for _ in range(mp_steps))     # spatial
        self.to_ssm = mlp([hidden, ssm_dim])
        self.ssm = DiagonalSSM(ssm_dim)                             # temporal
        self.decoder = mlp([ssm_dim, hidden, 3], layernorm=False)   # h -> accel
        self.h0_enc = mlp([e_dim + z_dim + 3 + 3, ssm_dim])         # [e,z,ivel,ipos]

    def init_hidden(self, e, z, init_vel, init_pos):
        return self.h0_enc(torch.cat([e, z, init_vel, init_pos], dim=-1))

    def step(self, p, v, h, e, z, edge_index, dt):
        node = self.node_enc(torch.cat([v, e, z], dim=-1))
        edge = self.edge_enc(G.edge_features(p, edge_index))
        x = node
        for layer in self.processor:                                # GNN message passing
            x, edge = layer(x, edge_index, edge)
        u = self.to_ssm(x)                                          # spatial-aware SSM input
        h = self.ssm.step(h, u, dt)                                # temporal recurrence
        a = torch.tanh(self.decoder(h)) * self.accel_scale         # acceleration
        return h, a


def build_graph(pos, cfg):
    if cfg.get("graph", "knn") == "radius":
        return G.radius_graph(pos, r=cfg.get("radius", 0.6),
                              max_neighbors=cfg.get("max_neighbors", 16))
    return G.knn_graph(pos, k=cfg.get("k", 6))


def ssm_rollout(model, p0, v0, e, z, init_vel, init_pos, steps, cfg, dt,
                grad=True, rebuild_graph=False, recenter=False, damping=1.0,
                bptt_start=0):
    """Roll out T = steps+1 frames from (p0, v0). e,z,init_* are per-anchor [M,·].

    Explicit (p,v,a) integration — position from previous position+velocity, only
    acceleration comes from the SSM hidden state. Returns positions [T, M, 3].

    `damping` ∈ (0,1] multiplies velocity each step (friction) so a persistent
    acceleration can't accumulate velocity/position unboundedly — the open-loop
    autonomous rollout stays bounded instead of exploding. 1.0 = no damping.

    `bptt_start`: steps before this index are run without gradient (p,v,h detached
    at that boundary). Enables truncated BPTT: rollout [0, steps] with gradient
    only over [bptt_start, steps], so gradient chain length = steps - bptt_start."""
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        h = model.init_hidden(e, z, init_vel, init_pos)
        p, v = p0, v0
        out = [p]
        edge_index = build_graph(p.detach(), cfg)
        for i in range(steps):
            if rebuild_graph:
                edge_index = build_graph(p.detach(), cfg)
            h, a = model.step(p, v, h, e, z, edge_index, dt)
            p_next = p + v * dt                    # p^{t+1} = p^t + v^t·dt
            v = damping * (v + a * dt)             # v^{t+1} = γ·(v^t + a^t·dt)
            p = p_next
            if i < bptt_start - 1:               # detach before gradient window
                p = p.detach()
                v = v.detach()
                h = [x.detach() for x in h] if isinstance(h, (list, tuple)) else h.detach()
            out.append(p)
        seq = torch.stack(out, dim=0)              # [T, M, 3]
        if recenter:                               # anti-drift: hold COM at rest
            seq = seq - seq.mean(1, keepdim=True) + seq[:1].mean(1, keepdim=True)
        return seq

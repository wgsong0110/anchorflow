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

import math

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

    def step_batch(self, h, u, dt):
        """h,u: [B,M,dim]; dt: [B] (one Δt per batch element, broadcast
        across all M anchors and the dim channels within it) -- lets a batch
        of B independent per-frame walks each advance by their own Δt in a
        single call, instead of one step() call per (frame, depth) pair."""
        dt_b = dt.reshape(-1, 1, 1)
        decay = torch.exp(-dt_b * F.softplus(self.log_rate))
        return decay * h + (1 - decay) * self.in_proj(u)


class SSMDynamics(nn.Module):
    def __init__(self, hidden=128, mp_steps=6, ssm_dim=128, e_dim=8, z_dim=8,
                 edge_in=4, accel_scale=0.1, use_ssm=True, predict_velocity=False):
        super().__init__()
        self.accel_scale = accel_scale
        self.use_ssm = use_ssm   # ablation: False -> memoryless per-step GNN (h=u, no recurrence)
        # ablation: True -> decoder output IS the (damped) velocity directly
        # (single integration: p'=p+v*dt) instead of an acceleration that gets
        # integrated into v first (double integration: p'=p+v*dt, v'=v+a*dt).
        self.predict_velocity = predict_velocity
        self.node_enc = mlp([3 + e_dim + z_dim, hidden, hidden])     # [v, e, z]
        self.edge_enc = mlp([edge_in, hidden, hidden])
        self.processor = nn.ModuleList(
            InteractionNetwork(hidden) for _ in range(mp_steps))     # spatial
        self.to_ssm = mlp([hidden, ssm_dim])
        self.ssm = DiagonalSSM(ssm_dim)                             # temporal
        self.decoder = mlp([ssm_dim, hidden, 3], layernorm=False)   # h -> accel
        # h -> per-anchor rotation (residual quaternion, SC-GS d_rot_as_res
        # convention: normalize([1,0,0,0] + d_rotation)). Directly predicted from
        # the same hidden state as acceleration -- no Procrustes/SVD anywhere, so
        # there is no degenerate-singular-value NaN path and gradient flows to
        # the anchor dynamics normally. Zero-init so rollout starts at identity
        # rotation (matches the accel decoder's zero-init rationale).
        self.rot_decoder = mlp([ssm_dim, hidden, 4], layernorm=False)
        last = [m for m in self.rot_decoder.modules() if isinstance(m, nn.Linear)][-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
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
        h = self.ssm.step(h, u, dt) if self.use_ssm else u         # temporal recurrence (or bypass)
        a = torch.tanh(self.decoder(h)) * self.accel_scale         # accel, or velocity if predict_velocity
        d_rot = self.rot_decoder(h)                                 # [M,4] residual quat
        return h, a, d_rot


def build_graph(pos, cfg):
    if cfg.get("graph", "knn") == "radius":
        return G.radius_graph(pos, r=cfg.get("radius", 0.6),
                              max_neighbors=cfg.get("max_neighbors", 16))
    return G.knn_graph(pos, k=cfg.get("k", 6))


def ssm_rollout(model, p0, v0, e, z, init_vel, init_pos, steps, cfg, dt,
                grad=True, rebuild_graph=False, recenter=False, damping=1.0,
                bptt_start=0, return_states=False, vel_smooth=0.1,
                return_rotations=False):
    """Roll out T = steps+1 frames from (p0, v0). Returns positions [T, M, 3].

    bptt_start: detach p/v/h before this step index (truncated BPTT).
    return_states: if True, also return list of (p,v,h) CPU tensors at each t.
    return_rotations: if True, also return per-anchor residual quaternions
        [T, M, 4] (raw, un-normalized; frame 0 is identity [1,0,0,0])."""
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        h = model.init_hidden(e, z, init_vel, init_pos)
        p, v = p0, v0
        out = [p]
        rot_out = [torch.zeros(p.shape[0], 4, device=p.device).index_fill_(1, torch.tensor([0], device=p.device), 1.0)] \
                  if return_rotations else None
        states = [(p.detach().cpu(), v.detach().cpu(), h.detach().cpu())] \
                 if return_states else None
        edge_index = build_graph(p.detach(), cfg)
        for i in range(steps):
            if rebuild_graph:
                edge_index = build_graph(p.detach(), cfg)
            h, a, d_rot = model.step(p, v, h, e, z, edge_index, dt)
            p_next = p + v * dt
            v = damping * a if model.predict_velocity else damping * (v + a * dt)
            # velocity smoothing: MPM P2G→G2P equivalent — neighbors enforce coherent velocity field
            src_e, dst_e = edge_index
            v_agg = torch.zeros_like(v).scatter_add_(0, dst_e.unsqueeze(1).expand(-1, 3), v[src_e])
            deg_v = torch.zeros(v.shape[0], device=v.device).scatter_add_(0, dst_e, torch.ones(dst_e.shape[0], device=v.device))
            v = (1 - vel_smooth) * v + vel_smooth * (v_agg / deg_v.unsqueeze(1).clamp(min=1))
            p = p_next
            if i < bptt_start - 1:
                p = p.detach(); v = v.detach(); h = h.detach()
            out.append(p)
            if return_rotations:
                rot_out.append(d_rot)
            if return_states:
                states.append((p.detach().cpu(), v.detach().cpu(), h.detach().cpu()))
        seq = torch.stack(out, dim=0)
        if recenter:
            seq = seq - seq.mean(1, keepdim=True) + seq[:1].mean(1, keepdim=True)
        if return_rotations:
            rot_seq = torch.stack(rot_out, dim=0)               # [T,M,4]
            if return_states:
                return seq, rot_seq, states
            return seq, rot_seq
        if return_states:
            return seq, states
        return seq


def ssm_rollout_from(model, p0, v0, h0, e, z, steps, cfg, dt,
                     grad=True, damping=1.0, vel_smooth=0.1, return_rotations=False):
    """Rollout from a given state (p0,v0,h0) for `steps` steps — no bptt needed
    because the caller is responsible for detaching the initial state."""
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        p, v, h = p0, v0, h0
        out = [p]
        rot_out = [torch.zeros(p.shape[0], 4, device=p.device).index_fill_(1, torch.tensor([0], device=p.device), 1.0)] \
                  if return_rotations else None
        edge_index = build_graph(p.detach(), cfg)
        for _ in range(steps):
            h, a, d_rot = model.step(p, v, h, e, z, edge_index, dt)
            p_next = p + v * dt
            v = damping * a if model.predict_velocity else damping * (v + a * dt)
            src_e, dst_e = edge_index
            v_agg = torch.zeros_like(v).scatter_add_(0, dst_e.unsqueeze(1).expand(-1, 3), v[src_e])
            deg_v = torch.zeros(v.shape[0], device=v.device).scatter_add_(0, dst_e, torch.ones(dst_e.shape[0], device=v.device))
            v = (1 - vel_smooth) * v + vel_smooth * (v_agg / deg_v.unsqueeze(1).clamp(min=1))
            p = p_next
            out.append(p)
            if return_rotations:
                rot_out.append(d_rot)
        seq = torch.stack(out, dim=0)               # [steps+1, M, 3]
        if return_rotations:
            return seq, torch.stack(rot_out, dim=0)  # + [steps+1, M, 4]
        return seq


# =============================================================================
# HopDynamics -- variable-Δt, sparse-hop dynamics (middle ground between the
# fully-sequential SSMDynamics rollout above and a fully non-autoregressive
# per-frame regression). Only hops between the frames actually sampled for a
# training step (instead of a fixed small dt for every one of the T frames),
# each hop conditioned explicitly on its own Δt (not assumed constant), and
# outputs the position DELTA directly (no separate velocity state to
# integrate). See lib/anchorflow/warp.py's anchor_arap_loss for the ARAP
# analogue this composes with; see hop_rollout()'s docstring for the
# fine/coarse consistency + initial-velocity-consistency losses this design
# is meant to support from the training script.
# =============================================================================

def _time_encoding(dt, n_freqs=6):
    """NeRF-style positional encoding of a scalar Δt (same Δt for every anchor
    in a hop -- it's the gap between two selected frames, not per-anchor).
    Returns [1 + 2*n_freqs]."""
    dt_t = torch.as_tensor(dt, dtype=torch.float32)
    freqs = 2.0 ** torch.arange(n_freqs, dtype=torch.float32, device=dt_t.device)
    args = dt_t * freqs * math.pi
    return torch.cat([dt_t.reshape(1), torch.sin(args), torch.cos(args)], dim=0)


def _time_encoding_batch(dt, n_freqs=6):
    """Batched version of _time_encoding: dt [B] (one Δt per batch element)
    -> [B, 1 + 2*n_freqs]."""
    dt_t = torch.as_tensor(dt, dtype=torch.float32)
    freqs = 2.0 ** torch.arange(n_freqs, dtype=torch.float32, device=dt_t.device)
    args = dt_t[:, None] * freqs[None, :] * math.pi
    return torch.cat([dt_t[:, None], torch.sin(args), torch.cos(args)], dim=-1)


class HopDynamics(nn.Module):
    def __init__(self, hidden=128, mp_steps=6, ssm_dim=128, e_dim=8,
                 edge_in=4, n_time_freqs=6):
        super().__init__()
        self.n_time_freqs = n_time_freqs
        time_dim = 1 + 2 * n_time_freqs
        # one-time spatial embedding over the (fixed) canonical anchor graph --
        # unlike SSMDynamics, this never needs recomputing mid-rollout since
        # there is no "current position" until a hop has actually been taken.
        self.node_enc = mlp([e_dim, hidden, hidden])
        self.edge_enc = mlp([edge_in, hidden, hidden])
        self.processor = nn.ModuleList(
            InteractionNetwork(hidden) for _ in range(mp_steps))
        self.hop_in = mlp([hidden + e_dim + ssm_dim + time_dim, hidden, ssm_dim])
        self.ssm = DiagonalSSM(ssm_dim)
        self.decoder = mlp([ssm_dim, hidden, 3], layernorm=False)      # -> Δp (delta, not velocity)
        last = [m for m in self.decoder.modules() if isinstance(m, nn.Linear)][-1]
        nn.init.zeros_(last.weight); nn.init.zeros_(last.bias)
        # rotation is NOT accumulated across hops -- like SSMDynamics.rot_decoder,
        # this gives the FULL residual quaternion for the frame just reached,
        # freshly from that hop's resulting h (matches the d_rot_as_res
        # convention used everywhere else in this codebase).
        self.rot_decoder = mlp([ssm_dim, hidden, 4], layernorm=False)
        last2 = [m for m in self.rot_decoder.modules() if isinstance(m, nn.Linear)][-1]
        nn.init.zeros_(last2.weight); nn.init.zeros_(last2.bias)

    def spatial_embed(self, e, edge_index, canonical):
        node = self.node_enc(e)
        edge = self.edge_enc(G.edge_features(canonical, edge_index))
        x = node
        for layer in self.processor:
            x, edge = layer(x, edge_index, edge)
        return x                                                       # [M, hidden]

    def hop(self, spatial, h, e, dt):
        """One hop of Δt (python float / 0-d tensor, shared by all anchors).
        Returns (d_p [M,3], d_rot [M,4], h_next [M,ssm_dim])."""
        te = _time_encoding(dt, self.n_time_freqs).to(h.device)
        te = te.expand(h.shape[0], -1)
        u = self.hop_in(torch.cat([spatial, e, h, te], dim=-1))
        h_next = self.ssm.step(h, u, float(dt))
        d_p = self.decoder(h_next)
        d_rot = self.rot_decoder(h_next)
        return d_p, d_rot, h_next

    def hop_batch(self, spatial, h, e, dt):
        """Batched hop: h [B,M,ssm_dim], dt [B] (one Δt per batch element,
        broadcast across all M anchors within it), spatial/e [M,...] shared
        across the batch. Lets B independent per-frame walks each advance one
        depth in a single call, instead of B separate hop() calls. Returns
        (d_p [B,M,3], d_rot [B,M,4], h_next [B,M,ssm_dim])."""
        B, M = h.shape[0], h.shape[1]
        dt_t = torch.as_tensor(dt, dtype=torch.float32, device=h.device)
        te = _time_encoding_batch(dt_t, self.n_time_freqs)          # [B, time_dim]
        te = te[:, None, :].expand(B, M, -1)
        spatial_b = spatial[None, :, :].expand(B, M, -1)
        e_b = e[None, :, :].expand(B, M, -1)
        u = self.hop_in(torch.cat([spatial_b, e_b, h, te], dim=-1))
        h_next = self.ssm.step_batch(h, u, dt_t)
        d_p = self.decoder(h_next)
        d_rot = self.rot_decoder(h_next)
        return d_p, d_rot, h_next


def hop_rollout(model, canonical, e, edge_index, times, dt_base, grad=True,
                h0=None, spatial=None):
    """Hop through `times` (sorted list of positive frame indices; t=0 is the
    implicit, trivial starting point -- canonical position, identity
    rotation, no network call). Each hop's Δt is the actual gap between
    consecutive visited frames (times[i] - times[i-1]) * dt_base, not a fixed
    per-step constant -- this is the key difference from ssm_rollout, which
    always advances by exactly dt_base and must visit every one of the T
    frames to reach frame T-1.

    Pass h0 explicitly (the directly-learned init_h) -- there is no
    ic_dir/ic_net/z anywhere in this mode, so h0 must always be supplied by
    the caller. Pass spatial to continue a chain from a previously-computed
    embedding (needed for the coarse/shadow-fine consistency check: the
    shadow chain branches from this chain's own real (p, h) at an
    intermediate time, reusing the same spatial embedding).

    Returns:
      p_by_t   dict {t: [M,3]}       (includes t=0 -> canonical)
      rot_by_t dict {t: [M,4]}       (includes t=0 -> identity [1,0,0,0])
      h_by_t   dict {t: [M,ssm_dim]} (includes t=0 -> h0; hidden state after
                                       the hop landing on t, for branching a
                                       shadow chain from any visited time)
    """
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        if spatial is None:
            spatial = model.spatial_embed(e, edge_index, canonical)
        M = canonical.shape[0]
        p_by_t = {0: canonical}
        rot_by_t = {0: torch.zeros(M, 4, device=canonical.device)
                       .index_fill_(1, torch.tensor([0], device=canonical.device), 1.0)}
        h_by_t = {0: h0}
        p, h, prev_t = canonical, h0, 0
        for t in times:
            if t == 0:
                continue
            dt_hop = (t - prev_t) * dt_base
            d_p, d_rot, h = model.hop(spatial, h, e, dt_hop)
            p = p + d_p
            p_by_t[t] = p
            rot_by_t[t] = d_rot
            h_by_t[t] = h
            prev_t = t
        return p_by_t, rot_by_t, h_by_t

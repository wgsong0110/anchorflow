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

from .dynamics import mlp, InteractionNetwork, MPNNLayer
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
                 edge_in=4, n_time_freqs=6, use_ssm=True, stateless=False, f_ext_dim=0,
                 gnn_layer="interaction"):
        super().__init__()
        # "interaction": InteractionNetwork (GNS-lineage, residual edge state
        #   carried/evolved layer to layer). "mpnn": MPNNLayer (Gilmer et al.
        #   textbook message passing, edge features are the raw input at
        #   every layer, no evolving edge state).
        assert gnn_layer in ("interaction", "mpnn")
        gnn_cls = InteractionNetwork if gnn_layer == "interaction" else MPNNLayer
        # f_ext_dim>0 (stateless mode only): P2G channel -- a per-anchor
        # aggregated external force/perturbation (see
        # AnchorSet.scatter_particle_to_anchor), concatenated into the
        # per-hop GNN node features alongside v/e/te. 0 = disabled (default,
        # backward compatible with checkpoints trained before this existed).
        self.f_ext_dim = f_ext_dim
        self.n_time_freqs = n_time_freqs
        # ablation: False -> memoryless hop, h_next = u (this hop's raw
        # hop_in output) instead of the SSM's decayed recurrent blend. Drops
        # any "momentum"/"gait phase" signal beyond the immediately previous
        # hop's own output -- can't represent rhythm/actuation state that
        # needs more than 1-hop lookback, but also removes the recurrent
        # state as a channel through which error can compound/blow up over
        # a long autoregressive rollout (h no longer accumulates across
        # hops at all, decayed or otherwise).
        self.use_ssm = use_ssm
        # stateless=True: a completely different (p,v)-state architecture,
        # see hop_pv() below -- no recurrent h at all (not even the 1-hop
        # lookback of use_ssm=False), GNN message passing RE-RUN every hop
        # (grounded in the CURRENT position/velocity, not a one-time
        # spatial embedding computed from canonical), and the network
        # predicts velocity directly (d_p = v_next * dt) instead of a raw
        # per-hop displacement -- so the position update scales with
        # elapsed time by construction, instead of the network having to
        # learn dt-scaling from scratch (a plausible contributor to the
        # d_p-magnitude blowups diagnosed earlier: a raw displacement
        # decoder has no guarantee its output scales sensibly across the
        # very different Δt values a sparse curriculum hop can have).
        self.stateless = stateless
        time_dim = 1 + 2 * n_time_freqs
        if stateless:
            # use_ssm here means something different from the h-based hop()'s
            # use_ssm=False ablation: it controls whether hop_pv ALSO keeps a
            # recurrent h (SSM memory -- gait phase / self-powered actuation
            # rhythm) ON TOP OF the every-hop GNN recompute, vs GNN-recompute
            # with NO memory at all (pure function of current p/v). The two
            # design axes (GNN once-vs-every-hop, SSM present-vs-absent) are
            # independent; this is the "keep SSM, also re-run GNN every hop"
            # cell of that 2x2 that hadn't been tried yet.
            node_in_dim = 3 + e_dim + time_dim + f_ext_dim + (ssm_dim if use_ssm else 0)  # [v,e,te,f_ext?,h?]
            self.pv_node_enc = mlp([node_in_dim, hidden, hidden])
            self.pv_edge_enc = mlp([edge_in, hidden, hidden])
            self.pv_processor = nn.ModuleList(
                gnn_cls(hidden) for _ in range(mp_steps))
            if use_ssm:
                self.pv_to_ssm = mlp([hidden, ssm_dim])
                self.pv_ssm = DiagonalSSM(ssm_dim)
            dec_dim = ssm_dim if use_ssm else hidden
            self.pv_decoder = mlp([dec_dim, hidden, 3], layernorm=False)      # -> v_next directly
            lastv = [m for m in self.pv_decoder.modules() if isinstance(m, nn.Linear)][-1]
            nn.init.zeros_(lastv.weight); nn.init.zeros_(lastv.bias)
            self.pv_rot_decoder = mlp([dec_dim, hidden, 4], layernorm=False)
            lastr = [m for m in self.pv_rot_decoder.modules() if isinstance(m, nn.Linear)][-1]
            nn.init.zeros_(lastr.weight); nn.init.zeros_(lastr.bias)
            self.pv_local_rot_decoder = mlp([dec_dim, hidden, 4], layernorm=False)
            lastl = [m for m in self.pv_local_rot_decoder.modules() if isinstance(m, nn.Linear)][-1]
            nn.init.zeros_(lastl.weight); nn.init.zeros_(lastl.bias)
            return
        # one-time spatial embedding over the (fixed) canonical anchor graph --
        # unlike SSMDynamics, this never needs recomputing mid-rollout since
        # there is no "current position" until a hop has actually been taken.
        self.node_enc = mlp([e_dim, hidden, hidden])
        self.edge_enc = mlp([edge_in, hidden, hidden])
        self.processor = nn.ModuleList(
            gnn_cls(hidden) for _ in range(mp_steps))
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
        # local_rotation: SC-GS's --local_frame mechanism (utils/time_utils.py
        # DeformNetworkNode.forward()) uses a THIRD per-node quaternion head,
        # distinct from d_rotation, to rigidly rotate each Gaussian's offset
        # from its nearest anchor before translating it (Ax = local_rot @
        # (x - node) + node + d_xyz, then LBS-blended over K neighbors) --
        # this is what actually lets non-anchor-coincident points (i.e. most
        # Gaussians) inherit a sensible rigid transform, not just d_xyz/d_rot
        # evaluated at the anchor itself. Purely additive: hop()/hop_batch()/
        # hop_rollout() are unchanged (other scripts unpack their exact
        # 3-tuple returns) -- callers that want local_rotation call
        # .local_rotation(h) themselves using the h_by_t hop_rollout already
        # returns (see train_hop_autoreg_anchortraj.py).
        self.local_rot_decoder = mlp([ssm_dim, hidden, 4], layernorm=False)
        last3 = [m for m in self.local_rot_decoder.modules() if isinstance(m, nn.Linear)][-1]
        nn.init.zeros_(last3.weight); nn.init.zeros_(last3.bias)

    def local_rotation(self, h):
        """Per-anchor local-frame quaternion from a hop's resulting hidden
        state h [*, ssm_dim] -> [*, 4]. Zero-initialized like rot_decoder."""
        return self.local_rot_decoder(h)

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
        h_next = self.ssm.step(h, u, float(dt)) if self.use_ssm else u
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
        h_next = self.ssm.step_batch(h, u, dt_t) if self.use_ssm else u
        d_p = self.decoder(h_next)
        d_rot = self.rot_decoder(h_next)
        return d_p, d_rot, h_next

    def hop_pv(self, e, edge_index, p, v, dt, h=None, f_ext=None):
        """Only valid when stateless=True. One hop of a (p,v)-explicit,
        GNN-every-hop rollout: message passing is re-run from scratch using
        the CURRENT p (edge features) and v (node features, alongside e and
        this hop's Δt encoding) -- unlike hop()'s spatial (computed once),
        anchors coordinate every single hop, not just at t=0.

        h [M, ssm_dim], required iff self.use_ssm: the SSM's recurrent
        memory (gait phase / self-powered actuation rhythm), carried
        alongside (p, v) -- lets this mode ALSO represent self-driven motion
        beyond what the current-instant GNN pass alone could infer, on top
        of the every-hop coordination. If self.use_ssm is False, h is
        ignored and this is the original fully memoryless hop_pv.

        f_ext [M, f_ext_dim], required iff self.f_ext_dim > 0: per-anchor
        aggregated external force (P2G, see
        AnchorSet.scatter_particle_to_anchor) -- the channel through which a
        force applied to particles that aren't necessarily coincident with
        any anchor still reaches this hop's decision.

        The network predicts v_next directly (not an acceleration, not a
        raw displacement); d_p = v_next * dt is the actual position update,
        so displacement scales with elapsed time by construction rather
        than something the network has to learn to represent correctly
        across whatever Δt a sparse curriculum hop happens to have.

        Returns (d_p [M,3], d_rot [M,4], v_next [M,3], local_rot [M,4],
        h_next [M,ssm_dim] or None if not self.use_ssm)."""
        te = _time_encoding(dt, self.n_time_freqs).to(p.device)
        te = te.expand(p.shape[0], -1)
        node_in = [v, e, te]
        if self.f_ext_dim > 0:
            node_in.append(f_ext)
        if self.use_ssm:
            node_in.append(h)
        node = self.pv_node_enc(torch.cat(node_in, dim=-1))
        edge = self.pv_edge_enc(G.edge_features(p, edge_index))
        x = node
        for layer in self.pv_processor:
            x, edge = layer(x, edge_index, edge)
        if self.use_ssm:
            u = self.pv_to_ssm(x)
            h_next = self.pv_ssm.step(h, u, float(dt))
            dec_in = h_next
        else:
            h_next = None
            dec_in = x
        v_next = self.pv_decoder(dec_in)
        d_rot = self.pv_rot_decoder(dec_in)
        local_rot = self.pv_local_rot_decoder(dec_in)
        d_p = v_next * float(dt)
        return d_p, d_rot, v_next, local_rot, h_next


def rest_edge_lengths(canonical, edge_index):
    """‖canonical[dst] - canonical[src]‖ per edge -> [E]. Precompute once;
    the anchor graph topology is fixed for a rollout (built from canonical)."""
    src, dst = edge_index
    return (canonical[dst] - canonical[src]).norm(dim=-1)


def xpbd_distance_project(pos, edge_index, rest_len, tolerance=0.15, iters=4):
    """Structural (not learned, not a training-time penalty) no-splitting
    constraint: project `pos` so every edge's length stays within
    rest_len * (1 ± tolerance) of its canonical length.

    This is the position-based-dynamics (PBD/XPBD) distance-constraint
    solve: find the position closest (in a least-squares sense) to the GNN's
    raw proposal subject to the per-edge distance bound, applied as a fixed
    -point iteration since there's no closed form for many overlapping
    constraints. Each sweep is a Jacobi step (all edges corrected
    simultaneously via scatter_add, not sequentially like classic
    Gauss-Seidel PBD) -- fully parallel, and differentiable (pure tensor ops,
    no learnable parameters), so gradients flow through it unchanged.

    Because it operates on the same fixed edge_index the rollout already
    builds from canonical, this holds for ANY d_p the network ever produces
    (including a randomly-initialized, untrained model) -- the guarantee is
    architectural, not something training has to learn to respect.

    Jacobi (simultaneous) correction needs per-node degree normalization to
    stay stable: without it, a node touched by k edges that all want to pull
    it the same way gets the FULL sum of k half-corrections in one sweep,
    overshooting by ~k -- harmless-looking for k=1 but divergent (-> NaN
    within a few sweeps) for the k=6-8 degree typical of a k-NN anchor
    graph. Dividing each node's accumulated correction by its own degree
    keeps every sweep a genuine averaging step (contractive) instead of an
    amplifying one."""
    src, dst = edge_index
    N = pos.shape[0]
    deg = torch.zeros(N, device=pos.device, dtype=pos.dtype)
    deg = deg.index_add(0, dst, torch.ones_like(dst, dtype=pos.dtype))
    deg = deg.index_add(0, src, torch.ones_like(src, dtype=pos.dtype)).clamp(min=1)
    lo = (rest_len * (1 - tolerance)).unsqueeze(-1)
    hi = (rest_len * (1 + tolerance)).unsqueeze(-1)
    for _ in range(iters):
        diff = pos[dst] - pos[src]                                   # [E,3]
        dist = diff.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        target = dist.clamp(min=lo, max=hi)
        correction = (target - dist) / dist * diff * 0.5             # split between endpoints
        delta = torch.zeros_like(pos)
        delta = delta.index_add(0, dst, correction)
        delta = delta.index_add(0, src, -correction)
        pos = pos + delta / deg.unsqueeze(-1)                        # per-node degree normalization
    return pos


def xpbd_directional_project(pos, edge_index, rest_len, direction, tolerance=0.15, iters=4,
                              min_dir_norm=1e-4):
    """Like xpbd_distance_project, but restricts each anchor's correction to
    its OWN fixed direction (the network's predicted d_p, normalized) instead
    of allowing a free 3D correction.

    xpbd_distance_project minimizes ‖p_new - p_pre‖ (p_pre = p + d_p), which
    is equivalent to minimizing ‖u - d_p‖ where u = p_new - p is the actual
    displacement -- close to d_p in a least-squares sense, but with many
    simultaneous neighbor constraints (k=6-8 in a k-NN anchor graph all
    pulling at once), the L2-closest constraint-satisfying point can end up
    with u pointing quite differently from d_p: neighbors correcting in
    different directions can rotate the net result away from what the
    network actually intended, not just shrink it.

    This variant guarantees cos_similarity(u, d_p) is exactly 1 (or -1, if
    pulled backward past the start) by construction: every correction is
    projected onto direction before being applied, so position can only
    ever move along the anchor's own intended line p + alpha*direction,
    never off of it. Constraint satisfaction is then a 1D search (scalar
    alpha per anchor) along that line instead of a free 3D search -- weaker
    (may not reach as tight a constraint-violation bound as the free
    version, since it has fewer degrees of freedom to satisfy neighbors
    with), but never redirects intent.

    Anchors with near-zero d_p (no meaningful intended direction) fall back
    to the free (xpbd_distance_project-style) correction via `min_dir_norm`,
    since forcing them onto an arbitrary near-zero-norm direction would be
    numerically unstable and physically meaningless."""
    src, dst = edge_index
    N = pos.shape[0]
    deg = torch.zeros(N, device=pos.device, dtype=pos.dtype)
    deg = deg.index_add(0, dst, torch.ones_like(dst, dtype=pos.dtype))
    deg = deg.index_add(0, src, torch.ones_like(src, dtype=pos.dtype)).clamp(min=1)
    lo = (rest_len * (1 - tolerance)).unsqueeze(-1)
    hi = (rest_len * (1 + tolerance)).unsqueeze(-1)
    dir_norm = direction.norm(dim=-1, keepdim=True)
    unit_dir = direction / dir_norm.clamp(min=1e-8)
    has_dir = (dir_norm > min_dir_norm).to(pos.dtype)                # [N,1]
    for _ in range(iters):
        diff = pos[dst] - pos[src]
        dist = diff.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        target = dist.clamp(min=lo, max=hi)
        correction = (target - dist) / dist * diff * 0.5
        delta = torch.zeros_like(pos)
        delta = delta.index_add(0, dst, correction)
        delta = delta.index_add(0, src, -correction)
        delta = delta / deg.unsqueeze(-1)
        delta_along = (delta * unit_dir).sum(-1, keepdim=True) * unit_dir
        delta = has_dir * delta_along + (1 - has_dir) * delta        # free fallback for ~zero-d_p anchors
        pos = pos + delta
    return pos


def hop_rollout(model, canonical, e, edge_index, times, dt_base, grad=True,
                h0=None, spatial=None, project_fn=None):
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

    project_fn, if given, is called as project_fn(p_pre, d_p) after every
    hop's raw update (p_pre = p + d_p) and must return the (possibly
    corrected) position to actually use -- e.g. xpbd_distance_project or
    xpbd_directional_project (both take extra args via a closure/lambda;
    d_p is passed through so a directional projector can derive its own
    per-anchor direction from it).

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
            if project_fn is not None:
                p = project_fn(p, d_p)                    # (proposed pos, this hop's raw d_p)
            p_by_t[t] = p
            rot_by_t[t] = d_rot
            h_by_t[t] = h
            prev_t = t
        return p_by_t, rot_by_t, h_by_t


def hop_rollout_pv(model, canonical, e, edge_index, times, dt_base, grad=True,
                   v0=None, h0=None, project_fn=None):
    """hop_rollout's counterpart for HopDynamics(stateless=True) models (see
    hop_pv()): explicit (p,v) state, GNN re-run every hop from the CURRENT
    p/v. v0 defaults to rest (zeros). h0 is required iff model.use_ssm (the
    "keep SSM, also re-run GNN every hop" combination) -- pass the same kind
    of learned init_h parameter hop_rollout's h0 expects; ignored/unused if
    model.use_ssm is False (the original fully memoryless hop_pv).

    Returns (p_by_t, rot_by_t, lrot_by_t) -- no h_by_t returned even when
    model.use_ssm (nothing currently needs to branch a shadow chain from an
    intermediate h in this mode; add if that changes)."""
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        M = canonical.shape[0]
        p = canonical
        v = v0 if v0 is not None else torch.zeros_like(canonical)
        h = h0
        p_by_t = {0: canonical}
        rot_by_t = {0: torch.zeros(M, 4, device=canonical.device)
                       .index_fill_(1, torch.tensor([0], device=canonical.device), 1.0)}
        lrot_by_t = {0: torch.zeros(M, 4, device=canonical.device)
                        .index_fill_(1, torch.tensor([0], device=canonical.device), 1.0)}
        prev_t = 0
        for t in times:
            if t == 0:
                continue
            dt_hop = (t - prev_t) * dt_base
            d_p, d_rot, v, lrot, h = model.hop_pv(e, edge_index, p, v, dt_hop, h=h)
            p_pre = p + d_p
            p = project_fn(p_pre, d_p) if project_fn is not None else p_pre
            p_by_t[t] = p
            rot_by_t[t] = d_rot
            lrot_by_t[t] = lrot
            prev_t = t
        return p_by_t, rot_by_t, lrot_by_t

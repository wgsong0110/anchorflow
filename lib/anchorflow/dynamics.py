"""GNS-style graph-network dynamics for anchor state.

Encode-Process-Decode (Sanchez-Gonzalez et al., ICML 2020), implemented with a
pure-PyTorch scatter so there is no PyG / torch-scatter build dependency.  The
:class:`InteractionNetwork` layer keeps the same (x, edge_index, edge_attr) ->
x signature as ``torch_geometric.nn.MessagePassing`` subclasses, so PyG is a
drop-in replacement if large-N neighbour sampling is later needed.

State / prediction convention (second-order, dt folded into units):

    v_t   = p_t   - p_{t-1}                      (per-step velocity)
    a_t   = p_{t+1} - 2 p_t + p_{t-1}            (per-step acceleration = target)
    p_{t+1} = p_t + v_t + a_pred                 (integration during rollout)

The network predicts the *normalised* acceleration for every non-fixed anchor.
Fixed anchors are held at their initial position during rollout.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from . import graph as G


# --------------------------------------------------------------------------- #
#  building blocks                                                             #
# --------------------------------------------------------------------------- #
def mlp(sizes, layernorm=True, act=nn.SiLU):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
    net = nn.Sequential(*layers)
    if layernorm:
        net = nn.Sequential(net, nn.LayerNorm(sizes[-1]))
    return net


class InteractionNetwork(nn.Module):
    """One message-passing step with residual edge and node updates."""

    def __init__(self, hidden, mlp_layers=2):
        super().__init__()
        self.edge_mlp = mlp([3 * hidden] + [hidden] * mlp_layers)   # [h_i,h_j,e]
        self.node_mlp = mlp([2 * hidden] + [hidden] * mlp_layers)   # [h_i,agg]

    def forward(self, h, edge_index, e):
        src, dst = edge_index                          # j -> i
        m = self.edge_mlp(torch.cat([h[dst], h[src], e], dim=-1))
        e = e + m                                      # residual edge update
        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, e)                      # sum messages at receiver i
        h = h + self.node_mlp(torch.cat([h, agg], dim=-1))          # residual node
        return h, e


class Normalizer(nn.Module):
    """Running mean/std normaliser (GNS-style), updated only in train mode."""

    def __init__(self, dim, max_acc=10**6, eps=1e-5):
        super().__init__()
        self.register_buffer("count", torch.tensor(0.0))
        self.register_buffer("sum", torch.zeros(dim))
        self.register_buffer("sqsum", torch.zeros(dim))
        self.max_acc = max_acc
        self.eps = eps

    def _accumulate(self, x):
        if self.count.item() < self.max_acc:
            self.count += x.shape[0]
            self.sum += x.sum(0)
            self.sqsum += (x * x).sum(0)

    def mean(self):
        return self.sum / self.count.clamp(min=1)

    def std(self):
        m = self.mean()
        var = (self.sqsum / self.count.clamp(min=1)) - m * m
        return var.clamp(min=0).sqrt().clamp(min=self.eps)

    def forward(self, x, accumulate=True):
        if self.training and accumulate:
            self._accumulate(x.detach())
        if self.count.item() == 0:
            return x
        return (x - self.mean()) / self.std()

    def inverse(self, x):
        if self.count.item() == 0:
            return x
        return x * self.std() + self.mean()


# --------------------------------------------------------------------------- #
#  the dynamics model                                                         #
# --------------------------------------------------------------------------- #
class GNSDynamics(nn.Module):
    """Autoregressive anchor dynamics.

    node input features : [velocity(3), fixed_flag(1)] (+ actuation latent z_i)
    edge input features : [rel_disp(3), dist(1)]            -> 4
    output              : normalised acceleration (3)

    ``latent_dim`` > 0 appends a per-node actuation latent z_i to the node input
    (the internal-drive signal for self-actuated motion — optimised under SDS).
    Only the kinematic part is Normalizer-standardised; z_i is passed raw so the
    running stats don't fight the latent's own optimisation. latent_dim=0 recovers
    the plain GNS used by the synth unit test.
    """

    def __init__(self, hidden=128, message_passing_steps=6,
                 latent_dim=0, base_node_in=4, edge_in=4, out_dim=3):
        super().__init__()
        self.latent_dim = latent_dim
        self.node_encoder = mlp([base_node_in + latent_dim, hidden, hidden])
        self.edge_encoder = mlp([edge_in, hidden, hidden])
        self.processor = nn.ModuleList(
            InteractionNetwork(hidden) for _ in range(message_passing_steps)
        )
        self.decoder = mlp([hidden, hidden, out_dim], layernorm=False)
        self.in_norm = Normalizer(base_node_in)
        self.out_norm = Normalizer(out_dim)

    # --- single-step prediction ------------------------------------------- #
    def predict_accel(self, pos, vel, fixed, edge_index, z=None):
        """Predict *un-normalised* per-node acceleration for one step."""
        kin = torch.cat([vel, fixed.float().unsqueeze(-1)], dim=-1)
        kin = self.in_norm(kin)
        node_feat = kin if z is None else torch.cat([kin, z], dim=-1)
        e = G.edge_features(pos, edge_index)
        h = self.node_encoder(node_feat)
        e = self.edge_encoder(e)
        for layer in self.processor:
            h, e = layer(h, edge_index, e)
        acc_norm = self.decoder(h)
        return self.out_norm.inverse(acc_norm), acc_norm

    def forward(self, pos, vel, fixed, edge_index, z=None):
        acc, _ = self.predict_accel(pos, vel, fixed, edge_index, z)
        return acc


# --------------------------------------------------------------------------- #
#  graph builder dispatch                                                     #
# --------------------------------------------------------------------------- #
def build_graph(pos, cfg):
    if cfg.get("graph", "knn") == "radius":
        return G.radius_graph(pos, r=cfg.get("radius", 0.6),
                              max_neighbors=cfg.get("max_neighbors", 16))
    return G.knn_graph(pos, k=cfg.get("k", 6))


# --------------------------------------------------------------------------- #
#  autoregressive rollout                                                     #
# --------------------------------------------------------------------------- #
def rollout(model, p0, p1, fixed, steps, cfg, rebuild_graph=True, z=None,
            grad=False):
    """Free-running rollout from two seed frames p0, p1.

    Returns predicted positions [steps+2, N, 3] (including the two seeds).
    ``z`` is the per-node actuation latent (or None). ``grad=True`` keeps the
    graph differentiable end-to-end (needed for SDS backprop through the rollout);
    ``grad=False`` runs under no_grad for evaluation.
    """
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        if not grad:
            model.eval()
        dev = p0.device
        fixed = fixed.to(dev)
        p_prev, p_cur = p0.clone(), p1.clone()
        out = [p_prev, p_cur]
        edge_index = build_graph(p_cur, cfg)
        fixed_pos = p0[fixed]
        for _ in range(steps):
            if rebuild_graph:
                edge_index = build_graph(p_cur.detach(), cfg)   # topology only
            vel = p_cur - p_prev
            acc = model(p_cur, vel, fixed, edge_index, z)
            p_next = p_cur + vel + acc
            if fixed.any():
                p_next = torch.where(fixed[:, None], fixed_pos_full(fixed_pos, fixed, p_next), p_next)
            out.append(p_next)
            p_prev, p_cur = p_cur, p_next
        return torch.stack(out, dim=0)


def fixed_pos_full(fixed_pos, fixed, like):
    """Scatter pinned-node positions back into an [N,3] frame (grad-safe)."""
    full = like.clone()
    full[fixed] = fixed_pos
    return full

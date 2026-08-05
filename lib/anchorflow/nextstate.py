"""Learn the anchor dynamics outright: state in, displacement out, no physics.

The previous line asked a network to solve an implicit step -- it was scored
against the momentum residual, supervised by an LBFGS solve of the incremental
potential, and built on top of a Newmark predictor. Physics was in the loss, the
targets and the output path. That is retired (doc/implicit-step-learning-retired.md).

Here the network is the dynamics. Its target is simply where the explicit
simulator's anchors ended up after the same interval, and at inference no
constitutive model, force or residual is evaluated at all -- the physics kernel
is only used afterwards to skin the Gaussians onto whatever anchors the network
produced.

The state is (position, velocity, elastic acceleration). Velocity is the last
step's displacement over dt -- a quantity the network itself produced, so it
means the same thing in training and in rollout. Acceleration is f_int(p)/m,
evaluated by the physics kernel at the CURRENT configuration, which likewise
means the same thing in both.

That consistency is the point. The retired line carried an acceleration in the
state that was the simulator's f/m while training and the integrator's readback
of the network's own output while rolling out -- two different quantities under
one name, and the network had no way to tell. Here the force is recomputed from
whatever configuration the network has produced, so it is always the real one.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .dynamics import mlp


class DtFiLM(nn.Module):
    """Fourier-encoded step size -> per-layer (gamma, beta) modulation.

    dt is one scalar shared by every anchor, so as a node feature it would waste
    encoder channels on a constant and compete with the per-anchor signal. FiLM
    lets it rescale and shift every channel of the residual stream at every
    layer instead. log10(dt) is what gets encoded, since the step sizes of
    interest span more than two decades. Zero-init, so it starts as the identity.
    """

    def __init__(self, hidden, n_sites, n_freq=6):
        super().__init__()
        self.n_freq, self.hidden, self.n_sites = n_freq, hidden, n_sites
        self.mlp = mlp([1 + 2 * n_freq, hidden, 2 * hidden * n_sites], layernorm=False)
        last = [m for m in self.mlp.modules() if isinstance(m, nn.Linear)][-1]
        nn.init.zeros_(last.weight); nn.init.zeros_(last.bias)

    def forward(self, dt, device):
        x = torch.tensor([math.log10(dt)], device=device)
        f = 2.0 ** torch.arange(self.n_freq, device=device) * math.pi
        enc = torch.cat([x, torch.sin(f * x), torch.cos(f * x)])
        out = self.mlp(enc).view(self.n_sites, 2, self.hidden)
        return 1.0 + out[:, 0], out[:, 1]


class GeoAttentionBias(nn.Module):
    """Per-head attention bias from each pair's relative offset and distance.

    Attention is permutation-invariant and knows nothing about where the anchors
    are; this is how geometry enters. Only relative quantities are used, and
    distances are divided by the mean pair distance, so the operator is
    translation- and scale-invariant without ever seeing absolute coordinates.
    """

    def __init__(self, heads, hidden=32):
        super().__init__()
        self.mlp = mlp([4, hidden, heads], layernorm=False)

    def forward(self, p):
        """p [B,M,3] -> bias [B,H,M,M]"""
        rel = p.unsqueeze(2) - p.unsqueeze(1)                 # [B,M,M,3]
        d = rel.norm(dim=-1, keepdim=True)
        s = d.mean(dim=(1, 2, 3), keepdim=True).detach().clamp(min=1e-12)
        b = self.mlp(torch.cat([rel / s, torch.log1p(d / s)], -1))   # [B,M,M,H]
        return b.permute(0, 3, 1, 2)


class GeoAttentionBlock(nn.Module):
    """Pre-norm transformer block over the anchor set."""

    def __init__(self, hidden, heads=4):
        super().__init__()
        assert hidden % heads == 0
        self.h, self.d = heads, hidden // heads
        self.n1, self.n2 = nn.LayerNorm(hidden), nn.LayerNorm(hidden)
        self.qkv = nn.Linear(hidden, 3 * hidden)
        self.proj = nn.Linear(hidden, hidden)
        self.ffn = mlp([hidden, 2 * hidden, hidden], layernorm=False)

    def forward(self, x, bias):
        """x [B,M,C], bias [B,H,M,M]"""
        B, M, _ = x.shape
        q, k, v = self.qkv(self.n1(x)).chunk(3, dim=-1)
        q, k, v = (t.view(B, M, self.h, self.d).transpose(1, 2) for t in (q, k, v))
        att = (q @ k.transpose(-1, -2)) / math.sqrt(self.d) + bias
        o = (att.softmax(-1) @ v).transpose(1, 2).reshape(B, M, -1)
        x = x + self.proj(o)
        return x + self.ffn(self.n2(x))


class NextStep(nn.Module):
    """(position, velocity, elastic acceleration) per anchor -> displacement over dt.

    Message passing was measured and rejected for this anchor set: at k=8
    neighbours and depth 4, perturbing one anchor moved 29 of 512 outputs against
    a k-NN graph diameter of 73 hops. Attention gives every anchor every other in
    one layer, and at M=512 the full map is ~4 MB.

    Displacements are carried in units of `scale` (a measured typical
    displacement) on both the input and the output side. Predicting a
    displacement rather than a position matters: anchor coordinates are O(1)
    while one step's displacement is ~4e-3, so emitting a position would require
    three and a half significant digits before the motion is resolved at all --
    measured on a trained network, the position error was 2.3x larger than the
    displacement it was supposed to predict.
    """

    def __init__(self, hidden=128, depth=4, heads=4, scale=1.0, vel_scale=1.0,
                  acc_scale=1.0):
        super().__init__()
        # The whole displacement, never du = v*dt + correction. Adding the
        # inertial part as a skip makes the rollout's velocity an accumulator,
        # v_{k+1} = v_k + c_k/dt, so an error in the correction is carried
        # forever and the drift compounds. Measured: with the skip the rollout
        # reached 689% of the reference's own motion by frame 30 and neither 3x
        # the data nor noise injection helped; without it, 56%. Emitting the
        # displacement outright means the velocity is re-estimated from the
        # current position and force every step, so nothing is carried.
        self.scale = scale
        self.vel_scale = vel_scale
        self.acc_scale = acc_scale
        self.node_enc = mlp([3 + 3 + 3, hidden, hidden])   # p, v, a
        self.bias = GeoAttentionBias(heads)
        self.blocks = nn.ModuleList(GeoAttentionBlock(hidden, heads) for _ in range(depth))
        self.film = DtFiLM(hidden, depth + 1)
        self.dec = mlp([hidden, hidden, 3], layernorm=False)
        last = [m for m in self.dec.modules() if isinstance(m, nn.Linear)][-1]
        nn.init.zeros_(last.weight); nn.init.zeros_(last.bias)

    def forward(self, p, v, a, dt):
        """p, v, a all [M,3] or [B,M,3]; returns the displacement over dt, same shape.

        A batch is several independent anchor sets, so attention and the
        geometric bias stay within each one -- there is no cross-state mixing.
        """
        squeeze = (p.dim() == 2)
        if squeeze:
            p, v, a = p.unsqueeze(0), v.unsqueeze(0), a.unsqueeze(0)
        h = self.node_enc(torch.cat([p, v / self.vel_scale, a / self.acc_scale], -1))
        gamma, beta = self.film(dt, p.device)
        h = gamma[0] * h + beta[0]
        bias = self.bias(p)
        for i, blk in enumerate(self.blocks):
            h = blk(gamma[i + 1] * h + beta[i + 1], bias)
        out = self.dec(h) * self.scale
        return out.squeeze(0) if squeeze else out

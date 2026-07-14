"""Learned (amortized) ARAP initial-condition completion.

A GNN over the anchor graph maps a partial handle spec -> a full, plausible anchor
pose (feedforward, replacing the per-step hard ARAP linear solve). Pretrained
self-supervised with (handle-consistency + ARAP energy + proximity-to-rest) over
random handles — no data needed — then fine-tuned end-to-end by MDS in the full
pipeline so it learns object-specific plausibility.

    p = canonical + CompletionGNN(canonical, handle_mask, handle_target)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .dynamics import mlp, InteractionNetwork
from . import graph as G
from . import reg as R


class CompletionGNN(nn.Module):
    def __init__(self, hidden=128, mp_steps=6, edge_in=4):
        super().__init__()
        self.node_enc = mlp([1 + 3, hidden, hidden])        # [handle_flag, target_disp]
        self.edge_enc = mlp([edge_in, hidden, hidden])
        self.processor = nn.ModuleList(
            InteractionNetwork(hidden) for _ in range(mp_steps))
        self.decoder = mlp([hidden, hidden, 3], layernorm=False)   # -> displacement
        last = [m for m in self.decoder.modules() if isinstance(m, nn.Linear)][-1]
        nn.init.zeros_(last.weight); nn.init.zeros_(last.bias)     # start at rest

    def forward(self, canonical, handle_mask, handle_target, edge_index):
        disp0 = torch.zeros_like(canonical)
        disp0[handle_mask] = (handle_target - canonical)[handle_mask]
        node = self.node_enc(torch.cat([handle_mask.float()[:, None], disp0], dim=-1))
        edge = self.edge_enc(G.edge_features(canonical, edge_index))
        x = node
        for layer in self.processor:
            x, edge = layer(x, edge_index, edge)
        return canonical + self.decoder(x)                  # full positions [M,3]


def _sample_handles(canonical, p_handle, pos_std, gen=None):
    """Random handle mask + target positions (for pretraining)."""
    M = canonical.shape[0]
    mask = torch.rand(M, device=canonical.device) < p_handle
    if not mask.any():
        mask[torch.randint(0, M, (1,), device=canonical.device)] = True
    target = canonical.clone()
    target[mask] += pos_std * torch.randn((int(mask.sum()), 3), device=canonical.device)
    return mask, target


def pretrain_completion(model, canonical, idx, w, edge_index, steps=2000,
                        p_handle=0.1, pos_std=0.1, lr=1e-3,
                        lam=(10.0, 1.0, 0.1), log_every=200):
    """Self-supervised: loss = λ_h·handle_mse + λ_arap·ARAP(rest->p) + λ_prox·‖p-rest‖².
    lam = (handle, arap, prox)."""
    lh, la, lp = lam
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for step in range(steps):
        opt.zero_grad()
        mask, target = _sample_handles(canonical, p_handle, pos_std)
        p = model(canonical, mask, target, edge_index)
        l_handle = ((p[mask] - target[mask]) ** 2).mean()
        l_arap = R.arap_loss(torch.stack([canonical, p]), idx, w)
        l_prox = ((p - canonical) ** 2).mean()
        loss = lh * l_handle + la * l_arap + lp * l_prox
        loss.backward()
        opt.step()
        if step % log_every == 0:
            print(f"[pretrain_comp {step}] handle={float(l_handle):.3e} "
                  f"arap={float(l_arap):.3e} prox={float(l_prox):.3e}")
    return model

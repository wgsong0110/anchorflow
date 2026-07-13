#!/usr/bin/env python
"""Cheap GPU plumbing test for the anchorflow torch path (no SVD/TRELLIS).

Validates end-to-end on a placeholder Gaussian cloud:
  rollout (grad) -> LBS warp -> covariance -> reg losses -> backward,
checking shapes, finiteness, and that gradients actually reach the GNN weights
AND the actuation latents z_i. This is the first thing to run on the cheap GPU
stage — it catches the majority of integration bugs before any SVD cost.

    PYTHONPATH=lib python exe/smoke_render.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))

import torch

from anchorflow.anchors import AnchorSet
from anchorflow.dynamics import GNSDynamics, rollout
from anchorflow import warp as W
from anchorflow import reg as R


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    print(f"[smoke] device={dev}")

    # placeholder Gaussian cloud (a unit-ish blob)
    N = 4000
    xyz = torch.randn(N, 3, device=dev) * 0.3
    cov6 = torch.zeros(N, 6, device=dev)
    cov6[:, 0] = cov6[:, 3] = cov6[:, 5] = 0.01           # small isotropic

    anchors, _ = AnchorSet.from_gaussians(xyz, node_num=128, latent_dim=8, K=4)
    anchors = anchors.to(dev)
    w_bind, idx_bind = anchors.cal_nn_weight(xyz)
    conn_idx, conn_w = R.connectivity(anchors.canonical, K=8)
    fixed = torch.zeros(anchors.num, dtype=torch.bool, device=dev)

    gnn = GNSDynamics(hidden=64, message_passing_steps=4, latent_dim=8).to(dev)
    T = 8
    gcfg = {"graph": "knn", "k": 4, "rebuild_graph": False}

    node_seq = rollout(gnn, anchors.canonical, anchors.canonical, fixed,
                       steps=T - 2, cfg=gcfg, z=anchors.z, grad=True)
    assert node_seq.shape == (T, anchors.num, 3), node_seq.shape
    assert torch.isfinite(node_seq).all(), "rollout produced non-finite"
    print(f"[smoke] rollout ok {tuple(node_seq.shape)} "
          f"motion={float((node_seq[-1]-node_seq[0]).norm(dim=-1).mean()):.4f}")

    pos_all = []
    for t in range(T):
        Rk = W.anchor_rotations(anchors.canonical, node_seq[t])
        p, c6, rot = W.lbs_warp(xyz, cov6, w_bind, idx_bind,
                                anchors.canonical, node_seq[t], Rk)
        assert p.shape == (N, 3) and c6.shape == (N, 6) and rot.shape == (N, 3, 3)
        assert torch.isfinite(p).all() and torch.isfinite(c6).all(), f"nan @frame {t}"
        pos_all.append(p)
    print(f"[smoke] LBS warp ok, {T} frames, pos/cov/rot finite")

    reg = R.total(node_seq, conn_idx, conn_w)
    assert torch.isfinite(reg), "reg non-finite"
    # dummy image-space-like loss on warped positions to exercise full backward
    loss = torch.stack(pos_all).pow(2).mean() + reg
    loss.backward()

    g_gnn = [p.grad for p in gnn.parameters() if p.grad is not None]
    g_z = anchors.z.grad
    assert g_gnn and all(torch.isfinite(g).all() for g in g_gnn), "GNN grad bad"
    assert g_z is not None and torch.isfinite(g_z).all(), "z_i grad bad"
    print(f"[smoke] backward ok: reg={float(reg):.4e} "
          f"gnn_grad={sum(g.abs().sum() for g in g_gnn):.3e} "
          f"z_grad={float(g_z.abs().sum()):.3e}")
    print("[smoke] PASS — anchorflow torch path is sound on GPU")


if __name__ == "__main__":
    main()

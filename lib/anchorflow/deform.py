"""Anchor -> Gaussian deformation (Priority 2 foundation).

Binds each Gaussian to its k nearest anchors at rest, then drives the Gaussians
from the anchors' per-frame motion.  Two modes:

    "translation"  Gaussian moves by the weighted average anchor translation.
                   Cheap, no rotation of the Gaussian covariance.
    "affine"       Estimate a local affine/rigid transform per Gaussian from the
                   displacement of its bound anchors (least-squares), and apply
                   it to both the centre and (rotation part) the covariance.
                   This is the SC-GS / LBS-with-local-frame style.

The binding is computed once from the rest pose; per frame you only pass the new
anchor positions.  Weights are inverse-distance with a softmax temperature.

This module is deliberately standalone from any specific 3DGS codebase: it maps
[G,3] Gaussian centres (and optional [G,3,3] covariances) -> deformed versions,
which the renderer of your choice then consumes.
"""

from __future__ import annotations

import torch

from . import graph as G


class AnchorBinding:
    """Precomputed skinning weights of Gaussians to anchors."""

    def __init__(self, gauss_rest, anchor_rest, k=4, temp=0.1):
        self.k = min(k, anchor_rest.shape[0])
        d = torch.cdist(gauss_rest, anchor_rest)               # [G, A]
        dist, idx = d.topk(self.k, largest=False)              # [G, k]
        w = torch.softmax(-dist / max(temp, 1e-6), dim=-1)     # inverse-dist weights
        self.idx = idx                                         # [G, k]
        self.w = w                                             # [G, k]
        self.gauss_rest = gauss_rest
        self.anchor_rest = anchor_rest
        # rest offset of each Gaussian from each bound anchor
        self.offset = gauss_rest[:, None, :] - anchor_rest[idx]   # [G, k, 3]

    # --- translation LBS --------------------------------------------------- #
    def deform_translation(self, anchor_now):
        disp = anchor_now[self.idx] - self.anchor_rest[self.idx]   # [G, k, 3]
        delta = (self.w[..., None] * disp).sum(1)                  # [G, 3]
        return self.gauss_rest + delta

    # --- local rigid/affine LBS ------------------------------------------- #
    def deform_affine(self, anchor_now, cov_rest=None, rigid=True):
        """Weighted local transform per Gaussian from bound-anchor motion.

        Solves for R,t minimising sum_k w_k || R a_k^rest + t - a_k^now ||^2
        (weighted Procrustes).  Applies it to the Gaussian centre; if cov_rest
        is given, rotates the covariance by R.
        """
        A_rest = self.anchor_rest[self.idx]                       # [G, k, 3]
        A_now = anchor_now[self.idx]                              # [G, k, 3]
        w = self.w[..., None]                                     # [G, k, 1]
        mu_r = (w * A_rest).sum(1) / w.sum(1)                     # [G, 3]
        mu_n = (w * A_now).sum(1) / w.sum(1)
        X = A_rest - mu_r[:, None]
        Y = A_now - mu_n[:, None]
        H = torch.einsum("gki,gkj->gij", w * X, Y)               # [G, 3, 3]
        U, S, Vt = torch.linalg.svd(H)
        R = torch.einsum("gij,gjk->gik", Vt.transpose(-1, -2), U.transpose(-1, -2))
        if rigid:                                                # fix reflections
            det = torch.linalg.det(R)
            flip = torch.ones_like(S)
            flip[:, -1] = det.sign()
            R = torch.einsum("gij,gj,gkj->gik", Vt.transpose(-1, -2), flip,
                             U.transpose(-1, -2))
        centre = torch.einsum("gij,gj->gi", R, self.gauss_rest - mu_r) + mu_n
        if cov_rest is None:
            return centre
        cov = torch.einsum("gij,gjk,glk->gil", R, cov_rest, R)   # R C R^T
        return centre, cov


def bind(gauss_rest, anchor_rest, k=4, temp=0.1):
    return AnchorBinding(gauss_rest, anchor_rest, k=k, temp=temp)


def extract_anchors_fps(gauss_centres, n_anchors, seed=0):
    """Farthest-point sampling of anchor centres from Gaussian centres.

    Returns the anchor positions [n_anchors, 3] and their indices into the
    Gaussian set.  Pure torch, O(n_anchors * G).
    """
    G_ = gauss_centres.shape[0]
    n_anchors = min(n_anchors, G_)
    dev = gauss_centres.device
    gen = torch.Generator(device="cpu").manual_seed(seed)
    first = torch.randint(0, G_, (1,), generator=gen).item()
    chosen = [first]
    dist = torch.full((G_,), float("inf"), device=dev)
    for _ in range(n_anchors - 1):
        d = (gauss_centres - gauss_centres[chosen[-1]]).norm(dim=-1)
        dist = torch.minimum(dist, d)
        chosen.append(int(dist.argmax()))
    idx = torch.tensor(chosen, device=dev)
    return gauss_centres[idx], idx

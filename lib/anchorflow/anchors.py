"""Control-node (anchor) state + SC-GS-faithful RBF-LBS binding.

Supersedes deform.py's simple softmax binding for the generation pipeline.
Faithful to SC-GS ControlNodeWarp.cal_nn_weight:
    KNN (squared dist) -> w = exp(-d2 / (2*radius[idx]^2)) * node_weight[idx],
    normalized over the K neighbours.

Node state:
    canonical   [M,3]   rest positions (buffer; the GNN moves a separate state)
    _radius     [M]     learnable log-radius (RBF falloff)   -> radius = exp
    _node_weight[M]     learnable logit                      -> weight = sigmoid
    z           [M,L]   actuation latent (internal-drive signal, SDS-optimised)
"""

from __future__ import annotations

import torch
import torch.nn as nn


def knn(x, nodes, K):
    """K nearest `nodes` for each `x`. Returns (dist2 [N,K], idx [N,K]).

    dist2 is the SQUARED euclidean distance (matches pytorch3d.knn_points /
    SC-GS's RBF and radius**2 conventions)."""
    K = min(K, nodes.shape[0])
    d2 = torch.cdist(x, nodes) ** 2
    dist2, idx = torch.topk(d2, K, dim=-1, largest=False)
    return dist2, idx


def fps(points, n, seed=0):
    """Farthest-point sampling -> indices [n] into `points`."""
    N = points.shape[0]
    n = min(n, N)
    dev = points.device
    gen = torch.Generator(device="cpu").manual_seed(seed)
    chosen = [int(torch.randint(0, N, (1,), generator=gen).item())]
    dist = torch.full((N,), float("inf"), device=dev)
    for _ in range(n - 1):
        d = (points - points[chosen[-1]]).norm(dim=-1)
        dist = torch.minimum(dist, d)
        chosen.append(int(dist.argmax()))
    return torch.tensor(chosen, device=dev, dtype=torch.long)


class AnchorSet(nn.Module):
    def __init__(self, canonical, latent_dim=8, e_dim=8, K=4, radius_init=None):
        super().__init__()
        M = canonical.shape[0]
        self.K = K
        self.register_buffer("canonical", canonical.clone())
        if radius_init is None:
            rng = (canonical.max(0).values - canonical.min(0).values).norm()
            radius_init = 0.1 * float(rng) + 1e-7
        self._radius = nn.Parameter(torch.full((M,), float(torch.log(torch.tensor(radius_init)))))
        self._node_weight = nn.Parameter(torch.zeros(M))
        # z : actuation/control latent (varied per-IC during MDS)
        self.z = nn.Parameter(0.01 * torch.randn(M, latent_dim)) if latent_dim > 0 \
            else None
        # e : intrinsic per-anchor identity embedding (learned, FIXED across ICs —
        #     "what this anchor is": which part / joint-vs-rigid / material-like)
        self.e = nn.Parameter(0.01 * torch.randn(M, e_dim)) if e_dim > 0 else None

    @property
    def radius(self):
        return torch.exp(self._radius)

    @property
    def node_weight(self):
        return torch.sigmoid(self._node_weight)

    @property
    def num(self):
        return self.canonical.shape[0]

    def cal_nn_weight(self, x, K=None):
        """RBF-LBS weights of Gaussians `x` [N,3] to control nodes.

        Returns (w [N,K], idx [N,K]) with w summing to 1 over K. Weights use the
        *canonical* node positions (the binding is fixed at rest, SC-GS style)."""
        K = K or self.K
        dist2, idx = knn(x, self.canonical, K)              # squared dist
        r = self.radius[idx]                                # [N,K]
        w = torch.exp(-dist2 / (2 * r ** 2)) * self.node_weight[idx]
        w = w + 1e-7
        w = w / w.sum(dim=-1, keepdim=True)
        return w, idx

    @classmethod
    def from_gaussians(cls, gaussian_xyz, node_num=512, latent_dim=8, e_dim=8, K=4, seed=0):
        idx = fps(gaussian_xyz, node_num, seed=seed)
        return cls(gaussian_xyz[idx].detach(), latent_dim=latent_dim, e_dim=e_dim, K=K), idx

    @classmethod
    def from_trajectory(cls, canonical, latent_dim=8, e_dim=8, K=4):
        """Anchors whose canonical positions come from a MoSca scaffold (not FPS)."""
        return cls(canonical.detach(), latent_dim=latent_dim, e_dim=e_dim, K=K)

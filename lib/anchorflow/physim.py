"""Differentiable mass-spring simulator on the anchor graph.

Each anchor is a mass point.  Spring stiffness is per-node (k_i); the
per-edge stiffness is the mean of the two endpoint values.  Damping is a
single global scalar.  Euler integration runs for T steps.

Learnable parameters
    _log_stiffness  [N]  per-node spring stiffness (softplus activation)
    _log_damping    []   global viscous damping coefficient (softplus)

External force
    f_ext  [3]  uniform body force applied to every node (wind, gravity …).
    Sampled randomly per training step; magnitude and direction are the
    "conditioning signal" analogous to SeqGen's cond_vel.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph import knn_graph


class SpringSim(nn.Module):
    def __init__(self, canonical: torch.Tensor, T: int = 25,
                 dt: float = 0.04, K: int = 6, mass: float = 1.0,
                 stiffness_init: float = 10.0, damping_init: float = 1.0):
        """
        canonical   [N, 3]  rest positions (registered as buffer, not learned)
        T                   number of frames to simulate (including frame 0)
        dt                  Euler timestep
        K                   knn degree for the spring graph
        mass                scalar node mass (fixed)
        """
        super().__init__()
        N = canonical.shape[0]
        self.T = T
        self.dt = dt
        self.mass = mass

        self.register_buffer("canonical", canonical.clone())

        # spring graph (fixed topology from rest config)
        edge_index = knn_graph(canonical.detach(), K)   # [2, E]
        self.register_buffer("edge_index", edge_index)

        src, dst = edge_index
        rest_len = (canonical[src] - canonical[dst]).norm(dim=-1)
        self.register_buffer("rest_lengths", rest_len)  # [E]

        # learnable params
        self._log_stiffness = nn.Parameter(
            torch.full((N,), float(torch.log(torch.tensor(stiffness_init)))))
        self._log_damping = nn.Parameter(
            torch.tensor(float(torch.log(torch.tensor(damping_init)))))

    @property
    def stiffness(self) -> torch.Tensor:
        return F.softplus(self._log_stiffness)              # [N] > 0

    @property
    def damping(self) -> torch.Tensor:
        return F.softplus(self._log_damping)                # scalar > 0

    def forward(self, f_ext: torch.Tensor) -> torch.Tensor:
        """Simulate under uniform external force f_ext [3].

        Returns trajectory [T, N, 3] where traj[0] == canonical.
        """
        src, dst = self.edge_index                          # [E] each
        k = self.stiffness                                  # [N]
        c = self.damping                                    # scalar
        L = self.rest_lengths                               # [E]

        # per-edge stiffness: mean of the two endpoint values
        k_edge = (k[src] + k[dst]) * 0.5                   # [E]

        x = self.canonical.clone()
        v = torch.zeros_like(x)
        traj = [x]

        for _ in range(self.T - 1):
            xi = x[src]                                     # [E, 3]
            xj = x[dst]                                     # [E, 3]
            diff = xj - xi                                  # [E, 3]
            dist = diff.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            direction = diff / dist                         # [E, 3] unit vec

            # Hookean spring: F = k * (|xj-xi| - L_rest) * direction
            stretch = dist.squeeze(-1) - L                  # [E]
            F_spring = k_edge[:, None] * stretch[:, None] * direction  # [E, 3]

            # accumulate spring forces (Newton's 3rd law)
            F = torch.zeros_like(x)
            F = F.scatter_add(0, src[:, None].expand_as(F_spring), -F_spring)
            F = F.scatter_add(0, dst[:, None].expand_as(F_spring),  F_spring)

            # body force + viscous damping
            F = F + f_ext.unsqueeze(0) - c * v

            # Euler step
            v = v + (self.dt / self.mass) * F
            x = x + self.dt * v
            traj.append(x)

        return torch.stack(traj, dim=0)                     # [T, N, 3]

    def save(self, path: str):
        torch.save({
            "state_dict": self.state_dict(),
            "T": self.T, "dt": self.dt, "mass": self.mass,
        }, path)

    @classmethod
    def load(cls, path: str, canonical: torch.Tensor, **kw) -> "SpringSim":
        ck = torch.load(path, map_location="cpu")
        obj = cls(canonical, T=ck["T"], dt=ck["dt"], mass=ck["mass"], **kw)
        obj.load_state_dict(ck["state_dict"])
        return obj

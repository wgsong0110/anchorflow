"""Cubic Hermite spline trajectory for T2N-style node deformation.

Each node follows a cubic Hermite spline over K keyframes:
    ξ(t) = h00(τ)·P_k + h10(τ)·Δt·Ṗ_k + h01(τ)·P_{k+1} + h11(τ)·Δt·Ṗ_{k+1}
where τ = (t - t_k) / (t_{k+1} - t_k) ∈ [0, 1].
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class CubicHermiteTrajectory(nn.Module):
    """Per-node cubic Hermite spline trajectory.

    Parameters (learned):
        P  [K, M, 3]  - keyframe positions
        Pd [K, M, 3]  - keyframe velocities (tangents)
    """

    def __init__(self, canonical: Tensor, K: int, T: int):
        super().__init__()
        M = canonical.shape[0]
        self.K = K
        self.T = T
        # Initialise all keyframes at canonical; velocities zero.
        self.P  = nn.Parameter(canonical.detach().unsqueeze(0).expand(K, -1, -1).clone())
        self.Pd = nn.Parameter(torch.zeros(K, M, 3))
        # Uniform keyframe times in [0, T-1]
        self.register_buffer("ts", torch.linspace(0, T - 1, K))

    # ------------------------------------------------------------------
    def _hermite(self, tau: Tensor, k: int) -> Tensor:
        """Evaluate spline at normalised time tau ∈ [0,1] within segment k."""
        tau2, tau3 = tau ** 2, tau ** 3
        h00 =  2*tau3 - 3*tau2 + 1
        h10 =    tau3 - 2*tau2 + tau
        h01 = -2*tau3 + 3*tau2
        h11 =    tau3 -   tau2
        dt = float(self.ts[k + 1] - self.ts[k])
        return (h00.view(-1, 1, 1) * self.P[k]
              + h10.view(-1, 1, 1) * (dt * self.Pd[k])
              + h01.view(-1, 1, 1) * self.P[k + 1]
              + h11.view(-1, 1, 1) * (dt * self.Pd[k + 1]))

    def forward(self, t: float | int) -> Tensor:
        """Return node positions [M, 3] at continuous time t."""
        ts = self.ts
        K = self.K
        # clamp to valid range
        t = float(max(float(ts[0]), min(float(ts[-1]), t)))
        # find segment k such that ts[k] <= t < ts[k+1]
        k = int(((t - float(ts[0])) / (float(ts[-1]) - float(ts[0]) + 1e-8)) * (K - 1))
        k = max(0, min(K - 2, k))
        while k < K - 2 and float(ts[k + 1]) <= t:
            k += 1
        t0, t1 = float(ts[k]), float(ts[k + 1])
        tau = torch.tensor([(t - t0) / max(t1 - t0, 1e-8)],
                           device=self.P.device, dtype=self.P.dtype)
        return self._hermite(tau, k).squeeze(0)  # [M, 3]

    def all_positions(self) -> Tensor:
        """Evaluate at all integer frames. Returns [T, M, 3]."""
        return torch.stack([self.forward(t) for t in range(self.T)], dim=0)

    # ------------------------------------------------------------------
    def init_from_tracklets(self, tracklets_3d: Tensor):
        """Least-squares initialise P from 3-D tracklets [T, M, 3].

        Fits keyframe positions via normal equations; sets velocities to
        finite-difference estimates at keyframe times.
        """
        T, M, _ = tracklets_3d.shape
        ts = self.ts  # [K]
        K = self.K
        dev = self.P.device
        x = tracklets_3d.to(dev)  # [T, M, 3]

        # Build Hermite design matrix H [T, K] at integer times 0..T-1
        t_all = torch.arange(T, device=dev, dtype=torch.float32)
        H_pos = torch.zeros(T, K, device=dev)
        H_vel = torch.zeros(T, K, device=dev)
        for k in range(K - 1):
            t0, t1 = float(ts[k]), float(ts[k + 1])
            dt = t1 - t0
            mask = (t_all >= t0) & (t_all < (t1 + (1 if k == K - 2 else 0)))
            if not mask.any():
                continue
            tau = (t_all[mask] - t0) / dt
            tau2, tau3 = tau ** 2, tau ** 3
            h00 =  2*tau3 - 3*tau2 + 1
            h10 = (   tau3 - 2*tau2 + tau) * dt
            h01 = -2*tau3 + 3*tau2
            h11 = (   tau3 -   tau2) * dt
            H_pos[mask, k]     += h00
            H_pos[mask, k + 1] += h01
            H_vel[mask, k]     += h10
            H_vel[mask, k + 1] += h11

        # Combined design matrix [T, 2K]
        H = torch.cat([H_pos, H_vel], dim=1)  # [T, 2K]
        # Solve least-squares for each node dim
        # params [2K, M, 3]
        params, _ = torch.linalg.lstsq(H, x.reshape(T, M * 3))[:2]
        params = params.reshape(2 * K, M, 3)
        with torch.no_grad():
            self.P.copy_(params[:K])
            self.Pd.copy_(params[K:])

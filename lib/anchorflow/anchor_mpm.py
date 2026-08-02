"""Grid-free, anchor-based elastodynamics: PhysGaussian's MPM with the
Eulerian background grid replaced by the sparse (Lagrangian) anchor set.

Anchors carry ONLY (position, velocity) -- no learned rotation/state. Per-hop
the deformation gradient F at every Gaussian is recovered from its K nearest
anchors via a weighted shape-matching (Procrustes-style affine) fit between
each anchor's REST and CURRENT position, exactly like Mueller et al. 2005
"Meshless Deformations Based on Shape Matching" generalized to a per-Gaussian
weighted neighborhood. The same fit also gives the Gaussian's current
position directly (no separate LBS blend needed).

Design choices (see anchorflow session notes, 2026-08-02):
  - Neighbor CONNECTIVITY (which K anchors influence a Gaussian) is fixed
    from canonical k-NN, computed once -- re-searching neighbors every step
    would be expensive and is not needed (large deformation is handled by
    the WEIGHTS adapting, not by connectivity changing).
  - Blend WEIGHTS are recomputed every step from actual current anchor
    positions (not frozen at canonical) -- this is what makes the method
    correct for large deformation. A distance-based kernel is automatically
    rotation-invariant (rotation preserves distances), so this doesn't
    reintroduce spurious strain under pure rotation; it only changes the
    weights when there's genuine local stretch, which is exactly wanted.
  - Because a Gaussian's OWN current position is circularly what the weight
    kernel would need, weights use the Gaussian's position from the END OF
    THE PREVIOUS STEP (one-step lag) -- standard explicit-integrator
    practice, matches the semi-implicit Euler used everywhere else in this
    codebase.
  - No internal elastic force is derived by hand: the constitutive law gives
    a scalar energy density per Gaussian; total elastic energy is summed and
    backpropagated (`torch.autograd.grad`) straight to anchor positions to
    get generalized forces -- no manual stress-divergence-to-anchor
    distribution formula needed.
"""
from __future__ import annotations

import torch

from .anchors import knn


def _polar_decompose(F, iters=8, eps=1e-6):
    """F = R @ S, R orthogonal (proper rotation, det=+1), S symmetric PSD.

    Newton-Schulz / Higham iteration (R_{k+1} = 0.5*(R_k + inv(R_k)^T)) instead
    of SVD -- torch.linalg.svd's backward is NaN at repeated singular values
    (near-isotropic F, e.g. at rest / near-rigid motion), which is exactly the
    common case here (F starts at/near I every rest configuration). The
    Newton iteration only needs matrix inverse, which has a well-behaved
    backward away from singular F (elastic F shouldn't be singular before the
    sim has already blown up). F: [...,3,3] -> R,S: [...,3,3]."""
    R = F
    eye = torch.eye(3, device=F.device, dtype=F.dtype).expand_as(F)
    for _ in range(iters):
        R_reg = R + eps * eye
        R = 0.5 * (R + torch.linalg.inv(R_reg).transpose(-1, -2))
    # proper-rotation fixup: flip the sign of R's worst-conditioned axis
    # wherever det(R) < 0 (reflection) -- rare for well-behaved elastic F,
    # but cheap to guard against.
    det = torch.linalg.det(R)
    flip = (det < 0).to(F.dtype)
    fix = eye.clone()
    fix[..., 2, 2] = 1.0 - 2.0 * flip
    R = R @ fix
    S = R.transpose(-2, -1) @ F
    S = 0.5 * (S + S.transpose(-2, -1))
    return R, S


def fixed_corotated_energy_density(F, mu, lam):
    """Per-particle Fixed Corotated elastic energy density (PhysGaussian /
    DreamPhysics's `kirchoff_stress_FCR` convention, integrated to a scalar
    potential): Psi(F) = mu*||F-R||_F^2 + (lam/2)*(J-1)^2.

    F: [N,3,3], mu/lam: scalar or [N] -> Psi: [N]."""
    R, _ = _polar_decompose(F)
    J = torch.linalg.det(F)
    dev = (F - R)
    frob2 = (dev * dev).sum(dim=(-2, -1))
    return mu * frob2 + 0.5 * lam * (J - 1.0) ** 2


def lame_from_E_nu(E, nu, scale=1e7):
    """DreamPhysics/PhysGaussian's exact scaling convention
    (`compute_mu_lam_from_E_nu`, mpm_utils.py) so fitted ficus E/nu values
    (config/physics/ficus_config.json) plug in directly."""
    mu = scale * E / (2.0 * (1.0 + nu))
    lam = scale * E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return mu, lam


class AnchorElasticSim:
    """Grid-free anchor elastodynamics driver.

    canonical Gaussian/anchor positions are fixed at construction (rest
    configuration). All physics state (anchor position/velocity, last-known
    Gaussian position for the weight lag) is passed in/out of `step()`
    explicitly rather than held as module state, so this composes cleanly
    with whatever owns the render loop.
    """

    def __init__(self, gaussian_canonical, anchor_canonical, K=8, kernel_eps=1e-4):
        """gaussian_canonical [N,3], anchor_canonical [M,3]."""
        self.K = min(K, anchor_canonical.shape[0])
        self.kernel_eps = kernel_eps
        self.gaussian_canonical = gaussian_canonical
        self.anchor_canonical = anchor_canonical
        # connectivity fixed once from canonical k-NN (see module docstring)
        _, idx = knn(gaussian_canonical, anchor_canonical, self.K)   # [N,K]
        self.nn_idx = idx
        self.anchor_nbr = anchor_canonical[idx]                      # [N,K,3] rest

    def _weights(self, gaussian_pos_prev, anchor_pos):
        """Distance kernel between each Gaussian's LAST-STEP position and its
        (fixed) candidate anchors' CURRENT positions -- recomputed every
        step, see module docstring for why this is both correct and cheap.
        gaussian_pos_prev [N,3], anchor_pos [M,3] -> w [N,K] (sums to 1)."""
        nbr_cur = anchor_pos[self.nn_idx]                            # [N,K,3]
        d = (gaussian_pos_prev.unsqueeze(1) - nbr_cur).norm(dim=-1)  # [N,K]
        w = 1.0 / (d + self.kernel_eps)
        return w / w.sum(dim=-1, keepdim=True)

    def _shape_match(self, anchor_pos, w):
        """Weighted Procrustes affine fit per Gaussian -> (F [N,3,3],
        gaussian_pos [N,3]). See module docstring / derivation: exact for
        pure rigid motion (F=R, no spurious strain) by construction."""
        nbr_rest = self.anchor_nbr                                   # [N,K,3]
        nbr_cur = anchor_pos[self.nn_idx]                            # [N,K,3]
        w_ = w.unsqueeze(-1)                                         # [N,K,1]
        rest_centroid = (w_ * nbr_rest).sum(dim=1)                   # [N,3]
        cur_centroid = (w_ * nbr_cur).sum(dim=1)                     # [N,3]
        q = nbr_rest - rest_centroid.unsqueeze(1)                    # [N,K,3]
        p = nbr_cur - cur_centroid.unsqueeze(1)                      # [N,K,3]
        # A = sum_k w_k p_k (x) q_k ; B = sum_k w_k q_k (x) q_k ; F = A @ B^-1
        A = torch.einsum("nk,nki,nkj->nij", w, p, q)                 # [N,3,3]
        B = torch.einsum("nk,nki,nkj->nij", w, q, q)                 # [N,3,3]
        # With only K neighbors, B is a scatter/covariance matrix of just K
        # points around their own weighted centroid -- for a Gaussian whose
        # nearest anchors happen to lie close to a plane/line (common with
        # small K on a sparse irregular anchor cloud), B is ill-conditioned
        # and inv(B) blows up along the flat direction (verified: exact
        # match to a ground-truth rotation for a well-conditioned B, ~15-20%
        # spurious stretch injected under PURE RIGID ROTATION for a
        # poorly-conditioned one -- same formula, different B). A uniform
        # (trace-based) Tikhonov regularizer was tried first and made things
        # WORSE for well-conditioned B (biases every direction, not just the
        # flat one) while still under-fixing the truly degenerate case.
        # Correct fix: eigenvalue clamping -- only inflate the SPECIFIC
        # flat direction(s), leave well-conditioned eigenvalues untouched.
        eigval, eigvec = torch.linalg.eigh(B)                        # ascending, B PSD
        lambda_max = eigval[..., -1:].clamp(min=1e-12)
        eigval_clamped = torch.clamp(eigval, min=0.05 * lambda_max)
        B_reg = eigvec @ torch.diag_embed(eigval_clamped) @ eigvec.transpose(-1, -2)
        F = A @ torch.linalg.inv(B_reg)
        gaussian_pos = cur_centroid + torch.einsum(
            "nij,nj->ni", F, self.gaussian_canonical - rest_centroid)
        return F, gaussian_pos

    def elastic_energy(self, anchor_pos, gaussian_pos_prev, gaussian_volume, mu, lam):
        """Total Fixed-Corotated elastic energy as a function of anchor
        positions (weights/connectivity are constants w.r.t. this
        differentiation -- see module docstring on the lagged weight
        convention). Returns (E scalar, F [N,3,3], gaussian_pos [N,3])."""
        w = self._weights(gaussian_pos_prev, anchor_pos)
        F, gaussian_pos = self._shape_match(anchor_pos, w)
        psi = fixed_corotated_energy_density(F, mu, lam)             # [N]
        E = (gaussian_volume * psi).sum()
        return E, F, gaussian_pos

    def step(self, anchor_pos, anchor_vel, anchor_mass, gaussian_pos_prev,
              gaussian_volume, mu, lam, dt, f_ext_particle=None, gravity=None,
              damping=1.0):
        """One semi-implicit Euler step.

        anchor_pos/vel/mass: [M,3]/[M,3]/[M]. gaussian_pos_prev: [N,3] (last
        step's Gaussian positions, for the weight lag). gaussian_volume:
        [N] (rest volume). mu/lam: scalar or [N] Lame params. f_ext_particle:
        optional [N,3] per-Gaussian external force (e.g. wind drag),
        P2G-scattered onto anchors with the SAME (lagged) weights used for
        the elastic energy. gravity: optional [3].

        Returns (anchor_pos', anchor_vel', gaussian_pos', F [N,3,3])."""
        anchor_pos = anchor_pos.detach().requires_grad_(True)
        E, F, gaussian_pos = self.elastic_energy(
            anchor_pos, gaussian_pos_prev, gaussian_volume, mu, lam)
        (f_elastic,) = torch.autograd.grad(E, anchor_pos, create_graph=False)
        f_elastic = -f_elastic

        f_total = f_elastic
        if f_ext_particle is not None:
            w = self._weights(gaussian_pos_prev, anchor_pos.detach())
            weighted = f_ext_particle.unsqueeze(1) * w.unsqueeze(-1)      # [N,K,3]
            f_ext_anchor = torch.zeros_like(anchor_pos)
            f_ext_anchor = f_ext_anchor.index_add(
                0, self.nn_idx.reshape(-1), weighted.reshape(-1, 3))
            f_total = f_total + f_ext_anchor
        if gravity is not None:
            f_total = f_total + anchor_mass.unsqueeze(-1) * gravity

        anchor_vel_next = damping * anchor_vel + dt * f_total / anchor_mass.unsqueeze(-1)
        anchor_pos_next = anchor_pos.detach() + dt * anchor_vel_next
        return anchor_pos_next, anchor_vel_next, gaussian_pos.detach(), F.detach()

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
    positions rather than frozen at canonical. A distance-based kernel is
    rotation-invariant, so this changes the weights only where there is
    genuine local stretch. This line used to claim the adaptation is what
    makes the method correct for large deformation; measured against the
    same simulation with the weights frozen at canonical, it moves the
    anchors by 0.24% of peak motion over 60 coarse steps at the config's
    impulse and 1.4% at three times it (exe/render_weight_convention.py).
    Real, and larger the more the object deforms, but a small correction --
    not what the method rests on.
  - Because a Gaussian's OWN current position is circularly what the weight
    kernel would need, weights use the Gaussian's position from the END OF
    THE PREVIOUS STEP (one-step lag) -- standard explicit-integrator
    practice, matches the semi-implicit Euler used everywhere else in this
    codebase. The cost is that the simulator is then not a function of its
    state: the same anchor positions and velocities step differently
    depending on the history that produced the cloud the weights are read
    from. Freezing the weights at canonical would remove that, and is the
    reason to consider it -- not accuracy.
  - No internal elastic force is derived by hand: the constitutive law gives
    a scalar energy density per Gaussian; total elastic energy is summed and
    backpropagated (`torch.autograd.grad`) straight to anchor positions to
    get generalized forces -- no manual stress-divergence-to-anchor
    distribution formula needed.
"""
from __future__ import annotations

import torch

from .anchors import knn
from eigen3x3 import eigh3x3


def _polar_decompose(F, eps=1e-8):
    """F = R @ S, R orthogonal, S symmetric PSD -- closed form via the
    (custom CUDA, verified) eigh3x3 kernel instead of the original 8-round
    Newton-Schulz iteration: the Newton loop cost 8 batched
    torch.linalg.inv launches PLUS their full autograd graph per energy
    evaluation, which profiling-by-elimination showed dominated the physics
    step (the user asked for custom-kernel optimization; eigh3x3 was already
    swapped into _shape_match, this was the remaining LAPACK hot spot).

    Closed form: F^T F is symmetric PSD -> eigh3x3 gives V, lambda;
    S = V diag(sqrt(lambda)) V^T,  S^{-1} = V diag(1/sqrt(lambda)) V^T,
    R = F @ S^{-1}. Degenerate/tiny eigenvalues are clamped so R falls back
    smoothly rather than dividing by ~0 (same spirit as eigh3x3's own
    scale-free guards). F: [...,3,3] -> R,S: [...,3,3]."""
    FtF = F.transpose(-2, -1) @ F
    eigval, eigvec = eigh3x3(FtF.reshape(-1, 3, 3))
    eigval = eigval.clamp(min=eps)
    sqrt_l = eigval.sqrt()
    S = eigvec @ torch.diag_embed(sqrt_l) @ eigvec.transpose(-1, -2)
    S_inv = eigvec @ torch.diag_embed(1.0 / sqrt_l) @ eigvec.transpose(-1, -2)
    R = F.reshape(-1, 3, 3) @ S_inv
    return R.reshape(F.shape), S.reshape(F.shape)


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

    def __init__(self, gaussian_canonical, anchor_canonical, K=8, radius=None):
        """gaussian_canonical [N,3], anchor_canonical [M,3]. radius: RBF
        falloff for the blend weights; defaults to the mean canonical
        Gaussian->neighbor-anchor distance (a natural scene-scale estimate)."""
        self.K = min(K, anchor_canonical.shape[0])
        self.gaussian_canonical = gaussian_canonical
        self.anchor_canonical = anchor_canonical
        # connectivity fixed once from canonical k-NN (see module docstring)
        dist2, idx = knn(gaussian_canonical, anchor_canonical, self.K)   # [N,K]
        self.nn_idx = idx
        self.anchor_nbr = anchor_canonical[idx]                      # [N,K,3] rest
        # RBF radius: an inverse-distance kernel (1/(d+eps), used originally)
        # COLLAPSES whenever a Gaussian nearly coincides with one of its
        # anchors -- measured on real ficus data, such Gaussians got weights
        # like [0.993, 0.001, ...], which makes the shape-matching scatter
        # matrix B degenerate (all neighbor offsets measured from a centroid
        # sitting on top of one anchor) and F ill-posed. A Gaussian RBF with
        # a scene-scale radius (same form as AnchorSet.cal_nn_weight, the
        # convention used elsewhere in this codebase) stays smooth and keeps
        # all K neighbors meaningfully weighted.
        self.radius = float(dist2.clamp(min=0).sqrt().mean()) if radius is None else float(radius)
        self.radius = max(self.radius, 1e-8)

    def _weights(self, gaussian_pos_prev, anchor_pos):
        """RBF kernel between each Gaussian's LAST-STEP position and its
        (fixed) candidate anchors' CURRENT positions -- recomputed every
        step, see module docstring for why this is both correct and cheap.
        gaussian_pos_prev [N,3], anchor_pos [M,3] -> w [N,K] (sums to 1)."""
        nbr_cur = anchor_pos[self.nn_idx]                            # [N,K,3]
        d2 = ((gaussian_pos_prev.unsqueeze(1) - nbr_cur) ** 2).sum(-1)  # [N,K]
        w = torch.exp(-d2 / (2.0 * self.radius ** 2)) + 1e-8
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
        # points around their own weighted centroid -- whenever those K
        # anchors don't spread out well in some direction (near-planar or
        # near-collinear local neighborhoods -- expected to be COMMON for a
        # branch-like structure such as ficus, where nearby anchors along a
        # twig are nearly collinear), B has a near-zero eigenvalue along
        # that direction: there is essentially NO DATA constraining how the
        # material stretches there. F = A @ B^-1 naively "solves" for that
        # direction anyway by dividing by a near-zero number, which
        # amplifies whatever noise exists in A into a huge, wrong strain --
        # confirmed experimentally: even mild B ill-conditioning (condition
        # ratio ~0.02, nowhere near numerically singular) was enough to
        # visibly break F under a PURE RIGID ROTATION test (no real strain
        # at all). Flooring/clamping B's small eigenvalues (tried first)
        # still lets A's noise leak into that direction, just scaled down --
        # not good enough (still ~30-45% det error at a 5% floor).
        #
        # Correct fix: for eigenvalue/eigenvector pairs (lambda_i, v_i) of B
        # below threshold, DON'T solve for F's action along v_i from data at
        # all -- explicitly set F @ v_i = v_i (identity: assume NO local
        # deformation along a direction with no supporting data). Above
        # threshold, use the normal data-driven F @ v_i = (A @ v_i) / lambda_i
        # (this is exactly what F = A @ B^-1 computes in B's eigenbasis).
        # Reconstruct F from its per-eigenvector action: F @ V = Fv  =>
        # F = Fv @ V^T (V orthogonal). This is a per-direction fallback, not
        # a blanket regularizer, so well-observed directions are untouched.
        eigval, eigvec = eigh3x3(B)                                   # ascending, B PSD; V's columns = v_i
        lambda_max = eigval[..., -1:].clamp(min=1e-12)
        well_observed = eigval > 0.2 * lambda_max                    # [N,3] bool, per v_i
        Av = A @ eigvec                                              # [N,3,3], column i = A @ v_i
        Fv_data = Av / eigval.clamp(min=1e-12).unsqueeze(-2)         # column i = (A@v_i)/lambda_i
        Fv_identity = eigvec                                        # column i = v_i (no deformation)
        Fv = torch.where(well_observed.unsqueeze(-2), Fv_data, Fv_identity)
        F = Fv @ eigvec.transpose(-1, -2)
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
              damping=1.0, fixed_mask=None, f_ext_anchor=None):
        """One semi-implicit Euler step.

        anchor_pos/vel/mass: [M,3]/[M,3]/[M]. gaussian_pos_prev: [N,3] (last
        step's Gaussian positions, for the weight lag). gaussian_volume:
        [N] (rest volume). mu/lam: scalar or [N] Lame params. f_ext_particle:
        optional [N,3] per-Gaussian external force (e.g. wind drag),
        P2G-scattered onto anchors with the SAME (lagged) weights used for
        the elastic energy. gravity: optional [3]. fixed_mask: optional
        [M] bool -- anchors held at their CURRENT position (zero velocity)
        every step, e.g. the pot/base region. Without this, gravity pulls
        every anchor including the base equally and the whole object just
        free-falls/drifts as a rigid body instead of the base staying put
        while only the flexible parts (branches) sway -- exactly what
        PhysGaussian's own boundary_conditions ("cuboid" pin regions in
        ficus_config.json) are for; this project skipped that piece
        initially, which is why the pot itself was moving.

        Returns (anchor_pos', anchor_vel', gaussian_pos', F [N,3,3])."""
        try:
            from anchorstep import fused_energy_force, HAVE_CUDA as _HAVE_FUSED
        except Exception:
            _HAVE_FUSED = False
        if _HAVE_FUSED and anchor_pos.is_cuda:
            # fused CUDA path: whole hot loop (weights, shape-match/eigh, F,
            # Fixed Corotated energy, ANALYTIC forces) in two kernels -- no
            # autograd graph. See lib/anchorstep. Falls through to the torch
            # reference below when unbuilt or on CPU. mu/lam may be scalars or
            # per-particle [N] tensors (region-varying material).
            anchor_pos = anchor_pos.detach()
            f_elastic, gaussian_pos, F, _psi = fused_energy_force(
                self.gaussian_canonical, gaussian_pos_prev, anchor_pos,
                self.anchor_canonical, self.nn_idx, gaussian_volume,
                self.radius, mu, lam)
        else:
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
        if f_ext_anchor is not None:
            # already-per-anchor external force (e.g. a distributed wind body
            # force); f_ext_particle above is the per-Gaussian P2G route.
            f_total = f_total + f_ext_anchor
        if gravity is not None:
            f_total = f_total + anchor_mass.unsqueeze(-1) * gravity

        anchor_vel_next = damping * anchor_vel + dt * f_total / anchor_mass.unsqueeze(-1)
        anchor_pos_next = anchor_pos.detach() + dt * anchor_vel_next
        if fixed_mask is not None:
            anchor_vel_next = torch.where(fixed_mask.unsqueeze(-1), torch.zeros_like(anchor_vel_next), anchor_vel_next)
            anchor_pos_next = torch.where(fixed_mask.unsqueeze(-1), anchor_pos.detach(), anchor_pos_next)
        return anchor_pos_next, anchor_vel_next, gaussian_pos.detach(), F.detach()

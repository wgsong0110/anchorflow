"""The anchor simulator with its discretisation as parameters, differentiably.

The simulator's anchors are placed by sampling and reach out isotropically to a
fixed radius. Neither choice knows anything about the material: on a branching
structure like ficus, an anchor sitting on a twig pulls in Gaussians from the
twig beside it, and the shape-matching fit then ties two pieces together that
bend independently. That shows up exactly where it should -- handed a state from
MPM's trajectory, the simulator agrees on where to go while the object is nearly
rigid (cosine 0.92 at frame 2) and points somewhere else entirely once it has
deformed (0.16 by frame 50, and no rescaling recovers it).

So make the discretisation trainable and fit it to MPM. Each anchor gets a
position, a rotation and three scales, which turn the isotropic kernel into a
Mahalanobis one: an anchor can be long along a branch and narrow across it.

Three things had to be built rather than reused.

The force, analytically. The fused CUDA kernel has no backward, and there is
nothing to differentiate through. Written out here it is short, because F is
linear in the anchor positions once the weights are fixed: the centroid term
cancels (the weighted rest offsets sum to zero by construction), leaving
f_k = -sum_i V_i w_ik (P_i Binv_i) q_ik.

The polar factor, stably. Fixed Corotated needs the rotation from F, and taking
it from an eigendecomposition of F^T F divides by the gap between eigenvalues --
which is zero wherever the motion is rigid, i.e. almost everywhere. The forward
value is fine and the gradient is NaN, which is what blocked every previous
attempt at fitting this simulator to anything. A scaled Newton iteration gets
the same factor out of matrix inverses alone.

The rest quantities, once. B, its eigendecomposition and the weights depend only
on the parameters, not on the state, so a rollout computes them once and reuses
them across substeps instead of per substep.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from eigen3x3 import eigh3x3


def polar_R(F, iters=8):
    """the rotation from F = R S, by scaled Newton iteration.

    R <- (gamma R + gamma^-1 R^-T) / 2, gamma = |det R|^(-1/3), which converges
    quadratically and touches only matrix inverses. The closed form via
    eigh(F^T F) is faster and its derivative divides by the difference between
    eigenvalues of F^T F -- zero under rigid motion, where this simulator spends
    most of its time, so its gradient is NaN exactly where it is needed.
    """
    R = F
    for _ in range(iters):
        Rinv_T = torch.linalg.inv(R).transpose(-1, -2)
        g = torch.linalg.det(R).abs().clamp(min=1e-12).pow(-1.0 / 3.0)
        g = g.unsqueeze(-1).unsqueeze(-1)
        R = 0.5 * (g * R + Rinv_T / g)
    return R


def quat_to_R(q):
    q = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=-1).reshape(*q.shape[:-1], 3, 3)


class AnchorFit(nn.Module):
    """anchor positions, orientations and extents as parameters of the physics.

    Connectivity -- which anchors a Gaussian belongs to -- is held fixed. It has
    to be: recomputing it as the anchors move makes the loss discontinuous, and
    a Gaussian changing hands mid-fit is a step in the objective that no
    gradient sees coming. It is refreshed between stages instead.
    """

    def __init__(self, sc, eig_floor=0.02, polar_iters=8, checkpoint_substeps=True):
        super().__init__()
        # a substep's graph is around 100 MB over this many Gaussians, mostly the
        # Newton iterates, and a coarse frame is 40 of them. Recomputing the
        # forward during the backward costs one extra evaluation and takes the
        # memory from linear in the rollout length to constant
        self.checkpoint_substeps = checkpoint_substeps
        sim = sc.sim
        self.eig_floor = eig_floor
        self.polar_iters = polar_iters
        # the opacity-rejected Gaussians carry zero volume, so they contribute
        # nothing to any force and only exist to be skinned. Dropping them here
        # is exact and removes a sixth of the work from every substep
        mat = torch.nonzero(sc.keep, as_tuple=False).squeeze(-1)
        self.register_buffer("mat", mat)
        self.register_buffer("Xc_all", sim.gaussian_canonical)
        self.register_buffer("idx_all", sim.nn_idx)
        self.register_buffer("Xc", sim.gaussian_canonical[mat])     # [Nm,3]
        self.register_buffer("idx", sim.nn_idx[mat])                # [Nm,K]
        self.register_buffer("vol", sc.volume[mat])
        self.register_buffer("mu", sc.mu[mat] if sc.mu.dim() else sc.mu)
        self.register_buffer("lam", sc.lam[mat] if sc.lam.dim() else sc.lam)
        self.register_buffer("fixed", sc.fixed_mask)
        self.register_buffer("dens_vol", sc.mass.new_zeros(mat.shape))
        self.M = sc.M
        self.dt = sc.sub_dt
        self.damping = sc.damping
        # the per-Gaussian mass that gets spread onto anchors. Taken from the
        # scene rather than recomputed, so a fitted anchor set redistributes the
        # same material instead of quietly creating some
        self.dens_vol.copy_((sc.volume * _density_of(sc))[mat])

        self.pos = nn.Parameter(sc.anchor_canonical.clone())
        self.log_s = nn.Parameter(torch.full((sc.M, 3), float(torch.log(
            torch.tensor(sim.radius))), device=sc.pos.device))
        q0 = torch.zeros(sc.M, 4, device=sc.pos.device)
        q0[:, 0] = 1.0
        self.quat = nn.Parameter(q0)

    @torch.no_grad()
    def init_from_geometry(self, spread=1.0):
        """orient and stretch each anchor along the material it holds.

        Isotropic scales make the kernel rotation-invariant, so the rotation has
        exactly zero gradient and never leaves its initial value -- the symmetry
        has to be broken by the initialisation rather than by the fit. The
        second moment of the Gaussians an anchor is responsible for is the
        natural thing to break it with: on a branch it comes out long along the
        branch and thin across it, which is the shape the whole exercise is
        about.

        The overall size is held at what the isotropic radius was, so this
        changes the shape of each anchor's reach and not how much it reaches.
        """
        w = self.weights()                                           # [N,K]
        flat = self.idx.reshape(-1)
        wf = w.reshape(-1)
        d = self.Xc.unsqueeze(1) - self.pos[self.idx]                # [N,K,3]
        df = d.reshape(-1, 3)
        tot = torch.zeros(self.M, device=w.device).index_add_(0, flat, wf).clamp(min=1e-12)
        C = torch.zeros(self.M, 3, 3, device=w.device).index_add_(
            0, flat, wf.reshape(-1, 1, 1) * (df.unsqueeze(-1) * df.unsqueeze(-2)))
        C = C / tot.reshape(-1, 1, 1)
        ev, evec = eigh3x3(C + 1e-12 * torch.eye(3, device=w.device))
        s = ev.clamp(min=1e-12).sqrt()
        # anchors holding almost nothing have a degenerate second moment and
        # would be initialised as slivers; those keep the isotropic radius
        thin = (ev[..., 0] < 1e-4 * ev[..., -1]) | (tot < 1e-8)
        s = s / s.mean(-1, keepdim=True).clamp(min=1e-12)             # shape only
        s = 1.0 + spread * (s - 1.0)
        s = s * self.log_s.exp().mean(-1, keepdim=True)
        s = torch.where(thin.unsqueeze(-1), self.log_s.exp(), s)
        self.log_s.copy_(s.clamp(min=1e-6).log())
        R = torch.where(thin.reshape(-1, 1, 1),
                        torch.eye(3, device=w.device).expand_as(evec), evec)
        self.quat.copy_(_R_to_quat(R))
        return int(thin.sum())

    # ---- the discretisation ------------------------------------------------
    def weights(self):
        """[N,K], normalised. Mahalanobis in each anchor's own frame, which is
        the isotropic kernel when the scales are equal and the rotation is
        identity -- the configuration this starts from."""
        a = self.pos[self.idx]                                       # [N,K,3]
        d = self.Xc.unsqueeze(1) - a
        R = quat_to_R(self.quat)[self.idx]                           # [N,K,3,3]
        s = self.log_s.exp()[self.idx]                               # [N,K,3]
        local = torch.einsum("nkji,nkj->nki", R, d) / s.clamp(min=1e-6)
        w = torch.exp(-0.5 * (local * local).sum(-1)) + 1e-8
        return w / w.sum(-1, keepdim=True)

    def rest(self, w):
        """what depends on the parameters alone: rest offsets, the inverse of
        the scatter matrix, and the anchor masses"""
        a = self.pos[self.idx]
        rc = (w.unsqueeze(-1) * a).sum(1)
        q = a - rc.unsqueeze(1)                                      # [N,K,3]
        B = torch.einsum("nk,nki,nkj->nij", w, q, q)
        ev, evec = eigh3x3(B)
        lmax = ev[..., -1:].clamp(min=1e-12)
        well = ev > self.eig_floor * lmax
        inv = torch.where(well, 1.0 / ev.clamp(min=1e-12), torch.zeros_like(ev))
        Binv = evec @ torch.diag_embed(inv) @ evec.transpose(-1, -2)
        # directions with no data get identity rather than a solve, so F leaves
        # them alone instead of amplifying whatever noise reached them
        blocked = evec @ torch.diag_embed((~well).to(ev.dtype)) @ evec.transpose(-1, -2)
        mass = torch.zeros(self.M, device=w.device, dtype=w.dtype).index_add_(
            0, self.idx.reshape(-1),
            (self.dens_vol.unsqueeze(-1) * w).reshape(-1)).clamp(min=1e-12)
        return rc, q, Binv, blocked, mass

    # ---- the physics -------------------------------------------------------
    def deformation(self, p, w, rc, q, Binv, blocked):
        nbr = p[self.idx]
        cc = (w.unsqueeze(-1) * nbr).sum(1)
        A = torch.einsum("nk,nki,nkj->nij", w, nbr - cc.unsqueeze(1), q)
        return A @ Binv + blocked, cc

    def force(self, p, w, rc, q, Binv, blocked):
        """-dE/dp, written out rather than taken by autograd.

        F is linear in p once the weights are fixed, and the centroid term drops
        because the weighted rest offsets sum to zero, so the whole derivative
        is one scatter of P Binv q.
        """
        F, _ = self.deformation(p, w, rc, q, Binv, blocked)
        R = polar_R(F, self.polar_iters)
        J = torch.linalg.det(F)
        Finv_T = torch.linalg.inv(F).transpose(-1, -2)
        mu = self.mu.unsqueeze(-1).unsqueeze(-1)
        lam = self.lam.unsqueeze(-1).unsqueeze(-1)
        P = 2 * mu * (F - R) + lam * (J - 1).unsqueeze(-1).unsqueeze(-1) * \
            J.unsqueeze(-1).unsqueeze(-1) * Finv_T
        PB = torch.einsum("nij,njk->nik", P, Binv)                   # Binv symmetric
        contrib = -self.vol.reshape(-1, 1, 1) * w.unsqueeze(-1) * \
            torch.einsum("nij,nkj->nki", PB, q)                      # [N,K,3]
        f = torch.zeros(self.M, 3, device=p.device, dtype=p.dtype).index_add_(
            0, self.idx.reshape(-1), contrib.reshape(-1, 3))
        return f

    def substep(self, p, v, w, rc, q, Binv, blocked, m, keep):
        a = self.force(p, w, rc, q, Binv, blocked) / m
        v = (v + self.dt * a) * self.damping * keep
        return p + self.dt * v, v

    def rollout(self, p, v, n, cache=None):
        """n substeps of semi-implicit Euler, differentiable in the parameters"""
        w, (rc, q, Binv, blocked, mass) = cache if cache is not None else self.prepare()
        m = mass.unsqueeze(-1)
        keep = (~self.fixed).unsqueeze(-1).to(p.dtype)
        use_ckpt = self.checkpoint_substeps and torch.is_grad_enabled()
        for _ in range(n):
            if use_ckpt:
                p, v = checkpoint(self.substep, p, v, w, rc, q, Binv, blocked, m, keep,
                                   use_reentrant=False)
            else:
                p, v = self.substep(p, v, w, rc, q, Binv, blocked, m, keep)
        return p, v

    def prepare(self):
        w = self.weights()
        return w, self.rest(w)

    @torch.no_grad()
    def skin(self, p):
        """every Gaussian, including the zero-volume ones the physics skips"""
        a = self.pos[self.idx_all]
        d = self.Xc_all.unsqueeze(1) - a
        R = quat_to_R(self.quat)[self.idx_all]
        sg = self.log_s.exp()[self.idx_all]
        local = torch.einsum("nkji,nkj->nki", R, d) / sg.clamp(min=1e-6)
        w = torch.exp(-0.5 * (local * local).sum(-1)) + 1e-8
        w = w / w.sum(-1, keepdim=True)
        rc = (w.unsqueeze(-1) * a).sum(1)
        q = a - rc.unsqueeze(1)
        B = torch.einsum("nk,nki,nkj->nij", w, q, q)
        ev, evec = eigh3x3(B)
        well = ev > self.eig_floor * ev[..., -1:].clamp(min=1e-12)
        inv = torch.where(well, 1.0 / ev.clamp(min=1e-12), torch.zeros_like(ev))
        Binv = evec @ torch.diag_embed(inv) @ evec.transpose(-1, -2)
        blocked = evec @ torch.diag_embed((~well).to(ev.dtype)) @ evec.transpose(-1, -2)
        nbr = p[self.idx_all]
        cc = (w.unsqueeze(-1) * nbr).sum(1)
        A = torch.einsum("nk,nki,nkj->nij", w, nbr - cc.unsqueeze(1), q)
        F = A @ Binv + blocked
        return cc + torch.einsum("nij,nj->ni", F, self.Xc_all - rc)


def _R_to_quat(R):
    """[...,3,3] -> [...,4], by the branch with the largest denominator so no
    case divides by something near zero"""
    m = R.reshape(-1, 3, 3)
    t = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    q = torch.zeros(m.shape[0], 4, device=m.device, dtype=m.dtype)
    big = t > 0
    r = (1.0 + t).clamp(min=1e-12).sqrt()
    q[big] = torch.stack([0.5 * r, (m[:, 2, 1] - m[:, 1, 2]) / (2 * r),
                          (m[:, 0, 2] - m[:, 2, 0]) / (2 * r),
                          (m[:, 1, 0] - m[:, 0, 1]) / (2 * r)], -1)[big]
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        sel = (~big) & (m[:, i, i] >= m[:, j, j]) & (m[:, i, i] >= m[:, k, k])
        if not sel.any():
            continue
        r = (1.0 + m[:, i, i] - m[:, j, j] - m[:, k, k]).clamp(min=1e-12).sqrt()
        col = torch.zeros_like(q)
        col[:, 0] = (m[:, k, j] - m[:, j, k]) / (2 * r)
        col[:, 1 + i] = 0.5 * r
        col[:, 1 + j] = (m[:, j, i] + m[:, i, j]) / (2 * r)
        col[:, 1 + k] = (m[:, k, i] + m[:, i, k]) / (2 * r)
        q[sel] = col[sel]
    return q.reshape(*R.shape[:-2], 4)


def _density_of(sc):
    """per-Gaussian density, as scene_setup built the anchor masses from"""
    dens = torch.full_like(sc.volume, float(sc.cfg["density"]))
    for reg in sc.cfg.get("additional_material_params", []):
        c = torch.tensor(reg["point"], device=sc.pos.device)
        s = torch.tensor(reg["size"], device=sc.pos.device)
        dens[((sc.pos - c).abs() <= s).all(-1)] = reg["density"]
    return dens

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



def det3(A):
    """determinant of a batch of 3x3, by the closed form.

    torch.linalg.det factorises, which for 3x3 is a batched LU launch and a
    backward that carries the factorisation with it. The polar iteration below
    calls this six times per evaluation over 171k matrices, so the difference is
    not incidental: the closed form is a handful of elementwise ops whose
    derivative autograd already knows.
    """
    a, b, c = A[..., 0, 0], A[..., 0, 1], A[..., 0, 2]
    d, e, f = A[..., 1, 0], A[..., 1, 1], A[..., 1, 2]
    g, h, i = A[..., 2, 0], A[..., 2, 1], A[..., 2, 2]
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def inv3(A, eps=0.0):
    """inverse of a batch of 3x3, via the adjugate.

    eps guards a singular matrix the way the callers did by hand: the reciprocal
    is taken of a determinant held off zero, rather than producing infinities the
    integrator then has to survive.
    """
    a, b, c = A[..., 0, 0], A[..., 0, 1], A[..., 0, 2]
    d, e, f = A[..., 1, 0], A[..., 1, 1], A[..., 1, 2]
    g, h, i = A[..., 2, 0], A[..., 2, 1], A[..., 2, 2]
    A00 = e * i - f * h; A01 = c * h - b * i; A02 = b * f - c * e
    A10 = f * g - d * i; A11 = a * i - c * g; A12 = c * d - a * f
    A20 = d * h - e * g; A21 = b * g - a * h; A22 = a * e - b * d
    det = a * A00 + b * A10 + c * A20
    if eps:
        det = torch.where(det.abs() < eps, torch.full_like(det, eps), det)
    inv = 1.0 / det
    return torch.stack([A00, A01, A02, A10, A11, A12, A20, A21, A22],
                        dim=-1).reshape(*A.shape) * inv.unsqueeze(-1).unsqueeze(-1)


def closest_rotation(F, iters=8, ridge=1e-6):
    """the nearest ROTATION to F, including where F has flipped over.

    The polar factor of a matrix with negative determinant is a reflection, not
    a rotation, and Fixed Corotated built on one does not restore an inverted
    element -- it drives it further in. That is what takes this simulator to
    NaN: an element inverts at the fourth substep of a frame and the forces run
    away over the next eight.

    The correction is the standard one: negate the singular direction with the
    smallest stretch, which is the axis the material folded through. It needs an
    eigendecomposition of F^T F, whose derivative divides by the gap between
    eigenvalues, so it is applied only to the elements that actually inverted
    and its gradient is not taken. Everywhere else -- effectively everything --
    the Newton factor and its gradient are used unchanged.
    """
    R = polar_R(F, iters, ridge)
    flat = R.reshape(-1, 3, 3)
    bad = det3(flat) < 0
    if bad.any():
        with torch.no_grad():
            Fb = F.reshape(-1, 3, 3)[bad]
            ev, evec = eigh3x3(Fb.transpose(-1, -2) @ Fb)
            u = evec[..., 0]                       # the smallest stretch
            H = torch.eye(3, device=F.device) - 2 * u.unsqueeze(-1) * u.unsqueeze(-2)
            fixed = flat[bad].detach() @ H
        out = flat.clone()
        out[bad] = fixed
        # values corrected, gradient left to the well-behaved branch
        R = (flat + (out - flat).detach()).reshape(F.shape)
    return R


class _PolarR(torch.autograd.Function):
    """the Newton polar factor, with its derivative in closed form.

    Differentiating the iteration itself means differentiating six chained 3x3
    inverses, and that is the single most expensive thing in a substep: the
    polar factor costs 4 ms forward and adds 22 ms once its backward is
    included (exe/probe_speed.py). The derivative does not need the iteration.

    From F = R S with S symmetric, a perturbation gives A = R^T dF = Omega S +
    dS with Omega = R^T dR skew and dS symmetric. The antisymmetric part is
    Omega S + S Omega = A - A^T, and for Omega = [w]x that is the 3x3 system

        (tr(S) I - S) w = vee(A - A^T),

    positive definite whenever S is. Pushing the incoming gradient G through it,
    with C = R^T G and c = vee(C - C^T) and b = (tr(S) I - S)^-1 c,

        dL/dF = R [b]x.

    One symmetric 3x3 solve, whatever the iteration count was.
    """

    @staticmethod
    def forward(ctx, F, iters, ridge):
        with torch.no_grad():
            R = _polar_newton(F, iters, ridge)
        ctx.save_for_backward(F, R)
        ctx.ridge = ridge
        return R

    @staticmethod
    def backward(ctx, g):
        F, R = ctx.saved_tensors
        sh = F.shape
        F3 = F.reshape(-1, 3, 3)
        R3 = R.reshape(-1, 3, 3)
        g3 = g.reshape(-1, 3, 3)
        Rt = R3.transpose(-1, -2)
        S = Rt @ F3
        S = 0.5 * (S + S.transpose(-1, -2))
        C = Rt @ g3
        cc = C - C.transpose(-1, -2)
        c = torch.stack([cc[:, 2, 1], cc[:, 0, 2], cc[:, 1, 0]], -1)
        tr = S.diagonal(dim1=-2, dim2=-1).sum(-1)
        eye = torch.eye(3, device=F.device, dtype=F.dtype)
        # PD for a proper polar factor; the ridge covers a flipped element, whose
        # value is corrected without its gradient anyway
        K = tr.reshape(-1, 1, 1) * eye - S
        K = K + ctx.ridge * tr.abs().clamp(min=1e-12).reshape(-1, 1, 1) * eye
        b = torch.einsum("nij,nj->ni", inv3(K), c)
        z = torch.zeros_like(b[:, 0])
        bx = torch.stack([
            torch.stack([z, -b[:, 2], b[:, 1]], -1),
            torch.stack([b[:, 2], z, -b[:, 0]], -1),
            torch.stack([-b[:, 1], b[:, 0], z], -1)], -2)
        return (R3 @ bx).reshape(sh), None, None


def polar_R(F, iters=8, ridge=1e-6, analytic=True):
    """the polar factor. analytic=False differentiates the iteration instead,
    which is what the closed form above is verified against."""
    if analytic and torch.is_grad_enabled() and F.requires_grad:
        return _PolarR.apply(F, iters, ridge)
    return _polar_newton(F, iters, ridge)


def _polar_newton(F, iters=8, ridge=1e-6):
    """the rotation from F = R S, by scaled Newton iteration.

    R <- (gamma R + gamma^-1 R^-T) / 2, gamma = |det R|^(-1/3), which converges
    quadratically and touches only matrix inverses. The closed form via
    eigh(F^T F) is faster and its derivative divides by the difference between
    eigenvalues of F^T F -- zero under rigid motion, where this simulator spends
    most of its time, so its gradient is NaN exactly where it is needed.
    """
    # a ridge in the direction of the identity: the polar factor of a singular
    # matrix is not defined, and inverting one gives infinities that reach the
    # parameters as NaN a few steps later
    n = F.reshape(*F.shape[:-2], 9).norm(dim=-1).clamp(min=1e-12)
    R = F + (ridge * n).reshape(*F.shape[:-2], 1, 1) * torch.eye(3, device=F.device)
    for _ in range(iters):
        Rinv_T = inv3(R, eps=1e-30).transpose(-1, -2)
        g = det3(R).abs().clamp(min=1e-12).pow(-1.0 / 3.0)
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
        J = det3(F)
        Finv_T = inv3(F, eps=1e-30).transpose(-1, -2)
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

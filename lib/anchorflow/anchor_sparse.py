"""The anchor simulator with a support that moves, and an anchor set that changes size.

Two limits of the fixed-neighbour version, both structural.

Which anchors hold a Gaussian was decided once, by Euclidean distance at the
canonical configuration, and frozen at eight. Orientation and extent could then
only redistribute weight among those eight -- an anchor could not reach a
Gaussian that was not already on its list, nor hand one off to a better-placed
anchor further along the same branch. Freezing it was not a preference: with a
hard k-nearest cutoff the loss jumps when a Gaussian changes hands, and no
gradient sees that coming.

Here an anchor holds whatever falls inside G(x) > c, and the weight is G(x) - c,
which reaches zero exactly at that boundary. Membership becomes a continuous
function of the parameters: a Gaussian entering an anchor's region arrives with
zero weight and leaves the same way, so the connectivity can follow the fit
instead of being pinned in front of it.

That also makes the anchor set editable. Removing an anchor drops its pairs and
the remaining weights renormalise smoothly; splitting one puts two half-sized
anchors where the fit is working hardest. Both are the same operation the
3DGS-style density control performs, on a set whose members are physics rather
than appearance.

The candidate search runs on the canonical Gaussian centres, which never move,
so the tree is built once; only the anchors' query regions change. The pair list
is a superset of the true support -- the exact ellipsoid test happens after --
which is what lets it be refreshed on a schedule rather than every step.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from eigen3x3 import eigh3x3

from .anchor_fit import _R_to_quat, closest_rotation, quat_to_R


class AnchorSparse(nn.Module):
    """anchors with elliptical, compactly supported reach and a variable count"""

    def __init__(self, sc, c=0.25, eig_floor=0.02, polar_iters=6, margin=1.25,
                 checkpoint_substeps=True, s_lo=0.25, s_hi=4.0, cfl_frac=0.05):
        super().__init__()
        from scipy.spatial import cKDTree

        sim = sc.sim
        dev = sc.pos.device
        self.c = c
        self.eig_floor = eig_floor
        self.polar_iters = polar_iters
        self.margin = margin
        self.checkpoint_substeps = checkpoint_substeps
        # an anchor that shrinks far enough holds almost nothing, its mass goes
        # with it, and the accelerations it sees go up without bound; one that
        # grows without limit swallows the object. Both ends were reachable --
        # the unbounded fit went to NaN around its 250th iteration
        self.s_lo = s_lo * sim.radius
        self.s_hi = s_hi * sim.radius
        self.cfl_limit = cfl_frac * sim.radius
        self.polar_ridge = 1e-6
        self.cfg = sc.cfg
        self.dt = sc.sub_dt
        self.damping = sc.damping
        self.dev = dev

        mat = torch.nonzero(sc.keep, as_tuple=False).squeeze(-1)
        self.register_buffer("mat", mat)
        self.register_buffer("Xc", sim.gaussian_canonical[mat].contiguous())
        self.register_buffer("vol", sc.volume[mat].contiguous())
        self.register_buffer("mu", sc.mu[mat].contiguous())
        self.register_buffer("lam", sc.lam[mat].contiguous())
        self.register_buffer("dens_vol", (sc.volume * _density_of(sc))[mat].contiguous())
        self.N = mat.shape[0]
        # the canonical centres never move, so this is built once and every
        # refresh is a query against it rather than a rebuild
        self.tree = cKDTree(self.Xc.detach().cpu().numpy())

        self.pos = nn.Parameter(sc.anchor_canonical.clone())
        self.log_s = nn.Parameter(torch.full((sc.M, 3), float(np.log(sim.radius)),
                                              device=dev))
        q0 = torch.zeros(sc.M, 4, device=dev); q0[:, 0] = 1.0
        self.quat = nn.Parameter(q0)
        # a stiffness carried by the anchor rather than by the material. The
        # config's E, nu and density stay exactly as they are and keep their
        # spatial pattern; what this scales is how stiffly a given piece of the
        # discretisation responds, which is an anchor property in the same sense
        # as its reach. Starts at one, so an unfitted set is the same simulator.
        self.log_k = nn.Parameter(torch.zeros(sc.M, device=dev))
        self.register_buffer("fixed", sc.fixed_mask.clone())
        self.register_buffer("B_ref", torch.zeros((), device=dev))
        self.refresh()
        self.set_B_ref()

    @torch.no_grad()
    def set_B_ref(self, quantile=0.5):
        """the neighbourhood size the floor refers to, fixed at the start.

        Taken once rather than per call: a reference that moves with the
        parameters is one the fit can shrink to escape the floor.
        """
        w = self.weights()
        a = self.pos[self.pair_a]
        rc = torch.zeros(self.N, 3, device=self.dev).index_add_(
            0, self.pair_g, w.unsqueeze(-1) * a)
        q = a - rc[self.pair_g]
        B = torch.zeros(self.N, 3, 3, device=self.dev).index_add_(
            0, self.pair_g, w.reshape(-1, 1, 1) * (q.unsqueeze(-1) * q.unsqueeze(-2)))
        self.B_ref.copy_((B.diagonal(dim1=-2, dim2=-1).sum(-1) / 3.0).quantile(quantile))
        return float(self.B_ref)

    @torch.no_grad()
    def clamp_(self):
        self.log_s.clamp_(min=float(np.log(self.s_lo)), max=float(np.log(self.s_hi)))
        self.quat.div_(self.quat.norm(dim=-1, keepdim=True).clamp(min=1e-12))

    # ---- support ----------------------------------------------------------
    @property
    def M(self):
        return self.pos.shape[0]

    @property
    def mahal_radius(self):
        """G(x) > c is the ellipsoid at this many standard deviations"""
        return float(np.sqrt(max(-2.0 * np.log(self.c), 1e-12)))

    @torch.no_grad()
    def refresh(self):
        """rebuild the candidate pairs.

        A sphere large enough to contain the ellipsoid, widened by a margin, so
        the list stays a superset as the parameters drift between refreshes --
        anything outside the true region has weight exactly zero and costs only
        arithmetic, but anything MISSING is a wrong loss rather than a rough one.
        """
        s = self.log_s.exp()
        rad = (self.mahal_radius * s.max(-1).values * self.margin).cpu().numpy()
        cand = self.tree.query_ball_point(self.pos.detach().cpu().numpy(), rad)
        gi = np.concatenate([np.asarray(c_, dtype=np.int64) for c_ in cand]) \
            if any(len(c_) for c_ in cand) else np.zeros(0, dtype=np.int64)
        aj = np.repeat(np.arange(len(cand), dtype=np.int64),
                        [len(c_) for c_ in cand])
        g = torch.from_numpy(gi).to(self.dev)
        a = torch.from_numpy(aj).to(self.dev)
        keep = self._mahal2(g, a) < (self.mahal_radius * self.margin) ** 2
        g, a = g[keep], a[keep]

        # a Gaussian outside every region has nothing to normalise against, so
        # it is given its nearest anchor back
        have = torch.zeros(self.N, dtype=torch.bool, device=self.dev)
        have[g] = True
        if (~have).any():
            orphan = torch.nonzero(~have, as_tuple=False).squeeze(-1)
            d = torch.cdist(self.Xc[orphan], self.pos)
            g = torch.cat([g, orphan])
            a = torch.cat([a, d.argmin(-1)])
        order = torch.argsort(g * self.M + a)
        self.register_buffer("pair_g", g[order].contiguous())
        self.register_buffer("pair_a", a[order].contiguous())
        return self.pair_g.shape[0]

    def _mahal2(self, g, a):
        d = self.Xc[g] - self.pos[a]
        R = quat_to_R(self.quat)[a]
        s = self.log_s.exp()[a].clamp(min=1e-6)
        local = torch.einsum("pji,pj->pi", R, d) / s
        return (local * local).sum(-1)

    def weights(self):
        """[P], normalised per Gaussian. Zero at the boundary of the region and
        zero outside it, so a Gaussian entering or leaving does so continuously"""
        w = (torch.exp(-0.5 * self._mahal2(self.pair_g, self.pair_a)) - self.c).clamp(min=0)
        tot = torch.zeros(self.N, device=self.dev, dtype=w.dtype).index_add_(
            0, self.pair_g, w)
        # a Gaussian whose only anchors sit exactly on the boundary would divide
        # by zero; it keeps whatever it has, evenly
        empty = tot <= 1e-12
        if empty.any():
            w = torch.where(empty[self.pair_g], torch.ones_like(w), w)
            tot = torch.zeros(self.N, device=self.dev, dtype=w.dtype).index_add_(
                0, self.pair_g, w)
        return w / tot[self.pair_g].clamp(min=1e-12)

    # ---- the rest configuration -------------------------------------------
    def prepare(self):
        w = self.weights()
        a = self.pos[self.pair_a]
        rc = torch.zeros(self.N, 3, device=self.dev).index_add_(
            0, self.pair_g, w.unsqueeze(-1) * a)
        q = a - rc[self.pair_g]
        B = torch.zeros(self.N, 3, 3, device=self.dev).index_add_(
            0, self.pair_g, w.reshape(-1, 1, 1) * (q.unsqueeze(-1) * q.unsqueeze(-2)))
        # The eigen route -- solve above a floor, identity below it -- is what the
        # fixed-neighbour version does, and its derivative divides by the gap
        # between B's eigenvalues. With a compact support the weights reach exact
        # zero, so B goes rank deficient and those eigenvalues coincide: the
        # forward is fine and the backward is NaN within two iterations.
        #
        # (B + eps I)^-1 does the same thing without an eigendecomposition. In
        # well-observed directions it is B^-1; in unobserved ones the leftover
        # eps (B + eps I)^-1 tends to the identity, which is exactly the fallback
        # the eigen version applies by hand, and it arrives smoothly rather than
        # at a threshold.
        # ...with a floor that does not vanish along with B. Relative to B alone
        # the regulariser disappears exactly where it is needed: a Gaussian at
        # the edge of the object can end up with two effective anchors almost on
        # top of each other, and shape matching over a neighbourhood that small
        # is an enormously stiff spring -- |f/m| reached 7.7e4 against the 175
        # this simulator runs at, from a trace(B) one fiftieth of the median.
        # Below the reference size a Gaussian is treated as having the reference
        # neighbourhood, which is the statement that nothing may be stiffer than
        # the discretisation itself.
        tr = (B.diagonal(dim1=-2, dim2=-1).sum(-1) / 3.0).clamp(min=1e-20)
        eps = (self.eig_floor * tr.clamp(min=self.B_ref)).reshape(-1, 1, 1)
        eye = torch.eye(3, device=self.dev)
        Binv = torch.linalg.inv(B + eps * eye)
        blocked = eps * Binv
        mass = torch.zeros(self.M, device=self.dev).index_add_(
            0, self.pair_a, self.dens_vol[self.pair_g] * w).clamp(min=1e-12)
        return w, rc, q, Binv, blocked, mass

    # ---- the physics -------------------------------------------------------
    def deformation(self, p, w, rc, q, Binv, blocked):
        pa = p[self.pair_a]
        cc = torch.zeros(self.N, 3, device=self.dev).index_add_(
            0, self.pair_g, w.unsqueeze(-1) * pa)
        pq = pa - cc[self.pair_g]
        A = torch.zeros(self.N, 3, 3, device=self.dev).index_add_(
            0, self.pair_g, w.reshape(-1, 1, 1) * (pq.unsqueeze(-1) * q.unsqueeze(-2)))
        return A @ Binv + blocked, cc

    def gaussian_pos(self, p, cache):
        w, rc, q, Binv, blocked, _ = cache
        F, cc = self.deformation(p, w, rc, q, Binv, blocked)
        return cc + torch.einsum("nij,nj->ni", F, self.Xc - rc)

    def stiffness(self, w):
        """[N] -- each Gaussian's multiplier, blended from the anchors holding it"""
        k = torch.zeros(self.N, device=self.dev).index_add_(
            0, self.pair_g, w * self.log_k.exp()[self.pair_a])
        return k.clamp(min=1e-3)

    def force(self, p, w, rc, q, Binv, blocked):
        F, _ = self.deformation(p, w, rc, q, Binv, blocked)
        R = closest_rotation(F, self.polar_iters, self.polar_ridge)
        J = torch.linalg.det(F)
        # the volumetric term carries J F^-T, which is unbounded as an element
        # approaches zero volume. A ridge keeps it finite through the crossing
        # rather than handing the integrator an infinity
        n = F.reshape(-1, 9).norm(dim=-1).clamp(min=1e-12).reshape(-1, 1, 1)
        Finv_T = torch.linalg.inv(F + 1e-6 * n * torch.eye(3, device=self.dev)
                                   ).transpose(-1, -2)
        k = self.stiffness(w).unsqueeze(-1).unsqueeze(-1)
        mu = self.mu.unsqueeze(-1).unsqueeze(-1) * k
        lam = self.lam.unsqueeze(-1).unsqueeze(-1) * k
        P = 2 * mu * (F - R) + lam * (J - 1).unsqueeze(-1).unsqueeze(-1) * \
            J.unsqueeze(-1).unsqueeze(-1) * Finv_T
        PB = (P @ Binv)[self.pair_g]
        contrib = -(self.vol[self.pair_g] * w).unsqueeze(-1) * \
            torch.einsum("pij,pj->pi", PB, q)
        return torch.zeros(self.M, 3, device=self.dev).index_add_(0, self.pair_a, contrib)

    def substep(self, p, v, w, rc, q, Binv, blocked, m, keep):
        a = self.force(p, w, rc, q, Binv, blocked) / m
        v = (v + self.dt * a) * self.damping * keep
        dp = self.dt * v
        # how far an anchor travels in one substep, against the spacing between
        # anchors. An explicit step is only meaningful while this is small, and
        # nothing else in the loss knows that: the fit is free to make the
        # discretisation stiffer than the substep can integrate, and it does --
        # 1.4% of the spacing at the start, 50% where the rollout blows up
        over = (dp.norm(dim=-1) / self.cfl_limit - 1.0).clamp(min=0)
        return p + dp, v, (over * over).mean()

    def rollout(self, p, v, n, cache):
        """returns (p, v, cfl) -- the last being how badly, on average, an anchor
        outran the substep it was integrated with"""
        w, rc, q, Binv, blocked, mass = cache
        m = mass.unsqueeze(-1)
        keep = (~self.fixed).unsqueeze(-1).to(p.dtype)
        use = self.checkpoint_substeps and torch.is_grad_enabled()
        pen = p.new_zeros(())
        for _ in range(n):
            if use:
                p, v, c = checkpoint(self.substep, p, v, w, rc, q, Binv, blocked,
                                      m, keep, use_reentrant=False)
            else:
                p, v, c = self.substep(p, v, w, rc, q, Binv, blocked, m, keep)
            pen = pen + c
        return p, v, pen / max(n, 1)

    # ---- moving between particles and anchors ------------------------------
    @torch.no_grad()
    def project(self, x, cache):
        """MPM particles -> this anchor set, by weighted average of displacement"""
        w = cache[0]
        d = (x - self.Xc)[self.pair_g]
        num = torch.zeros(self.M, 3, device=self.dev).index_add_(
            0, self.pair_a, w.unsqueeze(-1) * d)
        den = torch.zeros(self.M, device=self.dev).index_add_(
            0, self.pair_a, w).clamp(min=1e-12)
        p = self.pos + num / den.unsqueeze(-1)
        return torch.where(self.fixed.unsqueeze(-1), self.pos, p)

    @torch.no_grad()
    def project_v(self, vp, cache):
        w = cache[0]
        num = torch.zeros(self.M, 3, device=self.dev).index_add_(
            0, self.pair_a, w.unsqueeze(-1) * vp[self.pair_g])
        den = torch.zeros(self.M, device=self.dev).index_add_(
            0, self.pair_a, w).clamp(min=1e-12)
        v = num / den.unsqueeze(-1)
        return torch.where(self.fixed.unsqueeze(-1), torch.zeros_like(v), v)

    @torch.no_grad()
    def impulse_dv(self, force, cache):
        """the velocity an impulse gives each anchor.

        DreamPhysics applies it per particle as v += f dt / (rho V), so each
        anchor takes (sum of its weights) f dt of momentum -- independent of the
        particle, which is why the anchor set can change without changing what
        the impulse means.
        """
        w = cache[0]
        # [3] is the config's uniform push; [N,3] is a force that varies over the
        # object, which is what a random field is and what the fit is now drawn
        # from. Only the material Gaussians are pushed either way.
        if force.dim() == 1:
            wf = w.unsqueeze(-1) * force.reshape(1, 3)
        else:
            f = force[self.mat] if force.shape[0] != self.N else force
            wf = w.unsqueeze(-1) * f[self.pair_g]
        p2g = torch.zeros(self.M, 3, device=self.dev).index_add_(0, self.pair_a, wf)
        dv = p2g * self.dt / cache[5].unsqueeze(-1)
        return torch.where(self.fixed.unsqueeze(-1), torch.zeros_like(dv), dv)

    @torch.no_grad()
    def lift(self, p, v, cache):
        """this anchor state -> (x, v, F, C) for MPM to restart from"""
        w, rc, q, Binv, blocked, _ = cache
        F, cc = self.deformation(p, w, rc, q, Binv, blocked)
        x = cc + torch.einsum("nij,nj->ni", F, self.Xc - rc)
        va = v[self.pair_a]
        vc = torch.zeros(self.N, 3, device=self.dev).index_add_(
            0, self.pair_g, w.unsqueeze(-1) * va)
        dv = va - vc[self.pair_g]
        Ad = torch.zeros(self.N, 3, 3, device=self.dev).index_add_(
            0, self.pair_g, w.reshape(-1, 1, 1) * (dv.unsqueeze(-1) * q.unsqueeze(-2)))
        Fdot = Ad @ Binv
        vx = vc + torch.einsum("nij,nj->ni", Fdot, self.Xc - rc)
        C = Fdot @ torch.linalg.inv(F + 1e-6 * torch.eye(3, device=self.dev))
        return x.contiguous(), vx.contiguous(), F.reshape(-1, 9).contiguous(), \
            C.reshape(-1, 9).contiguous()

    # ---- editing the anchor set -------------------------------------------
    @torch.no_grad()
    def init_from_geometry(self, spread=1.0):
        """orient and stretch each anchor along the material it holds.

        With equal scales the kernel is rotation-invariant, so the orientation
        has exactly zero gradient and stays wherever it started; the symmetry has
        to be broken here rather than by the fit.
        """
        w = self.weights()
        d = self.Xc[self.pair_g] - self.pos[self.pair_a]
        tot = torch.zeros(self.M, device=self.dev).index_add_(
            0, self.pair_a, w).clamp(min=1e-12)
        C = torch.zeros(self.M, 3, 3, device=self.dev).index_add_(
            0, self.pair_a, w.reshape(-1, 1, 1) * (d.unsqueeze(-1) * d.unsqueeze(-2)))
        C = C / tot.reshape(-1, 1, 1)
        ev, evec = eigh3x3(C + 1e-12 * torch.eye(3, device=self.dev))
        s = ev.clamp(min=1e-12).sqrt()
        thin = (ev[..., 0] < 1e-4 * ev[..., -1]) | (tot < 1e-8)
        s = s / s.mean(-1, keepdim=True).clamp(min=1e-12)
        s = (1.0 + spread * (s - 1.0)) * self.log_s.exp().mean(-1, keepdim=True)
        self.log_s.copy_(torch.where(thin.unsqueeze(-1), self.log_s.exp(),
                                      s.clamp(min=1e-6)).log())
        R = torch.where(thin.reshape(-1, 1, 1),
                        torch.eye(3, device=self.dev).expand_as(evec), evec)
        self.quat.copy_(_R_to_quat(R))
        self.refresh()
        return int(thin.sum())

    @torch.no_grad()
    def _rebuild(self, pos, quat, log_s, log_k=None):
        self.pos = nn.Parameter(pos.contiguous())
        self.quat = nn.Parameter(quat.contiguous())
        self.log_s = nn.Parameter(log_s.contiguous())
        if log_k is None:
            log_k = torch.zeros(pos.shape[0], device=self.dev)
        self.log_k = nn.Parameter(log_k.contiguous())
        f = torch.zeros(pos.shape[0], dtype=torch.bool, device=self.dev)
        for bc in self.cfg.get("boundary_conditions", []):
            if bc["type"] == "cuboid":
                c = torch.tensor(bc["point"], device=self.dev)
                s = torch.tensor(bc["size"], device=self.dev)
                f |= ((pos - c).abs() <= s).all(-1)
        self.fixed = f
        self.refresh()

    @torch.no_grad()
    def densify_and_prune(self, grad_accum, split_frac=0.05, prune_share=0.05,
                          split_scale=1.6, max_anchors=2048, min_pairs=4):
        """split where the fit is pushing hardest, drop what holds nothing.

        The criterion is the accumulated gradient on an anchor's position, the
        same signal 3DGS densifies on: a large one means the loss wants that
        anchor to be in two places, which is what splitting gives it.

        Pinned anchors are boundary condition, not representation, and are left
        alone in both directions.
        """
        w = self.weights()
        held = torch.zeros(self.M, device=self.dev).index_add_(
            0, self.pair_a, w)
        n_pairs = torch.zeros(self.M, device=self.dev).index_add_(
            0, self.pair_a, torch.ones_like(w))

        dead = ((held < prune_share * held.median()) | (n_pairs < min_pairs)) & ~self.fixed
        alive = ~dead
        pos, quat, log_s = self.pos[alive], self.quat[alive], self.log_s[alive]
        log_k = self.log_k[alive]
        g = grad_accum[alive]
        fixed = self.fixed[alive]

        room = max(0, max_anchors - pos.shape[0])
        k = min(room, int(split_frac * pos.shape[0]))
        n_split = 0
        if k > 0:
            score = torch.where(fixed, torch.zeros_like(g), g)
            pick = torch.topk(score, k).indices
            s = log_s[pick].exp()
            R = quat_to_R(quat[pick])
            # split along the anchor's own long axis: that is the direction it
            # is stretched to cover, so two of them cover it at half the length
            axis = torch.gather(R, 2, s.argmax(-1).reshape(-1, 1, 1)
                                 .expand(-1, 3, 1)).squeeze(-1)
            off = 0.5 * s.max(-1).values.unsqueeze(-1) * axis
            new_s = (s / split_scale).clamp(min=1e-6).log()
            pos = torch.cat([pos, pos[pick] + off]); pos[pick] -= off
            quat = torch.cat([quat, quat[pick]])
            log_s = torch.cat([log_s, new_s]); log_s[pick] = new_s
            log_k = torch.cat([log_k, log_k[pick]])
            n_split = k
        self._rebuild(pos, quat, log_s, log_k)
        return int(dead.sum()), n_split


def _density_of(sc):
    dens = torch.full_like(sc.volume, float(sc.cfg["density"]))
    for reg in sc.cfg.get("additional_material_params", []):
        c = torch.tensor(reg["point"], device=sc.pos.device)
        s = torch.tensor(reg["size"], device=sc.pos.device)
        dens[((sc.pos - c).abs() <= s).all(-1)] = reg["density"]
    return dens

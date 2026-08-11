"""Build the anchor/Gaussian scene from a 3DGS ply + a DreamPhysics config.

Every experiment script had its own copy of this and they had already drifted
apart in small ways (which cloud defines MPM space, whether the impulse's P2G
weights are masked to the material subset). One place, one behaviour.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import torch

from .anchors import AnchorSet
from .anchor_mpm import AnchorElasticSim, lame_from_E_nu


@dataclass
class Scene:
    cfg: dict
    # Gaussians
    xyz_world: torch.Tensor       # [N,3] all Gaussians, world space
    pos: torch.Tensor             # [N,3] all Gaussians, MPM space
    keep: torch.Tensor            # [N] bool, which ones carry material
    volume: torch.Tensor          # [N] zero for the non-material ones
    mu: torch.Tensor              # [N]
    lam: torch.Tensor             # [N]
    # anchors
    anchor_canonical: torch.Tensor        # [M,3]
    mass: torch.Tensor                    # [M]
    fixed_mask: torch.Tensor              # [M] bool
    sim: AnchorElasticSim
    # misc
    gravity: torch.Tensor
    sub_dt: float
    damping: float
    to_mpm: object
    undo: object

    @property
    def M(self):
        return self.anchor_canonical.shape[0]

    @property
    def N(self):
        return self.pos.shape[0]

    def initial_velocity(self, force=None):
        """anchor velocity right after the config's particle_impulse.

        DreamPhysics applies it per particle as v += (f/m)*dt, and m*dv = f*dt is
        particle-independent, so each anchor receives (sum of its P2G weights) *
        f * dt of momentum. Only MATERIAL particles are pushed.
        """
        dev = self.pos.device
        v = torch.zeros(self.M, 3, device=dev)
        if force is None:
            for bc in self.cfg.get("boundary_conditions", []):
                if bc["type"] == "particle_impulse":
                    force = torch.tensor(bc["force"], device=dev)
        if force is not None:
            v = v + self.impulse_dv(force)
        v[self.fixed_mask] = 0
        return v

    def impulse_dv(self, force):
        """Velocity an impulse adds, per anchor. Separate from initial_velocity
        so the same kick can be delivered part-way through a run: the P2G
        weights are taken at the canonical configuration, so the momentum a
        given force deposits does not depend on where the object currently is,
        and a mid-run impulse means the same thing as the one at t=0.

        force is [3] -- the same vector on every material Gaussian, which is
        what the config's particle_impulse means -- or [N,3], a force that
        varies over the object (see random_force_field).
        """
        w = self.sim._weights(self.pos, self.anchor_canonical) * self.keep.unsqueeze(-1)
        if force.dim() == 1:
            wf = w.unsqueeze(-1) * force.view(1, 1, 3)
        else:
            wf = w.unsqueeze(-1) * (force * self.keep.unsqueeze(-1)).unsqueeze(1)
        p2g = torch.zeros(self.M, 3, device=self.pos.device).index_add_(
            0, self.sim.nn_idx.reshape(-1), wf.reshape(-1, 3))
        dv = p2g * self.sub_dt / self.mass.unsqueeze(-1)
        return torch.where(self.fixed_mask.unsqueeze(-1), torch.zeros_like(dv), dv)

    def random_force_field(self, gen, sigma, magnitude):
        """A force that varies over the object, smooth on a length scale sigma.

        Every impulse in this project so far has been one vector applied to the
        whole object, because that is what the config's particle_impulse is. So
        the training data contains only the lowest spatial frequency there is,
        and nothing the network has seen distinguishes bending one branch from
        shoving everything.

        Drawing an independent vector per anchor is not the fix -- neighbouring
        anchors would be pushed apart, which is not a force anything can apply
        and which the material absorbs locally without moving. Smoothing the
        draw over a length sigma sets how far apart two points must be before
        they are pushed differently: at the anchor spacing this excites one
        twig, and as sigma approaches the object's size it converges back to the
        uniform push, so the old behaviour is one end of the range rather than
        something replaced.

        Normalised on the RMS anchor force rather than the peak. Normalising on
        the peak makes a localised field deposit far less momentum than a global
        one -- measured, it dropped the streaming data's typical displacement by
        4.3x, which gives back the amplitude coverage the streaming data exists
        for. On the RMS, a field that is already uniform is unchanged (peak and
        RMS coincide), so the long-sigma end still reproduces the config's own
        impulse exactly, while a concentrated field is scaled up to push its
        smaller region harder.
        """
        AC = self.anchor_canonical
        g = torch.randn(self.M, 3, device=AC.device, generator=gen)
        d2 = torch.cdist(AC, AC) ** 2
        w = torch.exp(-d2 / (2.0 * float(sigma) ** 2))
        w = w / w.sum(-1, keepdim=True)
        f = w @ g
        rms = f.norm(dim=-1).pow(2).mean().sqrt().clamp(min=1e-12)
        f = f / rms * float(magnitude)
        wc = self.sim._canonical_weights()                      # [N,K]
        return (wc.unsqueeze(-1) * f[self.sim.nn_idx]).sum(1)   # [N,3]

    def random_poke(self, gen, radius, magnitude):
        """A force on one region of the object and nothing else.

        The smooth random fields cover spatial FREQUENCY -- at a short
        correlation length the whole object is forced, in patches of that size
        pointing different ways. They do not cover spatial SUPPORT: nothing in
        any training or held-out set here is a force applied to one part of the
        object with the rest left alone, which is what an actual interaction
        with one of these looks like.

        One random material Gaussian is the centre, the window is a Gaussian
        bump of the given radius rather than a hard edge (a discontinuous force
        is not something the discretisation handles meaningfully), and the
        direction is uniform over the region -- so this is a push on a part,
        not a texture over the whole.
        """
        dev = self.pos.device
        mat = torch.nonzero(self.keep, as_tuple=False).squeeze(-1)
        c = self.pos[mat[torch.randint(mat.shape[0], (1,), device=dev, generator=gen)]][0]
        d2 = ((self.pos - c) ** 2).sum(-1)
        w = torch.exp(-d2 / (2.0 * float(radius) ** 2)) * self.keep.float()
        q, r = torch.linalg.qr(torch.randn(3, 3, device=dev, generator=gen))
        q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)
        if torch.det(q) < 0:
            q[:, 0] = -q[:, 0]
        direction = q[:, 0]
        f = w.unsqueeze(-1) * direction.view(1, 3) * float(magnitude)
        return f, c, float(w.sum() / self.keep.sum())

    @property
    def extent(self):
        a = self.anchor_canonical
        return float((a.max(0).values - a.min(0).values).norm())

    def explicit_step(self, p, v, gp, n=1):
        """n explicit substeps; returns (p, v, gaussian_pos)."""
        g = self.gravity if self.gravity.abs().sum() > 0 else None
        with torch.enable_grad():
            for _ in range(n):
                p, v, gp, _ = self.sim.step(p, v, self.mass, gp, self.volume, self.mu,
                                             self.lam, self.sub_dt, gravity=g,
                                             damping=self.damping, fixed_mask=self.fixed_mask)
        return p, v, gp

    def elastic_accel(self, p, gp):
        """f_int(p)/m per anchor -- one fused-kernel call (~0.1 ms).

        This is the only physics the network sees, and it is computed the SAME
        way in training and in rollout: from the current anchor configuration.
        The previous line carried an acceleration in the state instead, which
        silently meant two different things -- the simulator's f/m while
        training, the integrator's readback of the network's own output while
        rolling out.
        """
        from anchorstep import fused_energy_force
        # the same settings the step uses. Left at the kernel's defaults this
        # computed a different physics than the simulator it is supposed to
        # describe -- eig floor 0.2 instead of the scene's, weights recomputed
        # from the cloud instead of the frozen ones, and the unobserved
        # directions of F frozen rather than rotating. On its own trajectory
        # that made |f/m| nearly ten times what the simulator had just used
        f, _, _, _ = fused_energy_force(
            self.sim.gaussian_canonical, gp, p.detach(), self.sim.anchor_canonical,
            self.sim.nn_idx, self.volume, self.sim.radius, self.mu, self.lam,
            eig_floor_frac=self.sim.eig_floor, w_in=self.sim.frozen_w,
            rot_fallback=self.sim.rot_fallback)
        a = f / self.mass.unsqueeze(-1)
        return torch.where(self.fixed_mask.unsqueeze(-1), torch.zeros_like(a), a)

    def skin(self, p, gp):
        """Gaussian positions implied by an anchor configuration (dt=0 step)."""
        with torch.no_grad():
            _, _, gp, _ = self.sim.step(p, torch.zeros_like(p), self.mass, gp, self.volume,
                                         self.mu, self.lam, 0.0, gravity=None, damping=1.0,
                                         fixed_mask=self.fixed_mask)
        return gp


def build(ply, config, n_anchors=512, K=8, n_grid=100, grid_lim=2.0, device="cuda",
          sh_degree=3, frozen_weights=False, radius_scale=1.0, eig_floor=0.2,
          rot_fallback=False, anchors=None):
    """frozen_weights holds the blend weights at their canonical values, which
    makes the step a function of the anchor positions and velocities alone --
    see AnchorElasticSim.freeze_weights. Without it the simulator is
    path-dependent and no learned stepper can fit it exactly."""
    from scene.gaussian_model import GaussianModel

    dev = device
    cfg = json.load(open(config))
    g = GaussianModel(sh_degree, fea_dim=0)
    g.load_ply(ply)
    xyz = g.get_xyz.detach().clone()
    op = g.get_opacity.detach().clone()

    # the opacity threshold selects MATERIAL, not what gets drawn: the rejected
    # kernels are interleaved through the foliage, so they are skinned like
    # everything else and simply carry zero volume (the fused kernel only ever
    # multiplies by volume), which leaves the physics identical.
    keep = op[:, 0] > cfg["opacity_threshold"]
    xw = xyz[keep]
    pmin, pmax = xw.min(0).values, xw.max(0).values
    mid = (pmin + pmax) / 2
    sc = 1.0 / (pmax - pmin).max()
    one = torch.tensor([1., 1., 1.], device=dev)
    to_mpm = lambda q: (q - mid) * sc + one
    undo = lambda q: (q - one) / sc + mid
    pos = to_mpm(xyz).contiguous()
    pm = pos[keep].contiguous()
    N = pos.shape[0]

    dx = grid_lim / n_grid
    vi = (pm / dx).long().clamp(0, n_grid - 1)
    flat = (vi[:, 0] * n_grid + vi[:, 1]) * n_grid + vi[:, 2]
    cnt = torch.zeros(n_grid ** 3, device=dev).index_add_(
        0, flat, torch.ones(pm.shape[0], device=dev))
    volume = torch.zeros(N, device=dev)
    volume[keep] = (dx ** 3) / cnt[flat]
    volume = volume.contiguous()

    E = torch.full((N,), float(cfg["E"]), device=dev)
    nu = torch.full((N,), float(cfg["nu"]), device=dev)
    dens = torch.full((N,), float(cfg["density"]), device=dev)
    for reg in cfg.get("additional_material_params", []):
        c = torch.tensor(reg["point"], device=dev)
        s = torch.tensor(reg["size"], device=dev)
        ins = ((pos - c).abs() <= s).all(-1)
        E[ins] = reg["E"]; nu[ins] = reg["nu"]; dens[ins] = reg["density"]
    mu, lam = lame_from_E_nu(E, nu)

    aset, _ = AnchorSet.from_gaussians(pm, node_num=n_anchors, latent_dim=0, e_dim=0, K=K)
    ac = aset.canonical.clone().contiguous()
    # anchors and the RBF radius come from the material cloud, so the zero-volume
    # Gaussians cannot shift the physics through either. The radius is the
    # sampled spacing even when the anchors are supplied: it sets how far a
    # Gaussian looks, and moving the anchors is not meant to change that.
    radius = AnchorElasticSim(pm, ac, K=K).radius * radius_scale
    if anchors is not None:
        ac = anchors.to(dev).float().contiguous()
    M = ac.shape[0]
    sim = AnchorElasticSim(pos, ac, K=K, radius=radius)
    sim.eig_floor = eig_floor
    sim.rot_fallback = rot_fallback
    w0 = sim._weights(pos, ac)
    if frozen_weights:
        sim.freeze_weights()
    mass = torch.zeros(M, device=dev).index_add_(
        0, sim.nn_idx.reshape(-1), ((dens * volume).unsqueeze(-1) * w0).reshape(-1)).clamp(min=1e-12)

    fixed = torch.zeros(M, dtype=torch.bool, device=dev)
    for bc in cfg.get("boundary_conditions", []):
        if bc["type"] == "cuboid":
            c = torch.tensor(bc["point"], device=dev)
            s = torch.tensor(bc["size"], device=dev)
            fixed |= ((ac - c).abs() <= s).all(-1)

    return Scene(cfg=cfg, xyz_world=xyz, pos=pos, keep=keep, volume=volume, mu=mu, lam=lam,
                  anchor_canonical=ac, mass=mass, fixed_mask=fixed, sim=sim,
                  gravity=torch.tensor(cfg["g"], dtype=torch.float32, device=dev),
                  sub_dt=float(cfg["substep_dt"]),
                  damping=float(cfg.get("grid_v_damping_scale", 1.0)),
                  to_mpm=to_mpm, undo=undo)

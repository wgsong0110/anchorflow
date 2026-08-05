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
            w = self.sim._weights(self.pos, self.anchor_canonical) * self.keep.unsqueeze(-1)
            wsum = torch.zeros(self.M, device=dev).index_add_(
                0, self.sim.nn_idx.reshape(-1), w.reshape(-1))
            v = v + (wsum.unsqueeze(-1) * force.unsqueeze(0) * self.sub_dt) / self.mass.unsqueeze(-1)
        v[self.fixed_mask] = 0
        return v

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
        f, _, _, _ = fused_energy_force(
            self.sim.gaussian_canonical, gp, p.detach(), self.sim.anchor_canonical,
            self.sim.nn_idx, self.volume, self.sim.radius, self.mu, self.lam)
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
          sh_degree=3):
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
    M = ac.shape[0]
    # anchors and the RBF radius come from the material cloud, so the zero-volume
    # Gaussians cannot shift the physics through either
    radius = AnchorElasticSim(pm, ac, K=K).radius
    sim = AnchorElasticSim(pos, ac, K=K, radius=radius)
    w0 = sim._weights(pos, ac)
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

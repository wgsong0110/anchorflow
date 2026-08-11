"""PhysGaussian's MPM, wrapped so it can be the teacher the student imitates.

The student's state is 512 anchor positions and velocities. MPM's is 171k
particles carrying a deformation gradient and an affine velocity. Making MPM a
teacher means moving between the two in both directions:

  * down. MPM's particles to anchor positions. A weighted average of the
    DISPLACEMENT, not of the position -- the weighted average of the rest
    particles around an anchor is not the anchor, so averaging positions puts a
    fixed offset into every frame including the first.

  * up. An anchor state to a full particle state. Positions and the deformation
    gradient come from the same weighted Procrustes fit the anchor simulator
    uses. Velocities and the affine matrix come from that fit applied to
    velocities: the fitted dF/dt gives the velocity gradient, which is what
    MPM's C is.

The lift is what makes DAgger possible. Behavioural cloning against recorded
trajectories is the regime that diverged in this project, and the fix was
labelling states the student actually reaches -- which needs an expert that can
be asked "from HERE, what happens next?". Handing the solver (x, v, F, C)
reproduces its own continuation to 2e-4 of a 0.59 displacement span, so the
solver is genuinely restartable; what the lift costs is whatever MPM's per
particle F carries that 512 anchors cannot say, and that is measured rather
than assumed (see exe/verify_mpm_teacher.py).
"""
from __future__ import annotations

import torch

from eigen3x3 import eigh3x3


class MPMTeacher:
    """MPM driven from, and reported in, the anchor state the student uses."""

    def __init__(self, sc, n_grid=100, grid_lim=2.0, affine=True):
        from mpm_solver_warp.mpm_solver_warp import MPM_Simulator_WARP

        self.sc = sc
        self.affine = affine
        self.dev = sc.pos.device
        cfg = sc.cfg
        self.mat = torch.nonzero(sc.keep, as_tuple=False).squeeze(-1)
        self.pos_m = sc.pos[self.mat].contiguous()
        self.vol_m = sc.volume[self.mat].contiguous()
        self.n = self.mat.shape[0]

        s = MPM_Simulator_WARP(10)
        s.load_initial_data_from_torch(
            self.pos_m, self.vol_m, torch.zeros((self.n, 6), device=self.dev),
            n_grid=n_grid, grid_lim=grid_lim)
        mp = {k: cfg[k] for k in ("E", "nu", "density", "material") if k in cfg}
        mp.update({"n_grid": n_grid, "grid_lim": grid_lim, "g": cfg.get("g", [0, 0, 0]),
                   "grid_v_damping_scale": cfg.get("grid_v_damping_scale", 1.0)})
        # region-varying material is what gives ficus a soft canopy on a stiff
        # trunk, and set_parameters_dict is the only place this solver takes it
        if "additional_material_params" in cfg:
            mp["additional_material_params"] = cfg["additional_material_params"]
        s.set_parameters_dict(mp)
        s.finalize_mu_lam()
        for bc in cfg.get("boundary_conditions", []):
            if bc["type"] == "cuboid":
                s.set_velocity_on_cuboid(bc["point"], bc["size"], [0.0, 0.0, 0.0],
                                          start_time=0.0, end_time=999.0, reset=1)
        self.solver = s

        # projection, fixed at the canonical configuration like the blend weights
        sim = sc.sim
        self.w = sim._canonical_weights()[self.mat]                 # [Nm,K]
        self.idx = sim.nn_idx[self.mat]                             # [Nm,K]
        self.den = torch.zeros(sc.M, device=self.dev).index_add_(
            0, self.idx.reshape(-1), self.w.reshape(-1)).clamp(min=1e-12)
        self.AC = sc.anchor_canonical
        self.fixed = sc.fixed_mask
        self.eye = torch.eye(3, device=self.dev).reshape(1, 9).repeat(self.n, 1).contiguous()

    # ---- MPM particles -> anchors -----------------------------------------
    def project(self, x):
        """[Nm,3] particle positions -> [M,3] anchor positions.

        The displacement is averaged, not the position: at rest this returns the
        canonical anchors exactly, which averaging positions does not.
        """
        d = x - self.pos_m
        num = torch.zeros(self.AC.shape[0], 3, device=self.dev).index_add_(
            0, self.idx.reshape(-1), (self.w.unsqueeze(-1) * d.unsqueeze(1)).reshape(-1, 3))
        p = self.AC + num / self.den.unsqueeze(-1)
        return torch.where(self.fixed.unsqueeze(-1), self.AC, p)

    def project_v(self, vp):
        """[Nm,3] particle velocities -> [M,3]. No rest state to subtract here,
        so this is the plain weighted average."""
        num = torch.zeros(self.AC.shape[0], 3, device=self.dev).index_add_(
            0, self.idx.reshape(-1), (self.w.unsqueeze(-1) * vp.unsqueeze(1)).reshape(-1, 3))
        v = num / self.den.unsqueeze(-1)
        return torch.where(self.fixed.unsqueeze(-1), torch.zeros_like(v), v)

    # ---- anchors -> MPM particles -----------------------------------------
    def lift(self, p, v):
        """anchor state -> (x, v, F, C), the particle state MPM restarts from.

        F is the weighted Procrustes fit the anchor simulator uses. The velocity
        field is that same fit applied to velocities, which gives dF/dt; MPM's C
        is the velocity gradient, dF/dt F^-1.
        """
        sim = self.sc.sim
        w = sim._weights(self.sc.pos, self.AC)
        F_all, x_all = sim._shape_match(p, w)
        F, x = F_all[self.mat], x_all[self.mat]

        wm = self.w
        nbr_rest = sim.anchor_nbr[self.mat]                          # [Nm,K,3]
        rc = (wm.unsqueeze(-1) * nbr_rest).sum(1)
        q = nbr_rest - rc.unsqueeze(1)
        B = torch.einsum("nk,nki,nkj->nij", wm, q, q)
        ev, evec = eigh3x3(B)
        lmax = ev[..., -1:].clamp(min=1e-12)
        well = ev > sim.eig_floor * lmax

        vn = v[self.idx]                                             # [Nm,K,3]
        vc = (wm.unsqueeze(-1) * vn).sum(1)
        dv = vn - vc.unsqueeze(1)
        Ad = torch.einsum("nk,nki,nkj->nij", wm, dv, q)
        # blocked directions have no data about how the velocity varies there,
        # so the fit says nothing rather than dividing by a near-zero eigenvalue
        Fdv = torch.where(well.unsqueeze(-2), (Ad @ evec) / ev.clamp(min=1e-12).unsqueeze(-2),
                          torch.zeros_like(Ad))
        Fdot = Fdv @ evec.transpose(-1, -2)

        off = (sim.gaussian_canonical[self.mat] - rc)
        vp = vc + torch.einsum("nij,nj->ni", Fdot, off)
        if self.affine:
            C = Fdot @ torch.linalg.inv(F + 1e-6 * torch.eye(3, device=self.dev))
        else:
            C = torch.zeros_like(F)
        return x.contiguous(), vp.contiguous(), F.reshape(-1, 9).contiguous(), \
            C.reshape(-1, 9).contiguous()

    # ---- running -----------------------------------------------------------
    def _set(self, x, v, F, C):
        self.solver.import_particle_x_from_torch(x)
        self.solver.import_particle_v_from_torch(v)
        self.solver.import_particle_F_from_torch(F)
        self.solver.import_particle_C_from_torch(C)

    def _advance(self, frames, dt_mult):
        out = []
        for _ in range(frames):
            for _ in range(dt_mult):
                self.solver.p2g2p(None, self.sc.sub_dt, device=self.dev)
            out.append(self.project(self.solver.export_particle_x_to_torch()))
        return out

    def trajectory(self, force, frames, dt_mult):
        """from rest, under an impulse -> anchor positions [frames+1, M, 3].

        The impulse is delivered as the anchors deliver it -- skinned back onto
        the particles -- so that a trajectory here and a trajectory from the
        anchor simulator start from identical motion and differ only in the
        physics that follows.
        """
        dv = self.sc.impulse_dv(force)
        v0 = (self.w.unsqueeze(-1) * dv[self.idx]).sum(1).contiguous()
        self._set(self.pos_m.clone(), v0, self.eye.clone(), torch.zeros_like(self.eye))
        return torch.stack([self.AC.clone()] + self._advance(frames, dt_mult))

    def query(self, p, v, k, dt_mult):
        """what MPM does for k coarse steps starting from an anchor state.

        This is the DAgger label: the student is rolled out, and wherever it
        gets to, the teacher is asked from exactly there.
        """
        self._set(*self.lift(p, v))
        return torch.stack(self._advance(k, dt_mult))

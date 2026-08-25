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
    """MPM driven from, and reported in, the anchor state the student uses.

    Everything here is under no_grad. The solver allocates its warp arrays with
    requires_grad=True, so export_particle_x_to_torch hands back a tensor that
    wants a graph; inside a trainer, where grad is on, that quietly retained one
    per frame over 171k particles and ran a 24 GB card out of memory on the
    first trajectory. Nothing differentiates through the teacher -- it is a data
    source.
    """

    def __init__(self, sc, n_grid=None, grid_lim=2.0, affine=True, margin=0.02,
                  sparse=None):
        from mpm_solver_warp.mpm_solver_warp import MPM_Simulator_WARP

        self.sc = sc
        self.affine = affine
        self.grid_lim = grid_lim
        # how close to the domain wall a lifted particle may sit. MPM writes to
        # the cells around each particle, so a particle at the very edge indexes
        # past the grid; warp does not bounds-check and the process dies with an
        # illegal memory access rather than an exception
        self.margin = margin * grid_lim
        self.dev = sc.pos.device
        # warp resolves devices by name and rejects a torch.device object
        self.wp_dev = str(self.dev)
        cfg = sc.cfg
        # the scene's own resolution: tear_bread asks for 150, wolf for 200
        n_grid = int(getattr(sc, "n_grid", None) or n_grid or 100)
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
        # Every boundary condition the scene asks for, dispatched the way
        # PhysGaussian's utils/decode_param.py dispatches them. Only cuboid was
        # handled here, which is all the ficus config uses; any other scene had
        # its material fall out of the grid, and warp reports that as an illegal
        # memory access rather than as a missing wall.
        for bc in cfg.get("boundary_conditions", []):
            t = bc["type"]
            t0 = float(bc.get("start_time", 0.0))
            t1 = float(bc.get("end_time", 1e3))
            if t == "cuboid":
                s.set_velocity_on_cuboid(bc["point"], bc["size"],
                                          bc.get("velocity", [0.0, 0.0, 0.0]),
                                          start_time=t0, end_time=t1,
                                          reset=int(bc.get("reset", 1)))
            elif t == "bounding_box":
                s.add_bounding_box()
            elif t == "enforce_particle_translation":
                s.enforce_particle_velocity_translation(
                    point=bc["point"], size=bc["size"], velocity=bc["velocity"],
                    start_time=t0, end_time=t1)
            elif t == "enforce_particle_velocity_rotation":
                s.enforce_particle_velocity_rotation(
                    point=bc["point"], normal=bc["normal"],
                    half_height_and_radius=bc["half_height_and_radius"],
                    rotation_scale=bc["rotation_scale"],
                    translation_scale=bc["translation_scale"],
                    start_time=t0, end_time=t1)
            elif t == "surface_collider":
                s.add_surface_collider(
                    point=bc["point"], normal=bc["normal"],
                    surface=bc.get("surface", "sticky"),
                    friction=float(bc.get("friction", 0.0)),
                    start_time=t0, end_time=t1)
            elif t == "release_particles_sequentially":
                s.release_particles_sequentially(
                    normal=bc["normal"], start_position=bc["start_position"],
                    end_position=bc["end_position"],
                    num_layers=bc["num_layers"],
                    start_time=t0, end_time=t1)
            elif t == "particle_impulse":
                # delivered by the caller, as an initial anchor velocity
                pass
            else:
                raise ValueError(f"boundary condition {t!r} is not handled")
        self.solver = s

        # projection, fixed at the canonical configuration like the blend weights.
        #
        # A fitted anchor set is a different set -- more of them, elsewhere, with
        # their own reach -- so it brings its own projection. Without this the
        # teacher would hand back states for the 512 sampled anchors while the
        # student is stepping 588 fitted ones.
        self.sparse = sparse
        if sparse is not None:
            self.AC = sparse.pos.detach()
            self.fixed = sparse.fixed
            self._scache = sparse.prepare()
            self.w = self.idx = self.den = None
        else:
            sim = sc.sim
            self.w = sim._canonical_weights()[self.mat]             # [Nm,K]
            self.idx = sim.nn_idx[self.mat]                         # [Nm,K]
            self.den = torch.zeros(sc.M, device=self.dev).index_add_(
                0, self.idx.reshape(-1), self.w.reshape(-1)).clamp(min=1e-12)
            self.AC = sc.anchor_canonical
            self.fixed = sc.fixed_mask
        self.eye = torch.eye(3, device=self.dev).reshape(1, 9).repeat(self.n, 1).contiguous()

    # ---- MPM particles -> anchors -----------------------------------------
    @torch.no_grad()
    def project(self, x):
        """[Nm,3] particle positions -> [M,3] anchor positions.

        The displacement is averaged, not the position: at rest this returns the
        canonical anchors exactly, which averaging positions does not.
        """
        if self.sparse is not None:
            return self.sparse.project(x, self._scache)
        d = x - self.pos_m
        num = torch.zeros(self.AC.shape[0], 3, device=self.dev).index_add_(
            0, self.idx.reshape(-1), (self.w.unsqueeze(-1) * d.unsqueeze(1)).reshape(-1, 3))
        p = self.AC + num / self.den.unsqueeze(-1)
        return torch.where(self.fixed.unsqueeze(-1), self.AC, p)

    @torch.no_grad()
    def project_v(self, vp):
        """[Nm,3] particle velocities -> [M,3]. No rest state to subtract here,
        so this is the plain weighted average."""
        if self.sparse is not None:
            return self.sparse.project_v(vp, self._scache)
        num = torch.zeros(self.AC.shape[0], 3, device=self.dev).index_add_(
            0, self.idx.reshape(-1), (self.w.unsqueeze(-1) * vp.unsqueeze(1)).reshape(-1, 3))
        v = num / self.den.unsqueeze(-1)
        return torch.where(self.fixed.unsqueeze(-1), torch.zeros_like(v), v)

    # ---- anchors -> MPM particles -----------------------------------------
    @torch.no_grad()
    def lift(self, p, v):
        """anchor state -> (x, v, F, C), the particle state MPM restarts from.

        F is the weighted Procrustes fit the anchor simulator uses. The velocity
        field is that same fit applied to velocities, which gives dF/dt; MPM's C
        is the velocity gradient, dF/dt F^-1.
        """
        if self.sparse is not None:
            return self.sparse.lift(p, v, self._scache)
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

    @torch.no_grad()
    def _in_domain(self):
        """cheap, and the only thing standing between a diverging query and a
        dead process: MPM writes to the cells around each particle and warp does
        not bounds-check, so a particle that leaves the grid takes the whole run
        down with CUDA error 700 rather than raising something catchable"""
        x = self.solver.export_particle_x_to_torch()
        lo, hi = x.min().item(), x.max().item()
        return lo == lo and self.margin < lo and hi < self.grid_lim - self.margin

    @torch.no_grad()
    def _advance(self, frames, dt_mult, check_every=8):
        """returns None if the run left the domain part-way, rather than dying.

        Checked mid-frame as well: a state that blows up does so within a few
        substeps, and 40 of them is long enough to go from plausible to out of
        bounds with nothing observed in between.
        """
        out = []
        for _ in range(frames):
            for k in range(dt_mult):
                self.solver.p2g2p(None, self.sc.sub_dt, device=self.wp_dev)
                if (k + 1) % check_every == 0 and not self._in_domain():
                    return None
            out.append(self.project(self.solver.export_particle_x_to_torch()))
        return out

    @torch.no_grad()
    def trajectory(self, force, frames, dt_mult):
        """from rest, under an impulse -> anchor positions [frames+1, M, 3].

        The impulse is delivered as the anchors deliver it -- skinned back onto
        the particles -- so that a trajectory here and a trajectory from the
        anchor simulator start from identical motion and differ only in the
        physics that follows.
        """
        if self.sparse is not None:
            dv = self.sparse.impulse_dv(force, self._scache)
            w_ = self._scache[0]
            v0 = torch.zeros(self.n, 3, device=self.dev).index_add_(
                0, self.sparse.pair_g, w_.unsqueeze(-1) * dv[self.sparse.pair_a]
            ).contiguous()
        else:
            dv = self.sc.impulse_dv(force)
            v0 = (self.w.unsqueeze(-1) * dv[self.idx]).sum(1).contiguous()
        self._set(self.pos_m.clone(), v0, self.eye.clone(), torch.zeros_like(self.eye))
        out = self._advance(frames, dt_mult)
        if out is None:
            raise RuntimeError("MPM left the grid on a from-rest trajectory; the "
                                "impulse is too large for this domain")
        return torch.stack([self.AC.clone()] + out)

    @torch.no_grad()
    def query(self, p, v, k, dt_mult):
        """what MPM does for k coarse steps starting from an anchor state.

        This is the DAgger label: the student is rolled out, and wherever it
        gets to, the teacher is asked from exactly there. Returns None when the
        student has left the domain MPM is defined on -- a state it cannot
        answer for is not a label, and handing it over kills the process
        outright (warp does not bounds-check its grid writes).
        """
        x, vp, F, C = self.lift(p, v)
        if not (torch.isfinite(x).all() and torch.isfinite(vp).all()
                and torch.isfinite(F).all() and torch.isfinite(C).all()):
            return None
        if x.min() < self.margin or x.max() > self.grid_lim - self.margin:
            return None
        # a deformation gradient the student's anchors imply but no material
        # could be in: MPM turns that into an enormous stress and the particles
        # leave the grid within a few substeps
        det = torch.linalg.det(F.reshape(-1, 3, 3))
        if det.min() < 0.05 or det.max() > 20.0:
            return None
        self._set(x, vp, F, C)
        out = self._advance(k, dt_mult)
        if out is None or not all(torch.isfinite(o).all() for o in out):
            return None
        return torch.stack(out)

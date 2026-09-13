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
                  sparse=None, encoder="ls", frame=False, horizon=None):
        import warp as wp
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
        # 이 선생이 굴릴 시간 창. 그 안에서 발동하지 않는 경계 조건은 등록하지
        # 않는다. 물리를 버리는 게 아니라 **닿지 않는 것을 빼는** 것이다.
        #
        # 빼야 하는 이유가 따로 있다: enforce_particle_velocity_rotation 은
        # 메서드 안에서 @wp.kernel 클로저를 정의하고, 그 커널 이름이 호출마다
        # 같다. warp 는 모듈 안에서 이름으로 커널을 찾으므로 두 번 이상 등록하면
        # 충돌해 "Failed to find forward kernel" 로 첫 스텝에서 죽는다 --
        # vasedeck 은 이 조건이 네 개다(0~0.3, 0.3~0.6, 0.6~0.9, 0.9~100).
        # 우리 창(60 프레임 x 40 substep)은 0.24 초라 첫 번째만 닿는다.
        self.horizon = float(horizon if horizon is not None
                             else 60 * 40 * float(sc.sub_dt))
        _dropped, _rot = [], 0
        for bc in cfg.get("boundary_conditions", []):
            t = bc["type"]
            if float(bc.get("start_time", 0.0)) >= self.horizon:
                _dropped.append(t)
                continue
            if t == "enforce_particle_velocity_rotation":
                _rot += 1
                if _rot > 1:
                    raise ValueError(
                        f"창 {self.horizon:g}s 안에 회전 구동기가 둘 이상이다. "
                        f"warp 가 같은 이름의 커널을 하나만 들 수 있어 첫 스텝에서 "
                        f"죽는다 -- horizon 을 줄이거나 config 를 고칠 것")
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
        if _dropped:
            print(f"[mpm] 창 {self.horizon:g}s 밖이라 등록하지 않은 경계 조건 "
                  f"{len(_dropped)}개: {sorted(set(_dropped))}", flush=True)
        # Some boundary conditions build their kernel when they are registered --
        # enforce_particle_velocity_rotation defines modify_particle_v_before_p2g
        # inside the call. warp compiles a module once and then refuses to add to
        # it, so a driver registered after anything else in that module has run
        # comes back as "Failed to find forward kernel" at the first step. Loading
        # the module here, with every driver already in it, is what keeps that from
        # depending on the order the caller happens to do things in.
        # force_load alone is not enough: warp builds a module once, and a driver
        # registered after that build adds a kernel the built module does not
        # carry -- the launch then fails with "Failed to find forward kernel".
        # Unloading first is what makes the rebuild actually happen.
        _mod = wp.context.get_module("mpm_solver_warp.mpm_solver_warp")
        if _mod is not None:
            # unload alone leaves the module gone, and force_load does not put
            # back the kernels a driver added after the first build -- that is
            # how ficus started failing on set_velocity_on_cuboid, which had
            # worked before this rebuild existed. Load the module back
            # explicitly, and if that cannot be done leave what was there.
            try:
                _mod.unload()
                _mod.load(self.wp_dev)
            except Exception:
                wp.force_load(device=self.wp_dev)
        else:
            wp.force_load(device=self.wp_dev)
        self.solver = s

        # projection, fixed at the canonical configuration like the blend weights.
        #
        # A fitted anchor set is a different set -- more of them, elsewhere, with
        # their own reach -- so it brings its own projection. Without this the
        # teacher would hand back states for the 512 sampled anchors while the
        # student is stepping 588 fitted ones.
        # 인코더 선택. "avg" 는 변위의 가중평균 -- 디코더의 전치이지 역이 아니다.
        # "ls" 는 skin 의 최소제곱 역(project_ls): 앵커도 가중치도 그대로이고
        # 새 파라미터도 학습도 없다. 표현 가능한 상태를 왕복시켰을 때 avg 는
        # 1~2% 를 잃는데(exe/probe_project_inverse.py) 그 잔차의 2/3 는 여기서
        # 사라진다. 기하 피팅은 이미 --encoder ls 를 쓸 수 있었지만, 선생은
        # 줄곧 avg 였다 -- 학생의 타깃 전부가 그 편향을 안고 있었다.
        # 피팅된 집합이 없는 경로에는 ls 가 없다(ls_factor 는 AnchorSparse 의 것).
        # 그 경우에만 가중평균으로 떨어진다.
        self.encoder = encoder if sparse is not None else "avg"
        self.sparse = sparse
        self._lsfac = None
        if sparse is not None:
            self.AC = sparse.pos.detach()
            self.fixed = sparse.fixed
            self._scache = sparse.prepare()
            if encoder == "ls":
                self._lsfac = sparse.ls_factor(self._scache)
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
        # 프레임 상태. 켜면 선생이 내주는 라벨이 앵커 위치 [M,3] 가 아니라
        # (p, u, s) [M,9] 가 되고, lift 도 F 를 형상 매칭이 아니라 (u,s) 에서
        # 조립한다. 프레임 복호로 맞춰진 기하 위에서는 이것이 유일하게 말이 되는
        # 라벨이다 -- (p,v) 만으로는 어느 프레임에서도 F 를 만들 수 없다.
        self.frame = bool(frame)
        self.det_floor, self.det_ceil = 0.05, 20.0
        self.sig_floor = 0.3
        self.bad_frac_max = 0.0    # 프레임이 아니면 예전대로 통째로 버린다
        self._fs = None
        if self.frame:
            if sparse is None:
                raise ValueError("frame=True 는 피팅된 앵커 집합(sparse)이 필요하다")
            from .frame_encode import FrameState
            self._mass_g = sc.volume[sc.keep].clone()
            self._fs = FrameState(sparse.pair_g, sparse.pair_a, sparse.N,
                                   sparse.M, self._mass_g)
            self._Yg = sparse.Xc - self._scache[1]
            r_ref = float(self._Yg.norm(dim=-1).pow(2).mean().sqrt())
            self._cx, self._cf = 1.0 / self.grid_lim, r_ref / self.grid_lim
            self.fs_gn, self.fs_cg = 6, 20
            self.frame_encoder = "closed"
            # "us" = (회전벡터 3, 로그신축 3), "F" = 자유 3x3 (9)
            self.frame_kind = "F"
            # 위반 입자를 버리지 않고 고치는 기준
            self.det_floor, self.det_ceil = 0.05, 20.0
            self.sig_floor = 0.3
            self.bad_frac_max = 0.05
            self.n_extra = 9

    # ---- 프레임 상태 --------------------------------------------------------
    @torch.no_grad()
    def project_frame(self, x, F):
        """MPM 의 (위치, 변형구배) -> 앵커의 (p, u, s) [M,9]"""
        # 기하가 맞춰진 인코더와 **같은 것**을 써야 한다. 학생의 라벨이 다른
        # 인코더에서 나오면 그 학생이 올라탈 바닥이 기하와 다르다.
        if self.frame_kind == "F":
            pj, Fa = self._fs.encode_closed_F(
                x, F.reshape(-1, 3, 3), self._scache[0], self._Yg,
                fixed=self.sparse.fixed, p_fix=self.sparse.pos.detach())
            return torch.cat([pj, Fa], -1)
        if self.frame_encoder == "closed":
            pj, uj, sj = self._fs.encode_closed(
                x, F.reshape(-1, 3, 3), self._scache[0], self._Yg,
                fixed=self.sparse.fixed, p_fix=self.sparse.pos.detach())
        else:
            pj, uj, sj, _ = self._fs.encode_joint(
                x, F.reshape(-1, 3, 3), self._scache[0], self._Yg,
                c_x=self._cx, c_F=self._cf, fixed=self.sparse.fixed,
                p_fix=self.sparse.pos.detach(), iters=self.fs_gn, cg_iters=self.fs_cg)
        return torch.cat([pj, uj, sj], -1)

    @torch.no_grad()
    def lift_frame(self, p, v, u, sfr):
        """자유 F 면 u 에 [M,9] 가 들어오고 sfr 은 안 쓴다."""
        """앵커의 (p, v, u, s) -> MPM 이 재시작할 (x, v, F, C).

        F 는 프레임에서 조립한다. 속도장의 dF/dt 는 아직 상태가 아니라(각속도·
        신축률을 안 든다) 앵커 속도의 형상 매칭 그대로이고, C = Fdot F^-1 은 그
        둘을 섞는다 -- 왕복 손실에서 속도 항을 다루는 방식과 같다.
        """
        Fm = (self._fs.decode_F(u, self._scache[0]) if self.frame_kind == "F"
              else self._fs.decode(u, sfr, self._scache[0]))
        cc = torch.zeros(self.sparse.N, 3, device=self.dev).index_add_(
            0, self.sparse.pair_g, self._scache[0].unsqueeze(-1) * p[self.sparse.pair_a])
        x = cc + torch.einsum("nij,nj->ni", Fm, self._Yg)
        _, vp, _, _ = self.sparse.lift(p, v, self._scache)
        from .anchor_sparse import inv3
        C = self.sparse.velocity_gradient(v, self._scache[0], self._scache[2],
                                           self._scache[3]) @ inv3(
            Fm + 1e-6 * torch.eye(3, device=self.dev), eps=1e-30)
        return x.contiguous(), vp.contiguous(), Fm.reshape(-1, 9).contiguous(), \
            C.reshape(-1, 9).contiguous()

    # ---- MPM particles -> anchors -----------------------------------------
    @torch.no_grad()
    def project(self, x):
        """[Nm,3] particle positions -> [M,3] anchor positions.

        The displacement is averaged, not the position: at rest this returns the
        canonical anchors exactly, which averaging positions does not.
        """
        if self.sparse is not None:
            if self.encoder == "ls":
                return self.sparse.project_ls(x, self._scache, self._lsfac)
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
            if self.encoder == "ls":
                return self.sparse.project_v_ls(vp, self._scache, self._lsfac)
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
        # (x, v, F, C) 만 넣어서는 솔버가 초기 상태로 돌아가지 않는다.
        #
        # **시뮬레이션 시각**이 남는다. p2g2p 가 self.time 을 전진시키고, 시간에
        # 의존하는 경계 조건 -- wolf 의 release_particles_sequentially, plane 과
        # vasedeck 의 enforce_particle_velocity_rotation, cuboid 의 start/end_time
        # -- 이 그 시각을 읽는다. 초기화하지 않으면 두 번째 궤적은 첫 궤적이
        # 끝난 시각에서 시작한다. 실측: wolf 에서 같은 임펄스를 세 번 굴리면
        # 변위가 0.00% -> 0.01% -> 1.74% 로 커졌다. ficus 는 cuboid 가 0~1e3 이라
        # 우리 구간에서 시간 의존이 없어 4 회에 0.4% 만 표류했고, 그래서 여태
        # 드러나지 않았다.
        self.solver.time = 0.0
        # 소성 상태도 import 함수가 없어 남는다. wolf 의 누적은 시각 탓이었지만
        # (Jp 만 지워서는 안 고쳐졌다) 소성 이력을 물려주는 것 자체가 틀렸다.
        jp = getattr(getattr(self.solver, "mpm_state", None), "particle_Jp", None)
        if jp is not None:
            jp.zero_()

    @torch.no_grad()
    def _in_domain(self):
        """cheap, and the only thing standing between a diverging query and a
        dead process: MPM writes to the cells around each particle and warp does
        not bounds-check, so a particle that leaves the grid takes the whole run
        down with CUDA error 700 rather than raising something catchable"""
        # min 과 max 를 각각 .item() 하면 GPU->CPU 동기화가 두 번 걸린다. 서브스텝마다
        # 부르는 자리라 그 두 배가 그대로 벽시계에 실린다 -- 한 번에 옮긴다.
        x = self.solver.export_particle_x_to_torch()
        lo, hi = torch.stack((x.min(), x.max())).tolist()
        return lo == lo and self.margin < lo and hi < self.grid_lim - self.margin

    @torch.no_grad()
    def _vel_safe(self, substeps):
        """다음 검사까지 격자를 벗어날 수 있는 속도인가.

        _in_domain 은 위치만 보고 check_every substep 마다 돈다. 고정점이 없는
        씬은 물체가 통째로 가속돼 그 사이에 격자를 넘어가고, warp 는 격자 쓰기에
        경계 검사가 없어 프로세스가 통째로 죽는다 -- 잡을 수 있는 예외가 아니다.
        속도로 미리 거르면 그 창을 닫을 수 있고, 비용은 리덕션 한 번이다.
        """
        v = self.solver.export_particle_v_to_torch()
        vmax = float(v.norm(dim=-1).max())
        return vmax == vmax and vmax * substeps * self.sc.sub_dt < self.margin

    @torch.no_grad()
    def _advance(self, frames, dt_mult, check_every=4):
        """returns None if the run left the domain part-way, rather than dying.

        Checked mid-frame as well: a state that blows up does so within a few
        substeps, and 40 of them is long enough to go from plausible to out of
        bounds with nothing observed in between.
        """
        out = []
        if not self._vel_safe(check_every):
            return None
        for _ in range(frames):
            for k in range(dt_mult):
                self.solver.p2g2p(None, self.sc.sub_dt, device=self.wp_dev)
                if (k + 1) % check_every == 0:
                    if not self._in_domain() or not self._vel_safe(check_every):
                        return None
            x = self.solver.export_particle_x_to_torch()
            if self.frame:
                out.append(self.project_frame(
                    x, self.solver.export_particle_F_to_torch()))
            else:
                out.append(self.project(x))
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
            # 도메인을 벗어난 궤적은 라벨이 아니라 버릴 표본이다. 예외로 던지면
            # 학습 전체가 죽는데, (K, r) 계열은 반경이 작을수록 같은 RMS 를 맞추려고
            # 힘이 몰려 이 경우가 정상적으로 생긴다 -- 기하 피팅 쪽은 처음부터
            # None 을 받아 버리고 넘어간다(106 중 102 만 남는 식). 호출자가 다시
            # 뽑을 수 있도록 같은 신호를 준다.
            return None
        # 정지 상태. 프레임이면 (p, u, s) = (AC, 0, 0) -- 회전 없음, 신축 1.
        if self.frame:
            if self.frame_kind == "F":
                eye9 = torch.eye(3, device=self.dev).reshape(1, 9).expand(
                    self.AC.shape[0], 9)
                rest = torch.cat([self.AC, eye9], -1)
            else:
                rest = torch.cat(
                    [self.AC, torch.zeros(self.AC.shape[0], 6, device=self.dev)], -1)
        else:
            rest = self.AC.clone()
        return torch.stack([rest] + out)

    @torch.no_grad()
    def query(self, p, v, k, dt_mult, u=None, sfr=None):
        """what MPM does for k coarse steps starting from an anchor state.

        This is the DAgger label: the student is rolled out, and wherever it
        gets to, the teacher is asked from exactly there. Returns None when the
        student has left the domain MPM is defined on -- a state it cannot
        answer for is not a label, and handing it over kills the process
        outright (warp does not bounds-check its grid writes).
        """
        x, vp, F, C = (self.lift_frame(p, v, u, sfr) if self.frame
                        else self.lift(p, v))
        if not (torch.isfinite(x).all() and torch.isfinite(vp).all()
                and torch.isfinite(F).all() and torch.isfinite(C).all()):
            return None
        if x.min() < self.margin or x.max() > self.grid_lim - self.margin:
            return None
        # a deformation gradient the student's anchors imply but no material
        # could be in: MPM turns that into an enormous stress and the particles
        # leave the grid within a few substeps
        #
        # 원래는 하나라도 걸리면 표본 전체를 버렸다. 자유 F 는 성분별 평균이라
        # 0.31% 의 가우시안에서 한 축이 눌리고 0.038% 는 det<0 이 되는데, 그러면
        # 거의 모든 표본이 버려진다 -- 실측으로 DAgger 수집이 5000 반복에 0 건이었다
        # (다른 학생은 150 건). DAgger 가 없으면 후반에 퇴행한다(STU_nodag: 32500 에서
        # 13.5% 최저 -> 60000 에서 18.4%). 그래서 버리는 대신 **위반한 입자의 F 만**
        # 특이값 바닥으로 투영한다. 0.3% 만 손대므로 라벨 품질 손실이 작다.
        Fm = F.reshape(-1, 3, 3)
        det = torch.linalg.det(Fm)
        bad = (det < self.det_floor) | (det > self.det_ceil)
        frac = float(bad.float().mean())
        if frac > self.bad_frac_max:
            return None                      # 너무 많으면 상태 자체가 망가진 것이다
        if bad.any():
            U, S, Vh = torch.linalg.svd(Fm[bad])
            S = S.clamp(min=self.sig_floor)
            # 반사(det<0)는 가장 작은 특이방향을 되돌려 회전으로 만든다
            flip = torch.linalg.det(U) * torch.linalg.det(Vh) < 0
            if flip.any():
                U = U.clone(); U[flip, :, -1] = -U[flip, :, -1]
            Fm = Fm.clone()
            Fm[bad] = U @ torch.diag_embed(S) @ Vh
            F = Fm.reshape(-1, 9).contiguous()
        self._set(x, vp, F, C)
        out = self._advance(k, dt_mult)
        if out is None or not all(torch.isfinite(o).all() for o in out):
            return None
        return torch.stack(out)

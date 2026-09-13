"""Build the anchor/Gaussian scene from a 3DGS ply + a DreamPhysics config.

Every experiment script had its own copy of this and they had already drifted
apart in small ways (which cloud defines MPM space, whether the impulse's P2G
weights are masked to the material subset). One place, one behaviour.
"""
from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass

import math

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
    n_grid: int
    # which Gaussians of the PLY survived sim_area, or None when the scene asks
    # for no crop. A renderer loads the PLY itself and has to be told, or it
    # holds two million splats against a cropped cloud of eighty thousand.
    crop: object
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
        raise NotImplementedError(
            "임펄스 계열 분할(균일 / 힘장 / 포크)은 제거했다. 이 프로젝트의 임펄스는 "
        "(포크 개수 K, 반경 r) 로만 결정되는 다중 포크 하나뿐이다 -- "
        "Scene.random_multi_poke 를 쓸 것. 기하 피팅·학생 학습·평가가 모두 "
        "같은 계열에서 뽑혀야 서로 견줄 수 있다.")

    def random_multi_poke(self, gen, k, radius, magnitude, peak_cap=10.0):
        """K 개의 국소 밀기. 균일·힘장·포크를 한 계열로 덮는다.

        K=1, r 작음        -> 포크(공간 지지)
        K 큼,  r ~ sigma    -> 힘장(공간 주파수: 전체가 r 크기 조각으로 밀림)
        K=1, r ~ 물체 크기  -> 균일

        지금 세 계열 어디에도 없는 중간 -- 국소적인 힘 두셋이 동시에 걸리는 경우
        -- 도 여기서 나온다. 실제 상호작용에 가까운 쪽이다.

        정규화는 반드시 RMS 다. 최댓값으로 맞추면 국소적인 draw 가 운반하는
        운동량이 작아져 변위가 4.3 배 줄어든다(random_force_field 참조) -- K 와 r
        을 독립으로 뽑는 이 계열에서는 그 편향이 그대로 커버리지 구멍이 된다.
        """
        dev = self.pos.device
        mat = torch.nonzero(self.keep, as_tuple=False).squeeze(-1)
        f = torch.zeros(self.pos.shape[0], 3, device=dev)
        for _ in range(int(k)):
            c = self.pos[mat[torch.randint(mat.shape[0], (1,), device=dev,
                                            generator=gen)]][0]
            d2 = ((self.pos - c) ** 2).sum(-1)
            w = torch.exp(-d2 / (2.0 * float(radius) ** 2)) * self.keep.float()
            q, r_ = torch.linalg.qr(torch.randn(3, 3, device=dev, generator=gen))
            q = q * torch.sign(torch.diagonal(r_)).unsqueeze(0)
            f = f + w.unsqueeze(-1) * q[:, 0].unsqueeze(0)
        m = f[self.keep]
        rms = m.norm(dim=-1).pow(2).mean().sqrt().clamp(min=1e-12)
        f = f / rms * float(magnitude)

        # 최댓값 상한. 반경이 앵커 간격 아래로 내려가면 힘을 받는 가우시안이 몇 개
        # 남지 않아, RMS 정규화가 그 몇 개에 극단적인 값을 몰아준다. 그 입자들이 한
        # 서브스텝에 격자를 벗어나면 warp 커널이 잘못된 메모리에 쓰고 프로세스가
        # 통째로 죽는다 -- 도메인 검사는 커널 밖에서 도는 것이라 못 막는다
        # (실측: mp_rmin=0.125 로 궤적 생성 중 CUDA error 700, 103/106 에서).
        # RMS 는 그대로 두고 꼭대기만 눌러, 균일하거나 반경이 넓은 draw 는 이 한계에
        # 닿지 않고 지나간다.
        peak = f.norm(dim=-1).max()
        cap = float(magnitude) * float(peak_cap)
        if float(peak) > cap:
            f = f * (cap / peak)
        return f

    def random_poke(self, gen, radius, magnitude):
        raise NotImplementedError(
            "임펄스 계열 분할(균일 / 힘장 / 포크)은 제거했다. 이 프로젝트의 임펄스는 "
        "(포크 개수 K, 반경 r) 로만 결정되는 다중 포크 하나뿐이다 -- "
        "Scene.random_multi_poke 를 쓸 것. 기하 피팅·학생 학습·평가가 모두 "
        "같은 계열에서 뽑혀야 서로 견줄 수 있다.")

    @property
    def extent(self):
        a = self.anchor_canonical
        return float((a.max(0).values - a.min(0).values).norm())

    def explicit_step(self, p, v, gp, n=1):
        """n explicit substeps; returns (p, v, gaussian_pos).

        The scene's own drivers are applied here, the way the fitted simulator
        applies them. They were missing, and this is the baseline every table is
        read against: on plane it meant the unfitted discretisation stood
        perfectly still -- 0.2% of MPM's propeller motion -- and still scored a
        lower mean error than the fitted one, because most of that scene barely
        moves and standing still is free on those particles.
        """
        from .anchor_sparse import apply_bcs_to, parse_bcs

        if not hasattr(self, "_bc"):
            self._bc = parse_bcs(self, p.device)
            self._t = 0.0
        wall, bcs = self._bc
        g = self.gravity if self.gravity.abs().sum() > 0 else None
        with torch.enable_grad():
            for _ in range(n):
                p, v, gp, _ = self.sim.step(p, v, self.mass, gp, self.volume, self.mu,
                                             self.lam, self.sub_dt, gravity=g,
                                             damping=self.damping, fixed_mask=self.fixed_mask)
                if wall is not None:
                    lo, hi = wall
                    v = torch.where((p < lo) & (v < 0), torch.zeros_like(v), v)
                    v = torch.where((p > hi) & (v > 0), torch.zeros_like(v), v)
                    p = p.clamp(min=lo, max=hi)
                if bcs:
                    p, v = apply_bcs_to(bcs, p, v, self._t)
                self._t += self.sub_dt
        return p, v, gp

    def reset_time(self):
        """the drivers are scripted against simulated time, so a fresh rollout
        has to start the clock again"""
        self._t = 0.0

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

    def skin(self, p, gp, grad=False):
        """Gaussian positions implied by an anchor configuration (dt=0 step).

        grad=True 는 그래프를 유지한다. 영상 손실로 앵커를 학습할 때 지도 신호가
        가우시안 -> 화면으로만 오므로, 여기서 끊기면 손실에 grad_fn 이 없다.
        """
        if grad:
            # step() 은 힘을 뽑느라 앵커를 detach 하고 가우시안도 detach 해서
            # 돌려준다. 학습 경로에서는 형상 매칭만 통과시키는 skin_only 를 쓴다.
            gp, _ = self.sim.skin_only(p, gp)
            return gp
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

    # PhysGaussian's preprocessing, in its order (gs_simulation.py): rotate about
    # the ORIGIN, crop to sim_area in that rotated frame, then normalise so the
    # largest extent is `scale` and the centre sits at (1,1,1). The boundary
    # conditions are authored in the frame this produces, so getting any step
    # wrong puts the material where no wall or driver reaches it -- which is what
    # made vasedeck's rotation driver act on empty space.
    for deg, ax in zip(cfg.get("rotation_degree", []) or [],
                        cfg.get("rotation_axis", []) or []):
        th = math.radians(float(deg))
        c_, s_ = math.cos(th), math.sin(th)
        if int(ax) == 0:
            R = [[1, 0, 0], [0, c_, -s_], [0, s_, c_]]
        elif int(ax) == 1:
            R = [[c_, 0, s_], [0, 1, 0], [-s_, 0, c_]]
        else:
            R = [[c_, -s_, 0], [s_, c_, 0], [0, 0, 1]]
        xyz = xyz @ torch.tensor(R, device=dev, dtype=xyz.dtype).T

    area = cfg.get("sim_area")
    crop = None
    if area is not None:
        inside = torch.ones(xyz.shape[0], dtype=torch.bool, device=dev)
        for i in range(3):
            inside &= (xyz[:, i] > area[2 * i]) & (xyz[:, i] < area[2 * i + 1])
        # outside the simulated area a Gaussian is not material and not drawn by
        # the physics; dropping it here keeps every downstream index consistent
        crop = inside
        xyz, op, keep = xyz[inside], op[inside], keep[inside]
        g._xyz = g._xyz[inside]
        g._features_dc = g._features_dc[inside]
        g._features_rest = g._features_rest[inside]
        g._opacity = g._opacity[inside]
        g._scaling = g._scaling[inside]
        g._rotation = g._rotation[inside]

    xw = xyz[keep]
    pmin, pmax = xw.min(0).values, xw.max(0).values
    mid = (pmin + pmax) / 2
    sc = float(cfg.get("scale") or 1.0) / (pmax - pmin).max()
    one = torch.tensor([1., 1., 1.], device=dev)
    to_mpm = lambda q: (q - mid) * sc + one
    undo = lambda q: (q - one) / sc + mid
    pos = to_mpm(xyz).contiguous()
    pm = pos[keep].contiguous()
    N = pos.shape[0]

    # the resolution the scene asks for; the teacher is built with the same one
    n_grid = int(cfg.get("n_grid", n_grid))
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
                 crop=crop,
                  anchor_canonical=ac, mass=mass, fixed_mask=fixed, sim=sim,
                  gravity=torch.tensor(cfg["g"], dtype=torch.float32, device=dev),
                  n_grid=n_grid,
                  sub_dt=float(cfg["substep_dt"]),
                  damping=float(cfg.get("grid_v_damping_scale", 1.0)),
                  to_mpm=to_mpm, undo=undo)

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

try:
    import sparsestep
except Exception:                                   # unbuilt, or not on the path
    sparsestep = None

from .anchor_fit import _R_to_quat, closest_rotation, det3, inv3, quat_to_R


class Traj:
    """a trajectory held on the CPU in half precision, handed out a frame at a
    time in the precision the physics wants.

    At the data volume the student is trained on, the set is 15.7 GB: it does not
    belong on a 24 GB card, and a single frame is 2 MB, so the transfer is
    nothing next to the forty substeps it feeds.

    It lives here rather than in the script that writes the cache because the
    cache is pickled: defined in a __main__, it can only be read back by the one
    program that wrote it.
    """

    def __init__(self, t, dev="cuda"):
        self.t = t
        self.dev = dev

    def __getitem__(self, i):
        # getattr, because a cache written before this class moved into the
        # library unpickles without the attribute it never had
        return self.t[i].to(getattr(self, "dev", "cuda"), torch.float32,
                             non_blocking=True)

    def __len__(self):
        return self.t.shape[0]

    @property
    def shape(self):
        return self.t.shape


def parse_bcs(sc, dev):
    """the scene's scripted conditions, as the anchors need them.

    This used to live inside AnchorSparse, which meant only the fitted simulator
    ever saw a driver. Scene.explicit_step -- the unfitted discretisation, and
    the baseline every comparison is read against -- stepped with gravity and
    nothing else. On plane that showed as an aeroplane frozen in place: its
    propeller moved 0.2% of what MPM's did, and the frozen panel then scored
    *better* on mean error than the fitted one, because four fifths of that
    scene barely moves and standing still gets those particles for free.

    -> (wall, bcs). wall is the bounding box as (lo, hi), or None.
    """
    wall, bcs = None, []
    for bc in sc.cfg.get("boundary_conditions", []):
        t = bc["type"]
        if t == "bounding_box":
            # the same three cells of padding, at the scene's resolution
            cell = 2.0 / float(getattr(sc, "n_grid", 100) or 100)
            wall = (3 * cell, 2.0 - 3 * cell)
        elif t == "enforce_particle_translation":
            bcs.append({
                "kind": "translate",
                "point": torch.tensor(bc["point"], device=dev),
                "size": torch.tensor(bc["size"], device=dev),
                "vel": torch.tensor(bc["velocity"], device=dev),
                "t0": float(bc.get("start_time", 0.0)),
                "t1": float(bc.get("end_time", 1e3))})
        elif t == "enforce_particle_velocity_rotation":
            bcs.append({
                "kind": "rotate",
                "point": torch.tensor(bc["point"], device=dev),
                "normal": torch.tensor(bc["normal"], device=dev, dtype=torch.float32),
                "hr": torch.tensor(bc["half_height_and_radius"], device=dev),
                "rot": float(bc["rotation_scale"]),
                "tr": float(bc["translation_scale"]),
                "t0": float(bc.get("start_time", 0.0)),
                "t1": float(bc.get("end_time", 1e3))})
        elif t == "surface_collider":
            bcs.append({
                "kind": "surface",
                "point": torch.tensor(bc["point"], device=dev),
                "normal": torch.tensor(bc["normal"], device=dev, dtype=torch.float32),
                "surface": bc.get("surface", "sticky"),
                "friction": float(bc.get("friction", 0.0)),
                "t0": float(bc.get("start_time", 0.0)),
                "t1": float(bc.get("end_time", 1e3))})
    for b in bcs:
        if "normal" in b:
            b["normal"] = b["normal"] / b["normal"].norm().clamp(min=1e-12)
    return wall, bcs


def apply_bcs_to(bcs, p, v, t):
    """the scripted conditions, on anchor positions and velocities, at time t"""
    for b in bcs:
        if not (b["t0"] <= t < b["t1"]):
            continue
        if b["kind"] == "translate":
            inside = ((p - b["point"]).abs() <= b["size"]).all(-1, keepdim=True)
            v = torch.where(inside, b["vel"].reshape(1, 3).expand_as(v), v)
        elif b["kind"] == "rotate":
            n = b["normal"]
            d = p - b["point"]
            along = (d * n).sum(-1, keepdim=True)
            radial = d - along * n
            inside = ((along.abs() <= b["hr"][0]) &
                      (radial.norm(dim=-1, keepdim=True) <= b["hr"][1]))
            spin = torch.cross(n.reshape(1, 3).expand_as(radial), radial, dim=-1)
            v = torch.where(inside, b["rot"] * spin + b["tr"] * n.reshape(1, 3), v)
        elif b["kind"] == "surface":
            n = b["normal"]
            dist = ((p - b["point"]) * n).sum(-1, keepdim=True)
            vn = (v * n).sum(-1, keepdim=True)
            hit = (dist < 0) & (vn < 0)
            if b["surface"] == "sticky":
                v = torch.where(hit, torch.zeros_like(v), v)
            else:
                vt = v - vn * n
                keep = (1.0 + b["friction"] * vn / vt.norm(dim=-1, keepdim=True)
                        .clamp(min=1e-12)).clamp(min=0)
                v = torch.where(hit, vt * keep, v)
            p = torch.where(hit, p - dist * n, p)
    return p, v


class AnchorSparse(nn.Module):
    """anchors with elliptical, compactly supported reach and a variable count"""

    def __init__(self, sc, c=0.25, eig_floor=0.02, polar_iters=6, margin=1.25,
                 checkpoint_substeps=True, s_lo=0.25, s_hi=4.0, cfl_frac=0.05,
                 quad=False, oriented=False, softmax_w=False, softmax_k=16):
        super().__init__()
        from scipy.spatial import cKDTree

        sim = sc.sim
        dev = sc.pos.device
        self.c = c
        # 가중치를 잘린 가우시안(G(x)-c) 대신 **가우시안당 최근접 k 개 위 softmax**
        # 로 줄지. 잘린 쪽은 붙는 앵커 수가 제각각이고 경계 밖이 정확히 0 이다.
        self.softmax_w = bool(softmax_w)
        self.softmax_k = int(softmax_k)
        self.eig_floor = eig_floor
        self.polar_iters = polar_iters
        self.margin = margin
        self.checkpoint_substeps = checkpoint_substeps
        # quadratic shape matching: the deformation a Gaussian is allowed to see
        # in its neighbourhood becomes affine PLUS quadratic. Nothing is carried
        # -- the extra freedom is still derived from where the anchors are -- so
        # the fit gains expressiveness without gaining anything that can drift.
        self.quad = quad
        # oriented anchors: an anchor stops being a point and becomes an oriented
        # ellipsoid that carries its own rotation and spins under torque. Its
        # second moment enters the shape matching directly, so a Gaussian no
        # longer has to infer the local frame from where its neighbours happen to
        # be -- which is what recovers only 17% of MPM's F. Three extra degrees
        # of freedom, held in SO(3) and driven by a restoring torque, rather than
        # the nine unconstrained ones that made a carried F drift.
        self.oriented = oriented
        self._o = self._wv = None
        # simulated time, which the scripted boundary conditions switch on. Reset
        # by project(); advanced by every substep.
        self._t = 0.0
        # a body force, which the ficus config sets to zero and every other
        # scene does not. Without it the simulator simply does not fall, and a
        # fit cannot recover a term that is missing from the model.
        self.register_buffer("gravity", sc.gravity.clone().reshape(1, 3))
        # the domain MPM's grid boundary enforces: within three cells of the
        # edge it zeroes the outward velocity component, so material slides
        # along the wall rather than through it. Without the same condition the
        # anchors fall out of the world and no fit can follow that.
        self.wall = None
        # boundary conditions that act on the material directly rather than
        # through the grid. MPM applies them per particle; an anchor stands for
        # the material around it, so the same test on the anchor's own position
        # is the faithful translation.
        # F carried in time, MPM style. None until a rollout starts it.
        self.acc_blend = 0.0
        self._Facc = None
        self._v_now = None
        self.wall, self.bcs = parse_bcs(sc, dev)
        if False:
          for bc in sc.cfg.get("boundary_conditions", []):
            t = bc["type"]
            if t == "bounding_box":
                cell = 2.0 / float(getattr(sc, "n_grid", 100) or 100)
                self.wall = (3 * cell, 2.0 - 3 * cell)
            elif t == "enforce_particle_translation":
                self.bcs.append({
                    "kind": "translate",
                    "point": torch.tensor(bc["point"], device=dev),
                    "size": torch.tensor(bc["size"], device=dev),
                    "vel": torch.tensor(bc["velocity"], device=dev),
                    "t0": float(bc.get("start_time", 0.0)),
                    "t1": float(bc.get("end_time", 1e3))})
            elif t == "enforce_particle_velocity_rotation":
                self.bcs.append({
                    "kind": "rotate",
                    "point": torch.tensor(bc["point"], device=dev),
                    "normal": torch.tensor(bc["normal"], device=dev, dtype=torch.float32),
                    "hr": torch.tensor(bc["half_height_and_radius"], device=dev),
                    "rot": float(bc["rotation_scale"]),
                    "tr": float(bc["translation_scale"]),
                    "t0": float(bc.get("start_time", 0.0)),
                    "t1": float(bc.get("end_time", 1e3))})
            elif t == "surface_collider":
                self.bcs.append({
                    "kind": "surface",
                    "point": torch.tensor(bc["point"], device=dev),
                    "normal": torch.tensor(bc["normal"], device=dev, dtype=torch.float32),
                    "surface": bc.get("surface", "sticky"),
                    "friction": float(bc.get("friction", 0.0)),
                    "t0": float(bc.get("start_time", 0.0)),
                    "t1": float(bc.get("end_time", 1e3))})
          for b in self.bcs:
            if "normal" in b:
                b["normal"] = b["normal"] / b["normal"].norm().clamp(min=1e-12)
        # an anchor that shrinks far enough holds almost nothing, its mass goes
        # with it, and the accelerations it sees go up without bound; one that
        # grows without limit swallows the object. Both ends were reachable --
        # the unbounded fit went to NaN around its 250th iteration
        self.s_lo = s_lo * sim.radius
        self.s_hi = s_hi * sim.radius
        self.sim_radius = float(sim.radius)
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
        # 스키닝에서 앵커가 자기 지분을 정하는 진폭. 이것이 없으면 한 가우시안을
        # 여러 앵커가 나눠 가질 때 지분이 오직 거리로만 정해지고, 지분을 늘리려면
        # log_s 를 키울 수밖에 없어 "얼마나 멀리 닿는가" 와 "얼마나 지배하는가"
        # 가 한 파라미터에 묶인다.
        self.log_amp = nn.Parameter(torch.zeros(sc.M, device=dev))
        self.register_buffer("fixed", sc.fixed_mask.clone())
        self.register_buffer("B_ref", torch.zeros((), device=dev))
        # 앵커끼리 직접 주고받는, 학습되는 중심력. 없으면 지금까지의 시뮬레이터
        # 그대로다 -- 켜더라도 출력이 0 으로 초기화되어 있어 시작은 동일하다.
        self.edge = None
        self.edge_only = False
        self.astress = None
        # 프레임 동역학: 앵커가 (회전, 신축) 을 상태로 들고 F 를 거기서 조립한다.
        # 켜지면 롤아웃 상태가 (p, v, o, s, wv, sv) 로 늘고, gaussian_pos 도
        # 위치의 미분이 아니라 그 프레임에서 F 를 받는다.
        self.fdyn = None
        self._fo = None
        self._fs = None
        # 평균은 앵커 몇 개만 한계를 크게 넘는 상황을 묻어버린다 -- 시뮬레이션은
        # 그 몇 개 때문에 터지는데도. "max" 는 최악의 앵커를 직접 벌한다.
        self.cfl_agg = "mean"
        # 앵커 질량의 바닥(중앙값 대비). 0 이면 끈다 -- 정확도를 해치므로 기본은 끔.
        self.mass_floor = 0.0
        # 이 비율 아래의 질량을 가진 앵커는 적분에서 제외한다
        self.mass_freeze = 0.0
        self._light = None
        # F^-T 의 릿지 비율. 1e-6 이면 거의 납작해진 요소에서 F^-T 가 1e6 까지
        # 가고, 부피항이 그걸 그대로 증폭해 응력이 폭주한다. 실측: F^-T=8154 에서
        # 응력 5e6, 가속도 3e6, 11 서브스텝 뒤 발산.
        self.finv_ridge = 1e-6
        # NaN/Inf 추적. 켜면 서브스텝마다 주요 중간값을 검사해 "어느 양이 몇 번째
        # 서브스텝에서 처음 무한이 됐는가" 를 기록한다. 검사 결과는 GPU 에 두고
        # 롤아웃이 끝난 뒤 한 번만 읽으므로 스텝마다 동기화하지 않는다.
        self.nan_trace = False
        self._trace = {}
        self._mag = {}
        self._who = {}
        # trace_reset 전에도 _chk 가 불릴 수 있으므로(초기 보정) 여기서 만든다
        self._clean = torch.ones((), device=dev)
        self._k = 0
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
        # log_amp 는 weights() 에서 tanh 로 유계라 잘라낼 것이 없다

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
        # 앵커 수는 밀도 제어로 바뀌므로 간선도 여기서 함께 다시 만든다
        if getattr(self, "edge", None) is not None:
            self.edge.rebuild(self.pos, self.sim_radius, self.dt)
        if getattr(self, "astress", None) is not None:
            self.astress.rebuild(self)
        return self.pair_g.shape[0]

    def _mahal2(self, g, a):
        d = self.Xc[g] - self.pos[a]
        R = quat_to_R(self.quat)[a]
        s = self.log_s.exp()[a].clamp(min=1e-6)
        local = torch.einsum("pji,pj->pi", R, d) / s
        return (local * local).sum(-1)

    def _weights_softmax(self):
        """가우시안마다 최근접 앵커 k 개를 골라 그 위에서 softmax 로 가중치를 준다.

        기본 방식은 G(x) - c 를 clamp 한 **잘린** 가우시안이라, 한 가우시안에
        붙는 앵커 수가 제각각이고 경계 밖은 정확히 0 이다. 여기서는 잘라내지 않고
        일반 가우시안을 쓰되 **최근접 k 개로만 제한**하고 그 위에서 정규화한다.
        지수 안의 값이 -0.5 * mahal^2 이므로 이 정규화가 곧 softmax 다.

        붙는 앵커 수가 k 로 고정되는 것이 잘린 방식과의 핵심 차이다.
        """
        m2 = self._mahal2(self.pair_g, self.pair_a)              # [P]
        logit = -0.5 * m2 + (2.0 * torch.tanh(self.log_amp / 2.0))[self.pair_a]
        k = self.softmax_k

        # 가우시안별 상위 k 개만 남긴다. 짝 목록은 가우시안별로 이어져 있으므로
        # 각 짝이 자기 가우시안 안에서 몇 번째인지(slot)를 구해 [N, kmax] 로 펴고
        # topk 로 문턱을 잡는다.
        cnt = torch.zeros(self.N, device=self.dev, dtype=torch.long).index_add_(
            0, self.pair_g, torch.ones_like(self.pair_g))
        kmax = int(cnt.max()) if cnt.numel() else 0
        if kmax > k:
            pos = torch.arange(self.pair_g.numel(), device=self.dev)
            first = torch.full((self.N,), pos.numel(), device=self.dev,
                               dtype=torch.long)
            first.scatter_reduce_(0, self.pair_g, pos, reduce="amin",
                                  include_self=True)
            slot = pos - first[self.pair_g]
            dense = torch.full((self.N, kmax), -1e30, device=self.dev,
                               dtype=logit.dtype)
            dense[self.pair_g, slot] = logit
            thr = dense.topk(k, dim=1).values[:, -1]                 # [N]
            logit = torch.where(logit >= thr[self.pair_g], logit,
                                torch.full_like(logit, -1e30))
        # 가우시안별 softmax
        mx = torch.full((self.N,), -1e30, device=self.dev, dtype=logit.dtype)
        mx = mx.index_reduce_(0, self.pair_g, logit, "amax", include_self=True)
        e = torch.exp(logit - mx[self.pair_g])
        tot = torch.zeros(self.N, device=self.dev, dtype=e.dtype).index_add_(
            0, self.pair_g, e)
        return e / tot[self.pair_g].clamp(min=1e-12)

    def weights(self):
        """[P], normalised per Gaussian. Zero at the boundary of the region and
        zero outside it, so a Gaussian entering or leaving does so continuously"""
        if self.softmax_w:
            return self._weights_softmax()
        w = (torch.exp(-0.5 * self._mahal2(self.pair_g, self.pair_a)) - self.c).clamp(min=0)
        # 진폭도 tanh 로 유계다. 한 앵커가 지분을 독점하면 나머지가 그래디언트를
        # 잃으므로 e^{-2} ~ e^{2} 안에 두되, 경계에서 죽지 않게 매끄럽게 준다.
        w = w * (2.0 * torch.tanh(self.log_amp / 2.0)).exp()[self.pair_a]
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
    @staticmethod
    def _qbasis(u):
        """[P,3] -> [P,9]: the affine part, then the six quadratic terms"""
        x, y, z = u[:, 0], u[:, 1], u[:, 2]
        return torch.stack([x, y, z, x * x, y * y, z * z,
                            x * y, y * z, z * x], -1)

    @staticmethod
    def _qjac(u):
        """[N,3] -> [N,9,3]: d(qbasis)/du, which turns T into a real F"""
        n = u.shape[0]
        x, y, z = u[:, 0], u[:, 1], u[:, 2]
        o, zz = torch.ones_like(x), torch.zeros_like(x)
        rows = [(o, zz, zz), (zz, o, zz), (zz, zz, o),
                (2 * x, zz, zz), (zz, 2 * y, zz), (zz, zz, 2 * z),
                (y, x, zz), (zz, z, y), (z, zz, x)]
        return torch.stack([torch.stack(r, -1) for r in rows], 1)

    def _driven_mask(self, p):
        """anchors a scripted driver acts on, at the configuration given.

        Time is ignored on purpose: a driver that starts at t=0.6 still needs
        anchors there when it arrives, and the fit runs long before that.
        """
        m = torch.zeros(p.shape[0], dtype=torch.bool, device=p.device)
        for b in self.bcs:
            if b["kind"] == "translate":
                m |= ((p - b["point"]).abs() <= b["size"]).all(-1)
            elif b["kind"] == "rotate":
                n = b["normal"]
                d = p - b["point"]
                along = (d * n).sum(-1)
                radial = (d - along.unsqueeze(-1) * n).norm(dim=-1)
                m |= (along.abs() <= b["hr"][0]) & (radial <= b["hr"][1])
        return m

    def apply_bcs(self, p, v, t):
        """the scripted boundary conditions, on the anchors, at simulated time t"""
        return apply_bcs_to(self.bcs, p, v, t)

    def _apply_bcs_unused(self, p, v, t):
        for b in self.bcs:
            if not (b["t0"] <= t < b["t1"]):
                continue
            if b["kind"] == "translate":
                inside = ((p - b["point"]).abs() <= b["size"]).all(-1, keepdim=True)
                v = torch.where(inside, b["vel"].reshape(1, 3).expand_as(v), v)
            elif b["kind"] == "rotate":
                n = b["normal"]
                d = p - b["point"]
                along = (d * n).sum(-1, keepdim=True)
                radial = d - along * n
                inside = ((along.abs() <= b["hr"][0]) &
                          (radial.norm(dim=-1, keepdim=True) <= b["hr"][1]))
                # a rigid spin about the axis, plus a slide along it
                spin = torch.cross(n.reshape(1, 3).expand_as(radial), radial, dim=-1)
                v = torch.where(inside, b["rot"] * spin + b["tr"] * n.reshape(1, 3), v)
            elif b["kind"] == "surface":
                n = b["normal"]
                dist = ((p - b["point"]) * n).sum(-1, keepdim=True)
                vn = (v * n).sum(-1, keepdim=True)
                hit = (dist < 0) & (vn < 0)
                if b["surface"] == "sticky":
                    v = torch.where(hit, torch.zeros_like(v), v)
                else:
                    vt = v - vn * n
                    # Coulomb friction against the normal impulse
                    keep = (1.0 + b["friction"] * vn / vt.norm(dim=-1, keepdim=True)
                            .clamp(min=1e-12)).clamp(min=0)
                    v = torch.where(hit, vt * keep, v)
                p = torch.where(hit, p - dist * n, p)
        return p, v

    def _boundaries(self, p, v):
        """the wall and the scripted conditions, in the order MPM applies them"""
        if self.wall is not None:
            lo, hi = self.wall
            v = torch.where((p < lo) & (v < 0), torch.zeros_like(v), v)
            v = torch.where((p > hi) & (v > 0), torch.zeros_like(v), v)
            p = p.clamp(min=lo, max=hi)
        if self.bcs:
            p, v = self.apply_bcs(p, v, self._t)
        return p, v

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
        self._chk_min("가우시안_trB", tr)
        self._chk_min("가우시안_detB", torch.linalg.det(B))
        eps = (self.eig_floor * tr.clamp(min=self.B_ref)).reshape(-1, 1, 1)
        eye = torch.eye(3, device=self.dev)
        Binv = inv3(B + eps * eye)
        blocked = eps * Binv
        mass = torch.zeros(self.M, device=self.dev).index_add_(
            0, self.pair_a, self.dens_vol[self.pair_g] * w)
        # 가우시안을 거의 안 쥔 앵커는 질량이 1e-11 까지 내려간다. 가우시안 경로는
        # 그런 앵커가 받는 힘도 같은 w 에 비례해 함께 0 이 되므로 상쇄되지만,
        # 앵커 그래프의 힘은 이웃의 부피에 비례할 뿐 자기 질량과 무관해서
        # 상쇄되지 않는다 -- 실측으로 힘 1.9e3 이 가속도 1.2e14 가 됐다.
        # 중앙값의 일정 비율을 바닥으로 두되, softplus 라 어디서도 미분이
        # 끊기지 않는다. 1e-12 하드 클램프는 사실상 바닥이 없는 것과 같았다.
        if self.mass_floor > 0:
            fl = (self.mass_floor * mass.median()).detach().clamp(min=1e-30)
            mass = fl * torch.nn.functional.softplus(mass / fl)
        # 질량이 거의 없는 앵커는 질량을 부풀리는 대신 적분에서 뺀다. 부풀리면
        # 없는 질량이 생겨 동역학 자체가 틀어지고, 실제로 mwRMSD 가 6.88% 에서
        # 10.4% 로 나빠졌다. 그 앵커는 가우시안을 거의 안 쥐어 물리에 기여하는
        # 바가 없으므로, 스키닝에는 그대로 두고 움직이지만 않게 한다.
        if self.mass_freeze > 0:
            with torch.no_grad():
                self._light = mass < self.mass_freeze * mass.median()
        mass = mass.clamp(min=1e-12)
        self._chk_min("앵커질량", mass)
        if self.oriented:
            # the anchor's own second moment, in its rest frame: the ellipsoid
            # the weight kernel already describes, as a covariance
            R0 = quat_to_R(self.quat)
            s2 = (self.log_s.exp() ** 2) / 5.0
            S = torch.einsum("mij,mj,mkj->mik", R0, s2, R0)          # [M,3,3]
            B = B + torch.zeros_like(B).index_add_(
                0, self.pair_g, w.reshape(-1, 1, 1) * S[self.pair_a])
            tr = (B.diagonal(dim1=-2, dim2=-1).sum(-1) / 3.0).clamp(min=1e-20)
            eps = (self.eig_floor * tr.clamp(min=self.B_ref)).reshape(-1, 1, 1)
            Binv = inv3(B + eps * eye)
            blocked = eps * Binv
            self._S = S
            # isotropic inertia from that moment, times the anchor's mass
            self._inertia = (mass * S.diagonal(dim1=-2, dim2=-1).sum(-1)
                             * (2.0 / 3.0)).clamp(min=1e-12).unsqueeze(-1)
        if self.quad:
            # the same construction one basis up. E = [I3 | 0] is what T has to
            # equal at rest, and the blocked term is built so that it does:
            # at rest A~ = E B~, so T = (E B~ + eps E)(B~ + eps I)^-1 = E.
            qt = self._qbasis(q)                                   # [P,9]
            Bq = torch.zeros(self.N, 9, 9, device=self.dev).index_add_(
                0, self.pair_g,
                w.reshape(-1, 1, 1) * (qt.unsqueeze(-1) * qt.unsqueeze(-2)))
            trq = (Bq.diagonal(dim1=-2, dim2=-1).sum(-1) / 9.0).clamp(min=1e-20)
            epsq = (self.eig_floor * trq.clamp(min=self.B_ref)).reshape(-1, 1, 1)
            eye9 = torch.eye(9, device=self.dev)
            Bqinv = torch.linalg.inv(Bq + epsq * eye9)
            E = torch.zeros(3, 9, device=self.dev)
            E[0, 0] = E[1, 1] = E[2, 2] = 1.0
            blockedq = epsq * (E @ Bqinv)                          # [N,3,9]
            u = self.Xc - rc
            # kept on self rather than in the cache tuple: every caller unpacks
            # six, and the quadratic path is the only thing that wants these
            self._q = (qt, Bqinv, blockedq, self._qbasis(u), self._qjac(u))
        return w, rc, q, Binv, blocked, mass

    # ---- the physics -------------------------------------------------------
    def _deform_quad(self, p, w):
        """T = A~ (B~ + eps I)^-1, and the F it implies at each Gaussian"""
        qt, Bqinv, blockedq, _, Ju = self._q
        pa = p[self.pair_a]
        cc = torch.zeros(self.N, 3, device=self.dev).index_add_(
            0, self.pair_g, w.unsqueeze(-1) * pa)
        pq = pa - cc[self.pair_g]
        A = torch.zeros(self.N, 3, 9, device=self.dev).index_add_(
            0, self.pair_g,
            w.reshape(-1, 1, 1) * (pq.unsqueeze(-1) * qt.unsqueeze(-2)))
        Tm = A @ Bqinv + blockedq                                  # [N,3,9]
        return Tm, cc

    def deformation(self, p, w, rc, q, Binv, blocked):
        if self.quad:
            Tm, cc = self._deform_quad(p, w)
            return Tm @ self._q[4], cc            # F = T J
        # the fused reduction is differentiable, so this is taken whether or not
        # grad is on -- it is the fit's hot path as much as the rollout's
        if self._pairs_ok():
            csr = sparsestep.build_csr(self.pair_g, self.pair_a, self.N, self.M)
            RS = None
            if self.oriented:
                o = self._o if self._o is not None else self._ident()
                RS = (quat_to_R(o) @ self._S).contiguous()
            cc, A = sparsestep.deform_diff(p, w, q, csr, self.N, self.M, RS)
            return A @ Binv + blocked, cc
        pa = p[self.pair_a]
        cc = torch.zeros(self.N, 3, device=self.dev).index_add_(
            0, self.pair_g, w.unsqueeze(-1) * pa)
        pq = pa - cc[self.pair_g]
        A = torch.zeros(self.N, 3, 3, device=self.dev).index_add_(
            0, self.pair_g, w.reshape(-1, 1, 1) * (pq.unsqueeze(-1) * q.unsqueeze(-2)))
        if self.oriented:
            Rd = quat_to_R(self._o if self._o is not None else self._ident())
            A = A + torch.zeros_like(A).index_add_(
                0, self.pair_g,
                w.reshape(-1, 1, 1) * (Rd @ self._S)[self.pair_a])
        return A @ Binv + blocked, cc

    def gaussian_pos(self, p, cache):
        w, rc, q, Binv, blocked, _ = cache
        if self.fdyn is not None and self._fo is not None:
            F, _, _ = self.fdyn.frame.blend(self._fo, self._fs, w, self.pair_g,
                                             self.pair_a, self.N, "nlerp")
            cc = torch.zeros(self.N, 3, device=self.dev).index_add_(
                0, self.pair_g, w.unsqueeze(-1) * p[self.pair_a])
            return cc + torch.einsum("nij,nj->ni", F, self.Xc - rc)
        if self.quad:
            Tm, cc = self._deform_quad(p, w)
            return cc + torch.einsum("nik,nk->ni", Tm, self._q[3])
        if self._fused_ok():
            csr, _, _ = self._fused(w)
            return sparsestep.skin(p, csr, w, q, Binv, blocked, self.Xc, rc)[0]
        F, cc = self.deformation(p, w, rc, q, Binv, blocked)
        return cc + torch.einsum("nij,nj->ni", F, self.Xc - rc)

    def stiffness(self, w):
        """[N] -- each Gaussian's multiplier, blended from the anchors holding it"""
        k = torch.zeros(self.N, device=self.dev).index_add_(
            0, self.pair_g, w * self.log_k.exp()[self.pair_a])
        return k.clamp(min=1e-3)

    # ---- the fused path ----------------------------------------------------
    #
    # Same physics, no autograd graph, and the [P,3,3] outer products stay in
    # registers instead of being written out and scattered. Only taken with
    # grad off: the fit differentiates through all of this and keeps the torch
    # path, which is also the reference the kernel is verified against
    # (exe/verify_sparsestep.py).
    def _fused_ok(self):
        if self.quad or self.oriented:   # the kernel knows neither extra term
            return False
        """the whole-force kernel: forward only, so grad must be off"""
        # 추적 중에는 커널을 쓰지 않는다. 커널 안은 들여다볼 수 없어, 발산이
        # 가우시안 경로에서 났을 때 F/J/P 중 어느 것이 먼저 터졌는지 알 수 없다.
        if self.nan_trace:
            return False
        return self._pairs_ok() and not torch.is_grad_enabled()

    def _pairs_ok(self):
        if self.quad:
            return False
        """the pair reductions, which carry their own backward"""
        return (sparsestep is not None and sparsestep.HAVE_CUDA
                and self.pos.is_cuda)

    def _fused(self, w):
        """the CSR layout and the stiffness-carrying moduli, cached per refresh.

        Both are functions of the pair list and the weights, neither of which
        moves between refreshes, so recomputing them per substep would be the
        thing the kernel was written to avoid.

        **강성도 키에 들어가야 한다.** mu*k 의 k = stiffness(w) 는 가중치뿐 아니라
        log_k 에도 의존하는데, 원래 키가 (짝, 가중치) 뿐이라 log_k 를 바꿔도 캐시가
        적중해 옛 강성이 그대로 쓰였다 -- 실측으로 log_k 를 0 에서 5 로(강성 148 배)
        올려도 no_grad 롤아웃 궤적이 모든 자릿수까지 동일했다. 이 경로는 grad 가
        꺼졌을 때만 타므로 피팅은 무사했지만, 평가·렌더링·진단이 전부 옛 강성을
        보고 있었다. _version 은 in-place 갱신(옵티마이저 스텝, fill_, +=)마다
        올라가므로 GPU 동기화 없이 값 변화를 잡는다.
        """
        key = (self.pair_g.data_ptr(), w.data_ptr(), int(w.shape[0]),
               self.log_k.data_ptr(), self.log_k._version)
        hit = getattr(self, "_fused_cache", None)
        if hit is not None and hit[0] == key:
            return hit[1]
        csr = sparsestep.build_csr(self.pair_g, self.pair_a, self.N, self.M)
        k = self.stiffness(w)
        val = (csr, (self.mu * k).contiguous(), (self.lam * k).contiguous())
        self._fused_cache = (key, val)
        return val

    def velocity_gradient(self, v, w, q, Binv):
        """per-Gaussian velocity gradient, built exactly the way F is.

        L_g = ( sum_a w_ga (v_a - vbar_g) q_ga^T ) B_g^-1

        Same neighbourhood, same weights, same inverse -- only the field is
        velocity instead of position. That is the quantity MPM uses to march its
        particle F forward.
        """
        if self._pairs_ok():
            # the deform kernel is exactly this sum; it does not care that the
            # field it gathers is a velocity rather than a position, and it does
            # not materialise the per-pair outer product, which is what put the
            # torch version over 40 GB at three and a half million pairs
            csr = sparsestep.build_csr(self.pair_g, self.pair_a, self.N, self.M)
            _, A = sparsestep.deform_diff(v, w, q, csr, self.N, self.M, None)
            return A @ Binv
        va = v[self.pair_a]
        vc = torch.zeros(self.N, 3, device=self.dev).index_add_(
            0, self.pair_g, w.unsqueeze(-1) * va)
        vq = va - vc[self.pair_g]
        A = torch.zeros(self.N, 3, 3, device=self.dev)
        for i in range(3):
            for j in range(3):
                A[:, i, j].index_add_(0, self.pair_g, w * vq[:, i] * q[:, j])
        return A @ Binv

    def blended_F(self, p, v, w, rc, q, Binv, blocked):
        """shape-matched F, optionally mixed with one carried forward in time.

        Shape matching reads the deformation out of where the anchors are right
        now, against where they started. It sees nothing finer than the
        neighbourhood it averages over, and the weights it averages with were
        fixed at rest -- under large deformation they are describing a
        neighbourhood that no longer exists. Measured on ficus, the F this
        produces recovers 17% of the deviation MPM's own F has, and the forces
        that follow are wrong by a factor of 158. Feeding MPM's F through the
        same stress law gives the right answer, so the constitutive model is not
        what is failing.

        An MPM particle instead carries F and marches it with the velocity
        gradient. For a perfectly resolved elastic material the two agree; they
        part company exactly where ours is weak. This mixes them, so the blend
        weight says how much of the answer comes from history rather than from
        the current configuration.
        """
        Fs, cc = self.deformation(p, w, rc, q, Binv, blocked)
        if self.acc_blend <= 0.0 or self._Facc is None:
            return Fs, cc
        b = float(self.acc_blend)
        return (1.0 - b) * Fs + b * self._Facc, cc

    def reset_carried(self):
        """a fresh trajectory must not inherit the last one's deformation"""
        self._Facc = None

    def advance_Facc(self, v, w, q, Binv):
        """F <- (I + dt L) F, the update MPM applies to its particles"""
        if self.acc_blend <= 0.0 or self._Facc is None:
            return
        L = self.velocity_gradient(v, w, q, Binv)
        eye = torch.eye(3, device=self.dev)
        self._Facc = (eye + self.dt * L) @ self._Facc

    def force(self, p, w, rc, q, Binv, blocked):
        if self.quad:
            qt, Bqinv, _, _, Ju = self._q
            F, _ = self.deformation(p, w, rc, q, Binv, blocked)
            R = closest_rotation(F, self.polar_iters, self.polar_ridge)
            J = det3(F)
            n_ = F.reshape(-1, 9).norm(dim=-1).clamp(min=1e-12).reshape(-1, 1, 1)
            Finv_T = inv3(F + 1e-6 * n_ * torch.eye(3, device=self.dev),
                           eps=1e-30).transpose(-1, -2)
            k = self.stiffness(w).unsqueeze(-1).unsqueeze(-1)
            mu = self.mu.unsqueeze(-1).unsqueeze(-1) * k
            lam = self.lam.unsqueeze(-1).unsqueeze(-1) * k
            P = 2 * mu * (F - R) + lam * (J - 1).unsqueeze(-1).unsqueeze(-1) * \
                J.unsqueeze(-1).unsqueeze(-1) * Finv_T
            # dE/dp_a = V w (P J^T B~^-1) q~, the same transpose the linear path
            # takes, one basis up
            PB = (P @ Ju.transpose(-1, -2) @ Bqinv)[self.pair_g]   # [P,3,9]
            contrib = -(self.vol[self.pair_g] * w).unsqueeze(-1) * \
                torch.einsum("pik,pk->pi", PB, qt)
            return torch.zeros(self.M, 3, device=self.dev).index_add_(
                0, self.pair_a, contrib)
        if self._fused_ok():
            csr, mu_k, lam_k = self._fused(w)
            return sparsestep.force(p, csr, w, q, Binv, blocked, self.vol,
                                     mu_k, lam_k, self.M,
                                     polar_iters=self.polar_iters,
                                     polar_ridge=self.polar_ridge,
                                     Facc=self._Facc, acc_blend=self.acc_blend)
        F, _ = self.blended_F(p, self._v_now, w, rc, q, Binv, blocked) \
            if self.acc_blend > 0.0 else self.deformation(p, w, rc, q, Binv, blocked)
        self._chk("가우시안_F", F)
        R = closest_rotation(F, self.polar_iters, self.polar_ridge)
        J = det3(F)
        self._chk("가우시안_R", R)
        self._chk("가우시안_J", J)
        # the volumetric term carries J F^-T, which is unbounded as an element
        # approaches zero volume. A ridge keeps it finite through the crossing
        # rather than handing the integrator an infinity
        n = F.reshape(-1, 9).norm(dim=-1).clamp(min=1e-12).reshape(-1, 1, 1)
        Finv_T = inv3(F + self.finv_ridge * n * torch.eye(3, device=self.dev),
                       eps=1e-30).transpose(-1, -2)
        k = self.stiffness(w).unsqueeze(-1).unsqueeze(-1)
        mu = self.mu.unsqueeze(-1).unsqueeze(-1) * k
        lam = self.lam.unsqueeze(-1).unsqueeze(-1) * k
        self._chk("가우시안_Finv", Finv_T)
        self._chk("가우시안_강성", k)
        P = 2 * mu * (F - R) + lam * (J - 1).unsqueeze(-1).unsqueeze(-1) * \
            J.unsqueeze(-1).unsqueeze(-1) * Finv_T
        self._chk("가우시안_P", P)
        PBn = P @ Binv
        if self._pairs_ok():
            csr = sparsestep.build_csr(self.pair_g, self.pair_a, self.N, self.M)
            return sparsestep.gather_diff(PBn, w, q, self.vol, csr, self.N, self.M)
        PB = PBn[self.pair_g]
        contrib = -(self.vol[self.pair_g] * w).unsqueeze(-1) * \
            torch.einsum("pij,pj->pi", PB, q)
        return torch.zeros(self.M, 3, device=self.dev).index_add_(0, self.pair_a, contrib)

    def _stress(self, p, w, rc, q, Binv, blocked):
        """F, and the PK1 that goes with it. Shared by the force and the torque,
        which otherwise each rebuild the deformation, the polar factor and the
        material law -- twice per substep for the same numbers."""
        F, _ = self.deformation(p, w, rc, q, Binv, blocked)
        R = closest_rotation(F, self.polar_iters, self.polar_ridge)
        J = det3(F)
        n_ = F.reshape(-1, 9).norm(dim=-1).clamp(min=1e-12).reshape(-1, 1, 1)
        Finv_T = inv3(F + 1e-6 * n_ * torch.eye(3, device=self.dev),
                       eps=1e-30).transpose(-1, -2)
        k = self.stiffness(w).unsqueeze(-1).unsqueeze(-1)
        mu = self.mu.unsqueeze(-1).unsqueeze(-1) * k
        lam = self.lam.unsqueeze(-1).unsqueeze(-1) * k
        return 2 * mu * (F - R) + lam * (J - 1).unsqueeze(-1).unsqueeze(-1) * \
            J.unsqueeze(-1).unsqueeze(-1) * Finv_T

    def force_torque(self, p, w, rc, q, Binv, blocked):
        """both, from one stress evaluation"""
        P = self._stress(p, w, rc, q, Binv, blocked)
        PB = P @ Binv
        if self._pairs_ok():
            csr = sparsestep.build_csr(self.pair_g, self.pair_a, self.N, self.M)
            f = sparsestep.gather_diff(PB, w, q, self.vol, csr, self.N, self.M)
        else:
            contrib = -(self.vol[self.pair_g] * w).unsqueeze(-1) * \
                torch.einsum("pij,pj->pi", PB[self.pair_g], q)
            f = torch.zeros(self.M, 3, device=self.dev).index_add_(
                0, self.pair_a, contrib)
        if self._pairs_ok():
            csr = sparsestep.build_csr(self.pair_g, self.pair_a, self.N, self.M)
            M = sparsestep.moment_diff(PB, w, self.vol, csr, self.N, self.M) @ self._S
        else:
            G = (self.vol.unsqueeze(-1).unsqueeze(-1) * PB)[self.pair_g]
            M = torch.zeros(self.M, 3, 3, device=self.dev).index_add_(
                0, self.pair_a, w.reshape(-1, 1, 1) * G) @ self._S
        o = self._o if self._o is not None else self._ident()
        Y = quat_to_R(o) @ M.transpose(-1, -2)
        tau = torch.stack([Y[:, 2, 1] - Y[:, 1, 2],
                           Y[:, 0, 2] - Y[:, 2, 0],
                           Y[:, 1, 0] - Y[:, 0, 1]], -1)
        return f, tau

    def _ident(self):
        o = torch.zeros(self.M, 4, device=self.dev)
        o[:, 0] = 1.0
        return o

    def torque(self, p, w, rc, q, Binv, blocked):
        """-dE/dR projected onto so(3), per anchor.

        E = sum_g V Psi(F), F = A B^-1, and A carries w R_a S_a, so
        dE/dR_a = sum_g w (V P B^-1) S_a. A rotation of R by an infinitesimal
        world-frame omega changes E by omega . vee(Y - Y^T) with Y = R (dE/dR)^T,
        which is the torque up to sign.
        """
        F, _ = self.deformation(p, w, rc, q, Binv, blocked)
        R = closest_rotation(F, self.polar_iters, self.polar_ridge)
        J = det3(F)
        n_ = F.reshape(-1, 9).norm(dim=-1).clamp(min=1e-12).reshape(-1, 1, 1)
        Finv_T = inv3(F + 1e-6 * n_ * torch.eye(3, device=self.dev),
                       eps=1e-30).transpose(-1, -2)
        k = self.stiffness(w).unsqueeze(-1).unsqueeze(-1)
        mu = self.mu.unsqueeze(-1).unsqueeze(-1) * k
        lam = self.lam.unsqueeze(-1).unsqueeze(-1) * k
        P = 2 * mu * (F - R) + lam * (J - 1).unsqueeze(-1).unsqueeze(-1) * \
            J.unsqueeze(-1).unsqueeze(-1) * Finv_T
        G = (self.vol.unsqueeze(-1).unsqueeze(-1) * (P @ Binv))[self.pair_g]
        M = torch.zeros(self.M, 3, 3, device=self.dev).index_add_(
            0, self.pair_a, w.reshape(-1, 1, 1) * G)
        M = M @ self._S
        o = self._o if self._o is not None else self._ident()
        Y = quat_to_R(o) @ M.transpose(-1, -2)
        return torch.stack([Y[:, 2, 1] - Y[:, 1, 2],
                            Y[:, 0, 2] - Y[:, 2, 0],
                            Y[:, 1, 0] - Y[:, 0, 1]], -1)

    def substep_o(self, p, v, o, wv, w, rc, q, Binv, blocked, m, keep):
        """the oriented substep: linear as before, plus a spin driven by torque"""
        self._o = o
        f, tau = self.force_torque(p, w, rc, q, Binv, blocked)
        a = f / m + self.gravity
        v = (v + self.dt * a) * self.damping * keep
        wv = (wv + self.dt * tau / self._inertia) * self.damping * keep
        dp = self.dt * v
        # q <- normalize(q + dt/2 omega (x) q)
        ox, oy, oz = wv[:, 0], wv[:, 1], wv[:, 2]
        qw, qx, qy, qz = o[:, 0], o[:, 1], o[:, 2], o[:, 3]
        do = 0.5 * torch.stack([-ox * qx - oy * qy - oz * qz,
                                 ox * qw + oy * qz - oz * qy,
                                 -ox * qz + oy * qw + oz * qx,
                                 ox * qy - oy * qx + oz * qw], -1)
        o = o + self.dt * do
        o = o / o.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        over = (dp.norm(dim=-1) / self.cfl_limit - 1.0).clamp(min=0)
        p = p + dp
        p, v = self._boundaries(p, v)
        return p, v, o, wv, (over * over).mean()

    def reset_state(self):
        """롤아웃 사이의 상태를 지운다.

        프레임 상태(o, s)는 한 표본의 여러 코스 프레임에 걸쳐 그래프를 물고 가야
        하지만, 표본이 바뀌면 지워야 한다. 안 그러면 다음 역전파가 이미 해제된
        그래프를 건드려 "backward through the graph a second time" 로 죽는다.
        """
        self._o = self._wv = None
        self._fo = self._fs = None
        self._Facc = None

    def substep_f(self, p, v, o, s, wv, sv, w, rc, q, Binv, blocked, m, keep,
                   wS, qS, BiS):
        """프레임 동역학의 서브스텝. 앵커 M 개만 다루므로 가우시안 비용이 없다.

        탄성 에너지를 앵커에 둔다: E = sum_a V_a Psi(R_a S_a). 가우시안별 F 를
        서브스텝마다 조립하면 17 만 x 480 이라 메모리가 터진다(실측). 가우시안
        블렌딩은 프레임 경계에서 한 번만 한다.
        """
        from .anchor_frame import quat_to_R
        fd, ast = self.fdyn, self.astress
        # 위치는 검증된 앵커 응력이 굴린다. 프레임의 탄성 에너지는 p 에 의존하지
        # 않으므로, 그것만으로는 앵커에 복원력이 걸리지 않는다 -- 실측으로 관성
        # 운동이 되어 rollout 이 29.95% 에 고정됐다.
        fp = ast(p, self.pos)
        a = fp / m + self.gravity
        self._chk("프레임_힘", fp); self._chk("프레임_가속도", a)
        v = (v + self.dt * a) * self.damping * keep
        dp = self.dt * v
        over = (dp.norm(dim=-1) / self.cfl_limit - 1.0).clamp(min=0)
        pen = (over * over).max() if self.cfl_agg == "max" else (over * over).mean()
        p = p + dp
        p, v = self._boundaries(p, v)
        # (o, s) 는 앵커 그래프의 변형을 1 차 이완으로 따라간다. alpha=1 이면
        # 형상 매칭과 동일하고, 작을수록 이력을 담는다.
        d = p[ast.eb] - p[ast.ea]
        A0 = torch.zeros(self.M, 3, 3, device=self.dev).index_add_(
            0, ast.ea, wS.reshape(-1, 1, 1) * (d.unsqueeze(-1) * qS.unsqueeze(-2)))
        F_shape = A0 @ BiS
        o, s = fd.relax(o, s, F_shape, torch.sigmoid(fd.a_o), torch.sigmoid(fd.a_s))
        self._chk("프레임_s", s)
        return p, v, o, s, wv, sv, pen

    def trace_reset(self):
        self._trace, self._mag, self._who, self._k = {}, {}, {}, 0
        # 아직 아무것도 무한이 되지 않았는가. 크기 기록을 여기에 걸어두면
        # 터진 뒤의 값이 최댓값을 덮어써서 원인을 가리는 일이 없어진다.
        self._clean = torch.ones((), device=self.dev)

    def _chk(self, name, x):
        """유한성과 함께 크기도 기록한다.

        "유한하다" 는 isfinite 를 통과했다는 뜻일 뿐이라, F 가 1e20 이어도
        무한 목록에는 안 뜬다. 어느 양이 먼저 터졌는지만으로는 원인을 못 짚어서
        직전까지의 최대 절대값을 함께 남긴다.
        """
        if not self.nan_trace:
            return
        bad = ~torch.isfinite(x).all()
        t = self._trace.get(name)
        if t is None:
            t = torch.full((), -1.0, device=self.dev)
        self._trace[name] = torch.where(bad & (t < 0),
                                         torch.full_like(t, float(self._k)), t)
        # 크기는 "전부 유한하던 동안" 만 기록한다
        xf = torch.nan_to_num(x.detach(), nan=0.0, posinf=0.0, neginf=0.0)
        m = xf.abs().max() * self._clean
        prev = self._mag.get(name)
        self._mag[name] = m if prev is None else torch.maximum(prev, m)
        # 초기 세 서브스텝은 따로 남긴다. 최댓값 하나로 뭉치면 표본과 서브스텝이
        # 섞여, F=2e24 인데 J=det(F)=2e11 같은 서로 모순된 값이 한 줄에 실린다.
        # "서브스텝 1 에서 이미 컸는가" 가 원인을 가르는 질문이라 그것만은 분리한다.
        if self._k <= 3:
            key = f"{name}@k{self._k}"
            pk = self._mag.get(key)
            self._mag[key] = m if pk is None else torch.maximum(pk, m)
        self._clean = self._clean * (~bad).to(self._clean.dtype)
        # 어느 원소가 터졌는지. 첫 기록만 남기므로 체크포인트 재계산이
        # 덮어쓰지 않는다.
        row = ~torch.isfinite(x.detach().reshape(x.shape[0], -1)).all(-1)
        wprev = self._who.get(name)
        if wprev is None:
            wprev = torch.full((), -1.0, device=self.dev)
        first = row.to(torch.float32).argmax().to(torch.float32)
        self._who[name] = torch.where(row.any() & (wprev < 0), first, wprev)

    def _chk_min(self, name, x):
        """작아져서 문제가 되는 양(B 의 trace 등)은 최솟값을 남긴다."""
        if not self.nan_trace:
            return
        m = torch.nan_to_num(x.detach(), nan=float("inf")).min()
        m = torch.where(self._clean > 0, m, torch.full_like(m, float("inf")))
        prev = self._mag.get(name)
        self._mag[name] = m if prev is None else torch.minimum(prev, m)

    def nan_report(self):
        """{양 이름: 처음 무한이 된 서브스텝}. 빈 dict 면 전부 유한했다."""
        return {n: int(v.item()) for n, v in self._trace.items()
                if float(v.item()) >= 0}

    def mag_report(self):
        """{양 이름: 무한이 되기 직전까지의 최대(또는 최소) 크기}."""
        return {n: float(v.item()) for n, v in self._mag.items()}

    def who_report(self):
        """{양 이름: 처음 무한이 된 원소 번호}. 앵커 번호로 읽으면 된다."""
        return {n: int(v.item()) for n, v in self._who.items()
                if float(v.item()) >= 0}

    def substep(self, p, v, w, rc, q, Binv, blocked, m, keep, Facc=None):
        # Facc travels through the call rather than sitting on self, so that
        # checkpointing covers it. On self it escapes, and every one of the 480
        # substeps in a frame keeps its own N x 3 x 3 alive -- 35 GB on ficus.
        self._Facc = Facc
        self._v_now = v
        # edge_only 면 가우시안 응력 경로를 아예 타지 않는다 -- 앵커끼리만
        if self.edge_only:
            a = self.gravity
        else:
            fg = self.force(p, w, rc, q, Binv, blocked)
            self._chk("가우시안_힘", fg)
            a = fg / m + self.gravity
        if self.astress is not None:
            fa = self.astress(p, self.pos)
            self._chk("앵커응력_힘", fa)
            a = a + fa / m
        if self.edge is not None:
            fe = self.edge(p, v, m, self.log_s, self.log_k)
            self._chk("간선력", fe)
            a = a + fe / m
        self._chk("가속도", a)
        v = (v + self.dt * a) * self.damping * keep
        self._chk("속도", v)
        if Facc is not None:
            L = self.velocity_gradient(v, w, q, Binv)
            Facc = (torch.eye(3, device=self.dev) + self.dt * L) @ Facc
        dp = self.dt * v
        # how far an anchor travels in one substep, against the spacing between
        # anchors. An explicit step is only meaningful while this is small, and
        # nothing else in the loss knows that: the fit is free to make the
        # discretisation stiffer than the substep can integrate, and it does --
        # 1.4% of the spacing at the start, 50% where the rollout blows up
        over = (dp.norm(dim=-1) / self.cfl_limit - 1.0).clamp(min=0)
        pen = (over * over).max() if self.cfl_agg == "max" else (over * over).mean()
        p = p + dp
        p, v = self._boundaries(p, v)
        self._chk("위치", p)
        if Facc is not None:
            return p, v, Facc, pen
        return p, v, pen

    def rollout(self, p, v, n, cache):
        """returns (p, v, cfl) -- the last being how badly, on average, an anchor
        outran the substep it was integrated with"""
        w, rc, q, Binv, blocked, mass = cache
        m = mass.unsqueeze(-1)
        held = self.fixed if self._light is None else (self.fixed | self._light)
        keep = (~held).unsqueeze(-1).to(p.dtype)
        if self.acc_blend > 0.0 and self._Facc is None:
            self._Facc = torch.eye(3, device=self.dev).expand(self.N, 3, 3).contiguous()
        use = self.checkpoint_substeps and torch.is_grad_enabled()
        if self.astress is not None:
            self.astress.prepare_rest(self.pos)
        pen = p.new_zeros(())
        if self.fdyn is not None:
            wS, qS, BiS = self.astress.rest(self.pos)
            o = self._fo if self._fo is not None else self._ident()
            sst = self._fs if self._fs is not None else torch.zeros(self.M, 3, device=self.dev)
            wv = torch.zeros(self.M, 3, device=self.dev)
            sv = torch.zeros(self.M, 3, device=self.dev)
            for _ in range(n):
                if use:
                    p, v, o, sst, wv, sv, c = checkpoint(
                        self.substep_f, p, v, o, sst, wv, sv, w, rc, q, Binv,
                        blocked, m, keep, wS, qS, BiS, use_reentrant=False)
                else:
                    p, v, o, sst, wv, sv, c = self.substep_f(
                        p, v, o, sst, wv, sv, w, rc, q, Binv, blocked, m, keep,
                        wS, qS, BiS)
                self._t += self.dt
                self._k += 1
                pen = torch.maximum(pen, c) if self.cfl_agg == "max" else pen + c
            self._fo, self._fs = o, sst
            return p, v, pen if self.cfl_agg == "max" else pen / max(n, 1)
        if self.oriented:
            # threaded through the checkpoint rather than kept on self, so the
            # recompute sees the same orientation the forward did
            o = self._o if self._o is not None else self._ident()
            wv = self._wv if self._wv is not None else torch.zeros(self.M, 3,
                                                                   device=self.dev)
            for _ in range(n):
                if use:
                    p, v, o, wv, c = checkpoint(self.substep_o, p, v, o, wv, w, rc,
                                                 q, Binv, blocked, m, keep,
                                                 use_reentrant=False)
                else:
                    p, v, o, wv, c = self.substep_o(p, v, o, wv, w, rc, q, Binv,
                                                     blocked, m, keep)
                self._t += self.dt
                pen = torch.maximum(pen, c) if self.cfl_agg == "max" else pen + c
            self._o, self._wv = o, wv
            return p, v, pen if self.cfl_agg == "max" else pen / max(n, 1)
        Fa = self._Facc if self.acc_blend > 0.0 else None
        for _ in range(n):
            if Fa is not None:
                if use:
                    p, v, Fa, c = checkpoint(self.substep, p, v, w, rc, q, Binv,
                                              blocked, m, keep, Fa,
                                              use_reentrant=False)
                else:
                    p, v, Fa, c = self.substep(p, v, w, rc, q, Binv, blocked,
                                                m, keep, Fa)
            elif use:
                p, v, c = checkpoint(self.substep, p, v, w, rc, q, Binv, blocked,
                                      m, keep, use_reentrant=False)
            else:
                p, v, c = self.substep(p, v, w, rc, q, Binv, blocked, m, keep)
            self._t += self.dt
            self._k += 1
            pen = torch.maximum(pen, c) if self.cfl_agg == "max" else pen + c
        self._Facc = Fa
        return p, v, pen if self.cfl_agg == "max" else pen / max(n, 1)

    # ---- moving between particles and anchors ------------------------------
    # ---- inverting the decoder --------------------------------------------
    #
    # skin() is linear in the anchor positions, and its coefficients are scalars:
    #
    #   x_g = sum_a c_ga p_a + b_g,   c_ga = w_ga (1 + s_ga - S_g),
    #   s_ga = q_ga . (Binv_g y_g),   S_g = sum_a w_ga s_ga,   y_g = Xc_g - rc_g
    #
    # (derivation: cc and A are both linear in p, F = A Binv + blocked, and
    # A_g (Binv_g y_g) collapses to a scalar-weighted sum because q_ga enters
    # only through its inner product with Binv_g y_g. At rest it checks out
    # exactly: A = B, F = (B + eps I) Binv = I, and x = Xc.)
    #
    # So the map is C (x) I_3 with C sparse [N, M], and the projection that
    # belongs with this decoder is its least-squares inverse rather than a local
    # average -- which is the ADJOINT, and lands the anchors somewhere the
    # simulator never goes: measured at 15x the internal stress of any state it
    # reaches on its own (exe/probe_accel_residual.py).
    #
    # C^T C is [M, M], a few hundred on a side, and one factorisation serves both
    # positions and velocities: the velocity decoder in lift() is the same map
    # without the blocked term, so it shares C and takes b = 0.

    def ls_factor(self, cache, ridge=1e-6):
        """(Cholesky of C^T C over the free anchors, C, b) for this rest state

        C 를 희소 대신 **조밀**하게 만든다. 희소 텐서의 autograd 지원이 제한적이라,
        희소 경로에서는 앵커 파라미터로 가는 기울기가 끊긴다. 그 끊김이 기하 피팅이
        내려가지 않던 원인이었다 -- 유한차분과 대조하면 9 개 좌표 중 3 개가 부호까지
        반대였고 크기는 최대 14 배 어긋났다(exe/probe_fit_gradient.py).

        앵커를 움직이면 (1) 상태가 이 인코더로 접히는 방식과 (2) 거기서 굴러가고
        펴지는 방식이 함께 바뀐다. no_grad 로 막아 두면 역전파가 (2) 만 보고,
        부분 도함수로 내려가려다 제자리를 돈다.

        비용은 [N, M] 한 장 -- ficus 에서 171,553 x 512 x 4B = 351MB 다.
        """
        w, rc, q, Binv, blocked, _ = cache
        y = self.Xc - rc                                    # [N,3]
        z = torch.einsum("nij,nj->ni", Binv, y)             # [N,3]
        s = (q * z[self.pair_g]).sum(-1)                    # [P]
        S = torch.zeros(self.N, device=self.dev).index_add_(0, self.pair_g, w * s)
        c = w * (1.0 + s - S[self.pair_g])                  # [P]
        b = torch.einsum("nij,nj->ni", blocked, y)          # [N,3]

        # 같은 (가우시안, 앵커) 짝이 여러 번 나올 수 있으므로 더해서 채운다
        # (희소 경로의 coalesce 와 같은 동작).
        flat = self.pair_g * self.M + self.pair_a
        C = torch.zeros(self.N * self.M, device=self.dev, dtype=c.dtype
                        ).index_add_(0, flat, c).view(self.N, self.M)
        G = C.t() @ C                                       # [M,M]

        free = ~self.fixed
        Gf = G[free][:, free]
        # a ridge on the diagonal: an anchor whose support fell to nothing leaves
        # a zero row, and the scale is taken from the matrix so it does not
        # depend on the units of the scene
        Gf = Gf + ridge * Gf.diagonal().mean().clamp(min=1e-20) * torch.eye(
            int(free.sum()), device=self.dev)
        return torch.linalg.cholesky(Gf), C, b, free

    def project_ls(self, x, cache, fac=None):
        """MPM particles -> the anchor state whose DECODING is closest to them"""
        L, C, b, free = self.ls_factor(cache) if fac is None else fac
        rhs = x - b - C @ torch.where(
            self.fixed.unsqueeze(-1), self.pos, torch.zeros_like(self.pos))
        r = (C.t() @ rhs)[free]                             # [free,3]
        pf = torch.cholesky_solve(r, L)
        p = torch.where(self.fixed.unsqueeze(-1), self.pos,
                        torch.zeros_like(self.pos)).clone()
        p = p.index_put((torch.nonzero(free, as_tuple=False).squeeze(-1),), pf)
        return p

    def project_v_ls(self, vp, cache, fac=None):
        """the same inverse for velocities: same C, and no constant term"""
        L, C, b, free = self.ls_factor(cache) if fac is None else fac
        r = (C.t() @ vp)[free]
        vf = torch.cholesky_solve(r, L)
        v = torch.zeros_like(self.pos)
        v = v.index_put((torch.nonzero(free, as_tuple=False).squeeze(-1),), vf)
        return v

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
        p = torch.where(self.fixed.unsqueeze(-1), self.pos, p)
        self._t = 0.0
        if self.oriented:
            # a projected MPM state says nothing about how the anchors are
            # turned, so they start unrotated and the torque takes over
            self._o, self._wv = self._ident(), torch.zeros(self.M, 3, device=self.dev)
        return p

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
        C = Fdot @ inv3(F + 1e-6 * torch.eye(3, device=self.dev), eps=1e-30)
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
    def _rebuild(self, pos, quat, log_s, log_k=None, log_amp=None):
        self.pos = nn.Parameter(pos.contiguous())
        self.quat = nn.Parameter(quat.contiguous())
        self.log_s = nn.Parameter(log_s.contiguous())
        if log_k is None:
            log_k = torch.zeros(pos.shape[0], device=self.dev)
        self.log_k = nn.Parameter(log_k.contiguous())
        if log_amp is None:
            log_amp = torch.zeros(pos.shape[0], device=self.dev)
        self.log_amp = nn.Parameter(log_amp.contiguous())
        f = torch.zeros(pos.shape[0], dtype=torch.bool, device=self.dev)
        for bc in self.cfg.get("boundary_conditions", []):
            if bc["type"] == "cuboid":
                c = torch.tensor(bc["point"], device=self.dev)
                s = torch.tensor(bc["size"], device=self.dev)
                f |= ((pos - c).abs() <= s).all(-1)
        self.fixed = f
        if getattr(self, "fdyn", None) is not None and self.fdyn.log_lam.numel() != pos.shape[0]:
            self.fdyn.resize(pos.shape[0])
        self.refresh()

    @torch.no_grad()
    def densify_and_prune(self, grad_accum, split_frac=0.05, prune_share=0.05,
                          split_scale=1.6, max_anchors=2048, min_pairs=4):
        """split where the fit is pushing hardest, drop what holds nothing.

        The criterion is the accumulated gradient on an anchor's position, the
        same signal 3DGS densifies on: a large one means the loss wants that
        anchor to be in two places, which is what splitting gives it.

        Pinned anchors are boundary condition, not representation, and are left
        alone in both directions. So are anchors standing inside a driver: the
        criterion is how much weight an anchor holds, and a propeller blade is
        thin, so the anchors on it hold little and were being cut. On plane that
        took the driven region from 75 of 512 anchors down to 28 of 436 -- the
        fit removed 63% of the representation from exactly the region carrying
        the motion, while cutting 15% elsewhere. Whatever the driver imposes has
        to land on an anchor or it does not happen at all.
        """
        w = self.weights()
        held = torch.zeros(self.M, device=self.dev).index_add_(
            0, self.pair_a, w)
        n_pairs = torch.zeros(self.M, device=self.dev).index_add_(
            0, self.pair_a, torch.ones_like(w))

        dead = ((held < prune_share * held.median()) | (n_pairs < min_pairs)) \
            & ~self.fixed & ~self._driven_mask(self.pos.detach())
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


class FittedScene:
    """A fitted anchor set, wearing the interface the student's trainer expects.

    The trainer was written against scene_setup.Scene: it asks for canonical
    anchors, an initial velocity, an explicit step, and a skinning. A fitted set
    answers all of those, but it is a different simulator underneath -- a
    different number of anchors in different places, with orientations, extents
    and stiffnesses that were learned. Wrapping it here means the trainer does
    not need to know which teacher it is imitating, and the two runs stay
    comparable down to the seed.

    What it cannot do is the fused CUDA step: that kernel has fixed neighbours,
    isotropic weights and one material stiffness. This is the torch path, about
    an order of magnitude slower per substep, which is why the trajectories are
    generated once and cached rather than drawn as needed.
    """

    def __init__(self, sc, fit):
        self.sc = sc
        self.fit = fit
        self._cache = fit.prepare()
        self.cfg = sc.cfg
        self.sub_dt = sc.sub_dt
        self.pos = sc.pos
        self.keep = sc.keep
        self.N = sc.N
        self.sim = sc.sim               # for radius, and the canonical Gaussians
        self.extent = sc.extent

    @property
    def M(self):
        return self.fit.M

    @property
    def anchor_canonical(self):
        return self.fit.pos.detach()

    @property
    def fixed_mask(self):
        return self.fit.fixed

    def refresh(self):
        self._cache = self.fit.prepare()

    @torch.no_grad()
    def initial_velocity(self, force=None):
        if force is None:
            for bc in self.cfg.get("boundary_conditions", []):
                if bc["type"] == "particle_impulse":
                    force = torch.tensor(bc["force"], device=self.fit.dev)
        return self.fit.impulse_dv(force, self._cache)

    @torch.no_grad()
    def impulse_dv(self, force):
        return self.fit.impulse_dv(force, self._cache)

    @torch.no_grad()
    def explicit_step(self, p, v, gp, n=1):
        p, v, _ = self.fit.rollout(p, v, n, self._cache)
        return p, v, gp

    @torch.no_grad()
    def _carry(self):
        """weights taking each zero-volume Gaussian to nearby material ones.

        A sixth of this cloud -- 32,377 of 203,930 for ficus -- was rejected on
        opacity, carries zero volume and belongs to no anchor's support. It is
        interleaved through the foliage and it does get drawn, so leaving it at
        rest makes it stand still while the tree moves around it, which reads as
        a ghost of the original shape. It has no physics of its own; the best
        available answer is the motion of the material immediately around it.

        Canonical, so it is built once: which particles are nearby is a property
        of the rest configuration.
        """
        if getattr(self, "_carry_cache", None) is None:
            from scipy.spatial import cKDTree

            rest = torch.nonzero(~self.sc.keep, as_tuple=False).squeeze(-1)
            if rest.numel() == 0:
                self._carry_cache = (rest, None, None)
                return self._carry_cache
            mat_pos = self.sc.pos[self.fit.mat]
            # a tree rather than cdist: 32k by 172k in float32 is 22 GB
            tree = cKDTree(mat_pos.cpu().numpy())
            d, i = tree.query(self.sc.pos[rest].cpu().numpy(), k=8)
            d = torch.from_numpy(d).float().to(self.fit.dev)
            w = 1.0 / d.clamp(min=1e-6)
            self._carry_cache = (rest, w / w.sum(-1, keepdim=True),
                                  torch.from_numpy(i).long().to(self.fit.dev))
        return self._carry_cache

    @torch.no_grad()
    def skin(self, p, gp):
        """the material Gaussians move, and the zero-volume ones are carried by
        the material around them, so anything that renders this sees a whole
        cloud rather than a moving tree inside a stationary copy of itself"""
        out = gp.clone()
        gm = self.fit.gaussian_pos(p, self._cache)
        out[self.fit.mat] = gm
        rest, w, idx = self._carry()
        if rest.numel():
            disp = gm - self.sc.pos[self.fit.mat]
            out[rest] = self.sc.pos[rest] + (w.unsqueeze(-1) * disp[idx]).sum(1)
        return out

    def __getattr__(self, name):
        """anything not overridden here is the underlying scene's.

        The impulse draws reach for random_force_field, random_poke and a few
        geometry properties, and every one of them is about the Gaussian cloud
        rather than the anchors -- which the fit did not touch.
        """
        return getattr(self.__dict__["sc"], name)

    def elastic_accel(self, p, gp):
        w, rc, q, Binv, blocked, mass = self._cache
        f = self.fit.force(p, w, rc, q, Binv, blocked)
        a = f / mass.unsqueeze(-1)
        return torch.where(self.fixed_mask.unsqueeze(-1), torch.zeros_like(a), a)


def load_fitted(sc, path, device="cuda"):
    """rebuild a fitted anchor set from a checkpoint and wrap it for the trainer"""
    b = torch.load(path, map_location=device, weights_only=False)
    fit = AnchorSparse(sc, c=b.get("c", 0.25), eig_floor=b.get("eig_floor", 0.02)).to(device)
    fit._rebuild(b["pos"].to(device), b["quat"].to(device), b["log_s"].to(device),
                  b["log_k"].to(device) if "log_k" in b else None,
                  b["log_amp"].to(device) if b.get("log_amp") is not None else None)
    # 질량 바닥·cfl 집계·J 포화는 시뮬레이터의 일부다. 저장·복원하지 않으면
    # 학습 때와 다른 시뮬레이터를 평가하게 되고, 실제로 질량 바닥이 빠진 채
    # 로드되어 평가가 전부 nan 이 났다.
    rt = b.get("runtime") or {}
    fit.mass_floor = float(rt.get("mass_floor", 0.0))
    fit.mass_freeze = float(rt.get("mass_freeze", 0.0))
    fit.finv_ridge = float(rt.get("finv_ridge", 1e-6))
    fit.cfl_agg = rt.get("cfl_agg", "mean")
    if b.get("astress"):
        # 앵커 응력으로 학습한 체크포인트는 힘 법칙 자체가 다르므로, 그것까지
        # 되살려야 저장 당시와 같은 시뮬레이터가 된다.
        from .anchor_stress import AnchorStress
        fit.astress = AnchorStress(eig_floor=fit.eig_floor, dev=device)
        fit.astress.eig_floor = float(rt.get("astress_eig", fit.eig_floor))
        fit.astress.J_max = float(rt.get("astress_jmax", 20.0))
        fit.astress.rebuild(fit)
        fit.astress.load_state_dict(b["astress"])
        fit.edge_only = True
    return FittedScene(sc, fit), b

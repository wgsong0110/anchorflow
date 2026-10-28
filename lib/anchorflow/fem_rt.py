"""FEM 실시간 판 (사용자 10-28 "일단 해봐" 승인 계획, doc/fem_realtime.md 절 H): 파이프라인(train_deform) 밖 단독 시뮬레이터.

train_deform 의 FEM 판(AF_FEM_TOL + AF_FEM_FAST=tet + AF_FEM_TETQ=1, 뉴턴 1 걸음)과 **같은 식**을 상태를 GPU 에 둔 채 바로 돈다
(망·노드 특징·조각 저장·파이프라인 순전파 없음). 한 출력 프레임 = 증분 포텐셜 풀이 한 번 (h = frame_dt, 서브스텝 1).

격자 = 세계 고정 Kuhn 사면체 격자 (SX.grid_for_nodes h_fix = node_h -- 파이프라인 node_feats·mom_core.frame 과 같은 격자).
프레임마다: 격자·locate (입자 -> 사면체 네 노드·barycentric) -> 시작점 U0 = 노드 관성 예측 h v_n + h² g (--ov_init predict 와 같다)
-> 뉴턴 걸음 (기울기·사면체 12x12 헤시안·|λ| 투영·조립·풀이·아르미호 선탐색, 실패하면 LM) -> G2P (x += Σλ U, F <- A_T F).

바닥 항 두 가지 (--floor):
  particle : 기존 정의 그대로 -- 입자 단위 softplus·stick 매핑 (바닥 후보 입자는 입자 12x12, 나머지는 사면체 적분점; fem_fast.FemFrame.tetq_*).
  node     : (사용자 10-28 승인 -- 바닥 항 정의 변경) 모든 입자가 사면체 적분점 (매핑 없음). 바닥은 **노드 단위** 항으로:
             비관통 장벽  Σ_n κ m_n/h² b(d_n; d̂)   (기존 --bc_mode barrier 의 b(d) = -(d-d̂)² ln(d/d̂), quad 연장 그대로)
                d_n(U) = c_n + n·U_n,   c_n = min_{p: n ∈ 사면체(p)} n·(x_p - p0)   (그 노드가 받치는 가장 낮은 입자의 스텝 시작 높이)
             노드는 세계 고정 격자점이라 바닥 아래에 있는 노드가 늘 있다 (입자 바로 아래 셀의 꼭짓점) -- 그래서 노드 자기 높이가 아니라
             "그 노드가 받치는 가장 낮은 입자의 여유" 를 쓴다. 입자 법선 변위 n·u_p = Σ λ_i n·U_i ≥ -Σ λ_i c_i ≥ -s_p 이므로
             모든 d_n > 0 이면 **어느 입자도 바닥 아래로 못 간다** (선형 보간의 볼록 결합).
             고착 (선택, --k_st > 0)  Σ_n k_st m_n/(2h²) w_n |(I - nnᵀ) U_n|²,  w_n = 1 - σ(c_n/ε_st)  (스텝 시작 값으로 고정 -- IPC 식 지연 마찰)
             (기존 stick 매핑 w = σ(s/ε) 를 노드로 옮긴 꼴: 닿은 노드의 접선 변위를 막는다)
"""
from __future__ import annotations

import math
import os
import time

import torch

from . import phys_resid as PR
from . import simplex as SX
from . import fem_fast as FF

F8 = torch.float64


# ---------------------------------------------------------------- 시간 계측
class Tick:
    def __init__(self, sync):
        self.sync = bool(sync) and torch.cuda.is_available()
        self.d = {}
        if self.sync:
            torch.cuda.synchronize()
        self.t = time.perf_counter()

    def __call__(self, k):
        if self.sync:
            torch.cuda.synchronize()
        t = time.perf_counter()
        self.d[k] = self.d.get(k, 0.0) + 1e3 * (t - self.t)
        self.t = t


# ---------------------------------------------------------------- 격자 (스텝 시작 입자 위치)
def _compact(keys, size):
    """정렬 없는 unique (값 범위가 작을 때): (오름차순 유일값, 역색인) -- torch.unique(sorted) 와 같은 결과."""
    mark = torch.zeros(size, dtype=torch.bool, device=keys.device)
    mark[keys] = True
    cm = torch.cumsum(mark, 0) - 1
    return mark.nonzero().squeeze(1), cm[keys]


class Grid:
    def __init__(self, x, node_h, fast=False):
        lo, lat, nn = SX.grid_for_nodes(x, 32, hz_ratio=SX.HZ_CUBE, h_fix=node_h)
        idx, lam, aux = SX.locate(x, lo, lat, nn)
        P = int(nn[0]) * int(nn[1]) * int(nn[2])
        if fast:
            uniq, rows = _compact(idx.reshape(-1), P)
            rows = rows.reshape(idx.shape)
        else:
            rows, uniq = SX.active_nodes(idx)
        self.rows, self.lam, self.uniq, self.lat, self.lo, self.nn = rows, lam, uniq, lat, lo, nn
        self.npos = SX.node_pos(lo, lat, nn, uniq)
        tid = SX.tet_id(lo, lat, nn, aux)
        ut, pinv = _compact(tid, P * 9) if fast else torch.unique(tid, return_inverse=True)
        T = int(ut.numel())
        N = x.shape[0]
        rep = torch.empty(T, dtype=torch.long, device=x.device).scatter_(0, pinv, torch.arange(N, device=x.device))
        self.tets, self.pinv, self.T, self.M, self.N = rows[rep], pinv, T, int(uniq.numel()), N
        X = self.npos[self.tets].double()
        Di = torch.linalg.inv((X[:, 1:] - X[:, :1]).transpose(1, 2))
        self.gl8 = torch.cat([-Di.sum(1, keepdim=True), Di], 1)                 # [T,4,3] ∇λ (float64)
        # 세계 정수 좌표 (멀티그리드용): lo = A imin
        imin = torch.round(lat.Ai.double() @ lo.double()).long()
        nnl = [int(nn[k]) for k in range(3)]
        k2 = uniq % nnl[2]
        k1 = (uniq // nnl[2]) % nnl[1]
        k0 = uniq // (nnl[1] * nnl[2])
        self.gi = torch.stack([k0, k1, k2], -1) + imin


def node_inertial(x, v, m, G, g, h):
    """DM.node_inertial 과 같은 식: h·v_n + h² g (v_n = barycentric 질량 가중 평균)."""
    w = G.lam * m.unsqueeze(-1)
    ws = torch.zeros(G.M, device=x.device, dtype=x.dtype).index_add_(0, G.rows.reshape(-1), w.reshape(-1))
    vs = torch.zeros(G.M, 3, device=x.device, dtype=x.dtype).index_add_(0, G.rows.reshape(-1), (w.unsqueeze(-1) * v.unsqueeze(1)).reshape(-1, 3))
    vn = vs / ws.clamp_min(1e-30).unsqueeze(-1)
    return h * vn + (h * h) * g, ws


# ---------------------------------------------------------------- --cfg_bc (train_deform --cfg_bc 와 같은 정의, 10-28 ficus)
def cfg_bc_sel(cfg, x0):
    """train_deform._cfg_bc_sel 그대로: 입자 구속의 입자 선택 (0 프레임 위치로 한 번) -> [(bc, mask[N])]."""
    dev, dt = x0.device, x0.dtype
    out = []
    for bc in (cfg.get("boundary_conditions") or []):
        tp = bc.get("type")
        if tp == "enforce_particle_translation":
            q = torch.tensor(bc["point"], device=dev, dtype=dt)
            sz = torch.tensor(bc["size"], device=dev, dtype=dt)
            out.append((bc, ((x0 - q).abs() < sz).all(-1)))
        elif tp == "enforce_particle_velocity_rotation":
            n = torch.tensor(bc["normal"], device=dev, dtype=dt)
            n = n / n.norm()
            q = torch.tensor(bc["point"], device=dev, dtype=dt)
            off = x0 - q
            vd = (off @ n).abs()
            hd = (off - (off @ n).unsqueeze(-1) * n).norm(dim=-1)
            hh, rr = (float(z) for z in bc["half_height_and_radius"])
            out.append((bc, (vd < hh) & (hd < rr)))
    return out


def _cfg_bc_rot_v(pos, bc):
    n = torch.tensor(bc["normal"], device=pos.device, dtype=pos.dtype)
    n = n / n.norm()
    q = torch.tensor(bc["point"], device=pos.device, dtype=pos.dtype)
    off = pos - q
    rp = off - (off @ n).unsqueeze(-1) * n
    return float(bc["rotation_scale"]) * torch.cross(rp, n.expand_as(rp), dim=-1) + float(bc["translation_scale"]) * n


def cfg_bc_nodes(cfg, sel, npos, rows, lam, mass, now, dt):
    """train_deform.cfg_bc_nodes (--interp bary) 그대로: 시각 now 에 묶이는 노드와 속도 -> (mask[M] bool, vel[M,3]) 또는 (None, None).
    입자 구속 = 그 입자의 질량 가중 barycentric 비율이 노드 질량의 절반을 넘는 노드, cuboid = 상자 안 (|npos - q| < size, q 는 v·(now - t0) 로 이동) 노드,
    cuboid reset = 끝시각 뒤 15 dt 동안 전체 노드 0. 묶인 노드 변위 = τ·vel (train_deform _on_pin, τ = 스텝 h)."""
    M = npos.shape[0]
    dev, dt_ = npos.device, npos.dtype
    msk = torch.zeros(M, dtype=torch.bool, device=dev)
    vel = torch.zeros(M, 3, device=dev, dtype=dt_)
    hit = False
    wl = lam.to(dt_) * mass.to(dt_).unsqueeze(1)
    den = None
    for bc, pm in sel:
        if not (float(bc.get("start_time", 0.0)) <= now < float(bc.get("end_time", 1e3))):
            continue
        if den is None:
            den = torch.zeros(M, device=dev, dtype=dt_).index_add_(0, rows.reshape(-1), wl.reshape(-1))
        num = torch.zeros(M, device=dev, dtype=dt_).index_add_(0, rows.reshape(-1), (wl * pm.to(dt_).unsqueeze(1)).reshape(-1))
        nm = num > 0.5 * den.clamp_min(1e-30)
        vv = (torch.tensor(bc["velocity"], device=dev, dtype=dt_).expand(M, 3) if bc["type"] == "enforce_particle_translation"
              else _cfg_bc_rot_v(npos, bc))
        vel = torch.where(nm.unsqueeze(-1), vv, vel)
        msk = msk | nm
        hit = True
    for bc in (cfg.get("boundary_conditions") or []):
        if bc.get("type") != "cuboid":
            continue
        t0_, t1_ = float(bc.get("start_time", 0.0)), float(bc.get("end_time", 1e3))
        if not (t0_ <= now < t1_):
            continue
        vb = torch.tensor(bc["velocity"], device=dev, dtype=dt_)
        q = torch.tensor(bc["point"], device=dev, dtype=dt_) + (now - t0_) * vb
        sz = torch.tensor(bc["size"], device=dev, dtype=dt_)
        nm = ((npos - q).abs() < sz).all(-1)
        vel = torch.where(nm.unsqueeze(-1), vb.expand(M, 3), vel)
        msk = msk | nm
        hit = True
    for bc in (cfg.get("boundary_conditions") or []):
        if bc.get("type") == "cuboid" and int(bc.get("reset", 0)) == 1:
            t1_ = float(bc.get("end_time", 1e3))
            if t1_ <= now < t1_ + 15.0 * float(dt):
                msk = torch.ones(M, dtype=torch.bool, device=dev)
                vel = torch.zeros(M, 3, device=dev, dtype=dt_)
                hit = True
    return (msk, vel) if hit else (None, None)


def pg_vdamp_scale(cfg, h):
    """train_deform._pg_vdamp 의 배율: PG grid_v_damping_scale s 를 서브스텝마다 곱한 만큼 -> s^(h/substep_dt) (s ≥ 1 이면 1)."""
    s_ = float(cfg.get("grid_v_damping_scale", 1.0))
    if s_ >= 1.0:
        return 1.0
    return s_ ** (float(h) / float(cfg["substep_dt"]))


# ---------------------------------------------------------------- 장벽 b(d; d̂) (phys_resid._barrier_b quad 연장과 같은 식) 과 도함수
def barrier3(d, dhat, eps_r=1e-3):
    eps = eps_r * dhat
    dc = d.clamp_min(eps)
    L = torch.log(dc / dhat)
    k = dc - dhat
    b = -(k ** 2) * L
    b1 = -2.0 * k * L - k * k / dc
    b2 = -2.0 * L - 4.0 * k / dc + k * k / (dc * dc)
    le = math.log(eps / dhat)
    ke = eps - dhat
    be = -(ke ** 2) * le
    bp = -2.0 * ke * le - (ke ** 2) / eps
    bpp = -2.0 * le - 4.0 * ke / eps + (ke ** 2) / (eps ** 2)
    lo = d < eps
    de = d - eps
    b = torch.where(lo, be + bp * de + 0.5 * bpp * de * de, b)
    b1 = torch.where(lo, bp + bpp * de, b1)
    b2 = torch.where(lo, torch.full_like(d, bpp), b2)
    on = d < dhat
    z = torch.zeros_like(d)
    return torch.where(on, b, z), torch.where(on, b1, z), torch.where(on, b2, z)


# ---------------------------------------------------------------- --floor node: 사면체 적분점 + 노드 바닥 항
class NodeFrame:
    def __init__(self, G, x, v, F, m, vol, cfg, h, kappa=1.0, dhat=0.01, k_st=0.0, eps_st=0.01, clear="min", m_floor=0.0, stick_w="min"):
        dev = x.device
        self.G, self.cfg, self.h = G, cfg, float(h)
        self.mu, self.lam_ = PR.lame(cfg["E"], cfg["nu"])
        h2 = self.h * self.h
        T, M = G.T, G.M
        g = torch.as_tensor(cfg["g"], device=dev, dtype=F8)
        k = m.double() / h2
        lm = G.lam.double()
        c = self.h * v.double() + h2 * g
        pinv = G.pinv
        self.M4 = torch.zeros(T, 4, 4, device=dev, dtype=F8).index_add_(0, pinv, k[:, None, None] * lm[:, :, None] * lm[:, None, :])
        self.B4 = torch.zeros(T, 4, 3, device=dev, dtype=F8).index_add_(0, pinv, k[:, None, None] * lm[:, :, None] * c[:, None, :])
        self.C0 = 0.5 * (k * (c * c).sum(-1)).sum()
        V = vol.double()
        self.VT = torch.zeros(T, device=dev, dtype=F8).index_add_(0, pinv, V)
        self.FT = torch.zeros(T, 3, 3, device=dev, dtype=F8).index_add_(0, pinv, V[:, None, None] * F.double()) / self.VT[:, None, None]
        self.gl = G.gl8
        self.mn = torch.zeros(M, device=dev, dtype=F8).index_add_(0, G.rows.reshape(-1), (G.lam.double() * m.double().unsqueeze(-1)).reshape(-1))
        self.planes = []
        for bc in (cfg.get("boundary_conditions") or []):
            if bc.get("type") == "surface_collider":
                pt = torch.as_tensor(bc["point"], device=dev, dtype=F8)
                nr = torch.as_tensor(bc["normal"], device=dev, dtype=F8)
                nr = nr / nr.norm()
                s = ((x.double() - pt) * nr).sum(-1)
                if clear == "lam":
                    # 노드 n 만 움직일 때 입자 p 가 바닥에 닿는 노드 변위 s_p/λ_pn 의 최소 (λ 작은 꼭짓점은 덜 묶인다; 보장은 볼록 결합이 아니라 근사)
                    sv = (s.unsqueeze(1) / G.lam.double().clamp_min(1e-3)).reshape(-1)
                else:
                    sv = s.unsqueeze(1).expand(-1, 4).reshape(-1)
                cn = torch.full((M,), float("inf"), device=dev, dtype=F8).scatter_reduce_(0, G.rows.reshape(-1), sv, reduce="amin")
                if k_st <= 0:
                    w = torch.zeros_like(cn)
                elif stick_w == "lam":
                    # 입자 고착 무게 (1 - σ(s_p/ε)) 를 λ 로 노드에 (최대): 바닥 입자를 강하게 받치는 꼭짓점만 묶인다
                    wp = (G.lam.double() * (1.0 - torch.sigmoid(s / eps_st)).unsqueeze(1)).reshape(-1)
                    w = torch.zeros(M, device=dev, dtype=F8).scatter_reduce_(0, G.rows.reshape(-1), wp, reduce="amax")
                else:
                    w = 1.0 - torch.sigmoid(cn / eps_st)
                self.planes.append((nr, cn, w))
        mw = self.mn.clamp_min(float(m_floor))                                  # 장벽·고착 무게 질량 (작은 질량 노드가 탄성에 밀려 뚫리지 않게 하한)
        self.kb = float(kappa) * mw / h2                                       # κ m_n/h²
        self.ks = float(k_st) * mw / h2                                        # k_st m_n/h²
        self.dhat, self.k_st = float(dhat), float(k_st)

    def _el(self, Ut):
        A = torch.eye(3, device=Ut.device, dtype=Ut.dtype) + (Ut.unsqueeze(-1) * self.gl.unsqueeze(-2)).sum(1)
        return A, PR.psi_of(A @ self.FT, self.cfg, self.h)[0]

    def energy(self, U):
        U = U.double()
        Ut = U[self.G.tets]
        E = 0.5 * ((self.M4 @ Ut) * Ut).sum() - (self.B4 * Ut).sum() + self.C0
        E = E + (self.VT * self._el(Ut)[1]).sum()
        for nr, cn, w in self.planes:
            d = cn + U @ nr
            E = E + (self.kb * barrier3(d, self.dhat)[0]).sum()
            if self.k_st > 0:
                Ut_ = U - (U @ nr).unsqueeze(-1) * nr
                E = E + 0.5 * (self.ks * w * (Ut_ * Ut_).sum(-1)).sum()
        return E

    def _el_direct(self, Ut, grad):
        """탄성 ψ 와 dE/dU_T 를 자동미분 없이 (PR._fcr_fwd·_fcr_grad 컴파일 커널 그대로; σ_min ≤ 0.01 사면체의 상수 기울기 처리만 생략)."""
        A = torch.eye(3, device=Ut.device, dtype=Ut.dtype) + (Ut.unsqueeze(-1) * self.gl.unsqueeze(-2)).sum(1)
        X = A @ self.FT
        psi, S = PR._fcr_fwd(X, self.mu, self.lam_, 0.01)
        if not grad:
            return psi, None
        P = PR._fcr_grad(X, S, self.mu, self.lam_)
        Gm = self.VT[:, None, None] * (P @ self.FT.transpose(1, 2))          # dE/dA
        return psi, torch.einsum("tcb,tib->tic", Gm, self.gl)

    def energy_fast(self, U):
        U = U.double()
        Ut = U[self.G.tets]
        E = 0.5 * ((self.M4 @ Ut) * Ut).sum() - (self.B4 * Ut).sum() + self.C0 + (self.VT * self._el_direct(Ut, False)[0]).sum()
        return E + self._floor_eg(U)[0]

    def _floor_eg(self, U):
        E = torch.zeros((), device=U.device, dtype=F8)
        g = torch.zeros_like(U)
        for nr, cn, w in self.planes:
            d = cn + U @ nr
            b, b1, _ = barrier3(d, self.dhat)
            E = E + (self.kb * b).sum()
            g = g + (self.kb * b1).unsqueeze(-1) * nr
            if self.k_st > 0:
                Ut_ = U - (U @ nr).unsqueeze(-1) * nr
                E = E + 0.5 * (self.ks * w * (Ut_ * Ut_).sum(-1)).sum()
                g = g + (self.ks * w).unsqueeze(-1) * Ut_
        return E, g

    def energy_grad_fast(self, U):
        U = U.double()
        Ut = U[self.G.tets]
        MU = self.M4 @ Ut
        psi, gel = self._el_direct(Ut, True)
        E = 0.5 * (MU * Ut).sum() - (self.B4 * Ut).sum() + self.C0 + (self.VT * psi).sum()
        gt = MU - self.B4 + gel
        g = torch.zeros_like(U).index_add_(0, self.G.tets.reshape(-1), gt.reshape(-1, 3))
        Ef, gf = self._floor_eg(U)
        return E + Ef, g + gf

    def energy_grad(self, U):
        U = U.double()
        Ut = U[self.G.tets].requires_grad_(True)
        with torch.enable_grad():
            Et = 0.5 * ((self.M4 @ Ut) * Ut).sum() - (self.B4 * Ut).sum() + self.C0 + (self.VT * self._el(Ut)[1]).sum()
            gt, = torch.autograd.grad(Et, Ut)
        g = torch.zeros_like(U).index_add_(0, self.G.tets.reshape(-1), gt.reshape(-1, 3))
        E = Et.detach()
        for nr, cn, w in self.planes:
            d = cn + U @ nr
            b, b1, _ = barrier3(d, self.dhat)
            E = E + (self.kb * b).sum()
            g = g + (self.kb * b1).unsqueeze(-1) * nr
            if self.k_st > 0:
                Ut_ = U - (U @ nr).unsqueeze(-1) * nr
                E = E + 0.5 * (self.ks * w * (Ut_ * Ut_).sum(-1)).sum()
                g = g + (self.ks * w).unsqueeze(-1) * Ut_
        return E, g

    def hess_gn(self, U):
        """가우스-뉴턴 탄성 헤시안 (항상 양반정 -- 고유분해 투영 불필요): ψ = μ|X-R|² + λ/2 (J-1)² (X = A F̄) 에서 R 을 고정하고 (J-1)∂cof 항을 뺀다
            ∂²E/∂A² ≈ V_T [2μ δ_cd (F̄F̄ᵀ)_be + λ q_cb q_de],  q = cof(X) F̄ᵀ = ∂J/∂A
        -> 사면체 12x12 = (M4 + 2μ V_T ∇λ F̄F̄ᵀ ∇λᵀ) ⊗ I3 + λ V_T w wᵀ,  w_(i,c) = Σ_b q_cb ∇λ_ib.  방향만 바뀐다 (에너지·선탐색 그대로)."""
        G = self.G
        T = G.T
        I3 = torch.eye(3, device=U.device, dtype=F8)
        Ut = U.double()[G.tets]
        A = I3 + (Ut.unsqueeze(-1) * self.gl.unsqueeze(-2)).sum(1)
        X = A @ self.FT
        q = PR._cof3(X) @ self.FT.transpose(1, 2)
        S = self.FT @ self.FT.transpose(1, 2)
        L4 = self.M4 + (2.0 * self.mu) * self.VT[:, None, None] * (self.gl @ S @ self.gl.transpose(1, 2))
        H = (L4[:, :, None, :, None] * I3[None, None, :, None, :]).reshape(T, 12, 12)
        w = torch.einsum("tcb,tib->tic", q, self.gl).reshape(T, 12)
        H = H + (self.lam_ * self.VT)[:, None, None] * (w.unsqueeze(-1) * w.unsqueeze(-2))
        return H, self._floor_hess(U)

    def _floor_hess(self, U):
        I3 = torch.eye(3, device=U.device, dtype=F8)
        Dn = torch.zeros(self.G.M, 3, 3, device=U.device, dtype=F8)
        for nr, cn, w in self.planes:
            d = cn + U.double() @ nr
            b2 = barrier3(d, self.dhat)[2]
            N = nr[:, None] * nr[None, :]
            Dn = Dn + (self.kb * b2)[:, None, None] * N
            if self.k_st > 0:
                Dn = Dn + (self.ks * w)[:, None, None] * (I3 - N)
        return Dn

    def hess(self, U, kdt=torch.float32, f32_b=False):
        """(사면체 12x12 [T,12,12] float64 (투영 전), 노드 3x3 [M,3,3] (양반정, 투영 뒤 더한다))."""
        G = self.G
        T = G.T
        I3 = torch.eye(3, device=U.device, dtype=F8)
        H = (self.M4[:, :, None, :, None] * I3[None, None, :, None, :]).reshape(T, 12, 12)
        Ut = U.double()[G.tets]
        A0 = I3 + (Ut.unsqueeze(-1) * self.gl.unsqueeze(-2)).sum(1)
        K = FF._k_manual_run(A0.to(kdt), self.FT.to(kdt), self.VT.to(kdt), float(self.mu), float(self.lam_))
        if f32_b:
            gl = self.gl.float()
            H = H + torch.einsum("tib,tcbde,tje->ticjd", gl, K.float().reshape(T, 3, 3, 3, 3), gl).reshape(T, 12, 12).double()
        else:
            H = H + torch.einsum("tib,tcbde,tje->ticjd", self.gl, K.reshape(T, 3, 3, 3, 3), self.gl).reshape(T, 12, 12)
        Dn = torch.zeros(G.M, 3, 3, device=U.device, dtype=F8)
        for nr, cn, w in self.planes:
            d = cn + U.double() @ nr
            b2 = barrier3(d, self.dhat)[2]
            N = nr[:, None] * nr[None, :]
            Dn = Dn + (self.kb * b2)[:, None, None] * N
            if self.k_st > 0:
                Dn = Dn + (self.ks * w)[:, None, None] * (I3 - N)
        return H, Dn

    def g2p(self, x, F, U):
        G = self.G
        Uf = U.to(x.dtype)
        u = (G.lam.unsqueeze(-1) * Uf[G.rows]).sum(1)
        A = torch.eye(3, device=x.device, dtype=F8) + (U.double()[G.tets].unsqueeze(-1) * self.gl.unsqueeze(-2)).sum(1)
        return x + u, (A.to(F.dtype)[G.pinv] @ F)


# ---------------------------------------------------------------- 고유값 투영 (train_deform _newton_target_asm 과 같은 식)
def project(He, floor=1e-6, f32=True, gate=False):
    """gate=True: 배치 Cholesky 가 되는 (양정치) 사면체는 그대로 두고 실패한 사면체만 고유분해 투영 (|λ| 투영은 양정치에서 항등 --
    다만 바닥 floor·max|λ| 아래 고유값을 올리지 않는 차이만 있다)."""
    He = 0.5 * (He + He.transpose(1, 2))
    if gate:
        _, info = torch.linalg.cholesky_ex(He.float())
        bad = (info != 0).nonzero().squeeze(1)
        neg = float(bad.numel()) / max(He.shape[0], 1)
        if bad.numel():
            Hb, _ = project(He[bad], floor, f32, gate=False)
            He = He.index_copy(0, bad, Hb)
        return He, neg
    if f32:
        ew, eV = torch.linalg.eigh(He.float())
        ew, eV = ew.double(), eV.double()
    else:
        ew, eV = torch.linalg.eigh(He)
    neg = float((ew.min(1).values < 0).double().mean())
    elo = floor * ew.abs().amax(1, keepdim=True)
    return (eV * torch.maximum(ew.abs(), elo).unsqueeze(1)) @ eV.transpose(1, 2), neg


# ---------------------------------------------------------------- 조립: CSR (cuSPARSE 곱) + 노드 블록 대각
class Csr:
    def __init__(self, tets, M):
        dev = tets.device
        T = tets.shape[0]
        dof = (tets.unsqueeze(-1) * 3 + torch.arange(3, device=dev)).reshape(T, 12)
        ri = dof.unsqueeze(-1).expand(T, 12, 12).reshape(-1)
        ci = dof.unsqueeze(1).expand(T, 12, 12).reshape(-1)
        n = 3 * M
        key = ri * n + ci
        uk, ix = torch.unique(key, return_inverse=True)
        crow = torch.zeros(n + 1, device=dev, dtype=torch.long)
        crow[1:] = torch.bincount(uk // n, minlength=n).cumsum(0)
        self.crow, self.col, self.ix, self.nnz, self.n, self.M = crow, uk % n, ix, int(uk.numel()), n, M
        # 노드 3x3 대각 블록의 CSR 위치 (모든 노드가 어떤 사면체엔 들어 있어 패턴에 있다)
        nd = torch.arange(M, device=dev)
        rn = (nd[:, None, None] * 3 + torch.arange(3, device=dev)[None, :, None]).expand(M, 3, 3).reshape(-1)
        cn = (nd[:, None, None] * 3 + torch.arange(3, device=dev)[None, None, :]).expand(M, 3, 3).reshape(-1)
        self.dpos = torch.searchsorted(uk, rn * n + cn)
        self.tets = tets

    def build(self, He, Dn=None):
        vals = torch.zeros(self.nnz, device=He.device, dtype=He.dtype).index_add_(0, self.ix, He.reshape(-1))
        if Dn is not None:
            vals.index_add_(0, self.dpos, Dn.reshape(-1).to(vals.dtype))
        return torch.sparse_csr_tensor(self.crow, self.col, vals, (self.n, self.n))


def block_diag(tets, He, M, Dn=None):
    T = tets.shape[0]
    ar = torch.arange(4, device=He.device)
    blk = He.reshape(T, 4, 3, 4, 3)[:, ar, :, ar, :]                         # [4,T,3,3]
    D = torch.zeros(M, 3, 3, device=He.device, dtype=He.dtype).index_add_(0, tets.t().reshape(-1), blk.reshape(-1, 3, 3))
    if Dn is not None:
        D = D + Dn.to(D.dtype)
    bad = ~(torch.diagonal(D, dim1=1, dim2=2) > 0).all(1)
    D[bad] = torch.eye(3, device=D.device, dtype=D.dtype)
    return D


def pcg(mv, pre, b, ncg, rtol, ck=50):
    """전처리 CG (train_deform._asm_pcg_nosync 와 같은 판정: |r|² < rtol²|b|², ck 회마다만 확인, 음곡률이면 멈춤). -> (x, 반복, |r|²/|b|², 음곡률)."""
    x = torch.zeros_like(b)
    r = b.clone()
    z = pre(r)
    d = z.clone()
    rz = (r * z).sum()
    b2 = (b * b).sum().clamp_min(1e-300)
    thr = b2 * (rtol * rtol)
    alive = torch.ones((), dtype=torch.bool, device=b.device)
    negf = torch.zeros((), dtype=torch.bool, device=b.device)
    nit = torch.zeros((), dtype=torch.long, device=b.device)
    one = torch.ones((), dtype=b.dtype, device=b.device)
    for kk in range(1, ncg + 1):
        Ad = mv(d)
        dAd = (d * Ad).sum()
        neg_now = alive & ~(dAd > 0)
        negf = negf | neg_now
        alive = alive & ~neg_now
        al = torch.where(alive, rz / torch.where(dAd > 0, dAd, one), torch.zeros_like(rz))
        x = x + al * d
        r = r - al * Ad
        rr = (r * r).sum()
        nit = nit + alive.long()
        alive = alive & ~(rr < thr)
        z = pre(r)
        rz2 = (r * z).sum()
        d = z + torch.where(alive, rz2 / rz, torch.zeros_like(rz)) * d
        rz = torch.where(alive, rz2, rz)
        if kk % ck == 0 and not bool(alive):
            break
    return x, int(nit), float((r * r).sum() / b2), bool(negf)


_GS = [None]


def pcg_graph(mv, pre, b, ncg, rtol, ck=4):
    """pcg 와 같은 식·같은 판정을 CUDA 그래프로: 반복 한 번 (곱 + 전처리 + 갱신) 을 잡아 두고 재생, ck 회 재생마다 수렴 확인.
    커널 띄우기 비용을 없앤다 (멀티그리드 V 사이클은 반복당 커널 수백 개). 잡기 실패하면 None."""
    dev, dt = b.device, b.dtype
    z = pre(b)
    S = dict(x=torch.zeros_like(b), r=b.clone(), d=z.clone(), rz=(b * z).sum(), thr=(b * b).sum().clamp_min(1e-300) * (rtol * rtol),
             alive=torch.ones((), dtype=torch.bool, device=dev), neg=torch.zeros((), dtype=torch.bool, device=dev),
             nit=torch.zeros((), dtype=torch.long, device=dev), one=torch.ones((), dtype=dt, device=dev), zero=torch.zeros((), dtype=dt, device=dev))

    def body():
        Ad = mv(S["d"])
        dAd = (S["d"] * Ad).sum()
        neg_now = S["alive"] & ~(dAd > 0)
        S["neg"].copy_(S["neg"] | neg_now)
        S["alive"].copy_(S["alive"] & ~neg_now)
        al = torch.where(S["alive"], S["rz"] / torch.where(dAd > 0, dAd, S["one"]), S["zero"])
        S["x"].add_(al * S["d"])
        S["r"].sub_(al * Ad)
        rr = (S["r"] * S["r"]).sum()
        S["nit"].add_(S["alive"].long())
        S["alive"].copy_(S["alive"] & ~(rr < S["thr"]))
        zz = pre(S["r"])
        rz2 = (S["r"] * zz).sum()
        beta = torch.where(S["alive"], rz2 / S["rz"], S["zero"])
        S["d"].copy_(zz + beta * S["d"])
        S["rz"].copy_(torch.where(S["alive"], rz2, S["rz"]))
    try:
        # torch.cuda.graph(...) 문맥은 들어갈 때 gc.collect()·empty_cache() 를 해서 한 번에 ~95 ms 가 든다 (10-28 실측) -- capture_begin/end 를 직접
        if _GS[0] is None:
            _GS[0] = torch.cuda.Stream()
        st = _GS[0]
        st.wait_stream(torch.cuda.current_stream())
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.stream(st):
            keep = {k: v.clone() for k, v in S.items()}
            body()
            for k, v in keep.items():
                S[k].copy_(v)
            gr.capture_begin()
            body()
            gr.capture_end()
        torch.cuda.current_stream().wait_stream(st)
    except Exception as e:                                                    # noqa: BLE001
        print(f"[pcg_graph] 잡기 실패 -> 그래프 없이: {type(e).__name__}: {str(e)[:200]}", flush=True)
        return None
    n = 0
    while n < ncg:
        for _ in range(ck):
            gr.replay()
        n += ck
        if not bool(S["alive"]):
            break
    b2 = (b * b).sum().clamp_min(1e-300)
    return S["x"], int(S["nit"]), float((S["r"] * S["r"]).sum() / b2), bool(S["neg"])


# ---------------------------------------------------------------- Kuhn 격자 기하 멀티그리드 (블록 ELL, 15 점 스텐실)
# 변 오프셋은 0/1 벡터 7 개와 그 음수 (simplex.EDGE_OFFSETS) + 자기 자신 = 15. 격자를 2 배로 성기게 하면 Freudenthal 분할이 포개지므로
# (가는 사면체가 성긴 사면체 하나 안에 든다) 가는 점 g = 2c + r (r ∈ {0,1}³) 은 성긴 변 (c, c+r) 의 중점 -> 선형 보간 P 의 행은 많아야 2 개,
# Galerkin Pᵀ A P 도 같은 15 점 스텐실에 든다 (강체 6 모드는 선형 보간이 정확히 담는다).
_OFFS = [(0, 0, 0)] + list(SX.EDGE_OFFSETS)
_TBL = {}


def _tbl(dev):
    if dev not in _TBL:
        off = torch.tensor(_OFFS, device=dev, dtype=torch.long)
        t = torch.full((27,), -1, device=dev, dtype=torch.long)
        t[(off[:, 0] + 1) * 9 + (off[:, 1] + 1) * 3 + (off[:, 2] + 1)] = torch.arange(15, device=dev)
        _TBL[dev] = (off, t)
    return _TBL[dev]


def off_cls(d):
    off, t = _tbl(d.device)
    ok = (d.abs() <= 1).all(-1)
    key = ((d[..., 0] + 1) * 9 + (d[..., 1] + 1) * 3 + (d[..., 2] + 1)).clamp(0, 26)
    return torch.where(ok, t[key], torch.full_like(key, -1))


class BEll:
    """블록 ELL 행렬 [M,15,3,3]: 노드 i 의 칸 k = 이웃 gi + off_k."""

    def __init__(self, gi, dtype=F8):
        dev = gi.device
        off, _ = _tbl(dev)
        M = gi.shape[0]
        g0 = gi.min(0).values - 1
        dims = gi.max(0).values - g0 + 2
        lin = lambda q: ((q[..., 0] - g0[0]) * dims[1] + (q[..., 1] - g0[1])) * dims[2] + (q[..., 2] - g0[2])
        lut = torch.full((int(dims.prod()),), -1, device=dev, dtype=torch.long)
        lut[lin(gi)] = torch.arange(M, device=dev)
        nb = lut[lin(gi.unsqueeze(1) + off.unsqueeze(0))]                     # [M,15]
        self.valid = nb >= 0
        self.nb = torch.where(self.valid, nb, torch.arange(M, device=dev).unsqueeze(1).expand(M, 15))
        self.gi, self.M, self.dtype = gi, M, dtype
        self.B = torch.zeros(M * 15, 3, 3, device=dev, dtype=dtype)

    def add_tets(self, tets, He):
        T = tets.shape[0]
        gi = self.gi
        cls = off_cls(gi[tets].unsqueeze(1) - gi[tets].unsqueeze(2))           # [T,4(i),4(j)]: off = g_j - g_i
        slot = tets.unsqueeze(-1) * 15 + cls
        blk = He.reshape(T, 4, 3, 4, 3).permute(0, 1, 3, 2, 4).reshape(-1, 3, 3)
        self.B.index_add_(0, slot.reshape(-1), blk.to(self.dtype))

    def add_diag(self, Dn):
        self.B[0::15] += Dn.to(self.dtype)

    def diag(self):
        return self.B[0::15]

    def mv(self, x):
        X = x.reshape(self.M, 3)[self.nb]                                      # [M,15,3]
        return torch.einsum("mkab,mkb->ma", self.B.view(self.M, 15, 3, 3), X.to(self.B.dtype)).reshape(-1).to(x.dtype)

    def to_csr_sorted(self, nn):
        """정렬 없이 CSR: 노드 번호가 격자 평탄 색인 순서라 이웃 칸을 평탄 오프셋 순으로 놓으면 행 안 열이 오름차순이다."""
        M = self.M
        dev = self.B.device
        off, _ = _tbl(dev)
        lin = (off[:, 0] * int(nn[1]) + off[:, 1]) * int(nn[2]) + off[:, 2]
        o = torch.argsort(lin)
        B = self.B.view(M, 15, 3, 3)[:, o]                                     # [M,15(정렬),3,3]
        nb = self.nb[:, o]
        ok = self.valid[:, o]
        cnt = ok.sum(1) * 3                                                    # 행마다 원소 수 (세 성분 행 같다)
        crow = torch.zeros(3 * M + 1, device=dev, dtype=torch.long)
        crow[1:] = cnt.repeat_interleave(3).cumsum(0)
        # 원소 순서: 행 (m, a) -> 칸 k (유효만) -> b
        Bt = B.permute(0, 2, 1, 3)                                             # [M,3(a),15,3(b)]
        okx = ok[:, None, :, None].expand(M, 3, 15, 3)
        vals = Bt[okx]
        col = (nb[:, None, :, None] * 3 + torch.arange(3, device=dev)).expand(M, 3, 15, 3)[okx]
        return torch.sparse_csr_tensor(crow, col, vals, (3 * M, 3 * M))

    def to_csr(self):
        M = self.M
        dev = self.B.device
        m = torch.arange(M, device=dev)[:, None, None, None].expand(M, 15, 3, 3)
        a_ = torch.arange(3, device=dev)[None, None, :, None].expand(M, 15, 3, 3)
        b_ = torch.arange(3, device=dev)[None, None, None, :].expand(M, 15, 3, 3)
        nbx = self.nb[:, :, None, None].expand(M, 15, 3, 3)
        ok = self.valid[:, :, None, None].expand(M, 15, 3, 3)
        r = (m * 3 + a_)[ok]
        c = (nbx * 3 + b_)[ok]
        vals = self.B.view(M, 15, 3, 3)[ok]
        n = 3 * M
        o = torch.argsort(r * n + c)
        crow = torch.zeros(n + 1, device=dev, dtype=torch.long)
        crow[1:] = torch.bincount(r, minlength=n).cumsum(0)
        return torch.sparse_csr_tensor(crow, c[o], vals[o], (n, n))

    def dense(self):
        M = self.M
        i = torch.arange(M, device=self.B.device).unsqueeze(1).expand(M, 15)
        Bv = self.B.view(M, 15, 3, 3) * self.valid[..., None, None].to(self.B.dtype)
        Af = torch.zeros(M * M, 3, 3, device=self.B.device, dtype=self.B.dtype)
        Af.index_add_(0, (i * M + self.nb).reshape(-1), Bv.reshape(-1, 3, 3))
        return Af.reshape(M, M, 3, 3).permute(0, 2, 1, 3).reshape(3 * M, 3 * M)


def coarsen(gi):
    c = torch.div(gi, 2, rounding_mode="floor")
    r = gi - 2 * c
    par = torch.stack([c, c + r], 1)                                             # [M,2,3]
    mid = r.any(-1)
    w = torch.stack([torch.where(mid, 0.5, 1.0), torch.where(mid, 0.5, 0.0)], 1).to(F8)
    q = par.reshape(-1, 3)
    q0 = q.min(0).values
    dims = q.max(0).values - q0 + 1
    key = ((q[:, 0] - q0[0]) * dims[1] + (q[:, 1] - q0[1])) * dims[2] + (q[:, 2] - q0[2])
    uk, inv = _compact(key, int(dims.prod()))
    k2 = uk % dims[2]
    k1 = (uk // dims[2]) % dims[1]
    k0 = uk // (dims[1] * dims[2])
    gc = torch.stack([k0, k1, k2], -1) + q0
    return gc, inv.reshape(-1, 2), w


def galerkin(Af, gc, pidx, w):
    Ac = BEll(gc, dtype=Af.dtype)
    M = Af.M
    i = torch.arange(M, device=gc.device).unsqueeze(1).expand(M, 15)
    j = Af.nb
    Bv = Af.B.view(M, 15, 3, 3)
    drop = 0
    for a_ in range(2):
        ra = pidx[i, a_]
        wa = w[i, a_]
        for b_ in range(2):
            cb = pidx[j, b_]
            wt = wa * w[j, b_] * Af.valid.to(F8)
            cls = off_cls(gc[cb] - gc[ra])
            ok = (cls >= 0) & (wt > 0)
            drop += int(((cls < 0) & (wt > 0)).sum())
            Ac.B.index_add_(0, (ra * 15 + cls.clamp_min(0))[ok], (wt[..., None, None].to(Af.dtype) * Bv)[ok])
    return Ac, drop


class KuhnMG:
    """기하 멀티그리드 V(ν,ν) 전처리: 블록 야코비 ω 매끄럽게, 가장 성긴 수준은 조밀 Cholesky (float64)."""

    def __init__(self, A0, nu=2, omega=0.6, nmin=400, maxlev=8, dtype=F8, A0csr=None, bell=False):
        self.levels = []
        A = A0
        self.drop = 0
        while True:
            D = A.diag().to(F8).clone()
            bad = ~(torch.diagonal(D, dim1=1, dim2=2) > 0).all(1)
            D[bad] = torch.eye(3, device=D.device, dtype=F8)
            Dinv = torch.linalg.inv(D).to(dtype)
            if A.M <= nmin or len(self.levels) >= maxlev:
                K = A.dense().to(F8)
                K = 0.5 * (K + K.t())
                dg = K.diagonal()
                K = K + torch.diag(torch.where(dg > 0, 1e-12 * dg, torch.ones_like(dg)))
                self.Lc = torch.linalg.cholesky(K)
                self.Ki = torch.cholesky_inverse(self.Lc).to(dtype)
                self.levels.append(dict(A=A, Dinv=Dinv, P=None))
                break
            gc, pidx, w = coarsen(A.gi)
            Ac, dr = galerkin(A, gc, pidx, w)
            self.drop += dr
            Cs = None if bell else (A0csr if (A is A0 and A0csr is not None) else A.to_csr())
            if Cs is not None and Cs.values().dtype != dtype:
                Cs = torch.sparse_csr_tensor(Cs.crow_indices(), Cs.col_indices(), Cs.values().to(dtype), Cs.shape)
            self.levels.append(dict(A=A, C=Cs, Dinv=Dinv, P=(pidx, w.to(dtype)), Mc=int(gc.shape[0])))
            A = Ac
        self.nu, self.omega, self.dtype = int(nu), float(omega), dtype
        self.sizes = [lv["A"].M for lv in self.levels]

    def _jac(self, lv, r):
        return (lv["Dinv"] @ r.reshape(-1, 3, 1)).reshape(-1)

    def _cycle(self, l, r):
        lv = self.levels[l]
        if lv["P"] is None:
            return self.Ki @ r
        C = lv["C"]
        mv = (lambda q: (C @ q.unsqueeze(-1)).squeeze(-1)) if C is not None else lv["A"].mv
        om = self.omega
        x = om * self._jac(lv, r)
        for _ in range(self.nu - 1):
            x = x + om * self._jac(lv, r - mv(x))
        res = (r - mv(x)).reshape(-1, 3)
        pidx, w = lv["P"]
        rc = torch.zeros(lv["Mc"], 3, device=r.device, dtype=r.dtype).index_add_(
            0, pidx.reshape(-1), (w.unsqueeze(-1) * res.unsqueeze(1)).reshape(-1, 3))
        ec = self._cycle(l + 1, rc.reshape(-1)).reshape(-1, 3)
        x = x + (w.unsqueeze(-1) * ec[pidx]).sum(1).reshape(-1)
        for _ in range(self.nu):
            x = x + om * self._jac(lv, r - mv(x))
        return x

    def __call__(self, r):
        return self._cycle(0, r.to(self.dtype)).to(r.dtype)


# ---------------------------------------------------------------- PD 행렬 (스칼라 L ⊗ I3) 미리 분해 (cholespy: CHOLMOD 분해 + GPU 삼각 풀이)
class PdFactor:
    """L = diag(m_n/h² + c_n) + w Σ_T V_T (∇λ ∇λᵀ)_T  (노드 스칼라, 세 성분 같은 행렬). w = pd_w·2μ (ARAP 의 PD 무게, F̄≈I),
    c_n = 바닥 항 등방 대각 (선택). 같은 노드 집합이면 분해를 재사용 (값이 바뀌어도 -- 전처리/준뉴턴 행렬이라 식은 그대로)."""

    def __init__(self):
        self.key = None
        self.solver = None
        self.n_fact = 0
        self.t_fact = 0.0

    def build(self, G, mn, VT, h, w, cdiag=None, reuse=True):
        key = (int(G.M), int(G.uniq.sum()), int(G.uniq[::97].sum()))
        if reuse and self.key == key and self.solver is not None:
            return False
        from cholespy import CholeskySolverD, MatrixType
        t0 = time.perf_counter()
        dev = mn.device
        T = G.T
        Lt = w * VT[:, None, None] * (G.gl8 @ G.gl8.transpose(1, 2))           # [T,4,4]
        ri = G.tets.unsqueeze(-1).expand(T, 4, 4).reshape(-1)
        ci = G.tets.unsqueeze(1).expand(T, 4, 4).reshape(-1)
        dg = mn / (h * h) + (cdiag if cdiag is not None else 0.0)
        nd = torch.arange(G.M, device=dev)
        ii = torch.cat([ri, nd])
        jj = torch.cat([ci, nd])
        vv = torch.cat([Lt.reshape(-1), dg.clamp_min(1e-12)])
        S = torch.sparse_coo_tensor(torch.stack([ii, jj]), vv, (G.M, G.M)).coalesce()
        idx = S.indices()
        self.solver = CholeskySolverD(G.M, idx[0].int().contiguous(), idx[1].int().contiguous(), S.values().contiguous(), MatrixType.COO)
        self.key = key
        self.n_fact += 1
        torch.cuda.synchronize()
        self.t_fact += time.perf_counter() - t0
        return True

    def solve(self, r):
        R = r.reshape(-1, 3).to(F8).contiguous()
        X = torch.zeros_like(R)
        self.solver.solve(R, X)
        return X.reshape(-1).to(r.dtype)

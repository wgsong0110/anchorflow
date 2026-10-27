"""운동량 잔차 변위 망 -- MPM 식 고정 사면체 격자 시뮬레이터 (사용자 10-27 확정 설계, doc/mom_lego.md).

격자 = 세계 고정 Kuhn 사면체 격자 (SX.grid_for_nodes h_fix = --node_h, hz = 1/√6; train_deform·xpbd 와 같은 격자).
매 스텝 (스텝 길이 h):
  (1) P2G   입자 -> 노드 barycentric: m_n = Σ λ m_p, p_n = Σ λ m_p v_p, v_n = p_n / m_n.
            사면체마다 F̄_T = Σ_{p∈T} V_p F_p / V_T, V_T = Σ_{p∈T} V_p (V_p = 입자 초기 부피 = m_p/ρ).
  (2) 응력  τ_T = stress(F̄_T). 지금은 **고정 FCR** (FcrStress: τ = P(F̄)F̄ᵀ, P = ∂ψ/∂F, ψ = phys_resid.psi_of 그대로 --
            deform_model.kirchhoff_tau). 나중에 학습 응력 모델로 바꿀 자리는 stress 객체 하나뿐이다.
            노드 내부력 f_n = −Σ_{T∋n} V_T τ_T ∇λ_{n,T}   (= −∂/∂Δx_n Σ_T V_T ψ(A_T F̄_T) 의 Δx=0 값, check_force 로 검산)
  (3) 변위 망 Δx_n = net(입력) -- 관성 예측 더하기 없음, 망 출력이 Δx 그 자체 (dt 비례 출력 크기 --dt_scale 은 기존과 같은 구조).
            입력 (노드, 고정 상수 변환만): symlog(f/m ; |g|) 3, v [m/s] 3, log(m_n/m_full) 1, 점유 (점유 사면체 수/24) 1,
            log10(h/h_ref) 1, 바닥 근접 (--bc_feat near 와 같은 [플래그, 거리/h_node]) 2 = 11 + dt FiLM.
  (4) G2P   x_p += Σ λ Δx_n, v_p = Σ λ Δx_n / h, F_p ← A_T F_p, A_T = I + Σ_i Δx_i ⊗ ∇λ_i.

손실 (탄성 에너지를 다시 계산하지 않는다; f 는 스텝 시작 상태에서 정해진 상수):
  잔차  r_n = m_n (Δx_n − h v_n)/h² − f_n − m_n g
        E_res = Σ_n h² |r_n|² / (2 m_n) = Σ_n m_n/(2h²) |Δx_n − x̃_n|²,   x̃_n = h v_n + h²(f_n/m_n + g)   [J]
        (잔차 제곱을 노드 질량으로 나눈 M⁻¹ 노름 × h²/2 -- 증분 포텐셜의 관성항과 같은 꼴·단위라 아래 장벽과 단위가 맞고,
         망 출력 속도 Δx/h 의 오차로 보면 ½Σm|δv|² 라 dt 에 따라 무게가 바뀌지 않는다)
  뒤집힘 E_inv = Σ_T V_T κ_inv μ b(det A_T; Ĵ, 선형 연장)        (기존 --inv_barrier κ --inv_jhat Ĵ --inv_ext linear 와 같은 꼴, det F 대신 det A_T)
  바닥  E_bc  = Σ_p κ_bc m_p/h² b(sd_p(x_p + Σλ Δx); d̂, 선형 연장)  (기존 --bc_mode barrier 의 bc_energy 와 같은 꼴, d̂ = 격자 dx/2)
  손실 = (E_res + E_inv + E_bc) / E_ref,  E_ref = M_tot |g| EXT  (고정 상수 -- 표본마다 바뀌는 양으로 나누지 않는다)
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import torch

from . import deform_model as DM
from . import phys_resid
from . import simplex as SX
from . import simplex_gnn as SGM

HZ = SX.HZ_CUBE
FEAT_NAMES = ["f/m_x symlog", "f/m_y symlog", "f/m_z symlog", "v_x", "v_y", "v_z", "log(m/m_full)", "점유(사면체수/24)",
              "log10(h/h_ref)", "바닥 플래그", "바닥 거리/h_node"]
N_FEAT = len(FEAT_NAMES)


def symlog(x, c):
    return torch.sign(x) * torch.log1p(x.abs() / c)


class Scene:
    """형상·물성·초기 상태·정답 (lego L000 .pt). xpbd_roll 과 같은 방식으로 읽는다 (질량 DM.traj_mass, 부피 = 질량/밀도,
    초기 x = x[0], v = 궤적 v[0] (--v0_traj), F = DM.traj_F[0], EXT = DM.extent(x[0]))."""

    def __init__(self, pt, node_h, dev="cuda", k_inv=1.0, jhat=0.3, k_bc=1.0, dhat=0.0):
        d = torch.load(pt, map_location="cpu", weights_only=False)
        self.d = d
        cfg = d["cfg"]
        self.cfg = cfg
        self.dev = dev
        self.frame_dt = float(cfg["frame_dt"])
        self.n_frames = int(d["x"].shape[0])
        self.m = DM.traj_mass(d, dev).float()
        rho = d.get("_rho")
        self.rho = float(cfg["density"])
        self.Vp = self.m / (rho.to(dev).float() if rho is not None else self.rho)
        self.EXT = float(d["_ext"])
        N = int(d["x"].shape[1])
        self.N = N
        self.idx = torch.arange(N)
        self.x0 = DM.take(d["x"][0], self.idx, dev).float()
        self.v0 = DM.take(d["v"][0], self.idx, dev).float()
        self.F0 = DM.traj_F(d, self.frame_dt, dev)[0].to(dev).float().reshape(N, 3, 3)
        self.g = torch.tensor([float(q) for q in cfg["g"]], device=dev)
        self.gmag = float(self.g.norm())
        self.mu, self.lam = phys_resid.lame(cfg["E"], cfg["nu"])
        self.planes = []
        for bc in (cfg.get("boundary_conditions") or []):
            if bc.get("type") == "surface_collider":
                n = torch.tensor(bc["normal"], device=dev, dtype=torch.float32)
                self.planes.append((torch.tensor(bc["point"], device=dev, dtype=torch.float32), n / n.norm()))
        self.node_h = float(node_h)
        lat = SX.lattice(self.node_h, HZ * self.node_h, dtype=torch.float64)
        self.v_node = float(torch.linalg.det(lat.A).abs())            # 격자점 하나당 부피 (꽉 찬 노드의 질량 = ρ·v_node)
        self.m_full = self.rho * self.v_node
        self.h_ref = self.frame_dt
        self.k_inv, self.jhat, self.k_bc = float(k_inv), float(jhat), float(k_bc)
        self.dhat = float(dhat) if dhat > 0 else 0.5 * float(cfg.get("grid_lim", 2.0)) / int(cfg["n_grid"])
        self.M_tot = float(self.m.sum())
        self.E_ref = self.M_tot * self.gmag * self.EXT

    def gt(self, i):
        return DM.take(self.d["x"][i], self.idx, self.dev).float()

    def describe(self):
        c = self.cfg
        return (f"입자 {self.N}, 프레임 {self.n_frames} (dt {self.frame_dt:g}), 재질 {phys_resid.mat_name(c)} E {c['E']:g} ν {c['nu']:g} "
                f"-> μ {self.mu:.6g} λ {self.lam:.6g}, ρ {self.rho:g}, g {self.g.tolist()}, 질량 합 {self.M_tot:.6g}, 부피 합 {float(self.Vp.sum()):.6g}, "
                f"EXT {self.EXT:.6f}, node_h {self.node_h}, 노드 부피 {self.v_node:.4e} m_full {self.m_full:.4e}, 바닥 {len(self.planes)} 개 d̂ {self.dhat:.5f}, "
                f"E_ref = M|g|EXT {self.E_ref:.6g}, F0 det {float(torch.linalg.det(self.F0).min()):.4f}~{float(torch.linalg.det(self.F0).max()):.4f}, "
                f"|v0| 평균 {float(self.v0.norm(dim=-1).mean()):.4f}")


# ---------------------------------------------------------------- 격자 구조 (스텝 시작 입자 위치로)
def frame(x, node_h):
    lo, lat, nn = SX.grid_for_nodes(x, 32, hz_ratio=HZ, h_fix=node_h)
    idx, lam, aux = SX.locate(x, lo, lat, nn)
    rows, uniq = SX.active_nodes(idx)
    npos = SX.node_pos(lo, lat, nn, uniq)
    tid = SX.tet_id(lo, lat, nn, aux)
    _ut, pinv = torch.unique(tid, return_inverse=True)
    T = int(_ut.numel())
    rep = torch.empty(T, dtype=torch.long, device=x.device).scatter_(0, pinv, torch.arange(x.shape[0], device=x.device))
    tets = rows[rep]                                              # [T,4] (입자 rows 와 같은 사슬 순서)
    X = npos[tets].double()
    Di = torch.linalg.inv((X[:, 1:] - X[:, :1]).transpose(1, 2))   # 행 i = ∇λ_{i+1}
    gl = torch.cat([-Di.sum(1, keepdim=True), Di], 1).float()      # [T,4,3]
    src, dst, cls = SX.edges_of(rows, uniq, nn)
    M = int(uniq.numel())
    ninc = torch.bincount(tets.reshape(-1), minlength=M).float()
    return SimpleNamespace(lo=lo, lat=lat, nn=nn, rows=rows, lam=lam, uniq=uniq, npos=npos, pinv=pinv, T=T, tets=tets, gl=gl,
                           src=src, dst=dst, cls=cls, M=M, ninc=ninc)


def p2g(S, v, F, m, Vp):
    wl = S.lam * m.unsqueeze(1)
    mn = torch.zeros(S.M, device=v.device, dtype=v.dtype).index_add_(0, S.rows.reshape(-1), wl.reshape(-1))
    pn = torch.zeros(S.M, 3, device=v.device, dtype=v.dtype).index_add_(
        0, S.rows.reshape(-1), (wl.unsqueeze(-1) * v.unsqueeze(1)).reshape(-1, 3))
    vn = pn / mn.clamp_min(1e-30).unsqueeze(-1)
    VT = torch.zeros(S.T, device=v.device, dtype=v.dtype).index_add_(0, S.pinv, Vp)
    Fb = torch.zeros(S.T, 3, 3, device=v.device, dtype=v.dtype).index_add_(0, S.pinv, Vp[:, None, None] * F)
    Fb = Fb / VT.clamp_min(1e-30)[:, None, None]
    return mn, vn, VT, Fb


class FcrStress:
    """셀 응력 모델 자리 -- 지금은 고정 FCR 탄성 (deform_model.kirchhoff_tau = phys_resid.psi_of 를 미분한 τ = P Fᵀ, 씬 E·ν).
    학습 응력 모델로 바꿀 때는 같은 호출 꼴 (F̄ [T,3,3] -> τ [T,3,3]) 의 객체로 갈아 끼운다."""
    name = "fcr"

    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, Fb):
        return DM.kirchhoff_tau(Fb, self.cfg)


def node_force(S, VT, tau):
    """f_n = −Σ_{T∋n} V_T τ_T ∇λ_{n,T}."""
    ft = -VT[:, None, None] * (S.gl @ tau.transpose(1, 2))       # [T,4,3] 행 i = (τ ∇λ_i)ᵀ
    return torch.zeros(S.M, 3, device=tau.device, dtype=tau.dtype).index_add_(0, S.tets.reshape(-1), ft.reshape(-1, 3))


def cell_A(S, dx):
    """A_T = I + Σ_i Δx_i ⊗ ∇λ_i [T,3,3]."""
    return torch.eye(3, device=dx.device, dtype=dx.dtype) + dx[S.tets].transpose(1, 2) @ S.gl


def node_feats(sc, S, mn, vn, f, h):
    a = f / mn.clamp_min(1e-30).unsqueeze(-1)
    near = DM.bc_near_feats(sc.cfg, S.rows, S.npos, S.M, sc.node_h, "near", 1.0)
    M = S.M
    return torch.cat([symlog(a, sc.gmag), vn, torch.log(mn.clamp_min(1e-30) / sc.m_full).unsqueeze(1), (S.ninc / 24.0).unsqueeze(1),
                      torch.full((M, 1), math.log10(h / sc.h_ref), device=vn.device), near.float()], 1).float()


def build_net(sc, hidden=128, levels=3, dev="cuda"):
    """gunet5_near 와 같은 몸통: Graph U-Net 3 수준 (h·2h·4h, 층 2+2+2+1+1, 폭 128), 거친 수준 바닥 near, dt FiLM, dt 비례 출력,
    출력 크기 scale = 0.02·EXT (build_net 과 같은 규칙), 0 초기화 출력."""
    SGM._MP_NORM, SGM._MP_MEAN = True, True
    return SGM.GraphUNetGNN(n_feat=N_FEAT, hidden=hidden, levels=levels, bc_mode="near", bc_near_h=1.0,
                            scale=0.02 * sc.EXT, dt_cond=True, dt_ref=sc.h_ref, dt_scale=True, n_mat=0, n_out=3).to(dev)


def net_dx(net, sc, S, feat, h):
    dt = torch.tensor(float(h), device=feat.device)
    return net(feat, S.src, S.dst, S.cls, dt, grid=(S.uniq, S.nn, S.lo, S.lat, sc.cfg))[0]


def prepare(sc, x, v, F, h, stress):
    """(1)·(2): 격자, P2G, 응력, 노드 힘, 입력 특징."""
    S = frame(x, sc.node_h)
    mn, vn, VT, Fb = p2g(S, v, F, sc.m, sc.Vp)
    tau = stress(Fb)
    f = node_force(S, VT, tau)
    feat = node_feats(sc, S, mn, vn, f, h)
    return SimpleNamespace(S=S, mn=mn, vn=vn, VT=VT, Fb=Fb, tau=tau, f=f, feat=feat, h=float(h))


def explicit_dx(sc, P):
    """명시적 오일러 해 x̃ = h v + h²(f/m + g) (잔차 0 인 Δx)."""
    h = P.h
    return h * P.vn + h * h * (P.f / P.mn.clamp_min(1e-30).unsqueeze(-1) + sc.g)


def loss_terms(sc, P, dx, xp):
    """-> (손실 성분 dict (E_ref 로 나눈 무차원), det A_T, 스텝 끝 입자 바닥 거리 [planes 별 목록])."""
    h, S = P.h, P.S
    hh = h * h
    xt = explicit_dx(sc, P)
    E_res = (P.mn / (2.0 * hh) * ((dx - xt) ** 2).sum(-1)).sum()
    J = torch.linalg.det(cell_A(S, dx))
    E_inv = (P.VT * (sc.k_inv * sc.mu) * phys_resid._barrier_b(J, sc.jhat, ext_kind="linear")).sum()
    E_bc = torch.zeros((), device=dx.device)
    sds = []
    if sc.planes:
        xq = xp + (S.lam.unsqueeze(-1) * dx[S.rows]).sum(1)
        for p0, n in sc.planes:
            sd = (xq - p0) @ n
            sds.append(sd.detach())
            E_bc = E_bc + (sc.k_bc * sc.m / hh * phys_resid._barrier_b(sd, sc.dhat, ext_kind="linear")).sum()
    return dict(res=E_res / sc.E_ref, inv=E_inv / sc.E_ref, bc=E_bc / sc.E_ref), J.detach(), sds


def g2p(P, x, F, dx):
    S = P.S
    A = cell_A(S, dx)
    x2 = x + (S.lam.unsqueeze(-1) * dx[S.rows]).sum(1)
    v2 = (S.lam.unsqueeze(-1) * (dx / P.h)[S.rows]).sum(1)
    F2 = A[S.pinv] @ F
    return x2, v2, F2, A


def step(net, sc, x, v, F, h, stress):
    P = prepare(sc, x, v, F, h, stress)
    dx = net_dx(net, sc, P.S, P.feat, h)
    x2, v2, F2, A = g2p(P, x, F, dx)
    detA = torch.linalg.det(A)
    return x2, v2, F2, dict(nodes=P.S.M, tets=P.S.T, inv_tets=int((detA <= 0).sum()), detA_min=float(detA.min()))


def check_force(sc, P, stress, n_dir=2, gen=None):
    """f_n 검산: −∂/∂Δx Σ_T V_T ψ(A_T F̄_T) (자동미분, float64) 와 비교 -> 상대차."""
    S = P.S
    dx = torch.zeros(S.M, 3, device=P.f.device, dtype=torch.float64, requires_grad=True)
    A = torch.eye(3, device=dx.device, dtype=dx.dtype) + dx[S.tets].transpose(1, 2) @ S.gl.double()
    psi, _ = phys_resid.psi_of(A @ P.Fb.double(), sc.cfg, 1.0)
    E = (P.VT.double() * psi).sum()
    g, = torch.autograd.grad(E, dx)
    f_ad = -g
    return float((f_ad - P.f.double()).norm() / f_ad.norm().clamp_min(1e-300))

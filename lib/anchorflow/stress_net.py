"""셀 응력 모델 (사용자 10-27 확정 설계, doc/stress_lego.md).

셀 = 세계 고정 Kuhn 사면체 격자의 사면체 하나 (simplex.py, mom_core.frame 과 같은 격자·같은 사면체 색인 규약).
셀마다 그 셀에 든 입자들의 **집합** (셀 국소 좌표, 변형구배 F_p, 물성 μ_p·λ_p, 부피 V_p) 을 받아 (순서 불변, 입자 수 가변)
셀 안 고정 적분점 K 개의 Kirchhoff 응력 τ_q (대칭 3x3) 를 낸다. 노드 내부력

    f_n = −Σ_T Σ_q V_q τ_q ∇w_n(x_q),     V_q = ω_q V_T,  V_T = Σ_{p∈T} V_p,  Σ_q ω_q = 1

(적분점 위치 λ_q·가중치 ω_q 고정). 정답은 입자 적분 f_n^true = −Σ_p V_p τ_FCR(F_p) ∇w_n(x_p) (기존 고정 FCR = deform_model.kirchhoff_tau).

보간 w_n (--interp):
  bary     사면체 4 꼭짓점 barycentric (∇w 가 사면체 안 상수 -> K = 1 로도 정확할 수 있다)
  bspline2 MPM 과 같은 2 차 B-스플라인 27 노드 (simplex.bspline27 과 같은 식; ∇w 가 셀 안에서 변한다)
보존: 두 보간 모두 Σ_n ∇w_n = 0 (단위분할) -> Σ_n f_n = 0 (선운동량), Σ_n x_n ⊗ ∇w_n = I (선형 재현) -> Σ_n x_n × f_n = −ε:Σ_q V_q τ_q
      = 0 (τ 대칭이면 각운동량). check 모드에서 수치 확인.

회전 공변: 셀 대표 회전 R = polar(F̄_T) (F̄_T = Σ V_p F_p / V_T, det R = +1) 로 입력을 F'_p = Rᵀ F_p 로 정규화하고 출력 τ = R τ' Rᵀ 로 되돌린다.
  F_p -> Q F_p 이면 F̄ -> Q F̄, polar 유일성으로 R -> Q R, F' 그대로 -> τ' 그대로 -> τ -> Q τ Qᵀ (정확히 공변; 위치는 격자 고정이라 그대로).

색인 규약: 셀 국소 노드 = 셀 정수좌표 ci 기준 오프셋 o ∈ {−1,0,1,2}³ (bary 는 {0,1}³ 의 사슬 4 꼭짓점, bspline2 는 floor(y−½)+{0,1,2}),
  국소 키 = (o0+1)·16 + (o1+1)·4 + (o2+1) (64 칸). 전역 노드 = 격자 정수좌표 ci+o (simplex 의 평탄 색인과 같은 정수좌표).
  사면체 꼭짓점 순서·∇λ 는 simplex.locate 의 사슬 순서 (v0 = ci, v_{r+1} = v_r + e_{perm[r]}) 그대로 -- mom_core.frame 의 tets·gl 과 같다.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from . import deform_model as DM
from . import simplex as SX

HZ = SX.HZ_CUBE
NLOC = 64

# ---------------------------------------------------------------- 적분점 (사면체 barycentric λ_q, 가중치 ω_q)
_A4, _B4 = 0.5854101966249685, 0.1381966011250105
QUAD = {
    1: ([[0.25, 0.25, 0.25, 0.25]], [1.0]),
    4: ([[_A4, _B4, _B4, _B4], [_B4, _A4, _B4, _B4], [_B4, _B4, _A4, _B4], [_B4, _B4, _B4, _A4]], [0.25] * 4),   # 2 차 정확
}


def quad_rule(K, dev, dtype=torch.float32):
    if K not in QUAD:
        raise SystemExit(f"[적분점] K={K} 규칙 없음 (있는 것: {sorted(QUAD)})")
    l, w = QUAD[K]
    return torch.tensor(l, device=dev, dtype=dtype), torch.tensor(w, device=dev, dtype=dtype)


# ---------------------------------------------------------------- FCR 응력 (기존 고정 FCR 를 입자별 μ·λ 로)
_CFG_MU = {"E": 2.0, "nu": 0.0, "material": "jelly"}        # μ = 1, λ = 0
_CFG_B = {"E": 1.0, "nu": 0.25, "material": "jelly"}        # μ = 0.4, λ = 0.4


def tau_fcr(F, mu, lam):
    """τ = P(F)Fᵀ (deform_model.kirchhoff_tau 그대로, ψ 가 (μ, λ) 에 선형이라 두 번 불러 입자별 μ·λ 로 합친다). F [...,3,3]."""
    sh = F.shape
    F2 = F.reshape(-1, 3, 3)
    t_mu = DM.kirchhoff_tau(F2.double(), _CFG_MU)
    t_lam = (DM.kirchhoff_tau(F2.double(), _CFG_B) - 0.4 * t_mu) / 0.4
    mu = torch.as_tensor(mu, device=F.device, dtype=torch.float64).reshape(-1, 1, 1)
    lam = torch.as_tensor(lam, device=F.device, dtype=torch.float64).reshape(-1, 1, 1)
    return (mu * t_mu + lam * t_lam).to(F.dtype).reshape(sh)


def _polar_svd(Fd):
    U, S, Vh = torch.linalg.svd(Fd)
    d = torch.linalg.det(U @ Vh)
    U = torch.cat([U[..., :2], U[..., 2:] * d[:, None, None]], -1)
    return U @ Vh


def _inv_T3(X):
    """X^{-T} (3x3 여인수 닫힌 식) 과 det."""
    c0 = torch.cross(X[..., 1, :], X[..., 2, :], dim=-1)
    c1 = torch.cross(X[..., 2, :], X[..., 0, :], dim=-1)
    c2 = torch.cross(X[..., 0, :], X[..., 1, :], dim=-1)
    cof = torch.stack([c0, c1, c2], -2)                      # 여인수 행렬 (행 i = 행 j×k)
    det = (X[..., 0, :] * c0).sum(-1)
    return cof / det[..., None, None], det


def polar_R(F, iters=10, tol=1e-10):
    """극분해 회전 (det R = +1). F [T,3,3] -> R [T,3,3] (float64 로 계산).

    배율 붙인 뉴턴 반복 X ← ½(γX + γ⁻¹X^{-T}), γ = |det X|^{-1/3} (닫힌 식 3x3 역행렬, 일괄 SVD 보다 훨씬 빠르다).
    det F ≤ 0 (또는 거의 특이) 이거나 반복 뒤 ‖XᵀX − I‖ > tol 인 셀만 SVD 로 (det R = +1 이 되게 최소 특이벡터 부호를 뒤집는다).
    """
    Fd = F.double()
    X = Fd
    _, d0 = _inv_T3(Fd)
    for _ in range(iters):
        Xit, det = _inv_T3(X)
        g = det.abs().clamp_min(1e-300).pow(-1.0 / 3.0)[..., None, None]
        X = 0.5 * (g * X + Xit / g)
    I = torch.eye(3, device=F.device, dtype=torch.float64)
    bad = (d0 <= 1e-6 * Fd.flatten(1).norm(dim=1).pow(3).clamp_min(1e-300)) | ((X.transpose(1, 2) @ X - I).flatten(1).abs().amax(1) > tol) \
        | ~torch.isfinite(X).flatten(1).all(1)
    if bool(bad.any()):
        X = X.clone()
        X[bad] = _polar_svd(Fd[bad])
    return X.to(F.dtype)


def sym6(t):
    """[...,6] -> 대칭 [...,3,3] (xx yy zz yz xz xy)."""
    a, b, c, d, e, f = t.unbind(-1)
    return torch.stack([torch.stack([a, f, e], -1), torch.stack([f, b, d], -1), torch.stack([e, d, c], -1)], -2)


# ---------------------------------------------------------------- 격자 기하
_E3 = {}


def _eye(dev):
    if dev not in _E3:
        _E3[dev] = torch.eye(3, device=dev, dtype=torch.long)
    return _E3[dev]


def tet_verts(perm):
    """perm [T,3] (순위 -> 축) -> 사면체 4 꼭짓점의 셀 국소 정수좌표 [T,4,3] (v0 = 0, 사슬 순서 = simplex.locate)."""
    dirs = _eye(perm.device)[perm]
    return torch.cat([torch.zeros_like(dirs[:, :1]), dirs.cumsum(1)], 1)


def bary_grads(lat, perm):
    """∇λ_i (세계 좌표) [T,4,3] -- mom_core.frame 의 gl 과 같은 식 (Di = inv((X_{1..3} − X_0)ᵀ))."""
    A = lat.A.double()
    X = tet_verts(perm).double() @ A.T                               # [T,4,3]
    Di = torch.linalg.inv((X[:, 1:] - X[:, :1]).transpose(1, 2))
    return torch.cat([-Di.sum(1, keepdim=True), Di], 1).to(lat.A.dtype)


def lam_to_y(lamb, perm):
    """barycentric [P,4] + perm [P,3] -> 셀 국소 격자좌표 y − ci [P,3]."""
    return (lamb.unsqueeze(-1) * tet_verts(perm).to(lamb.dtype)).sum(1)


def y_to_lam(f, perm):
    """셀 국소 격자좌표 f [P,3] (사면체 perm 안) -> barycentric [P,4] (simplex.locate 와 같은 식)."""
    sv = f.gather(1, perm)
    return torch.stack([1.0 - sv[:, 0], sv[:, 0] - sv[:, 1], sv[:, 1] - sv[:, 2], sv[:, 2]], -1)


def interp_local(kind, f, perm, lat, gl=None):
    """점 (셀 국소 격자좌표 f [P,3], 그 점이 속한 셀의 perm [P,3]) -> (국소 노드 오프셋 [P,n,3] long, ∇w (세계) [P,n,3], w [P,n]).

    bary: n = 4 (사슬 꼭짓점, gl = bary_grads 의 행 [P,4,3] 를 넘기면 그대로 쓴다)
    bspline2: n = 27, simplex.bspline27 과 같은 식 (base = floor(f − ½), ∇w = (∂w/∂y) Ai).
    """
    if kind == "bary":
        off = tet_verts(perm)
        g = gl if gl is not None else bary_grads(lat, perm)
        return off, g.to(f.dtype), y_to_lam(f, perm)
    if kind == "bspline2":
        b = torch.floor(f - 0.5)
        u = f - b
        wq = torch.stack([0.5 * (1.5 - u) ** 2, 0.75 - (u - 1.0) ** 2, 0.5 * (u - 0.5) ** 2], 1)
        dq = torch.stack([u - 1.5, -2.0 * (u - 1.0), u - 0.5], 1)
        o = SX._o27(f.device)
        w0, w1, w2 = wq[:, o[:, 0], 0], wq[:, o[:, 1], 1], wq[:, o[:, 2], 2]
        d0, d1, d2 = dq[:, o[:, 0], 0], dq[:, o[:, 1], 1], dq[:, o[:, 2], 2]
        w = w0 * w1 * w2
        dwdy = torch.stack([d0 * w1 * w2, w0 * d1 * w2, w0 * w1 * d2], -1)
        off = b.long().unsqueeze(1) + o.unsqueeze(0)
        return off, dwdy @ lat.Ai.to(f.dtype), w
    raise SystemExit(f"[보간] 모르는 --interp {kind}")


def loc_key(off):
    return (off[..., 0] + 1) * 16 + (off[..., 1] + 1) * 4 + (off[..., 2] + 1)


def cell_forces(kind, seg, f, perm_pt, VT_or_Vp, tau, T, lat, gl_pt=None):
    """국소 노드 힘 [T,64,3]: f_{T,n} = −Σ_{점∈T} V τ ∇w_n. 점 = 입자 (정답) 또는 적분점 (모델).
    seg [P] 셀 번호, f [P,3] 셀 국소 격자좌표, perm_pt [P,3], VT_or_Vp [P] 점 부피 (적분점이면 ω_q V_T), tau [P,3,3]."""
    off, g, _w = interp_local(kind, f, perm_pt, lat, gl_pt)
    key = loc_key(off) + seg.unsqueeze(1) * NLOC                      # [P,n]
    fv = -VT_or_Vp.reshape(-1, 1, 1) * (g @ tau.transpose(1, 2))     # [P,n,3] 행 = (τ ∇w)ᵀ
    out = torch.zeros(T * NLOC, 3, device=f.device, dtype=fv.dtype)
    out.index_add_(0, key.reshape(-1), fv.reshape(-1, 3))
    return out.reshape(T, NLOC, 3)


_OFFS = None


def local_offsets(dev):
    o = torch.arange(NLOC, device=dev)
    return torch.stack([o // 16 - 1, (o // 4) % 4 - 1, o % 4 - 1], -1)          # [64,3]


def global_assemble(fl, ci):
    """국소 힘 [T,64,3] + 셀 정수좌표 ci [T,3] -> (전역 노드 정수좌표 [M,3], 힘 [M,3]) (쓰인 노드만)."""
    gi = ci.unsqueeze(1) + local_offsets(ci.device).unsqueeze(0)               # [T,64,3]
    used = fl.abs().sum(-1) > 0
    gi, fv = gi[used], fl[used]
    B = 1 << 20
    key = ((gi[:, 0] + 512) * B + (gi[:, 1] + 512)) * B + (gi[:, 2] + 512)
    uk, inv = torch.unique(key, return_inverse=True)
    F = torch.zeros(uk.numel(), fl.shape[-1], device=fl.device, dtype=fl.dtype).index_add_(0, inv, fv)
    ijk = torch.zeros(uk.numel(), 3, device=fl.device, dtype=torch.long)
    ijk[inv] = gi
    return ijk, F


# ---------------------------------------------------------------- 셀 묶음 (입자 평탄 배열 + 셀 번호)
def seg_sum(x, seg, T):
    return torch.zeros((T,) + tuple(x.shape[1:]), device=x.device, dtype=x.dtype).index_add_(0, seg, x)


def batch_from_particles(x, F, Vp, mu, lam, node_h, grid=None):
    """실제 상태 (입자 위치 x [N,3], F [N,3,3], V_p, μ_p, λ_p) -> 셀 묶음.
    격자 = mom_core.frame 과 같은 규칙 (grid_for_nodes(x, 32, hz=1/√6, h_fix=node_h)), 셀 번호 = unique(tet_id) 순서 (= mom_core 의 pinv)."""
    lo, lat, nn = grid if grid is not None else SX.grid_for_nodes(x, 32, hz_ratio=HZ, h_fix=node_h)
    idx, lamb, aux = SX.locate(x, lo, lat, nn)
    ci_p, rank, perm_p = aux
    tid = SX.tet_id(lo, lat, nn, aux)
    _ut, seg = torch.unique(tid, return_inverse=True)
    T = int(_ut.numel())
    rep = torch.empty(T, dtype=torch.long, device=x.device).scatter_(0, seg, torch.arange(x.shape[0], device=x.device))
    y = (x - lo) @ lat.Ai.T
    f = (y - ci_p).clamp(0.0, 1.0)
    N = x.shape[0]
    return dict(seg=seg, T=T, ci=ci_p[rep], perm=perm_p[rep], f=f, F=F, Vp=Vp.reshape(N), mu=mu.expand(N) if mu.numel() == 1 else mu,
                lam=lam.expand(N) if lam.numel() == 1 else lam, lat=lat, lo=lo, nn=nn, idx=idx, rep=rep)


def truth_forces(kind, b, lat):
    """정답 국소 힘 (입자 적분, 기존 고정 FCR) [T,64,3], 입자 τ."""
    tau = tau_fcr(b["F"], b["mu"], b["lam"])
    gl = bary_grads(lat, b["perm"])[b["seg"]] if kind == "bary" else None
    return cell_forces(kind, b["seg"], b["f"], b["perm"][b["seg"]], b["Vp"], tau, b["T"], lat, gl), tau


def cell_means(b):
    seg, T = b["seg"], b["T"]
    VT = seg_sum(b["Vp"], seg, T)
    Fb = seg_sum(b["Vp"][:, None, None] * b["F"], seg, T) / VT.clamp_min(1e-30)[:, None, None]
    muc = seg_sum(b["Vp"] * b["mu"], seg, T) / VT.clamp_min(1e-30)
    lamc = seg_sum(b["Vp"] * b["lam"], seg, T) / VT.clamp_min(1e-30)
    return VT, Fb, muc, lamc


def quad_forces(kind, b, lat, tau_q, VT, K):
    """적분점 응력 τ_q [T,K,3,3] -> 국소 힘 [T,64,3]."""
    T = b["T"]
    lq, wq = quad_rule(K, tau_q.device, tau_q.dtype)
    perm = b["perm"]
    fq = lam_to_y(lq.unsqueeze(0).expand(T, -1, -1).reshape(-1, 4), perm.repeat_interleave(K, 0))   # [T*K,3]
    seg = torch.arange(T, device=tau_q.device).repeat_interleave(K)
    gl = bary_grads(lat, perm).repeat_interleave(K, 0) if kind == "bary" else None
    Vq = (VT.unsqueeze(1) * wq.unsqueeze(0)).reshape(-1)
    return cell_forces(kind, seg, fq, perm.repeat_interleave(K, 0), Vq, tau_q.reshape(-1, 3, 3), T, lat, gl)


def baseline_tau(b, K):
    """기준: 셀 평균 F 하나로 낸 FCR 응력 (모든 적분점에 같은 값) [T,K,3,3]."""
    VT, Fb, muc, lamc = cell_means(b)
    t = tau_fcr(Fb, muc, lamc)
    return t.unsqueeze(1).expand(-1, K, -1, -1).contiguous(), VT


# ---------------------------------------------------------------- 모델
def mlp(i, h, o, n=2, act=nn.SiLU):
    L = [nn.Linear(i, h), act()]
    for _ in range(n - 2):
        L += [nn.Linear(h, h), act()]
    L += [nn.Linear(h, o)]
    return nn.Sequential(*L)


class StressNet(nn.Module):
    """집합 인코더 (입자 MLP ⇄ 셀 문맥 n_round 회, V_p 가중 합 풀링) + 적분점별 주의 풀링 -> τ'_q = τ_FCR(F̄') + μ_c S(Δ_q), τ_q = R τ'_q Rᵀ.

    입자 입력 (회전 정규화 뒤, 고정 상수 변환만): λ 4, f − ½ 3, F'_p − I 9, (F'_p − F̄') 9, log(V_p/V_tet) 1, log(μ_p/μ_ref) 1, ν_p 1 = 28
    셀 입력: 사면체 perm one-hot 6, F̄' − I 9, log(V_T/V_tet) 1, log(1+n)/log(64) 1, log(μ_c/μ_ref) 1, ν_c 1 = 19
    """

    N_PIN, N_CIN = 28, 19

    def __init__(self, K, kind, lat, mu_ref, H=128, n_round=2, out_scale=0.1):
        super().__init__()
        self.K, self.kind, self.H, self.n_round = int(K), kind, int(H), int(n_round)
        self.register_buffer("A", lat.A.clone().float())
        self.v_tet = float(torch.linalg.det(lat.A.double()).abs()) / 6.0
        self.mu_ref = float(mu_ref)
        self.out_scale = float(out_scale)
        self.pin = mlp(self.N_PIN, H, H, 3)
        self.cin = nn.ModuleList([mlp(2 * H + self.N_CIN, H, H, 3) for _ in range(n_round)])
        self.pupd = nn.ModuleList([mlp(2 * H, H, H, 2) for _ in range(n_round - 1)])
        self.att = nn.Linear(H, self.K)
        self.q_emb = nn.Parameter(torch.randn(self.K, H) * 0.1)
        self.out = mlp(3 * H, H, 6, 3)
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, b, return_parts=False):
        seg, T = b["seg"], b["T"]
        dev = b["F"].device
        Vp = b["Vp"]
        VT, Fb, muc, lamc = cell_means(b)
        R = polar_R(Fb)                                               # [T,3,3]
        Rt = R.transpose(1, 2)
        Fp = Rt[seg] @ b["F"]                                         # F'_p
        Fbn = Rt @ Fb                                                 # F̄' (대칭)
        I3 = torch.eye(3, device=dev, dtype=Fp.dtype)
        perm_p = b["perm"][seg]
        lamb = y_to_lam(b["f"], perm_p)
        nu_p = b["lam"] / (2.0 * (b["lam"] + b["mu"]))
        nu_c = lamc / (2.0 * (lamc + muc))
        pin = torch.cat([lamb, b["f"] - 0.5, (Fp - I3).reshape(-1, 9), (Fp - Fbn[seg]).reshape(-1, 9),
                         torch.log(Vp / self.v_tet).unsqueeze(1), torch.log(b["mu"] / self.mu_ref).unsqueeze(1), nu_p.unsqueeze(1)], 1)
        dt = Vp.dtype
        cnt = torch.zeros(T, device=dev, dtype=dt).index_add_(0, seg, torch.ones_like(Vp))
        pid = b["perm"][:, 0] * 3 + b["perm"][:, 1]                   # 0..8 중 6 개 값
        oh = torch.zeros(T, 9, device=dev, dtype=dt).scatter_(1, pid.unsqueeze(1), 1.0)[:, [1, 2, 3, 5, 6, 7]]
        cfe = torch.cat([oh, (Fbn - I3).reshape(-1, 9), torch.log(VT / self.v_tet).unsqueeze(1),
                         (torch.log1p(cnt) / math.log(64.0)).unsqueeze(1), torch.log(muc / self.mu_ref).unsqueeze(1), nu_c.unsqueeze(1)], 1)
        wv = (Vp / VT.clamp_min(1e-30)[seg]).unsqueeze(1)            # 셀 안 부피 비율 (합 1)
        h = self.pin(pin)
        for r in range(self.n_round):
            pm = seg_sum(wv * h, seg, T)                              # 부피 가중 평균
            ps = seg_sum((Vp / self.v_tet).unsqueeze(1) * h, seg, T)  # 부피 가중 합 (V_tet 단위)
            c = self.cin[r](torch.cat([pm, ps, cfe], 1))
            if r < self.n_round - 1:
                h = h + self.pupd[r](torch.cat([h, c[seg]], 1))
        lg = self.att(h)                                              # [N,K]
        mx = torch.full((T, self.K), -1e30, device=dev, dtype=dt).scatter_reduce(0, seg.unsqueeze(1).expand(-1, self.K), lg, "amax")
        e = wv * torch.exp(lg - mx[seg])                             # [N,K]
        den = seg_sum(e, seg, T).clamp_min(1e-30)
        pq = torch.zeros(T, self.K, self.H, device=dev, dtype=dt).index_add_(0, seg, e.unsqueeze(-1) * h.unsqueeze(1)) / den.unsqueeze(-1)
        o = self.out(torch.cat([pq, c.unsqueeze(1).expand(-1, self.K, -1), self.q_emb.unsqueeze(0).expand(T, -1, -1)], -1))   # [T,K,6]
        base = tau_fcr(Fbn, muc, lamc)                                # [T,3,3]
        tq_n = base.unsqueeze(1) + (self.out_scale * muc).reshape(T, 1, 1, 1) * sym6(o)
        tq = R.unsqueeze(1) @ tq_n @ Rt.unsqueeze(1)
        if return_parts:
            return tq, dict(VT=VT, R=R, base=base, tq_n=tq_n)
        return tq, VT


# ---------------------------------------------------------------- 자기지도 셀 생성 (실제 궤적 무관)
GEN_DEFAULTS = dict(
    n_max=640, p_logn=0.85,
    p_mode=[0.3, 0.25, 0.25, 0.2],               # 균등 / 몇 점으로 수축 / 부분 사면체 / 정규 부분격자
    shrink_min=0.02, sharp_max=4.0, sub_s_min=0.12, sub_s_max=0.9,
    v_lo=4e-9, v_hi=6e-5, p_vcell=0.5,         # V_p 모양 (상대 크기); 실제 lego log10 V_p [−8.12, −5.10] (test_1a9bcd9 cover)
    occ_lo=0.3, occ_hi=30.0,                     # 셀 점유 상한 V_T/V_tet ≤ logU[occ_lo, occ_hi] (실제 lego [0.0057, 22], 99.9 % 12) -- 넘는 셀만 V_p 를 비례로 줄인다
    S_max=0.7, g_max=0.6, sig_max=0.4, p_sig0=0.15, p_sameF=0.15,   # p_sameF: 셀 안 F 가 모두 같은 셀 (실제 롤아웃에 흔하다: 같이 머문 입자는 같은 A_T 를 받는다)
    p_out=0.5, out_frac=0.3, out_ang_min=1e-2, out_ang_max=math.pi, out_str_max=0.7,
    det_min=0.05, p_inv=0.1, p_lego=0.4, E_fac=100.0, nu_max=0.45, p_mix=0.1,
)


def _logu(gen, lo, hi, n, dev):
    u = torch.rand(n, generator=gen, device=dev)
    return torch.exp(math.log(lo) + u * (math.log(hi) - math.log(lo)))


def _unit(gen, n, dev):
    q = torch.randn(n, 3, generator=gen, device=dev)
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def _rodrigues(axis, ang):
    K = torch.zeros(*axis.shape[:-1], 3, 3, device=axis.device, dtype=axis.dtype)
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 2] = -axis[..., 2], axis[..., 1], -axis[..., 0]
    K = K - K.transpose(-1, -2)
    s, c = torch.sin(ang)[..., None, None], torch.cos(ang)[..., None, None]
    I = torch.eye(3, device=axis.device, dtype=axis.dtype).expand_as(K)
    return I + s * K + (1 - c) * (K @ K)


def _rand_rot(gen, n, dev):
    q = torch.randn(n, 4, generator=gen, device=dev)
    q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    return torch.stack([torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
                        torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                        torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)


def _dirichlet1(gen, n, dev):
    e = -torch.log(torch.rand(n, 4, generator=gen, device=dev).clamp_min(1e-12))
    return e / e.sum(-1, keepdim=True)


_PERMS = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]


class CellGen:
    """무작위 셀 B 개 -> 셀 묶음 (batch_from_particles 와 같은 꼴, ci = 0).

    위치 (셀 모드별): 균등 (Dirichlet(1)) / 몇 점으로 수축 (중심 1~4 개, 꼭짓점 쪽 치우침, 중심 + t(균등 − 중심), t ~ logU[shrink_min, 1]) /
      부분 사면체 (꼭짓점 4 개를 셀 안 균등으로 뽑은 작은 사면체 안 균등 -- 표면 셀처럼 셀 일부만 참) /
      정규 부분격자 (세계 축 정렬 간격 s·node_h, s ~ logU[sub_s_min, sub_s_max], 무작위 원점·약간 흔들림 -- PG 채우기 입자처럼).
    입자 수: 정규 부분격자 외에는 확률 p_logn 로 logU[1, n_max], 나머지 U{1..n_max}.
    V_p: 확률 p_vcell 로 셀 안 같은 값, 나머지 입자별 logU[v_lo, v_hi]; 셀 합이 점유 상한 logU[occ_lo, occ_hi] × V_tet 을 넘으면 비례로 줄인다.
    F: F_p = R_out,p · R_c U_c (I + G·(y_p − ȳ)) (I + σ N_p) (I + ε_out,p),
       R_c ∈ SO(3) 균등, U_c = Q diag(e^ε) Qᵀ, |ε_i| ≤ S ~ logU[1e-4, S_max]; G [3,3,3] 크기 g ~ logU[1e-4, g_max] (격자 단위당);
       σ ~ logU[1e-4, sig_max] (확률 p_sig0 로 0); 확률 p_out 로 입자 비율 U[0, out_frac] 가 추가 회전 (각 logU[out_ang_min, out_ang_max])
       + 추가 늘임 (|ε| ≤ logU[1e-3, out_str_max]), 그중 p_inv 는 한 축 반사 (뒤집힌 입자). 나머지 입자 det F_p < det_min 이면 그 셀 변형을 절반으로 줄여 다시 (최대 8 번).
    확률 p_sameF 로 셀 안 모든 입자 F 가 같다 (G = σ = 이상 입자 = 0).
    물성: 확률 p_lego 로 씬 값 (E, ν), 나머지 E ~ logU[E/E_fac, E·E_fac], ν ~ U[0, nu_max]; 확률 p_mix 로 셀 안 두 물성 섞기.
    """

    def __init__(self, lat, E_ref, nu_ref, dev="cuda", **kw):
        self.lat, self.dev = lat, dev
        self.o = dict(GEN_DEFAULTS, **kw)
        self.E_ref, self.nu_ref = float(E_ref), float(nu_ref)
        self.perms = torch.tensor(_PERMS, device=dev, dtype=torch.long)

    @staticmethod
    def lame(E, nu):
        return E / (2.0 * (1.0 + nu)), E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    def sample(self, B, gen):
        o, dev, lat = self.o, self.dev, self.lat
        perm = self.perms[torch.randint(0, 6, (B,), generator=gen, device=dev)]
        mode = torch.multinomial(torch.tensor(o["p_mode"], device=dev), B, replacement=True, generator=gen)
        nmax = int(o["n_max"])
        n_log = torch.round(_logu(gen, 1.0, nmax + 0.49, B, dev)).long()
        n_uni = torch.randint(1, nmax + 1, (B,), generator=gen, device=dev)
        n = torch.where(torch.rand(B, generator=gen, device=dev) < o["p_logn"], n_log, n_uni).clamp(1, nmax)
        # 정규 부분격자 후보 (셀마다 최대 nmax 개를 채택)
        sub = mode == 3
        lam_list = []
        if bool(sub.any()):
            nb = int(sub.sum())
            s = _logu(gen, o["sub_s_min"], o["sub_s_max"], nb, dev) * lat.s       # 세계 간격
            # 셀 국소 좌표 [0,1]³ 를 덮는 세계 상자: 정수 격자 좌표 범위 -> 세계 꼭짓점
            cor = torch.tensor([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], device=dev, dtype=torch.float32) @ lat.A.T
            lo_, hi_ = cor.min(0).values, cor.max(0).values
            m = int(math.ceil(float((hi_ - lo_).max()) / float(s.min()))) + 2
            m = min(m, 24)
            g1 = torch.arange(m, device=dev, dtype=torch.float32)
            gg = torch.stack(torch.meshgrid(g1, g1, g1, indexing="ij"), -1).reshape(-1, 3)        # [m³,3]
            org = lo_ - s.unsqueeze(1) * torch.rand(nb, 3, generator=gen, device=dev)
            jit = 0.05 * s.reshape(-1, 1, 1) * torch.randn(nb, gg.shape[0], 3, generator=gen, device=dev)
            xw = org.unsqueeze(1) + s.reshape(-1, 1, 1) * gg.unsqueeze(0) + jit                    # [nb,m³,3]
            fy = xw @ lat.Ai.T
            ps = perm[sub]
            sv = fy.gather(2, ps.unsqueeze(1).expand(-1, fy.shape[1], -1))
            inside = (sv[..., 0] <= 1) & (sv[..., 2] >= 0) & (sv[..., 0] >= sv[..., 1]) & (sv[..., 1] >= sv[..., 2])
            # 셀마다 무작위 순서로 앞 nmax 개 (없으면 중심 한 점)
            key = torch.rand(inside.shape, generator=gen, device=dev) + (~inside).float() * 2
            order = key.argsort(1)
            cnt = inside.sum(1).clamp(max=nmax)
            n_sub = cnt.clamp_min(1)
            n = n.clone()
            n[sub] = n_sub
            nk = min(nmax, int(order.shape[1]))
            sel = order[:, :nk]
            fsel = fy.gather(1, sel.unsqueeze(-1).expand(-1, -1, 3))                                # [nb,nk,3]
            okm = torch.arange(nk, device=dev).unsqueeze(0) < cnt.unsqueeze(1)
            cen = lam_to_y(torch.full((nb, 4), 0.25, device=dev), ps)
            fsel = torch.where(okm.unsqueeze(-1), fsel, cen.unsqueeze(1))
            sub_f = fsel                                                                           # 셀마다 n_sub 개를 앞에서
        N = int(n.sum())
        seg = torch.repeat_interleave(torch.arange(B, device=dev), n)
        perm_p = perm[seg]
        lam_u = _dirichlet1(gen, N, dev)
        md = mode[seg]
        # 몇 점으로 수축: 셀마다 중심 k ∈ {1..4} 개 (꼭짓점 쪽으로 치우친 무작위 점: Dirichlet(1) 표본을 지수 s ~ U[1, sharp_max] 로 날카롭게),
        # 입자는 중심 하나를 골라 t ~ logU[shrink_min, 1] 만큼 균등점 쪽으로 (꼭짓점 하나에 몰린 셀·마주 보는 꼭짓점에 갈린 셀 포함)
        ek = -torch.log(torch.rand(B, 4, 4, generator=gen, device=dev).clamp_min(1e-12))
        shp = 1.0 + (o["sharp_max"] - 1.0) * torch.rand(B, 1, 1, generator=gen, device=dev)
        ek = ek ** shp
        cen4 = ek / ek.sum(-1, keepdim=True)                                                      # [B,4(중심),4(λ)]
        kc = torch.randint(1, 5, (B,), generator=gen, device=dev)
        pick = (torch.rand(N, generator=gen, device=dev) * kc[seg].float()).long().clamp(max=3)
        cen_p = cen4[seg, pick]
        t = _logu(gen, o["shrink_min"], 1.0, B, dev)
        lam_s = cen_p + t[seg].unsqueeze(1) * (lam_u - cen_p)
        # 부분 사면체: 꼭짓점 4 개 (셀 안 균등) 의 볼록 결합
        Vs = torch.stack([_dirichlet1(gen, B, dev) for _ in range(4)], 1)                         # [B,4(꼭짓점),4(λ)]
        lam_b = (lam_u.unsqueeze(-1) * Vs[seg]).sum(1)
        lamb = torch.where((md == 1).unsqueeze(1), lam_s, torch.where((md == 2).unsqueeze(1), lam_b, lam_u))
        f = lam_to_y(lamb, perm_p)
        if bool(sub.any()):
            pos = torch.arange(N, device=dev) - torch.repeat_interleave(torch.cumsum(n, 0) - n, n)
            sidx = torch.full((B,), -1, device=dev, dtype=torch.long)
            sidx[sub] = torch.arange(int(sub.sum()), device=dev)
            ms = sub[seg]
            f = f.clone()
            f[ms] = sub_f[sidx[seg[ms]], pos[ms]]
            f = f.clamp(0.0, 1.0)
        # 부피
        vc = _logu(gen, o["v_lo"], o["v_hi"], B, dev)
        vp = _logu(gen, o["v_lo"], o["v_hi"], N, dev)
        Vp = torch.where((torch.rand(B, generator=gen, device=dev) < o["p_vcell"])[seg], vc[seg], vp)
        cap = _logu(gen, o["occ_lo"], o["occ_hi"], B, dev) * float(torch.linalg.det(lat.A.double()).abs() / 6.0)
        Vp = Vp * (cap / seg_sum(Vp, seg, B)).clamp(max=1.0)[seg]                                  # 셀 점유 상한 V_T ≤ cap (넘으면 비례로 줄임)
        # 물성
        def mat(k):
            E = _logu(gen, self.E_ref / o["E_fac"], self.E_ref * o["E_fac"], k, dev)
            nu = torch.rand(k, generator=gen, device=dev) * o["nu_max"]
            lg = torch.rand(k, generator=gen, device=dev) < o["p_lego"]
            return torch.where(lg, torch.full_like(E, self.E_ref), E), torch.where(lg, torch.full_like(nu, self.nu_ref), nu)
        Ea, nua = mat(B)
        Eb, nub = mat(B)
        mix = (torch.rand(B, generator=gen, device=dev) < o["p_mix"])[seg] & (torch.rand(N, generator=gen, device=dev) < 0.5)
        E = torch.where(mix, Eb[seg], Ea[seg])
        nu = torch.where(mix, nub[seg], nua[seg])
        mu, lam = self.lame(E, nu)
        # F
        Rc = _rand_rot(gen, B, dev)
        Q = _rand_rot(gen, B, dev)
        S = _logu(gen, 1e-4, o["S_max"], B, dev)
        eps = (2 * torch.rand(B, 3, generator=gen, device=dev) - 1)
        gm = _logu(gen, 1e-4, o["g_max"], B, dev)
        G = torch.randn(B, 3, 3, 3, generator=gen, device=dev) / math.sqrt(27.0)
        sig = _logu(gen, 1e-4, o["sig_max"], B, dev) * (torch.rand(B, generator=gen, device=dev) >= o["p_sig0"]).float()
        same = torch.rand(B, generator=gen, device=dev) < o["p_sameF"]
        gm = gm * (~same).float()
        sig = sig * (~same).float()
        Np = torch.randn(N, 3, 3, generator=gen, device=dev)
        fo = torch.rand(B, generator=gen, device=dev) * o["out_frac"] * (torch.rand(B, generator=gen, device=dev) < o["p_out"]).float()
        fo = fo * (~same).float()
        isout = torch.rand(N, generator=gen, device=dev) < fo[seg]
        oang = _logu(gen, o["out_ang_min"], o["out_ang_max"], N, dev) * isout.float()
        Ro = _rodrigues(_unit(gen, N, dev), oang)
        ostr = _logu(gen, 1e-3, o["out_str_max"], N, dev) * isout.float()
        oeps = (2 * torch.rand(N, 3, generator=gen, device=dev) - 1) * ostr.unsqueeze(1)
        Qo = _rand_rot(gen, N, dev)
        # 뒤집힌 입자 (실제 lego 롤아웃에 det F_p 최소 −1.0 이 있다): 이상 입자 중 p_inv 를 한 축 반사 + 압축 s ~ U[0.05, 1]
        isinv = isout & (torch.rand(N, generator=gen, device=dev) < o["p_inv"])
        rs = 0.05 + 0.95 * torch.rand(N, generator=gen, device=dev)
        Dv = torch.ones(N, 3, device=dev)
        Dv[:, 2] = torch.where(isinv, -rs, torch.ones_like(rs))
        Qi = _rand_rot(gen, N, dev)
        Finv = Qi @ torch.diag_embed(Dv) @ Qi.transpose(1, 2)
        cen = seg_sum(f, seg, B) / n.unsqueeze(1).float()
        dy = f - cen[seg]
        I3 = torch.eye(3, device=dev)
        sc = torch.ones(B, device=dev)
        for _ in range(8):
            Uc = Q @ torch.diag_embed(torch.exp(eps * (S * sc).unsqueeze(1))) @ Q.transpose(1, 2)
            Gp = torch.einsum("pijk,pk->pij", G[seg], dy) * (gm * sc)[seg].reshape(-1, 1, 1)
            Eo = Qo @ torch.diag_embed(torch.exp(oeps * sc[seg].unsqueeze(1))) @ Qo.transpose(1, 2)
            Fp = (Ro @ (Rc[seg] @ Uc[seg] @ (I3 + Gp))) @ (I3 + (sig * sc)[seg].reshape(-1, 1, 1) * Np) @ Eo
            dt_ = torch.linalg.det(Fp).masked_fill(isinv, 1e30)                                   # 일부러 뒤집은 입자는 판정에서 뺀다
            dmin = torch.full((B,), 1e30, device=dev).scatter_reduce(0, seg, dt_, "amin")
            bad = dmin < o["det_min"]
            if not bool(bad.any()):
                break
            sc = torch.where(bad, sc * 0.5, sc)
        Fp = Fp @ Finv
        info = dict(mode=mode, n=n, S=S * sc, g=gm * sc, sig=sig * sc, out=fo)
        return dict(seg=seg, T=B, ci=torch.zeros(B, 3, device=dev, dtype=torch.long), perm=perm, f=f, F=Fp, Vp=Vp, mu=mu, lam=lam), info


# ---------------------------------------------------------------- 분포 통계 (실제 vs 생성 비교용)
@torch.no_grad()
def cell_stats(b, lat, max_keep=200000, seed=0):
    """셀·입자 통계 -> {이름: 1 차원 텐서 (cpu)}. 위치는 격자 단위, F 는 셀 회전 정규화 뒤."""
    seg, T = b["seg"], b["T"]
    VT, Fb, muc, lamc = cell_means(b)
    R = polar_R(Fb)
    Rt = R.transpose(1, 2)
    Fp = Rt[seg] @ b["F"]
    Fbn = Rt @ Fb
    cnt = torch.zeros(T, device=Fb.device).index_add_(0, seg, torch.ones_like(b["Vp"]))
    cen = seg_sum(b["f"], seg, T) / cnt.unsqueeze(1)
    lamb = y_to_lam(b["f"], b["perm"][seg])
    lc = seg_sum(lamb, seg, T) / cnt.unsqueeze(1)
    rad = (seg_sum(((b["f"] - cen[seg]) @ lat.A.T.float() / lat.s).pow(2).sum(1), seg, T) / cnt).sqrt()
    dev_p = (Fp - Fbn[seg]).flatten(1).norm(dim=1)
    dmax = torch.zeros(T, device=Fb.device).scatter_reduce(0, seg, dev_p, "amax")
    sF = (seg_sum(b["Vp"] * dev_p ** 2, seg, T) / VT).sqrt()
    ls = torch.log(torch.linalg.svdvals(Fbn.double())).float()
    Rp = polar_R(Fp)
    ang = torch.arccos(((Rp.diagonal(dim1=1, dim2=2).sum(1) - 1) / 2).clamp(-1, 1))
    v_tet = float(torch.linalg.det(lat.A.double()).abs()) / 6.0
    out = {"셀 입자 수": cnt, "셀 중심 |λ̄−¼|": (lc - 0.25).norm(dim=1), "셀 퍼짐 반경/node_h": rad,
           "V_T/V_tet": VT / v_tet, "log10 V_p": torch.log10(b["Vp"]),
           "F̄' log 특이값 최소": ls[:, -1], "F̄' log 특이값 최대": ls[:, 0], "det F̄": torch.linalg.det(Fb),
           "입자 |F'_p − F̄'|": dev_p, "셀 최대 |F'_p − F̄'|": dmax, "셀 σ_F (부피가중)": sF,
           "입자 회전각 (셀 회전 뺀 뒤, 도)": ang * 180 / math.pi, "입자 det F_p": torch.linalg.det(b["F"])}
    g = torch.Generator(device="cpu").manual_seed(seed)
    for k, t in list(out.items()):
        t = t.reshape(-1).float().cpu()
        if t.numel() > max_keep:
            t = t[torch.randperm(t.numel(), generator=g)[:max_keep]]
        out[k] = t
    return out


# ---------------------------------------------------------------- mom_core (운동량 잔차 변위 망) 에 끼우는 자리
class MomStressAdapter:
    """mom_core.prepare 의 `tau = stress(Fb); f = node_force(S, VT, tau)` 자리에 끼우는 꼴 (bary 보간 전용; mom_core 격자가 barycentric).

        tau_eff = adapter.tau_T(S, x, F, Vp, mu, lam)      # [S.T,3,3], S = mom_core.frame(x, node_h)
        f = mom_core.node_force(S, VT, tau_eff)             # VT = Σ_{p∈T} V_p (mom_core.p2g 의 VT 그대로)

    bary 에서 ∇λ 가 사면체 안 상수라 −Σ_q ω_q V_T τ_q ∇λ = −V_T (Σ_q ω_q τ_q) ∇λ -> 사면체당 유효 응력 하나로 줄어든다.
    셀 번호는 S.pinv (unique(tet_id) 순서), 사면체 꼭짓점·∇λ 는 S.tets·S.gl 과 같은 사슬 순서 (check 모드에서 대조).
    """

    def __init__(self, net):
        assert net.kind == "bary", "mom_core 격자는 barycentric -- bspline2 모델은 전역 조립 (quad_forces + global_assemble) 을 쓴다"
        self.net = net

    @torch.no_grad()
    def tau_T(self, S, x, F, Vp, mu, lam):
        _idx, _lam, (ci, rank, perm) = SX.locate(x, S.lo, S.lat, S.nn)
        y = (x - S.lo) @ S.lat.Ai.T
        f = (y - ci).clamp(0.0, 1.0)
        rep = torch.empty(S.T, dtype=torch.long, device=x.device).scatter_(0, S.pinv, torch.arange(x.shape[0], device=x.device))
        N = x.shape[0]
        mu = torch.as_tensor(mu, device=x.device, dtype=x.dtype).expand(N)
        lam = torch.as_tensor(lam, device=x.device, dtype=x.dtype).expand(N)
        b = dict(seg=S.pinv, T=S.T, ci=ci[rep], perm=perm[rep], f=f, F=F, Vp=Vp, mu=mu, lam=lam)
        tq, _VT = self.net(b)
        _l, wq = quad_rule(self.net.K, x.device, tq.dtype)
        return (wq.reshape(1, -1, 1, 1) * tq).sum(1)

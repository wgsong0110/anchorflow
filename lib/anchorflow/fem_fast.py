"""FEM pcg 실시간 판 (사용자 10-27): train_deform `_fwd` 의 증분 포텐셜을 **같은 식 그대로** 입자 단위 작은 함수로 다시 쓴다.

프레임 안에서 고정인 것 (스텝 시작 x, v, F^n, m, V, 입자 -> 사면체 rows·λ, 노드 위치) 을 한 번 잡아 두면 노드 변위 U 의 에너지는

    E(U) = Σ_p [ w_p m_p/(2h²) |q_p(U) - x_p - h v_p - h² g|²  +  V_p ψ(J_p(U) F_p) ]          (AF_NO_BC: 접촉항 0)
    u_p = Σ_i λ_pi U_{r_pi},   G_p = Σ_i U_{r_pi} ⊗ ∇λ_i(T_p),   (q_p, J_p) = bc_maps(x_p + u_p, I + G_p)   (--bc_stick -> --bc_softplus)

이다 (pts_ip_energy + deform_model.bc_maps 와 같은 함수·같은 분기). 망·격자 탐색·특징 계산이 없어 평가가 싸다.
헤시안 (hess_mode 의 _FcrPsiH 그대로):
  * mode "particle": 입자 12x12 를 국소 탐침 12 회 역전파 (기존 _newton_target_asm 과 같은 방식, 작은 그래프)
  * mode "tet":      바닥 매핑이 항등인 입자 (두 매핑 모두 s/ε > 20 -- torch softplus threshold 와 같은 경계라 **정확히** 항등)는
                     관성 = w m/h² (λλᵀ⊗I), 탄성 = Bᵀ_T (Σ_{p∈T} V_p K_p) B_T, K_p = ∂²ψ(A F_p)/∂A² (9x9, 탐침 9 회) 로 사면체 단위 축약.
                     매핑에 걸린 입자(바닥 후보)만 입자 12x12 (탐침 12 회, 그 부분집합만). 근사 없음 (같은 식의 재배열).
"""
import torch
from . import phys_resid

_I3 = {}


def _eye(dev, dt):
    k = (str(dev), dt)
    if k not in _I3:
        _I3[k] = torch.eye(3, device=dev, dtype=dt)
    return _I3[k]


class FemFrame:
    def __init__(self, x, v, F, m, vol, rw, lm, npos, fixd, cfg, h, eps_sp=0.0, eps_st=0.0, stick_xn=False, dtype=None):
        dt = dtype or x.dtype
        self.dt, self.dev = dt, x.device
        self.x, self.v, self.F = x.to(dt), v.to(dt), F.to(dt)
        self.m, self.rw, self.lm = m.to(dt), rw, lm.to(dt)
        N = x.shape[0]
        self.vol = (vol.to(dt) if torch.is_tensor(vol) else torch.full((N,), float(vol), device=x.device, dtype=dt))
        self.h = float(h)
        g = torch.as_tensor(cfg["g"], device=x.device, dtype=dt)
        self.c = self.h * self.v + (self.h * self.h) * g                       # Δu 목표 (중력 포함, pts_ip_energy 와 같다)
        self.w = (~fixd[rw].all(1)).to(dt) if fixd is not None else torch.ones(N, device=x.device, dtype=dt)
        self.mu, self.lam = phys_resid.lame(cfg["E"], cfg["nu"])
        self.cfg = cfg
        self.eps_sp, self.eps_st, self.stick_xn = float(eps_sp), float(eps_st), bool(stick_xn)
        self.bcs = []
        for bc in (cfg.get("boundary_conditions") or []):
            if bc.get("type") == "surface_collider":
                pt = torch.as_tensor(bc["point"], device=x.device, dtype=dt)
                nr = torch.as_tensor(bc["normal"], device=x.device, dtype=dt)
                self.bcs.append((pt, nr / nr.norm().clamp_min(1e-12)))
        # 사면체 (rows 의 유일 행) 와 ∇λ
        self.uq, self.inv = torch.unique(rw, dim=0, return_inverse=True)
        X = npos[self.uq].to(torch.float64)                                    # [T,4,3]
        Dm = (X[:, 1:] - X[:, :1]).transpose(1, 2)                             # 열 = X_i - X_0
        Di = torch.linalg.inv(Dm)                                              # 행 i = ∇λ_{i+1}
        gl = torch.cat([-Di.sum(1, keepdim=True), Di], 1)                      # [T,4,3]
        self.glT = gl.to(dt)
        self.gl = self.glT[self.inv]                                           # [N,4,3]
        import os as _os
        self.kmode = _os.environ.get("AF_FEM_FAST_K", "autograd")            # compile = vmap(jacfwd) 닫힌 식을 torch.compile 한 커널

    # ---- 입자 단위 에너지 (Up = 입자별 네 노드 변위 사본 [n,4,3]) ----
    def _maps(self, x0, q, Jf):
        """deform_model.bc_maps 와 같은 순서·같은 분기: --bc_stick -> --bc_softplus."""
        I3 = _eye(q.device, q.dtype)
        if self.eps_st > 0:
            for pt, nr in self.bcs:
                eps = self.eps_st
                s = ((q - pt) * nr).sum(-1)
                far = s / eps > 20.0
                sg = torch.sigmoid(s / eps)
                w = torch.where(far, torch.ones_like(s), sg)
                dw = torch.where(far, torch.zeros_like(s), sg * (1.0 - sg) / eps)
                d = q - x0
                dt_ = d - (d * nr).sum(-1, keepdim=True) * nr
                q = q - (1.0 - w).unsqueeze(-1) * dt_
                nn_ = nr.unsqueeze(-1) * nr.unsqueeze(0)
                Pt = I3 - nn_
                Jst = nn_ + w[:, None, None] * Pt + dw[:, None, None] * (dt_.unsqueeze(-1) * nr.view(1, 1, 3))
                Jf = Jst @ Jf
                if self.stick_xn:
                    Jf = Jf + (1.0 - w)[:, None, None] * Pt
        if self.eps_sp > 0:
            for pt, nr in self.bcs:
                eps = self.eps_sp
                s = ((q - pt) * nr).sum(-1)
                sp = torch.nn.functional.softplus(s, beta=1.0 / eps, threshold=20.0)
                q = q + (sp - s).unsqueeze(-1) * nr
                sg = torch.where(s / eps > 20.0, torch.ones_like(s), torch.sigmoid(s / eps))
                nn_ = nr.unsqueeze(-1) * nr.unsqueeze(0)
                Jf = (I3 + (sg - 1.0)[:, None, None] * nn_) @ Jf
        return q, Jf

    def particle_energy(self, Up, idx=None):
        sl = slice(None) if idx is None else idx
        x, lm, gl = self.x[sl], self.lm[sl], self.gl[sl]
        u = (lm.unsqueeze(-1) * Up).sum(1)
        G = (Up.unsqueeze(-1) * gl.unsqueeze(-2)).sum(1)                       # G_ab = Σ_i U_ia ∇λ_i,b
        q, Jf = self._maps(x, x + u, _eye(x.device, x.dtype) + G)
        psi = phys_resid.psi_of(Jf @ self.F[sl], self.cfg, self.h)[0]
        d = q - x - self.c[sl]
        return 0.5 * self.w[sl] * self.m[sl] / (self.h * self.h) * (d * d).sum(-1) + self.vol[sl] * psi

    def energy(self, U):
        return self.particle_energy(U[:, :3].to(self.dt)[self.rw]).sum()

    def near(self, U):
        """바닥 매핑이 항등이 아닌 입자 (어느 매핑이든 s/ε ≤ 20). stick 의 s 는 스텝 끝 q (매핑 전), softplus 의 s 는 stick 뒤 q --
        stick 은 법선 좌표를 안 바꾸므로 둘 다 매핑 전 q 의 법선 거리다."""
        u = (self.lm.unsqueeze(-1) * U[:, :3].to(self.dt)[self.rw]).sum(1)
        q = self.x + u
        nr_ = torch.zeros(q.shape[0], dtype=torch.bool, device=q.device)
        for pt, nr in self.bcs:
            s = ((q - pt) * nr).sum(-1)
            for e in (self.eps_sp, self.eps_st):
                if e > 0:
                    nr_ |= s / e <= 20.0
        return nr_

    def energy_grad(self, U):
        """(E, dE/dU [M,3]) -- 입자 국소 기울기를 노드로 모은다."""
        Up = U[:, :3].to(self.dt)[self.rw].detach().requires_grad_(True)
        with torch.enable_grad():
            E = self.particle_energy(Up).sum()
            gL, = torch.autograd.grad(E, Up)
        g = torch.zeros(U.shape[0], 3, device=U.device, dtype=self.dt).index_add_(0, self.rw.reshape(-1), gL.reshape(-1, 3))
        return E.detach(), g, gL

    def hess_particle(self, U, idx=None):
        """입자 12x12 (float64) -- 국소 탐침 12 회 역전파 (hess_mode: _FcrPsiH)."""
        rw = self.rw if idx is None else self.rw[idx]
        Up = U[:, :3].to(self.dt)[rw].detach().requires_grad_(True)
        n = Up.shape[0]
        HL = torch.empty(n, 12, 12, device=U.device, dtype=torch.float64)
        with torch.enable_grad(), phys_resid.hess_mode():
            E = self.particle_energy(Up, idx).sum()
            gL, = torch.autograd.grad(E, Up, create_graph=True)
            gLf = gL.reshape(n, 12)
            for j in range(12):
                HL[:, :, j] = torch.autograd.grad(gLf[:, j].sum(), Up, retain_graph=True)[0].reshape(n, 12).double()
        return HL

    def hess_tet(self, U):
        """사면체 12x12 [T,12,12] (float64, 투영 전) -- 매핑 항등 입자는 사면체 축약, 바닥 후보만 입자 12x12."""
        T = self.uq.shape[0]
        near = self.near(U)
        far = (~near).nonzero().squeeze(1)
        nidx = near.nonzero().squeeze(1)
        H = torch.zeros(T, 12, 12, device=U.device, dtype=torch.float64)
        # 관성 (매핑 항등): w m/h² Σ λ_i λ_j δ_ab  -> 사면체 합
        lmf = self.lm[far].double()
        wm = (self.w[far] * self.m[far]).double() / (self.h * self.h)
        LL = torch.zeros(T, 4, 4, device=U.device, dtype=torch.float64).index_add_(
            0, self.inv[far], wm[:, None, None] * lmf.unsqueeze(2) * lmf.unsqueeze(1))
        I3 = torch.eye(3, device=U.device, dtype=torch.float64)
        H += (LL[:, :, None, :, None] * I3[None, None, :, None, :]).reshape(T, 12, 12)
        # 탄성 (매핑 항등): K_p = ∂²ψ(A F_p)/∂A² (A = I + G_T), Σ V_p K_p 를 사면체로 모은 뒤 B_T
        Up = U[:, :3].to(self.dt)[self.rw[far]]
        G = (Up.unsqueeze(-1) * self.gl[far].unsqueeze(-2)).sum(1)
        A0 = (_eye(Up.device, self.dt) + G).detach()
        n = A0.shape[0]
        if self.kmode == "compile":
            K = _k_compiled(A0, self.F[far], self.vol[far], float(self.mu), float(self.lam))
        else:
            ZA = torch.zeros_like(A0).requires_grad_(True)
            K = torch.empty(n, 9, 9, device=U.device, dtype=torch.float64)
            with torch.enable_grad(), phys_resid.hess_mode():
                psi = phys_resid.psi_of((A0 + ZA) @ self.F[far], self.cfg, self.h)[0]
                gA, = torch.autograd.grad((self.vol[far] * psi).sum(), ZA, create_graph=True)
                gAf = gA.reshape(n, 9)
                for j in range(9):
                    K[:, :, j] = torch.autograd.grad(gAf[:, j].sum(), ZA, retain_graph=True)[0].reshape(n, 9).double()
        KT = torch.zeros(T, 9, 9, device=U.device, dtype=torch.float64).index_add_(0, self.inv[far], K)
        # B_T [9 (a,b), 12 (i,c)] = δ_ac ∇λ_i,b  ->  H_el[(i,c),(j,d)] = Σ_{b,e} ∇λ_i,b K[(c,b),(d,e)] ∇λ_j,e
        gl = self.glT.double()
        K4 = KT.reshape(T, 3, 3, 3, 3)                                          # [T, c, b, d, e]
        Hel = torch.einsum("tib,tcbde,tje->ticjd", gl, K4, gl).reshape(T, 12, 12)
        H += Hel
        if nidx.numel():
            H.index_add_(0, self.inv[nidx], self.hess_particle(U, nidx))
        return H, dict(n_near=int(nidx.numel()), n_far=int(far.numel()))


# ---- K_p = V ∂²ψ(A F)/∂A² 를 원소별 닫힌 식의 전진 미분으로 (AF_FEM_FAST_K=compile) ----
# hess_mode 의 _FcrPsiH 와 같은 정의: 기울기 = _fcr_grad_h(AF) Fᵀ (극분해 Newton 10 회를 그대로 미분), σ_min ≤ 0.01 입자는 기울기가
# 상수라 K = 0. vmap(jacfwd) 를 torch.compile 로 한 커널에 묶는다 (반복마다 커널 수백 개를 띄우던 것을 없앤다).
_KC = [None]


def _k_one(Av, Fp, mu, lam):
    Am = Av.reshape(3, 3)
    P = phys_resid._fcr_grad_h((Am @ Fp).unsqueeze(0), mu, lam)[0]
    return (P @ Fp.transpose(0, 1)).reshape(9)


def _k_batch(A, F, vol, mu: float, lam: float):
    J = torch.func.vmap(torch.func.jacfwd(_k_one), in_dims=(0, 0, None, None))(A.reshape(-1, 9), F, mu, lam)   # [n,9,9]
    AF = A @ F
    C = (AF.unsqueeze(-1) * AF.unsqueeze(-2)).sum(-3)
    smin = phys_resid._sym_eig3(C).clamp_min(1e-12).sqrt()[:, 0]
    return torch.where((smin <= 0.01)[:, None, None], torch.zeros_like(J), J) * vol[:, None, None]


def _k_compiled(A, F, vol, mu, lam):
    if _KC[0] is None:
        try:
            _KC[0] = torch.compile(_k_batch, dynamic=True)
        except Exception as e:                                               # noqa: BLE001
            print(f"[fem_fast] torch.compile 실패, 그대로: {e}", flush=True)
            _KC[0] = _k_batch
    out = []
    for i in range(0, A.shape[0], 131072):
        out.append(_KC[0](A[i:i + 131072], F[i:i + 131072], vol[i:i + 131072], mu, lam).double())
    return torch.cat(out) if out else torch.zeros(0, 9, 9, device=A.device, dtype=torch.float64)

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


class _Tick:
    """AF_FEM_PROF=1 일 때만 동기화해 구간 시간 (ms) 을 모은다."""
    def __init__(self):
        import os as _os, time as _time
        self.on = bool(_os.environ.get("AF_FEM_PROF")) and torch.cuda.is_available()
        self._time = _time
        self.d = {}
        if self.on:
            torch.cuda.synchronize()
        self.t = _time.time()

    def __call__(self, k):
        if self.on:
            torch.cuda.synchronize()
            t = self._time.time()
            self.d[k] = 1e3 * (t - self.t)
            self.t = t


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
        self.near_mode = _os.environ.get("AF_FEM_FAST_NEAR", "autograd")     # chain = 바닥 후보도 manual K + 매핑만 든 작은 그래프 (정확히 같은 식)

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
        tk = _Tick()
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
        tk("관성")
        # 탄성 (매핑 항등): K_p = ∂²ψ(A F_p)/∂A² (A = I + G_T), Σ V_p K_p 를 사면체로 모은 뒤 B_T
        Up = U[:, :3].to(self.dt)[self.rw[far]]
        G = (Up.unsqueeze(-1) * self.gl[far].unsqueeze(-2)).sum(1)
        A0 = (_eye(Up.device, self.dt) + G).detach()
        n = A0.shape[0]
        if self.kmode == "manual":
            K = _k_manual_run(A0, self.F[far], self.vol[far], float(self.mu), float(self.lam))
        elif self.kmode == "compile":
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
        tk("K(항등)")
        KT = torch.zeros(T, 9, 9, device=U.device, dtype=torch.float64).index_add_(0, self.inv[far], K)
        # B_T [9 (a,b), 12 (i,c)] = δ_ac ∇λ_i,b  ->  H_el[(i,c),(j,d)] = Σ_{b,e} ∇λ_i,b K[(c,b),(d,e)] ∇λ_j,e
        gl = self.glT.double()
        K4 = KT.reshape(T, 3, 3, 3, 3)                                          # [T, c, b, d, e]
        Hel = torch.einsum("tib,tcbde,tje->ticjd", gl, K4, gl).reshape(T, 12, 12)
        H += Hel
        tk("사면체 합·B")
        if nidx.numel():
            H.index_add_(0, self.inv[nidx], self.hess_near_chain(U, nidx) if self.near_mode == "chain" else self.hess_particle(U, nidx))
        tk("바닥 후보")
        self.last_t = tk.d
        return H, dict(n_near=int(nidx.numel()), n_far=int(far.numel()), t=tk.d)

    def hess_near_chain(self, U, idx):
        """바닥 후보 입자 12x12 (AF_FEM_FAST_NEAR=chain): 연쇄법칙으로 정확히 같은 헤시안을 싼 그래프로.
            ∂²[V ψ(J_f(u) F)] = J'ᵀ K J' + Σ_ab P_ab ∂²J_f,ab/∂u²,   J' = ∂vec J_f/∂u (9x12), K = V ∂²ψ(AF)/∂A² (manual 커널),
            P = V ∂ψ/∂A (hess_mode 의 1 차 기울기 그대로, σ_min ≤ 0.01 이면 _FcrPsi 상수 기울기)
        관성 항과 Σ P ∂²J_f 는 매핑만 든 작은 함수 φ(u) = 관성 + <P(고정), J_f(u)> 의 이중 역전파 12 회 (극분해 반복이 그래프에 없다)."""
        rw = self.rw[idx]
        x, lm, gl, Fp = self.x[idx], self.lm[idx], self.gl[idx], self.F[idx]
        n = rw.shape[0]
        I3 = _eye(x.device, self.dt)
        Up = U[:, :3].to(self.dt)[rw].detach().requires_grad_(True)
        with torch.enable_grad():
            u = (lm.unsqueeze(-1) * Up).sum(1)
            G = (Up.unsqueeze(-1) * gl.unsqueeze(-2)).sum(1)
            q, Jf = self._maps(x, x + u, I3 + G)
            Jf9 = Jf.reshape(n, 9)
            JJ = torch.empty(n, 9, 12, device=x.device, dtype=torch.float64)
            for a_ in range(9):
                JJ[:, a_, :] = torch.autograd.grad(Jf9[:, a_].sum(), Up, retain_graph=True)[0].reshape(n, 12).double()
        A0 = Jf.detach()
        ZA = torch.zeros_like(A0).requires_grad_(True)
        with torch.enable_grad(), phys_resid.hess_mode():
            psi = phys_resid.psi_of((A0 + ZA) @ Fp, self.cfg, self.h)[0]
            Pc, = torch.autograd.grad((self.vol[idx] * psi).sum(), ZA)
        K = _k_manual_run(A0, Fp, self.vol[idx], float(self.mu), float(self.lam))
        H = JJ.transpose(1, 2) @ K @ JJ
        Up2 = U[:, :3].to(self.dt)[rw].detach().requires_grad_(True)
        with torch.enable_grad():
            u = (lm.unsqueeze(-1) * Up2).sum(1)
            G = (Up2.unsqueeze(-1) * gl.unsqueeze(-2)).sum(1)
            q, Jf = self._maps(x, x + u, I3 + G)
            d = q - x - self.c[idx]
            phi = (0.5 * self.w[idx] * self.m[idx] / (self.h * self.h) * (d * d).sum(-1)).sum() + (Pc.detach() * Jf).sum()
            gq, = torch.autograd.grad(phi, Up2, create_graph=True)
            gqf = gq.reshape(n, 12)
            for j in range(12):
                H[:, :, j] += torch.autograd.grad(gqf[:, j].sum(), Up2, retain_graph=True)[0].reshape(n, 12).double()
        return H


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
        try:
            out.append(_KC[0](A[i:i + 131072], F[i:i + 131072], vol[i:i + 131072], mu, lam).double())
        except Exception as e:                                               # noqa: BLE001
            if _KC[0] is _k_batch:
                raise
            # torch 2.5 dynamo 가 vmap(jacfwd) 안 matmul 을 못 넘긴다 (10-27 실측) -> 컴파일 없이 같은 함수 (값 같음)
            print(f"[fem_fast] 컴파일 K 실행 실패 -> 컴파일 없이: {type(e).__name__}", flush=True)
            _KC[0] = _k_batch
            out.append(_KC[0](A[i:i + 131072], F[i:i + 131072], vol[i:i + 131072], mu, lam).double())
    return torch.cat(out) if out else torch.zeros(0, 9, 9, device=A.device, dtype=torch.float64)


# ---- 손으로 쓴 전진 미분 (AF_FEM_FAST_K=manual): _fcr_grad_h 의 방향 미분을 원소별 식으로 직접 -- functorch 없이 torch.compile 이 한 커널로 묶는다 ----
def _cof_lin(A, dA):
    """cof(A) 의 A 방향 dA 미분 (각 원소가 2x2 소행렬식이라 곱의 미분)."""
    a, b, c = A[..., 0, 0], A[..., 0, 1], A[..., 0, 2]
    d, e, f = A[..., 1, 0], A[..., 1, 1], A[..., 1, 2]
    g, h, i = A[..., 2, 0], A[..., 2, 1], A[..., 2, 2]
    da, db, dc = dA[..., 0, 0], dA[..., 0, 1], dA[..., 0, 2]
    dd, de, df = dA[..., 1, 0], dA[..., 1, 1], dA[..., 1, 2]
    dg, dh, di = dA[..., 2, 0], dA[..., 2, 1], dA[..., 2, 2]
    out = torch.stack([de * i + e * di - df * h - f * dh, df * g + f * dg - dd * i - d * di, dd * h + d * dh - de * g - e * dg,
                       dc * h + c * dh - db * i - b * di, da * i + a * di - dc * g - c * dg, db * g + b * dg - da * h - a * dh,
                       db * f + b * df - dc * e - c * de, dc * d + c * dd - da * f - a * df, da * e + a * de - db * d - b * dd], -1)
    return out.reshape(dA.shape)


def _det_rows(Q):
    return (Q[..., 0, 0] * (Q[..., 1, 1] * Q[..., 2, 2] - Q[..., 1, 2] * Q[..., 2, 1])
            - Q[..., 0, 1] * (Q[..., 1, 0] * Q[..., 2, 2] - Q[..., 1, 2] * Q[..., 2, 0])
            + Q[..., 0, 2] * (Q[..., 1, 0] * Q[..., 2, 1] - Q[..., 1, 1] * Q[..., 2, 0]))


def _k_manual(A, F, vol, mu: float, lam: float):
    """K = V ∂(_fcr_grad_h(AF) Fᵀ)/∂A [n,9,9] -- phys_resid._polar_q (Newton 10 회) 와 _fcr_grad_h 를 그대로 따라가며 9 방향 전진 미분."""
    n = A.shape[0]
    X = A @ F                                                              # [n,3,3]
    I9 = torch.eye(9, device=A.device, dtype=A.dtype).reshape(9, 3, 3)
    dX = I9.unsqueeze(0) @ F.unsqueeze(1)                                  # [n,9,3,3]  (E_j F)
    Q = X.unsqueeze(1)                                                     # [n,1,3,3]
    dQ = dX
    for _ in range(10):
        C = phys_resid._cof3(Q)
        dC = _cof_lin(Q, dQ)
        dt = _det_rows(Q)                                                  # [n,1]
        ddt = (C * dQ).sum((-1, -2))                                       # [n,9]  d det = cof : dQ
        big = dt.abs() > 1e-30
        ds = torch.where(big, dt, torch.full_like(dt, 1e-30))
        dds = torch.where(big, ddt, torch.zeros_like(ddt))
        gm = ds.abs().pow(-1.0 / 3.0)
        dgm = -(1.0 / 3.0) * gm * dds / ds
        sg = ds * gm
        dsg = dds * gm + ds * dgm
        Qn = 0.5 * (gm[..., None, None] * Q + C / sg[..., None, None])
        dQ = 0.5 * (dgm[..., None, None] * Q + gm[..., None, None] * dQ + dC / sg[..., None, None]
                    - C * (dsg / (sg * sg))[..., None, None])
        Q = Qn
    Xb = X.unsqueeze(1)
    cof = phys_resid._cof3(Xb)
    dcof = _cof_lin(Xb, dX)
    det = (Xb[..., 0, :] * cof[..., 0, :]).sum(-1)                         # _fcr_grad_h 와 같은 식
    ddet = (cof * dX).sum((-1, -2))
    dP = 2.0 * mu * (dX - dQ) + (lam * ddet)[..., None, None] * cof + (lam * (det - torch.sign(det)))[..., None, None] * dcof
    dG = dP @ F.transpose(1, 2).unsqueeze(1)                               # [n,9(j),3,3]
    K = dG.reshape(n, 9, 9).transpose(1, 2)                                # K[:, a, j] = ∂gA_a/∂A_j
    Cx = (X.unsqueeze(-1) * X.unsqueeze(-2)).sum(-3)
    smin = phys_resid._sym_eig3(Cx).clamp_min(1e-12).sqrt()[:, 0]
    return torch.where((smin <= 0.01)[:, None, None], torch.zeros_like(K), K) * vol[:, None, None]


_KM = [None]


def _k_manual_run(A, F, vol, mu, lam):
    import os as _os
    if _KM[0] is None:
        _KM[0] = _k_manual
        if not _os.environ.get("AF_FEM_FAST_NOCOMPILE"):
            try:
                _KM[0] = torch.compile(_k_manual, dynamic=True)
            except Exception as e:                                           # noqa: BLE001
                print(f"[fem_fast] manual K compile 실패: {e}", flush=True)
    out = []
    for i in range(0, A.shape[0], 131072):
        try:
            out.append(_KM[0](A[i:i + 131072], F[i:i + 131072], vol[i:i + 131072], float(mu), float(lam)).double())
        except Exception as e:                                               # noqa: BLE001
            if _KM[0] is _k_manual:
                raise
            print(f"[fem_fast] manual K 컴파일 실행 실패 -> 컴파일 없이: {type(e).__name__}: {str(e)[:200]}", flush=True)
            _KM[0] = _k_manual
            out.append(_KM[0](A[i:i + 131072], F[i:i + 131072], vol[i:i + 131072], float(mu), float(lam)).double())
    return torch.cat(out) if out else torch.zeros(0, 9, 9, device=A.device, dtype=torch.float64)


# ---- 행렬 없는 블록 PCG + CUDA 그래프 (AF_FEM_PCG_MF=1, 실시간 판 선택지) ----
# A v = Σ_T scatter(H_T v_T) (사면체 12x12 투영 헤시안 그대로, 고정 자유도는 행·열 0 + 대각 1) -- 조립 CSR 과 같은 행렬 (합산 순서만 다름).
# PCG 본체 ck 회를 CUDA 그래프 하나로 잡아 재생 (커널 띄우기·호스트 동기화 없음), 재생마다 한 번 수렴 확인. 판정식은 _asm_pcg 와 같다.
class MFPcg:
    def __init__(self, He, dof, fx3, Dinv, M_, ck=50):
        self.He, self.dof, self.M, self.ck = He, dof.reshape(-1), M_, int(ck)
        self.keep = (~fx3).to(He.dtype)
        self.fx = fx3.to(He.dtype)
        self.Dinv = Dinv
        self.T = He.shape[0]
        self.graph = None

    def mv(self, v):
        vk = v * self.keep
        y = torch.bmm(self.He, vk[self.dof].reshape(self.T, 12, 1)).reshape(-1)
        out = torch.zeros_like(v).index_add_(0, self.dof, y)
        return out * self.keep + v * self.fx

    def pre(self, r):
        return (self.Dinv @ r.reshape(self.M, 3, 1)).reshape(-1)

    def _body(self):
        S = self.S
        Ad = self.mv(S["d"])
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
        z = self.pre(S["r"])
        rz2 = (S["r"] * z).sum()
        beta = torch.where(S["alive"], rz2 / S["rz"], S["zero"])
        S["d"].copy_(z + beta * S["d"])
        S["rz"].copy_(torch.where(S["alive"], rz2, S["rz"]))

    def solve(self, b, ncg, rtol, use_graph=True):
        dev, dt = b.device, b.dtype
        r = b.clone()
        z = self.pre(r)
        self.S = S = dict(x=torch.zeros_like(b), r=r, d=z.clone(), rz=(r * z).sum(), thr=(b * b).sum().clamp_min(1e-300) * (rtol * rtol),
                          alive=torch.ones((), dtype=torch.bool, device=dev), neg=torch.zeros((), dtype=torch.bool, device=dev),
                          nit=torch.zeros((), dtype=torch.long, device=dev), one=torch.ones((), dtype=dt, device=dev),
                          zero=torch.zeros((), dtype=dt, device=dev))
        nrep = (ncg + self.ck - 1) // self.ck
        if use_graph and b.is_cuda:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):                     # 데우기 (그래프 잡기 전 한 번; 상태를 바꾸므로 되돌린다)
                keep = {k: v.clone() for k, v in S.items()}
                self._body()
                for k, v in keep.items():
                    S[k].copy_(v)
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for _ in range(self.ck):
                    self._body()
            for _ in range(nrep):
                g.replay()
                if not bool(S["alive"]):
                    break
        else:
            for _ in range(nrep):
                for _ in range(self.ck):
                    self._body()
                if not bool(S["alive"]):
                    break
        return S["x"], int(S["nit"]), float((S["r"] * S["r"]).sum()), bool(S["neg"])

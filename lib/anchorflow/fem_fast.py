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
        self.tq = None                                                         # AF_FEM_TETQ=1: tetq_setup 이 채운다

    # ---- AF_FEM_TETQ=1 (사용자 10-27 확정): 사면체 적분점 하나 -- MPM 처럼 고정 사면체 격자가 grid, 격자 안 계산을 FEM 이 ----
    # 에너지가 바뀌는 새 모델 (입자 적분점 -> 사면체 적분점 1 개). 바닥 매핑(stick·softplus)은 입자마다 J_f 를 바꾸는 정의라 **그대로 두려고**
    # 바닥 후보 입자(매핑이 항등이 아닌 입자, s/ε ≤ 20) 는 지금 빠른 경로와 똑같이 입자 단위 (관성·탄성 모두, 닫힌 식 헤시안) 로 남기고,
    # 매핑이 항등인 입자만 사면체로 모은다 (P2G):
    #   관성   Σ_p w m/(2h²) |Σ_i λ_i U_i - c_p|² = Σ_T [½ Σ_ij M_T,ij U_i·U_j - Σ_i b_T,i·U_i] + C0   (일관 질량 -- 정확, 근사 아님)
    #   탄성   Σ_T V_T ψ(A_T F̄_T),  V_T = Σ_{p∈T} V_p,  F̄_T = Σ V_p F_p / V_T  (입자 부피 가중 평균),  A_T = I + Σ_i U_i ⊗ ∇λ_i
    # 분할(바닥 후보 / 항등)은 뉴턴 걸음 시작 U 에서 정해 그 걸음(헤시안·선탐색) 동안 고정한다. 입자가 없는(항등 입자가 없는) 사면체는 탄성에서 빠진다.
    # G2P 는 기존 파이프라인 그대로 (입자 위치·속도 barycentric 보간, F_p <- J_f F_p 입자별 누적).
    def tetq_setup(self, U):
        f8 = torch.float64
        if not isinstance(self.cfg.get("E"), (int, float)) or not isinstance(self.cfg.get("nu"), (int, float)):
            raise SystemExit("AF_FEM_TETQ: 물성이 입자별이면 사면체 부피 가중 평균이 필요하다 (지금은 균일 물성만)")
        dev = self.dev
        near = self.near(U)
        fi = (~near).nonzero().squeeze(1)
        ni = near.nonzero().squeeze(1)
        T = self.uq.shape[0]
        inv = self.inv[fi]
        V = self.vol[fi].to(f8)
        VT = torch.zeros(T, device=dev, dtype=f8).index_add_(0, inv, V)
        FT = torch.zeros(T, 3, 3, device=dev, dtype=f8).index_add_(0, inv, V[:, None, None] * self.F[fi].to(f8))
        k = (self.w[fi] * self.m[fi]).to(f8) / (self.h * self.h)
        lm = self.lm[fi].to(f8)
        c = self.c[fi].to(f8)
        M4 = torch.zeros(T, 4, 4, device=dev, dtype=f8).index_add_(0, inv, k[:, None, None] * lm[:, :, None] * lm[:, None, :])
        B4 = torch.zeros(T, 4, 3, device=dev, dtype=f8).index_add_(0, inv, k[:, None, None] * lm[:, :, None] * c[:, None, :])
        C0 = 0.5 * (k * (c * c).sum(-1)).sum()
        act = (VT > 0).nonzero().squeeze(1)
        self.tq = dict(near=near, fi=fi, ni=ni, act=act, uq=self.uq[act], gl=self.glT[act].to(f8), VT=VT[act],
                       FT=FT[act] / VT[act][:, None, None], M4=M4[act], B4=B4[act], C0=C0)
        return self.tq

    def _tq_energy(self, Ut):
        """사면체 부분 에너지 (float64): Ut [Ta,4,3] = 활성 사면체 네 노드 변위."""
        q = self.tq
        inert = 0.5 * ((q["M4"] @ Ut) * Ut).sum() - (q["B4"] * Ut).sum() + q["C0"]
        A = _eye(Ut.device, Ut.dtype) + (Ut.unsqueeze(-1) * q["gl"].unsqueeze(-2)).sum(1)
        psi = phys_resid.psi_of(A @ q["FT"], self.cfg, self.h)[0]
        return inert + (q["VT"] * psi).sum()

    def _tq_near_energy(self, U, grad=False):
        ni = self.tq["ni"]
        if not ni.numel():
            return None, None
        Up = U[:, :3].to(self.dt)[self.rw[ni]].detach()
        if not grad:
            return self.particle_energy(Up, ni).sum().double(), None
        Up.requires_grad_(True)
        with torch.enable_grad():
            En = self.particle_energy(Up, ni).sum()
            gn, = torch.autograd.grad(En, Up)
        return En.detach().double(), gn

    def tetq_energy(self, U):
        E = self._tq_energy(U[:, :3].double()[self.tq["uq"]])
        En, _ = self._tq_near_energy(U)
        return E if En is None else E + En

    def tetq_energy_grad(self, U):
        q = self.tq
        Ut = U[:, :3].double()[q["uq"]].requires_grad_(True)
        with torch.enable_grad():
            Et = self._tq_energy(Ut)
            gt, = torch.autograd.grad(Et, Ut)
        g = torch.zeros(U.shape[0], 3, device=U.device, dtype=torch.float64).index_add_(0, q["uq"].reshape(-1), gt.reshape(-1, 3))
        En, gn = self._tq_near_energy(U, grad=True)
        if En is not None:
            g.index_add_(0, self.rw[q["ni"]].reshape(-1), gn.reshape(-1, 3).double())
            Et = Et + En
        return Et.detach(), g.to(self.dt), None

    def hess_tetq(self, U):
        """사면체 12x12 [T,12,12] (float64, 투영 전): 활성 사면체 = M_T ⊗ I + B_Tᵀ K_T B_T (K_T = V_T ∂²ψ(A_T F̄_T)/∂A², 사면체 수만큼만
        manual 커널), 바닥 후보 입자는 지금 빠른 경로와 같은 입자 12x12 (닫힌 식) 를 자기 사면체에 더한다."""
        q = self.tq
        T = self.uq.shape[0]
        tk = _Tick()
        f8 = torch.float64
        Ta = int(q["act"].numel())
        I3 = torch.eye(3, device=U.device, dtype=f8)
        Hq = (q["M4"][:, :, None, :, None] * I3[None, None, :, None, :]).reshape(Ta, 12, 12)
        tk("관성")
        Ut = U[:, :3].double()[q["uq"]]
        A0 = I3 + (Ut.unsqueeze(-1) * q["gl"].unsqueeze(-2)).sum(1)
        K = _k_manual_run(A0, q["FT"], q["VT"], float(self.mu), float(self.lam))
        tk("K(사면체)")
        Hq = Hq + torch.einsum("tib,tcbde,tje->ticjd", q["gl"], K.reshape(Ta, 3, 3, 3, 3), q["gl"]).reshape(Ta, 12, 12)
        H = torch.zeros(T, 12, 12, device=U.device, dtype=f8)
        H[q["act"]] = Hq
        tk("사면체 합·B")
        ni = q["ni"]
        if ni.numel():
            if self.near_mode == "analytic":
                H.index_add_(0, self.inv[ni], self.hess_near_analytic(U, ni))
            elif self.near_mode in ("chain", "chainfwd"):
                H.index_add_(0, self.inv[ni], self.hess_near_chain(U, ni, fwd=(self.near_mode == "chainfwd")))
            else:
                H.index_add_(0, self.inv[ni], self.hess_particle(U, ni))
        tk("바닥 후보")
        self.last_t = tk.d
        return H, dict(n_near=int(ni.numel()), n_far=int(q["fi"].numel()), n_act=Ta, t=tk.d)

    def tetq_diag(self, U):
        """축약 근사 정도 (AF_FEM_TETQ_DIAG=1, 계측 구간 밖): 같은 A_T 에서 입자 적분점 Σ V_p ψ(A_T F_p) vs 사면체 적분점 Σ V_T ψ(A_T F̄_T)
        (매핑 항등 입자만 -- 바닥 후보는 두 모델이 같다), U=0 (A=I) 에서도. F 퍼짐 = 부피 가중 평균 |F_p - F̄_T|_F."""
        q = self.tq
        f8 = torch.float64
        fi = q["fi"]
        with torch.no_grad():
            T = self.uq.shape[0]
            FTf = torch.zeros(T, 3, 3, device=U.device, dtype=f8)
            FTf[q["act"]] = q["FT"]
            Ut = U[:, :3].double()[self.uq]
            AT = torch.eye(3, device=U.device, dtype=f8) + (Ut.unsqueeze(-1) * self.glT.double().unsqueeze(-2)).sum(1)
            inv = self.inv[fi]
            Fp, V, Fb, Ap = self.F[fi].double(), self.vol[fi].double(), FTf[inv], AT[inv]
            ps = lambda X: phys_resid.psi_of(X, self.cfg, self.h)[0].double()
            Ep, Et = float((V * ps(Ap @ Fp)).sum()), float((q["VT"] * ps(AT[q["act"]] @ q["FT"])).sum())
            E0p, E0t = float((V * ps(Fp)).sum()), float((q["VT"] * ps(q["FT"])).sum())
            dF = (Fp - Fb).flatten(1).norm(dim=1)
            vs = float(V.sum())
            return dict(E_el_part=Ep, E_el_tet=Et, E_el_rel=(Et - Ep) / max(abs(Ep), 1e-30),
                        E0_el_part=E0p, E0_el_tet=E0t, E0_el_rel=(E0t - E0p) / max(abs(E0p), 1e-30),
                        dF_mean=float((V * dF).sum()) / vs, dF_max=float(dF.max()),
                        F_mean=float((V * Fb.flatten(1).norm(dim=1)).sum()) / vs,
                        n_act=int(q["act"].numel()), n_tet=int(T), n_near=int(q["ni"].numel()), n_far=int(fi.numel()))

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
        if self.tq is not None:
            return self.tetq_energy(U)
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
        if self.tq is not None:
            return self.tetq_energy_grad(U)
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
            if self.near_mode == "analytic":
                H.index_add_(0, self.inv[nidx], self.hess_near_analytic(U, nidx))
            else:
                H.index_add_(0, self.inv[nidx], self.hess_near_chain(U, nidx, fwd=(self.near_mode == "chainfwd"))
                             if self.near_mode in ("chain", "chainfwd") else self.hess_particle(U, nidx))
        tk("바닥 후보")
        self.last_t = tk.d
        return H, dict(n_near=int(nidx.numel()), n_far=int(far.numel()), t=tk.d)

    def _maps1(self, x0, up, lm, gl):
        """입자 하나 (vmap 용): 국소 변위 up [12] -> (q [3], J_f [3,3]) -- _maps 를 묶음 1 로 그대로."""
        U4 = up.reshape(4, 3)
        u = (lm.unsqueeze(-1) * U4).sum(0)
        G = (U4.unsqueeze(-1) * gl.unsqueeze(-2)).sum(0)
        q, Jf = self._maps(x0.unsqueeze(0), (x0 + u).unsqueeze(0), (_eye(up.device, up.dtype) + G).unsqueeze(0))
        return q[0], Jf[0]

    def _near_fwd(self, Up, x, lm, gl, c, wm, Pc):
        """AF_FEM_FAST_NEAR=chainfwd: J' 와 φ 의 헤시안을 vmap(jacfwd) 전진 미분으로 (역전파 21 회 대신 한 번에)."""
        h2 = self.h * self.h

        def jf(up, x0, l_, g_):
            return self._maps1(x0, up, l_, g_)[1].reshape(9)

        def phi(up, x0, l_, g_, c_, wm_, P_):
            q, Jf = self._maps1(x0, up, l_, g_)
            d = q - x0 - c_
            return 0.5 * wm_ / h2 * (d * d).sum() + (P_ * Jf).sum()
        n = Up.shape[0]
        JJ = torch.func.vmap(torch.func.jacfwd(jf))(Up.reshape(n, 12), x, lm, gl)                       # [n,9,12]
        Hp = torch.func.vmap(torch.func.jacfwd(torch.func.jacfwd(phi)))(Up.reshape(n, 12), x, lm, gl, c, wm, Pc)   # [n,12,12]
        return JJ.double(), Hp.double()

    def hess_near_analytic(self, U, idx):
        """AF_FEM_FAST_NEAR=analytic: 바닥 후보 입자 12x12 를 닫힌 식으로 (자동미분 없음, float64). 매핑은 법선 좌표 s = n·(x+u-p) 의
        스칼라 함수 w(s) (stick), sg(s)·sp(s) (softplus) 와 u, G 의 다항식이라 y = (u[3], vec G[9]) 공간 헤시안을 직접 쓴 뒤 L(λ, ∇λ) 로 옮긴다.
            r = q - x - c = u - (1-w) P u + (sp - s) n,   J_f = Mn(s) K1,  Mn = I + (sg-1) N,  K1 = (N + w P + w' (Pu) nᵀ)(I+G) [+ (1-w) P  (xn)]
        헤시안 = k [R_uᵀR_u + Σ r·∂²r] + ∂²<P_A, J_f> + J'ᵀ K J'  (K = manual 커널, P_A = hess_mode 1 차 기울기). 바닥 면이 하나일 때만."""
        if len(self.bcs) != 1:
            raise SystemExit("AF_FEM_FAST_NEAR=analytic: 바닥 면(surface_collider) 하나일 때만")
        f8 = torch.float64
        rw = self.rw[idx]
        x, lm, gl, Fp = self.x[idx], self.lm[idx], self.gl[idx], self.F[idx]
        n = rw.shape[0]
        Up = U[:, :3].to(self.dt)[rw].detach()
        u = (lm.unsqueeze(-1) * Up).sum(1)
        G = (Up.unsqueeze(-1) * gl.unsqueeze(-2)).sum(1)
        I3f = _eye(x.device, self.dt)
        A0 = self._maps(x, x + u, I3f + G)[1].detach()                       # 실행과 같은 dtype 으로 (K·P_A 의 점)
        ZA = torch.zeros_like(A0).requires_grad_(True)
        with torch.enable_grad(), phys_resid.hess_mode():
            psi = phys_resid.psi_of((A0 + ZA) @ Fp, self.cfg, self.h)[0]
            Pc, = torch.autograd.grad((self.vol[idx] * psi).sum(), ZA)
        K = _k_manual_run(A0, Fp, self.vol[idx], float(self.mu), float(self.lam))   # [n,9,9] f64
        pt, nr = self.bcs[0]
        xn = 1.0 if (self.stick_xn and self.eps_st > 0) else 0.0
        k = (self.w[idx] * self.m[idx]).to(f8) / (self.h * self.h)
        return _near_core_run(Pc.to(f8), K, x.to(f8), u.to(f8), G.to(f8), lm.to(f8), gl.to(f8), self.c[idx].to(f8), k,
                              pt.to(f8), nr.to(f8), float(self.eps_st), float(self.eps_sp), xn)

    def hess_near_chain(self, U, idx, fwd=False):
        """바닥 후보 입자 12x12 (AF_FEM_FAST_NEAR=chain): 연쇄법칙으로 정확히 같은 헤시안을 싼 그래프로.
            ∂²[V ψ(J_f(u) F)] = J'ᵀ K J' + Σ_ab P_ab ∂²J_f,ab/∂u²,   J' = ∂vec J_f/∂u (9x12), K = V ∂²ψ(AF)/∂A² (manual 커널),
            P = V ∂ψ/∂A (hess_mode 의 1 차 기울기 그대로, σ_min ≤ 0.01 이면 _FcrPsi 상수 기울기)
        관성 항과 Σ P ∂²J_f 는 매핑만 든 작은 함수 φ(u) = 관성 + <P(고정), J_f(u)> 의 이중 역전파 12 회 (극분해 반복이 그래프에 없다)."""
        rw = self.rw[idx]
        x, lm, gl, Fp = self.x[idx], self.lm[idx], self.gl[idx], self.F[idx]
        n = rw.shape[0]
        I3 = _eye(x.device, self.dt)
        if fwd:
            Upd = U[:, :3].to(self.dt)[rw].detach()
            u = (lm.unsqueeze(-1) * Upd).sum(1)
            G = (Upd.unsqueeze(-1) * gl.unsqueeze(-2)).sum(1)
            A0 = self._maps(x, x + u, I3 + G)[1].detach()
        else:
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
        if fwd:
            JJ, Hphi = self._near_fwd(Upd, x, lm, gl, self.c[idx], self.w[idx] * self.m[idx], Pc.detach())
            return JJ.transpose(1, 2) @ K @ JJ + Hphi
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



# ---- 바닥 후보 12x12 닫힌 식 본체 (FemFrame.hess_near_analytic 에서 분리; AF_FEM_NEAR_COMPILE=1 이면 torch.compile -- 같은 식) ----
def _near_core(Pc, K, x8, u8, G8, lm8, gl8, c8, k, pt, nv, eps_st: float, eps_sp: float, xn: float):
    f8 = torch.float64
    n = x8.shape[0]
    x = x8
    I3 = torch.eye(3, device=x8.device, dtype=f8)
    N = nv[:, None] * nv[None, :]
    P = I3 - N
    s = ((x8 + u8 - pt) * nv).sum(-1)                                    # [n]

    def sigs(e):
        far = s / e > 20.0
        sg_ = torch.sigmoid(s / e)
        one, zero = torch.ones_like(s), torch.zeros_like(s)
        d0 = torch.where(far, one, sg_)
        d1 = torch.where(far, zero, sg_ * (1 - sg_) / e)
        d2 = torch.where(far, zero, sg_ * (1 - sg_) * (1 - 2 * sg_) / e ** 2)
        d3 = torch.where(far, zero, sg_ * (1 - sg_) * (1 - 6 * sg_ + 6 * sg_ * sg_) / e ** 3)
        return d0, d1, d2, d3
    if eps_st > 0:
        w, w1, w2, w3 = sigs(eps_st)
    else:
        w, w1, w2, w3 = torch.ones_like(s), torch.zeros_like(s), torch.zeros_like(s), torch.zeros_like(s)
    if eps_sp > 0:
        g, g1, g2, _ = sigs(eps_sp)
        sp = torch.nn.functional.softplus(s, beta=1.0 / eps_sp, threshold=20.0)
    else:
        g, g1, g2, sp = torch.ones_like(s), torch.zeros_like(s), torch.zeros_like(s), s
    a = u8 @ P                                                           # P u (P 대칭) [n,3]
    IG = I3 + G8
    r = u8 - (1 - w)[:, None] * a + (sp - s)[:, None] * nv - c8         # q - x - c
    # 관성 (y_u 3x3)
    Ru = I3 - (1 - w)[:, None, None] * P + w1[:, None, None] * a[:, :, None] * nv[None, None, :] + (g - 1)[:, None, None] * N
    Pr = r @ P
    Huu = k[:, None, None] * (Ru.transpose(1, 2) @ Ru + (w2 * (r * a).sum(-1) + g1 * (r @ nv))[:, None, None] * N
                               + w1[:, None, None] * (Pr[:, :, None] * nv[None, None, :] + nv[None, :, None] * Pr[:, None, :]))
    # K1, Jst, Mn
    Jst = N + w[:, None, None] * P + w1[:, None, None] * a[:, :, None] * nv[None, None, :]
    K1 = Jst @ IG + xn * (1 - w)[:, None, None] * P
    Mn = I3 + (g - 1)[:, None, None] * N
    # u 방향 (기저 e_c, ds = n_c) 의 dK1, d2K1
    E = I3                                                               # e_c 행
    ds = nv                                                              # [3] (방향 c 에서 ds = n_c)
    PE = P                                                               # P e_c = P[:, c]
    # dK1[c] = (w1 ds_c P + w2 ds_c a nᵀ + w1 (P e_c) nᵀ)(I+G) - xn w1 ds_c P      -> [n,3(c),3,3]
    aN = a[:, :, None] * nv[None, None, :]                               # a nᵀ [n,3,3]
    PcN = PE.t()[:, :, None] * nv[None, None, :]                         # [3(c),3,3] = (P e_c) nᵀ
    dK1 = ((w1[:, None, None, None] * ds[None, :, None, None] * P[None, None] + w2[:, None, None, None] * ds[None, :, None, None] * aN[:, None]
            + w1[:, None, None, None] * PcN[None]) @ IG[:, None] - xn * w1[:, None, None, None] * ds[None, :, None, None] * P[None, None])
    dsd = ds[:, None] * ds[None, :]                                      # [3,3] ds_c ds_d
    # d2K1[c,d] = (w2 ds_c ds_d P + w3 ds_c ds_d a nᵀ + w2 ds_c (P e_d) nᵀ + w2 ds_d (P e_c) nᵀ)(I+G) - xn w2 ds_c ds_d P
    d2K1 = ((w2[:, None, None, None, None] * dsd[None, :, :, None, None] * P + w3[:, None, None, None, None] * dsd[None, :, :, None, None] * aN[:, None, None]
             + w2[:, None, None, None, None] * (ds[None, :, None, None, None] * PcN[None, None, :] + ds[None, None, :, None, None] * PcN[None, :, None]))
            @ IG[:, None, None] - xn * w2[:, None, None, None, None] * dsd[None, :, :, None, None] * P)
    # Φ = <Pc, Mn K1>: uu
    PN = (Pc * (N @ K1)).sum((-1, -2))                                   # <Pc, N K1> [n]
    PNdK = (Pc[:, None] * (N @ dK1)).sum((-1, -2))                       # [n,3] <Pc, N dK1[c]>
    PMd2 = (Pc[:, None, None] * (Mn[:, None, None] @ d2K1)).sum((-1, -2))  # [n,3,3]
    Phi_uu = (g2 * PN)[:, None, None] * dsd + g1[:, None, None] * (ds[None, :, None] * PNdK[:, None, :] + ds[None, None, :] * PNdK[:, :, None]) + PMd2
    # uG: <Pc, g1 ds_c N (Jst dG) + Mn (w1 ds_c P + w2 ds_c a nᵀ + w1 (P e_c) nᵀ) dG>,  dG = 기저 E_(a,b)
    Bc = (w1[:, None, None, None] * ds[None, :, None, None] * P + w2[:, None, None, None] * ds[None, :, None, None] * aN[:, None]
          + w1[:, None, None, None] * PcN[None])                         # [n,3,3,3]
    Mc = g1[:, None, None, None] * ds[None, :, None, None] * (N @ Jst)[:, None] + Mn[:, None] @ Bc   # [n,3(c),3,3]: 계수 행렬 X 에서 <Pc, X dG>
    # <Pc, X E_ab> = (Xᵀ Pc)_ab  -> [n,3(c),9]
    Phi_uG = (Mc.transpose(-1, -2) @ Pc[:, None]).reshape(n, 3, 9)
    # J' (y 공간): u 방향 dJf = g1 ds_c N K1 + Mn dK1[c],  G 방향 dJf = Mn Jst E_ab
    dJu = g1[:, None, None, None] * ds[None, :, None, None] * (N @ K1)[:, None] + Mn[:, None] @ dK1     # [n,3,3,3]
    MJ = Mn @ Jst                                                        # [n,3,3]
    Eab = torch.eye(9, device=x.device, dtype=f8).reshape(9, 3, 3)
    dJG = MJ[:, None] @ Eab[None]                                        # [n,9,3,3]
    JJ = torch.cat([dJu.reshape(n, 3, 9), dJG.reshape(n, 9, 9)], 1).transpose(1, 2)   # [n,9(Jf),12(y)]
    Hy = JJ.transpose(1, 2) @ K @ JJ
    Hy[:, :3, :3] += Huu + Phi_uu
    Hy[:, :3, 3:] += Phi_uG
    Hy[:, 3:, :3] += Phi_uG.transpose(1, 2)
    # y -> 국소 12 (i,c): y_u[a] = Σ λ_i u_ia,  y_G[a,b] = Σ_i u_ia ∇λ_ib
    Lu = lm8[:, None, :, None] * I3[None, :, None, :]                    # [n,3(a),4(i),3(c)]
    LG = I3[None, :, None, None, :] * gl8.permute(0, 2, 1)[:, None, :, :, None]   # [n,3(a),3(b),4(i),3(c)] = δ_ac ∇λ_ib
    L = torch.cat([Lu.reshape(n, 3, 12), LG.reshape(n, 9, 12)], 1)      # [n,12(y),12]
    return L.transpose(1, 2) @ Hy @ L


_NC = [None]


def _near_core_run(*args):
    import os as _os
    if _NC[0] is None:
        _NC[0] = _near_core
        if _os.environ.get("AF_FEM_NEAR_COMPILE"):
            try:
                _NC[0] = torch.compile(_near_core, dynamic=True)
            except Exception as e:                                           # noqa: BLE001
                print(f"[fem_fast] 바닥 후보 컴파일 실패: {e}", flush=True)
    try:
        return _NC[0](*args)
    except Exception as e:                                                   # noqa: BLE001
        if _NC[0] is _near_core:
            raise
        print(f"[fem_fast] 바닥 후보 컴파일 실행 실패 -> 컴파일 없이: {type(e).__name__}: {str(e)[:200]}", flush=True)
        _NC[0] = _near_core
        return _near_core(*args)

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

    def solve(self, b, ncg, rtol, use_graph=True, x0=None):
        dev, dt = b.device, b.dtype
        x = torch.zeros_like(b)
        r = b.clone()
        if x0 is not None:                                   # warm start: s x0, s = x0ᵀb / x0ᵀA x0
            Ax = self.mv(x0)
            den = (x0 * Ax).sum()
            sc = torch.where(den > 0, (x0 * b).sum() / torch.where(den > 0, den, torch.ones_like(den)), torch.zeros_like(den))
            x, r = sc * x0, b - sc * Ax
        z = self.pre(r)
        self.S = S = dict(x=x, r=r, d=z.clone(), rz=(r * z).sum(), thr=(b * b).sum().clamp_min(1e-300) * (rtol * rtol),
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


# ---- 2 단계 전처리 (AF_FEM_PCG_PRE=add|mult, 실시간 판 선택지) ----
# 노드 격자를 agg x agg x agg 칸으로 묶고 (세계 고정 격자 -> 기하 집합), 묶음마다 강체 6 모드 (병진 3 + 회전 3) 로 거친 공간 P 를 만든다.
# 거친 행렬 Ac = Pᵀ A P 는 사면체 12x12 투영 헤시안에서 바로 (Qᵀ H_T Q, Q = 네 노드의 P 블록) 모아 조밀 행렬로, 역행렬을 한 번 만들어 둔다.
#   add : z = D⁻¹ r + P Ac⁻¹ Pᵀ r                                    (항상 대칭 양정치)
#   mult: z1 = ω D⁻¹ r, z2 = z1 + P Ac⁻¹ Pᵀ (r - A z1), z = z2 + ω D⁻¹ (r - A z2)   (대칭 V(1,1), SpMV 2 회 더)
# 전처리만 바뀐다 -- 푸는 식 (A p = g) 과 수렴 판정은 그대로.
class TwoLevel:
    def __init__(self, He, uq, npos, fixd, Dinv, mv, node_h, agg=3, mode="add", omega=0.5, f32=True):
        dev = He.device
        f8 = torch.float64
        M_ = npos.shape[0]
        T = He.shape[0]
        self.mode, self.omega, self.M, self.Dinv, self.mv = mode, float(omega), M_, Dinv, mv
        X = npos.to(f8)
        ijk = torch.floor((X - X.min(0).values) / (agg * node_h) + 1e-6).long()
        nk = ijk.max(0).values + 1
        key = (ijk[:, 0] * nk[1] + ijk[:, 1]) * nk[2] + ijk[:, 2]
        used = torch.zeros(M_, dtype=torch.bool, device=dev)
        used[uq.reshape(-1)] = True
        used &= ~fixd
        key = torch.where(used, key, torch.full_like(key, -1))
        uk, ag = torch.unique(key, return_inverse=True)
        off = 1 if bool((uk < 0).any()) else 0                               # -1 묶음 = 안 쓰는·고정 노드 (P 행 0)
        Na = int(uk.numel()) - off
        ag = ag - off
        cnt = torch.zeros(Na + 1, device=dev, dtype=f8).index_add_(0, (ag + 1).clamp_min(0), torch.ones(M_, device=dev, dtype=f8))[1:]
        cen = torch.zeros(Na + 1, 3, device=dev, dtype=f8).index_add_(0, (ag + 1).clamp_min(0), X)[1:] / cnt.clamp_min(1)[:, None]
        agc = ag.clamp_min(0)
        r = (X - cen[agc]) / node_h                                          # 회전 모드 크기를 병진과 맞춘다
        Sk = torch.zeros(M_, 3, 3, device=dev, dtype=f8)                     # Sk v = r × v
        Sk[:, 0, 1], Sk[:, 0, 2] = -r[:, 2], r[:, 1]
        Sk[:, 1, 0], Sk[:, 1, 2] = r[:, 2], -r[:, 0]
        Sk[:, 2, 0], Sk[:, 2, 1] = -r[:, 1], r[:, 0]
        P = torch.cat([torch.eye(3, device=dev, dtype=f8).expand(M_, 3, 3), -Sk], -1)   # [M,3,6]: 변위 = t + θ × r
        P = P * (ag >= 0).to(f8)[:, None, None]
        self.P, self.ag, self.Na = P, agc, Na
        Pt = P[uq]                                                           # [T,4,3,6]
        Q = torch.zeros(T, 4, 3, 4, 6, device=dev, dtype=f8)
        ar = torch.arange(4, device=dev)
        Q[:, ar, :, ar, :] = Pt.permute(1, 0, 2, 3)
        Q = Q.reshape(T, 12, 24)
        Hc = Q.transpose(1, 2) @ He @ Q                                      # [T,24,24]
        ci = (agc[uq].unsqueeze(-1) * 6 + torch.arange(6, device=dev)).reshape(T, 24)
        n6 = 6 * Na
        Ac = torch.zeros(n6 * n6, device=dev, dtype=f8).index_add_(0, (ci.unsqueeze(-1) * n6 + ci.unsqueeze(1)).reshape(-1), Hc.reshape(-1)).reshape(n6, n6)
        Ac = 0.5 * (Ac + Ac.t())
        dg = Ac.diagonal()
        Ac += torch.diag(torch.where(dg > 0, 1e-8 * dg, torch.ones_like(dg)))
        self.f32 = False
        if f32:
            Af = Ac.float()
            jd = Af.diagonal().mean()
            for jit in (0.0, 1e-6, 1e-5, 1e-4):                                # 전처리일 뿐이라 작은 대각 이동은 허용 (양정치 유지)
                L, info = torch.linalg.cholesky_ex(Af + jit * jd * torch.eye(n6, device=dev) if jit else Af)
                if int(info) == 0:
                    self.Ai = torch.cholesky_inverse(L)
                    self.f32, self.jit = True, jit
                    break
        if not self.f32:
            L = torch.linalg.cholesky(Ac)
            self.Ai = torch.cholesky_inverse(L)
        self.n6 = n6

    def coarse(self, r):
        rc = torch.zeros(self.Na, 6, device=r.device, dtype=r.dtype).index_add_(
            0, self.ag, (self.P.transpose(1, 2) @ r.reshape(self.M, 3, 1)).squeeze(-1))
        yc = (self.Ai @ rc.reshape(-1).to(self.Ai.dtype)).to(r.dtype).reshape(self.Na, 6)
        return (self.P @ yc[self.ag].unsqueeze(-1)).reshape(-1)

    def jac(self, r):
        return (self.Dinv @ r.reshape(self.M, 3, 1)).reshape(-1)

    def __call__(self, r):
        if self.mode == "add":
            return self.jac(r) + self.coarse(r)
        z1 = self.omega * self.jac(r)
        z2 = z1 + self.coarse(r - self.mv(z1))
        return z2 + self.omega * self.jac(r - self.mv(z2))


# ---- ELL 희소 곱 + CUDA 그래프 PCG (AF_FEM_PCG_ELL=1, 실시간 판 선택지) ----
# 조립 CSR (고정 자유도 행·열 0 + 대각 1 포함 -- 같은 행렬) 을 행마다 길이를 맞춘 ELL [3M, K] (모자란 칸은 열 = 자기 행, 값 0) 로 옮겨
# y = Σ_k val[:, k] x[col[:, k]] (gather + 곱 + 합, 원자 합 없음). 반복이 커널 띄우기 비용에 묶여 있어 (CSR 0.32 ms/회 vs 자료 20 MB)
# PCG 본체 ck 회를 CUDA 그래프 하나로 잡아 재생한다 (MFPcg 와 같은 본체·같은 판정). 색인은 CSR 패턴이 같으면 재사용.
_ELL = {}


def ell_from_csr(A, key=None):
    """key: 패턴이 같다는 호출자 표식 (예: CSR 캐시 객체 id + 프레임). 없으면 포인터·크기."""
    crow, col = A.crow_indices(), A.col_indices()
    key = (key, crow.data_ptr(), col.data_ptr(), int(col.numel()))
    if _ELL.get("key") != key:
        n = crow.numel() - 1
        ln = crow[1:] - crow[:-1]
        K = int(ln.max())
        row = torch.repeat_interleave(torch.arange(n, device=col.device), ln)
        pos = torch.arange(col.numel(), device=col.device) - crow[:-1][row]
        flat = row * K + pos
        ecol = torch.arange(n, device=col.device).unsqueeze(1).expand(n, K).contiguous().reshape(-1)
        ecol[flat] = col
        _ELL.clear()
        _ELL.update(key=key, n=n, K=K, flat=flat, col=ecol.reshape(n, K))
    E = _ELL
    val = torch.zeros(E["n"] * E["K"], device=col.device, dtype=A.values().dtype)
    val[E["flat"]] = A.values()
    return E["col"], val.reshape(E["n"], E["K"])


class EllPcg(MFPcg):
    def __init__(self, A, Dinv, M_, ck=50, key=None):
        self.ecol, self.eval_ = ell_from_csr(A, key)
        self.Dinv, self.M, self.ck = Dinv, M_, int(ck)
        self.graph = None

    def mv(self, v):
        return (self.eval_ * v[self.ecol]).sum(1)

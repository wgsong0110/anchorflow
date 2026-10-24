"""XPBD 한 프레임 -- 우리 표현(세계 고정 Kuhn 사면체 격자, 입자 = 셀 안 barycentric, 입자 상태 x·v·F) 그대로 (사용자 10-24).

망/증분 포텐셜 최적화 대신 한 프레임을 XPBD (Macklin et al. 2016, small steps 2019) 로 진행한다.

  격자    프레임 시작 입자 위치로 train_deform 과 같은 격자 (SX.grid_for_nodes, h_fix = --node_h, hz = 1/√6) ->
          점유 사면체의 꼭짓점 = XPBD 입자(노드). 노드 질량 M_i = Σ_p λ_ip m_p (관성 예측자 node_inertial 과 같은 가중),
          노드 초기 속도 v_i = Σ λ m v / M_i (같은 P2G).
  구속    사면체 c 마다 증분 포텐셜 탄성과 같은 fixed corotated (phys_resid jelly FCR, ψ = μ‖F−R‖² + λ/2 (J−1)², 씬 E·ν):
            회전  C_R = ‖F − R‖_F, R = polar(F),  컴플라이언스 α_R = 1/(2μV)
            부피  C_V = det F − 1,                 컴플라이언스 α_V = 1/(λV)
          (½C_R²/α_R + ½C_V²/α_V = V ψ_FCR). 셀 안 입자마다 F_p 가 다르므로 셀 사상 J (= D_s D_m⁻¹, train_deform 의 I + ∇u)
          에 대해 Σ_p V_p ψ(J F_p) 를 셀 하나의 두 구속으로 **정확히** 모은다 (rot="particle", 기본):
            C_R = sqrt(Σ_{p∈c} V_p ‖J F_p − R_p‖² / V),  R_p = polar(J F_p)   (모든 F_p 가 같으면 ‖JF − R(JF)‖)
            ∂C_R/∂J = Σ V_p (J F_p − R_p) F_pᵀ / (V C_R) -- 극분해 미분 dR 을 빼는 표준 corotated 꼴이지만 **정확한** 미분이다
                   ((J F_p − R_p) : dR_p = 0: R_pᵀ J F_p 가 대칭이고 dR = R Ω, Ω 반대칭)
            C_V = sqrt(m₂) det J − m₁/sqrt(m₂),  m_k = Σ V_p (det F_p)^k / V   (Σ V_p (det J det F_p − 1)² 와 상수만 다르다 -- 정확)
          (비교용 rot="cell": 셀 공통 회전 R_c = polar(J M̄), C_R² = ‖J M̄ − R_c‖² + tr(J (B̄ − M̄M̄ᵀ) Jᵀ) -- 셀 안 F 가 다르면 상계.
           격자를 매 프레임 다시 짜면 셀 안 F 의 섞임이 바뀌어 이 상계가 에너지를 밀어 넣는다 (lego 진단 diag_9cc3478 에서 접촉 뒤 발산))
          ∂C/∂노드 = ∂C/∂J · D_m⁻ᵀ (열 j = 노드 j+1, 노드 0 = −합).
          V = Σ_{p∈c} V_p⁰ (입자 초기 부피 = 증분 포텐셜 탄성항과 같은 부피; 꽉 찬 셀이면 격자 사면체 부피와 같다).
          두 구속은 사면체마다 2x2 로 **함께** 푼다 (결합 풀이). C_R < 1e-12 인 셀은 회전 구속을 건너뛴다 (기울기 0/0).
  풀이    가우스-자이델: 셀 정수좌표 홀짝(8) x Kuhn 순열(6) = 48 색. 같은 색 사면체는 꼭짓점을 공유하지 않아 한꺼번에 푼다.
  바닥    cfg 의 surface_collider 평면. 사면체마다 그 안 입자 중 가장 깊은 것 하나의 비침투 구속 n·(Σλ x − p) ≥ 0
          (하드, α = 0) + 정적 마찰 (Müller 2020 식: 이번 서브스텝 접선 변위 |d_t| < μ_s · 침투 깊이이면 d_t 를 지운다).
          μ_s = inf (기본) 이면 닿은 입자는 접선으로 움직이지 않는다 = PG sticky / 기본 실행의 미끄러짐 방지 매핑과 같은 뜻.
  고정    --cfg_bc 의 노드 하드 구속 (cuboid 등): 역질량 0, 위치 = 시작 + v_bc τ.
  서브스텝 N, 반복 K, λ 는 서브스텝마다 0, 속도 = (x − x_prev)/Δt_s, 감쇠 없음 (vdamp 를 주면 서브스텝마다 곱한다).
  끝      입자 x = Σ λ x_i, v = Σ λ v_i (마지막 서브스텝 노드 속도), F_p ← J_c F_p (누적).
"""
from __future__ import annotations

import math

import torch

from . import simplex as SX

HZ = SX.HZ_CUBE
# Kuhn 순열 (perm[0], perm[1]) -> 0..5
_PMAP = {(0, 1): 0, (0, 2): 1, (1, 0): 2, (1, 2): 3, (2, 0): 4, (2, 1): 5}


def lame(E, nu):
    """phys_resid.lame 과 같은 Lamé 값 (μ, λ)."""
    mu = float(E) / (2.0 * (1.0 + float(nu)))
    la = float(E) * float(nu) / ((1.0 + float(nu)) * (1.0 - 2.0 * float(nu)))
    return mu, la


def polar_R(A):
    """극분해 회전 (SVD, det < 0 이면 마지막 특이벡터 부호를 뒤집어 det R = +1) [T,3,3]."""
    U, _S, Vh = torch.linalg.svd(A)
    d = torch.sign(torch.linalg.det(U @ Vh))
    d = torch.where(d == 0, torch.ones_like(d), d)
    U = torch.cat([U[..., :2], U[..., 2:] * d[:, None, None]], -1)
    return U @ Vh


def psi_fcr(F, mu, la):
    """ψ = μ‖F − R‖² + λ/2 (det F − 1)² (float64, 특이값 자르기 없음)."""
    R = polar_R(F)
    return mu * ((F - R) ** 2).sum((-1, -2)) + 0.5 * la * (torch.linalg.det(F) - 1.0) ** 2


def cof_cols(J):
    """∂det J/∂J (열 c_j 의 외적) [T,3,3]."""
    c0, c1, c2 = J[:, :, 0], J[:, :, 1], J[:, :, 2]
    return torch.stack([torch.cross(c1, c2, dim=-1), torch.cross(c2, c0, dim=-1), torch.cross(c0, c1, dim=-1)], -1)


def cell_constraints_particle(J, pinv, T, Fp, Vp, V, m1, m2):
    """입자별 회전 구속 값 (C_R, C_V) -- Solver._elastic rot="particle" 과 같은 식 (시험용)."""
    A = J[pinv] @ Fp
    E_ = A - polar_R(A)
    CR = (torch.zeros(T, dtype=J.dtype).index_add_(0, pinv, Vp * (E_ * E_).sum((1, 2))) / V).sqrt()
    s2 = m2.clamp_min(1e-300).sqrt()
    return CR, s2 * torch.linalg.det(J) - m1 / s2


def cell_constraints(J, B, Mb, m1, m2):
    """셀 구속 값 (C_R, C_V) -- Solver._elastic rot="cell" 과 같은 식 (시험·진단용)."""
    JM = J @ Mb
    Rc = polar_R(JM)
    Cv = B                                           # aggregates 의 둘째 값 = 셀 안 F 의 공분산 B̄ − M̄M̄ᵀ (같으면 0)
    CR = (((JM - Rc) ** 2).sum((1, 2)) + (J * (J @ Cv)).sum((1, 2)).clamp_min(0.0)).sqrt()
    s2 = m2.clamp_min(1e-300).sqrt()
    return CR, s2 * torch.linalg.det(J) - m1 / s2


def build_lattice(x, node_h, hz_ratio=HZ):
    """프레임 시작 입자 위치 -> 격자 구조 (train_deform node_feats 와 같은 격자·locate·active_nodes)."""
    lo, lat, nn = SX.grid_for_nodes(x, 32, hz_ratio=hz_ratio, h_fix=node_h)
    idx, lam, aux = SX.locate(x, lo, lat, nn)
    rows, uniq = SX.active_nodes(idx)
    npos = SX.node_pos(lo, lat, nn, uniq)
    tid = SX.tet_id(lo, lat, nn, aux)
    utid, pinv = torch.unique(tid, return_inverse=True)
    T = int(utid.numel())
    rep = torch.empty(T, dtype=torch.long, device=x.device).scatter_(0, pinv, torch.arange(x.shape[0], device=x.device))
    tets = rows[rep]                                     # [T,4] 사슬 순서 v0..v3 (입자 rows 와 같은 순서)
    ci, rank, perm = aux
    par = (ci[rep] % 2)
    pk = perm[rep, 0] * 3 + perm[rep, 1]
    lut = torch.full((9,), -1, dtype=torch.long, device=x.device)
    for (p0, p1), k in _PMAP.items():
        lut[p0 * 3 + p1] = k
    col = (par[:, 0] * 4 + par[:, 1] * 2 + par[:, 2]) * 6 + lut[pk]
    if bool((lut[pk] < 0).any()):
        raise RuntimeError("Kuhn 순열 색 실패")
    groups = [torch.nonzero(col == c).squeeze(1) for c in range(48)]
    groups = [g for g in groups if g.numel()]
    # 색마다 그 사면체에 든 입자와 색 안 사면체 번호 (입자별 회전 구속용)
    colp = col[pinv]
    loc = torch.full((T,), -1, dtype=torch.long, device=x.device)
    pgroups = []
    for g in groups:
        loc[g] = torch.arange(g.numel(), device=x.device)
        pc = torch.nonzero(colp == col[g[0]]).squeeze(1)
        pgroups.append((pc, loc[pinv[pc]]))
    return dict(lo=lo, lat=lat, nn=nn, rows=rows, lam=lam, aux=aux, npos=npos, pinv=pinv, tets=tets, col=col, groups=groups,
                pgroups=pgroups, T=T)


def check_coloring(tets, groups):
    """같은 색 사면체끼리 꼭짓점이 겹치지 않는지 -> 겹친 색 수."""
    bad = 0
    for g in groups:
        v = tets[g].reshape(-1)
        if torch.unique(v).numel() != v.numel():
            bad += 1
    return bad


def tet_rest(npos, tets, dtype=torch.float64):
    X = npos[tets].to(dtype)
    Dm = (X[:, 1:] - X[:, :1]).transpose(1, 2)            # 열 = X_j − X_0
    return torch.linalg.inv(Dm), torch.linalg.det(Dm).abs() / 6.0


def tet_J(xs, tets, DmI):
    X = xs[tets]
    Ds = (X[:, 1:] - X[:, :1]).transpose(1, 2)
    return Ds @ DmI


def aggregates(pinv, T, Vp, F):
    """셀 모음 V, 공분산 B̄ − M̄M̄ᵀ = Σ V (F − M̄)(F − M̄)ᵀ / V, M̄ = Σ V F / V, m1, m2 (float64)."""
    dt = torch.float64
    Vp = Vp.to(dt); F = F.to(dt)
    V = torch.zeros(T, device=F.device, dtype=dt).index_add_(0, pinv, Vp)
    a = torch.linalg.det(F)
    m1 = torch.zeros(T, device=F.device, dtype=dt).index_add_(0, pinv, Vp * a)
    m2 = torch.zeros(T, device=F.device, dtype=dt).index_add_(0, pinv, Vp * a * a)
    Mb = torch.zeros(T, 3, 3, device=F.device, dtype=dt).index_add_(0, pinv, Vp[:, None, None] * F)
    Vc = V.clamp_min(1e-300)
    Mb = Mb / Vc[:, None, None]
    # B̄ − M̄M̄ᵀ 를 편차로 바로 모은다 (B̄ 와 M̄M̄ᵀ 를 따로 빼면 같은 F 셀에서 반올림 1e-16 이 sqrt 로 1e-8 이 된다)
    dF = F - Mb[pinv]
    Cv = torch.zeros(T, 3, 3, device=F.device, dtype=dt).index_add_(0, pinv, Vp[:, None, None] * (dF @ dF.transpose(1, 2)))
    return V, Cv / Vc[:, None, None], Mb, m1 / Vc, m2 / Vc


def p2g(rows, lam, m, v, M_nodes):
    dt = torch.float64
    wl = lam.to(dt) * m.to(dt).unsqueeze(1)
    M = torch.zeros(M_nodes, device=v.device, dtype=dt).index_add_(0, rows.reshape(-1), wl.reshape(-1))
    P = torch.zeros(M_nodes, 3, device=v.device, dtype=dt)
    for j in range(rows.shape[1]):
        P.index_add_(0, rows[:, j], wl[:, j:j + 1] * v.to(dt))
    return M, P / M.clamp_min(1e-300).unsqueeze(1)


def planes_of(cfg):
    out = []
    for bc in (cfg.get("boundary_conditions") or []):
        if bc.get("type") == "surface_collider" and float(bc.get("start_time", 0.0)) <= 0.0:
            n = torch.tensor(bc["normal"], dtype=torch.float64)
            out.append((torch.tensor(bc["point"], dtype=torch.float64), n / n.norm(), str(bc.get("surface", "sticky"))))
    return out


class Solver:
    """한 프레임 XPBD. S = build_lattice(...) 결과 + 노드 상태."""

    def __init__(self, mu, la, n_sub, n_iter, planes=(), mu_s=math.inf, vdamp=1.0, sweeps=10, tol=0.0, rot="particle"):
        self.mu, self.la = float(mu), float(la)
        self.rot = rot                                     # "particle" (정확, 기본) | "cell" (셀 공통 회전 상계, 비교용)
        self.sweeps, self.tol = int(sweeps), float(tol)    # 바닥 쓸기 최대 횟수 (반복마다), 침투 허용 [m]
        self.N, self.K = int(n_sub), int(n_iter)
        self.planes = list(planes)
        self.mu_s = float(mu_s)
        self.vdamp = float(vdamp)                 # 서브스텝마다 노드 속도에 곱함 (1 = 감쇠 없음)

    def _elastic(self, xs, w, tets, DmI, V, B, Mb, m1, m2, lR, lV, g, dts2, pg=None, Fp=None, Vp=None):
        r = tets[g]
        Di = DmI[g]
        J = tet_J(xs, r, Di)
        Tg = g.numel()
        if self.rot == "particle":
            # 입자별: C_R² = Σ V_p ‖J F_p − R_p‖² / V, ½∂C_R²/∂J = Σ V_p (J F_p − R_p) F_pᵀ / V
            pc, lc = pg
            Fq = Fp[pc]
            A = J[lc] @ Fq
            E_ = A - polar_R(A)
            vq = Vp[pc]
            Vg0 = V[g].clamp_min(1e-300)
            ss = torch.zeros(Tg, device=xs.device, dtype=xs.dtype).index_add_(0, lc, vq * (E_ * E_).sum((1, 2))) / Vg0
            GG = torch.zeros(Tg, 3, 3, device=xs.device, dtype=xs.dtype).index_add_(0, lc, vq[:, None, None] * (E_ @ Fq.transpose(1, 2))) / Vg0[:, None, None]
            CR = ss.clamp_min(0.0).sqrt()
        else:
            Mg = Mb[g]
            JM = J @ Mg
            Rc = polar_R(JM)
            # C_R² = ‖J M̄ − R_c‖² + tr(J (B̄ − M̄M̄ᵀ) Jᵀ), ½∂C_R²/∂J = (JM̄ − R_c)M̄ᵀ + J(B̄ − M̄M̄ᵀ)   (B = 공분산, aggregates)
            JC = J @ B[g]
            CR = (((JM - Rc) ** 2).sum((1, 2)) + (J * JC).sum((1, 2)).clamp_min(0.0)).sqrt()
            GG = (JM - Rc) @ Mg.transpose(1, 2) + JC
        live = CR > 1e-12
        GR = torch.where(live[:, None, None], GG / CR.clamp_min(1e-300)[:, None, None],
                         torch.zeros_like(J)) @ Di.transpose(1, 2)       # ∂C_R/∂D_s, 열 j = 노드 j+1
        s2 = m2[g].clamp_min(1e-300).sqrt()
        detJ = torch.linalg.det(J)
        CV = s2 * detJ - m1[g] / s2
        GV = (s2[:, None, None] * cof_cols(J)) @ Di.transpose(1, 2)
        gR = GR.transpose(1, 2)                                   # [T,3(노드1..3),3]
        gV = GV.transpose(1, 2)
        gR = torch.cat([-gR.sum(1, keepdim=True), gR], 1)         # [T,4,3]
        gV = torch.cat([-gV.sum(1, keepdim=True), gV], 1)
        wr = w[r]
        Vg = V[g].clamp_min(1e-300)
        aR = 1.0 / (2.0 * self.mu * Vg * dts2)
        aV = 1.0 / (self.la * Vg * dts2)
        sRR = (wr * (gR * gR).sum(-1)).sum(1) + aR
        sVV = (wr * (gV * gV).sum(-1)).sum(1) + aV
        sRV = (wr * (gR * gV).sum(-1)).sum(1)
        bR = torch.where(live, -CR - aR * lR[g], torch.zeros_like(CR))
        bV = -CV - aV * lV[g]
        det = (sRR * sVV - sRV * sRV).clamp_min(1e-300)
        dR = (sVV * bR - sRV * bV) / det
        dV = (sRR * bV - sRV * bR) / det
        lR[g] += dR
        lV[g] += dV
        dx = wr.unsqueeze(-1) * (gR * dR[:, None, None] + gV * dV[:, None, None])
        xs.index_add_(0, r.reshape(-1), dx.reshape(-1, 3))

    def _contact(self, xs, xprev, w, S, st):
        """바닥 접촉: 사면체마다 가장 깊은 입자 하나를 사영하는 쓸기를, 침투 입자가 없어지거나 self.sweeps 번이 될 때까지
        (한 쓸기는 각 사면체의 최심 입자만 고친다 -- 같은 사면체의 다음으로 깊은 입자는 다음 쓸기에서)."""
        for _sw in range(self.sweeps):
            if not self._contact_sweep(xs, xprev, w, S, st):
                break
            st["contact_sweeps"] = max(st.get("contact_sweeps", 0), _sw + 1)

    def _contact_sweep(self, xs, xprev, w, S, st):
        if not self.planes:
            return False
        hit = False
        rows, lam, pinv, tets, col = S["rows"], S["lam64"], S["pinv"], S["tets"], S["col"]
        T = S["T"]
        for (p0, n, _kind) in self.planes:
            p0 = p0.to(xs.device); n = n.to(xs.device)
            xp = (lam.unsqueeze(-1) * xs[rows]).sum(1)
            s = (xp - p0) @ n
            smin = torch.full((T,), math.inf, device=xs.device, dtype=xs.dtype).scatter_reduce_(0, pinv, s, "amin")
            act = smin < -self.tol
            if not bool(act.any()):
                continue
            hit = True
            cand = (s == smin[pinv]) & act[pinv]
            ci = torch.nonzero(cand).squeeze(1)
            pst = torch.full((T,), -1, device=xs.device, dtype=torch.long).scatter_reduce_(0, pinv[ci], ci, "amax")
            A = torch.nonzero(pst >= 0).squeeze(1)
            st["contact_tets"] = max(st.get("contact_tets", 0), int(A.numel()))
            for c in torch.unique(col[A]).tolist():
                Ac = A[col[A] == c]
                pp = pst[Ac]
                r = tets[Ac]
                lp = lam[pp]                                    # rows[pp] == tets[Ac] (같은 사면체, 같은 사슬 순서)
                X = xs[r]
                q = (lp.unsqueeze(-1) * X).sum(1)
                sc = (q - p0) @ n
                neg = sc < 0
                wr = w[r]
                W = (wr * lp * lp).sum(1)
                ok = neg & (W > 0)
                Wc = W.clamp_min(1e-300)
                dl = torch.where(ok, -sc / Wc, torch.zeros_like(sc))
                corr = (wr * lp).unsqueeze(-1) * n.view(1, 1, 3) * dl[:, None, None]
                d = q - (lp.unsqueeze(-1) * xprev[r]).sum(1)
                dtan = d - (d @ n).unsqueeze(-1) * n
                stick = ok & (dtan.norm(dim=-1) < self.mu_s * (-sc))
                corr = corr - torch.where(stick[:, None, None], (wr * lp).unsqueeze(-1) * dtan.unsqueeze(1) / Wc[:, None, None],
                                          torch.zeros_like(corr))
                xs.index_add_(0, r.reshape(-1), corr.reshape(-1, 3))
        return hit

    def frame(self, S, x0n, v0n, w, g, dt, pin=None, pin_vel=None, V=None, B=None, Mb=None, m1=None, m2=None, DmI=None,
              Fp=None, Vp=None):
        """노드 시작 위치 x0n, 속도 v0n (float64) -> (끝 위치, 끝 속도, 통계)."""
        tets, groups = S["tets"], S["groups"]
        T = S["T"]
        dts = float(dt) / self.N
        dts2 = dts * dts
        xs = x0n.clone()
        vs = v0n.clone()
        free = w > 0
        gv = torch.as_tensor(g, device=xs.device, dtype=xs.dtype)
        st = {}
        lD = torch.zeros(T, device=xs.device, dtype=xs.dtype)
        lH = torch.zeros_like(lD)
        for s in range(self.N):
            xprev = xs.clone()
            vs = vs + dts * gv * free.unsqueeze(-1).to(vs.dtype)
            xs = xs + dts * vs
            if pin is not None:
                xs[pin] = x0n[pin] + (s + 1) * dts * pin_vel[pin]
            lD.zero_(); lH.zero_()
            for _k in range(self.K):
                for gi, pg in zip(groups, S.get("pgroups") or [None] * len(groups)):
                    self._elastic(xs, w, tets, DmI, V, B, Mb, m1, m2, lD, lH, gi, dts2, pg=pg, Fp=Fp, Vp=Vp)
                self._contact(xs, xprev, w, S, st)
            vs = (xs - xprev) / dts
            if self.vdamp != 1.0:
                vs = vs * self.vdamp
        st["lamR_absmax"] = float(lD.abs().max()) if T else 0.0
        return xs, vs, st


def advance_frame(solver, x, v, F, m, Vp, node_h, g, dt, bc_fn=None, hz_ratio=HZ, m_floor=0.05):
    """입자 상태 (x, v, F) 한 프레임 -> (x2, v2, F2, J_p, 통계). 계산은 float64, 반환도 float64.

    bc_fn(npos, rows, lam) -> (노드 마스크, 노드 속도) 또는 (None, None) -- --cfg_bc 노드 하드 구속.
    질량이 중앙값의 m_floor 배 미만인 노드(입자가 가장자리에만 걸친 꼭짓점)는 그 바닥값을 역질량에만 쓴다 (관성 예측·속도는 P2G 그대로).
    바닥값이 없으면 그런 노드는 역질량이 커서, 입자 접촉 사영이 λ 작은 꼭짓점을 1/λ 배로 크게 옮겨 셀이 뒤집힌다.
    """
    dt64 = torch.float64
    x = x.to(dt64); v = v.to(dt64); F = F.to(dt64)
    S = build_lattice(x, node_h, hz_ratio)
    S["lam64"] = S["lam"].to(dt64)
    npos = S["npos"].to(dt64)
    Mn = npos.shape[0]
    M, vn = p2g(S["rows"], S["lam"], m, v, Mn)
    Mf = M.clamp_min(float(M.median()) * m_floor)
    w = 1.0 / Mf
    pin = pv = None
    if bc_fn is not None:
        pin, pv = bc_fn(npos, S["rows"], S["lam64"])
        if pin is not None:
            w = torch.where(pin, torch.zeros_like(w), w)
            vn = torch.where(pin.unsqueeze(-1), pv.to(dt64), vn)
            pv = pv.to(dt64)
    DmI, Vt = tet_rest(npos, S["tets"])
    V, B, Mb, m1, m2 = aggregates(S["pinv"], S["T"], Vp, F)
    xs, vs, st = solver.frame(S, npos, vn, w, g, dt, pin=pin, pin_vel=pv, V=V, B=B, Mb=Mb, m1=m1, m2=m2, DmI=DmI,
                              Fp=F, Vp=Vp.to(dt64))
    lam = S["lam64"]; rows = S["rows"]
    x2 = (lam.unsqueeze(-1) * xs[rows]).sum(1)
    v2 = (lam.unsqueeze(-1) * vs[rows]).sum(1)
    Jt = tet_J(xs, S["tets"], DmI)
    detJ = torch.linalg.det(Jt)
    Jp = Jt[S["pinv"]]
    F2 = Jp @ F
    st.update(nodes=int(Mn), tets=int(S["T"]), colors=len(S["groups"]), pinned=int(pin.sum()) if pin is not None else 0,
              detJ_min=float(detJ.min()), tets_inv=int((detJ <= 0).sum()),
              fill_ratio_mean=float((V / Vt).mean()), light_nodes=int((M < float(M.median()) * m_floor).sum()))
    return x2, v2, F2, Jp, st

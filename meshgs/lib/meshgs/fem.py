"""사면체 FEM. 탄성 전용(VR-GS 계열)과 소성(von Mises / Drucker-Prager) 둘 다.

소성은 곱셈 분해 F = F_e F_p 와 리턴 매핑으로 넣는다 -- PhysGaussian 의 MPM 이
쓰는 것과 같은 구성식이다. 그래야 "구성식이 달라서 실패했다"는 반박이 막힌다.
남는 차이는 **고정 연결성 사면체 메시**뿐이고, 그것이 이 시험의 대상이다.

적분은 준암시적(semi-implicit) 오일러다. 큰 소성 변형에서 요소가 뒤집히면
det F < 0 이 되고 그 지점부터 힘이 발산하는데, 그 발생 자체가 측정 대상이라
인공적으로 감싸지 않는다.
"""
from __future__ import annotations

import torch


def _svd3(F):
    U, S, Vh = torch.linalg.svd(F)
    det = torch.linalg.det(U @ Vh)
    neg = det < 0
    if neg.any():
        U = U.clone(); S = S.clone()
        U[neg, :, -1] = -U[neg, :, -1]
        S[neg, -1] = -S[neg, -1]
    return U, S, Vh, neg


def pk1_neohookean(F, mu, lam):
    """안정화 neo-Hookean 의 1st Piola-Kirchhoff."""
    J = torch.linalg.det(F)
    Finv_T = torch.linalg.inv(F).transpose(-1, -2)
    return mu * (F - Finv_T) + lam * torch.log(J.clamp(min=1e-6))[:, None, None] * Finv_T


def return_map(F, kind, mu, yield_stress=None, friction_angle=None,
               diag=None):
    """소성 리턴 매핑.

    돌려주는 것은 (F_e, 소성 발생 여부, 보정 행렬 C) 이고
    C = F_trial^{-1} F_e 다. 리턴 매핑은 U, V 를 건드리지 않고 특이값만 바꾸므로
    C = V diag(s_new/s) V^T 로 **역행렬 없이** 얻는다. F_trial 을 직접 역행렬로
    풀면 요소가 찌그러진 순간 특이행렬이 되어 터진다(실측).
    """
    if kind == "none":
        eye = torch.eye(3, device=F.device).expand_as(F)
        return F, torch.zeros(F.shape[0], dtype=torch.bool, device=F.device), eye
    U, S_raw, Vh, refl = _svd3(F)
    S = S_raw.clamp(min=1e-4)
    eps = torch.log(S)
    if kind == "von_mises":                       # 금속
        tr = eps.sum(-1, keepdim=True)
        dev = eps - tr / 3.0
        nrm = dev.norm(dim=-1, keepdim=True)
        thr = yield_stress / (2.0 * mu)
        over = (nrm > thr).squeeze(-1)
        scale = torch.where(nrm > thr, thr / nrm.clamp(min=1e-12),
                            torch.ones_like(nrm))
        eps_e = dev * scale + tr / 3.0
    elif kind == "drucker_prager":                # 모래
        tr = eps.sum(-1, keepdim=True)
        dev = eps - tr / 3.0
        nrm = dev.norm(dim=-1, keepdim=True)
        sin_phi = torch.sin(torch.tensor(friction_angle * 3.14159265 / 180.0,
                                         device=F.device))
        alpha = (2.0 / 3.0) ** 0.5 * 2.0 * sin_phi / (3.0 - sin_phi)
        eps_e = eps.clone()
        expand = (tr > 0).squeeze(-1)             # 인장이면 응력 0 (모래는 못 당긴다)
        eps_e[expand] = 0.0
        dp = nrm.squeeze(-1) + alpha * tr.squeeze(-1)
        shear = (~expand) & (dp > 0)
        if shear.any():
            s = (nrm.squeeze(-1)[shear] - dp[shear]) / nrm.squeeze(-1)[shear].clamp(min=1e-12)
            eps_e[shear] = dev[shear] * s.unsqueeze(-1) + tr[shear] / 3.0
        over = expand | shear
    else:
        raise ValueError(kind)
    S_new = torch.exp(eps_e)
    Fe = U @ torch.diag_embed(S_new) @ Vh
    C = Vh.transpose(-1, -2) @ torch.diag_embed(S_new / S) @ Vh
    if diag is not None:
        # 진단용 중간값. 반사 보정이 만든 음수 특이값을 clamp 가 지우는지,
        # S_new/S 가 어디까지 커지는지 보려면 이 둘을 원본째로 봐야 한다.
        diag.update(S_raw=S_raw, S=S, S_new=S_new, refl=refl, over=over, C=C)
    return Fe, over, C


class TetFEM:
    def __init__(self, V0, T, density=200.0, E=1e5, nu=0.3,
                 plastic="none", yield_stress=1e2, friction_angle=25.0,
                 damping=2.0):
        self.V0, self.T = V0, T
        self.mu = E / (2 * (1 + nu))
        self.lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        self.plastic, self.ys, self.phi = plastic, yield_stress, friction_angle
        self.damping = damping
        D = torch.stack([V0[T][:, 0] - V0[T][:, 3], V0[T][:, 1] - V0[T][:, 3],
                         V0[T][:, 2] - V0[T][:, 3]], -1)
        self.Dm_inv = torch.linalg.inv(D)
        self.vol = torch.linalg.det(D).abs() / 6.0
        self.mass = torch.zeros(V0.shape[0], device=V0.device)
        self.mass.index_add_(0, T.reshape(-1),
                             (density * self.vol / 4.0).repeat_interleave(4))
        self.mass = self.mass.clamp(min=1e-8)
        # 소성 변형 구배 (누적). 탄성만 쓸 때는 항등원 그대로다.
        self.Fp_inv = torch.eye(3, device=V0.device).expand(T.shape[0], 3, 3).clone()

    def deform_grad(self, V):
        D = torch.stack([V[self.T][:, 0] - V[self.T][:, 3],
                         V[self.T][:, 1] - V[self.T][:, 3],
                         V[self.T][:, 2] - V[self.T][:, 3]], -1)
        return D @ self.Dm_inv

    def step(self, V, vel, dt, fixed=None, gravity=None, diag=None):
        F_total = self.deform_grad(V)
        F_e_trial = F_total @ self.Fp_inv
        F_e = F_e_trial
        if self.plastic != "none":
            F_new, _, C = return_map(F_e, self.plastic, self.mu, self.ys,
                                     self.phi, diag=diag)
            self.Fp_inv = self.Fp_inv @ C          # 역행렬 없이 누적
            F_e = F_new
        P = pk1_neohookean(F_e, self.mu, self.lam)
        if diag is not None:
            diag.update(F_total=F_total, F_e_trial=F_e_trial, F_e=F_e, P=P,
                        Fp_inv=self.Fp_inv)
        H = -(self.vol[:, None, None] * P) @ self.Dm_inv.transpose(-1, -2)
        f = torch.zeros_like(V)
        f.index_add_(0, self.T[:, 0], H[:, :, 0])
        f.index_add_(0, self.T[:, 1], H[:, :, 1])
        f.index_add_(0, self.T[:, 2], H[:, :, 2])
        f.index_add_(0, self.T[:, 3], -(H[:, :, 0] + H[:, :, 1] + H[:, :, 2]))
        if gravity is not None:
            f = f + self.mass[:, None] * gravity
        a = f / self.mass[:, None]
        if diag is not None:
            diag.update(force=f, accel=a)
        vel = (vel + dt * a) * (1.0 - self.damping * dt)
        if fixed is not None:
            vel = vel * (~fixed).float()[:, None]
        return V + dt * vel, vel

    def quality(self, V):
        """뒤집힘·품질 진단. (뒤집힌 비율, 최소 det F, 최대 종횡비)"""
        F = self.deform_grad(V)
        det = torch.linalg.det(F)
        S = torch.linalg.svdvals(F)
        ar = (S[:, 0] / S[:, 2].clamp(min=1e-12))
        return float((det <= 0).float().mean()), float(det.min()), float(ar.max())

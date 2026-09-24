"""Backward Euler 의 증분 포텐셜(incremental potential). Phase 2 학습의 목적함수.

    x^{n+1} = argmin_x  E(x),
    E(x) = sum_p m_p/(2h^2) |x_p - xtil_p|^2 + sum_p V_p Psi(F_p(x)) - sum_p m_p g.x_p
    xtil = x^n + h v^n

정류점이 곧 backward Euler 잔차 r = M(x-xtil)/h^2 + grad Psi - f_ext = 0 이므로,
E 를 낮추는 것과 잔차를 0 으로 보내는 것이 같다. 손실로는 E 를 쓴다 -- |r|^2 은
강성 때문에 조건수가 사납고, E 는 최소점이 정확히 구하려는 해다.

구성모델은 **PhysGaussian 의 정의를 그대로** 옮겼다 (mpm_solver_warp/mpm_utils.py):
  jelly(0)  고정 코로테이션 FCR   tau = 2mu(F-R)F^T + lam J(J-1) I
  metal(1)  StVK-Hencky + von Mises 사영
  foam(3)   StVK-Hencky + 점소성 사영 (plastic_viscosity)
셋 다 에너지가 **특이값만의 함수**라, SVD 대신 C = F^T F 의 고유값으로 간다.
eigvalsh 의 역전파는 고유벡터 차이(1/(li-lj))를 타지 않아 겹친 특이값에서도
안전하다 -- 회전 대칭인 배치에서 SVD 역전파가 터지는 것을 피하는 길이다.
"""
import math

import torch

__all__ = ["lame", "psi_of", "plastic_step", "ip_energy", "residual",
           "smooth_noise", "mat_name"]


def lame(E, nu):
    E, nu = float(E), float(nu)
    return E / (2.0 * (1.0 + nu)), E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))


def mat_name(cfg):
    return str(cfg.get("material", "jelly"))


def _sig(F):
    """특이값 [N,3] (오름차순). C = F^T F 의 고유값으로 구한다."""
    C = F.transpose(-1, -2) @ F
    return torch.linalg.eigvalsh(C.double()).clamp_min(1e-12).sqrt().to(F.dtype)


def _psi_fcr(sig, mu, lam):
    J = sig.prod(-1)
    return mu * ((sig - 1.0) ** 2).sum(-1) + 0.5 * lam * (J - 1.0) ** 2


def _psi_hencky(eps, mu, lam):
    return mu * (eps ** 2).sum(-1) + 0.5 * lam * (eps.sum(-1) ** 2)


def _vm_project(eps, mu, lam, ys):
    """PG von_mises_return_mapping 의 주응력 공간 판본."""
    tr = eps.sum(-1, keepdim=True)
    tau = 2.0 * mu * eps + lam * tr
    dev = tau - tau.sum(-1, keepdim=True) / 3.0
    over = dev.norm(dim=-1, keepdim=True) > ys
    ehat = eps - tr / 3.0
    n = ehat.norm(dim=-1, keepdim=True) + 1e-6
    dg = (n - ys / (2.0 * mu)).clamp_min(0.0)
    return torch.where(over, eps - (dg / n) * ehat, eps)


def _visco_project(eps, mu, lam, ys, eta, dt):
    """PG viscoplasticity_return_mapping_with_StVK 의 주응력 공간 판본."""
    tr = eps.sum(-1, keepdim=True)
    ehat = eps - tr / 3.0
    s = 2.0 * mu * ehat
    sn = s.norm(dim=-1, keepdim=True)
    y = sn - math.sqrt(2.0 / 3.0) * ys
    b = (2.0 * eps).exp()                      # sig^2
    mu_hat = mu * b.sum(-1, keepdim=True) / 3.0
    s_new = sn - y / (1.0 + eta / (2.0 * mu_hat * dt).clamp_min(1e-12))
    eps_new = (s_new / sn.clamp_min(1e-12)) * s / (2.0 * mu) + tr / 3.0
    return torch.where(y > 0, eps_new, eps)


def psi_of(F_trial, cfg, dt):
    """(에너지밀도 [N], 주응력 공간의 소성 보정 dlog [N,3] 또는 None)."""
    mu, lam = lame(cfg["E"], cfg["nu"])
    m = mat_name(cfg)
    sig = _sig(F_trial).clamp_min(0.01)        # PG 도 0.01 로 자른다
    if m in ("jelly", "elastic_damage", "watermelon"):
        return _psi_fcr(sig, mu, lam), None
    eps = sig.log()
    ys = float(cfg.get("yield_stress", 0.0))
    if m in ("metal", "plasticine"):
        eps2 = _vm_project(eps, mu, lam, ys)
    elif m == "foam":
        eps2 = _visco_project(eps, mu, lam, ys,
                              float(cfg.get("plastic_viscosity", 0.0)), dt)
    else:
        raise ValueError(f"아직 옮기지 않은 재질: {m}")
    return _psi_hencky(eps2, mu, lam), (eps2 - eps)


@torch.no_grad()
def plastic_step(F_trial, dlog):
    """소성 사영을 배치에 반영한다. F_e = F_trial V diag(exp(dlog)) V^T.

    보정은 C 의 고유기저에서 대각이라 F_trial 오른쪽에 곱하면 된다. 사영 자체는
    비매끄러워 기울기를 타면 튀므로 **여기서만** 떼어낸다 (다중 스텝에서 상태를
    이어 나르는 용도). 에너지의 기울기는 psi_of 를 통해 그대로 흐른다.
    """
    if dlog is None:
        return F_trial
    C = (F_trial.transpose(-1, -2) @ F_trial).double()
    _, V = torch.linalg.eigh(C)
    P = V @ torch.diag_embed(dlog.double().exp()) @ V.transpose(-1, -2)
    return (F_trial.double() @ P).to(F_trial.dtype)


def ip_energy(x2, xtil, F_trial, mass, vol, cfg, h, free=None, g=None,
              norm=None):
    """증분 포텐셜. 구속 입자는 관성·중력 항에서 뺀다 (반력이 미지수다).

    구속 입자의 **위치는** F_trial 을 통해 Psi 에 들어간다 -- 손잡이가 변형을
    일으키는 경로가 바로 그것이라, 여기서 빼면 안 된다.
    """
    psi, dlog = psi_of(F_trial, cfg, h)
    e_el = (vol * psi).sum()
    if free is None:
        free = slice(None)
    d = x2[free] - xtil[free]
    e_in = (0.5 * mass[free] / (h * h) * (d * d).sum(-1)).sum()
    e_g = torch.zeros((), device=x2.device, dtype=x2.dtype)
    if g is not None:
        e_g = -(mass[free] * (x2[free] * g).sum(-1)).sum()
    tot = e_in + e_el + e_g
    if norm is not None:
        tot = tot / norm
    return tot, dlog, (float(e_in), float(e_el), float(e_g))


def residual(E, x2, mass, ext):
    """|r| 을 질량으로 정규화해 길이 단위로 돌려준다 (물체 크기 대비 %).

    r = dE/dx 이고 r_p = m_p (x_p - xtil_p)/h^2 + ... 이므로, h^2/m_p 를 곱하면
    "이 입자가 얼마나 더 움직여야 했는가" 라는 길이가 된다.
    """
    gx, = torch.autograd.grad(E, x2, retain_graph=True, create_graph=False)
    return (gx.norm(dim=-1) / mass.clamp_min(1e-20) / ext)


def smooth_noise(x, sigma, ext, gen, n_wave=6):
    """매끄러운 저주파 변위장과 그 기울기. u(x) = sum_j A_j sin(k_j.x + phi_j).

    학생의 실제 드리프트가 저주파라 이 모양이 맞고, 무엇보다 **grad u 가 해석적**
    이라 F <- (I + grad u) F 로 배치와 변형구배를 일관되게 흔들 수 있다. x 만 흔들고
    F 를 두면 Psi 가 무의미해진다.
    """
    dev, dt = x.device, x.dtype
    k_dir = torch.randn(n_wave, 3, generator=gen, device=dev, dtype=dt)
    k_dir = k_dir / k_dir.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    # 파장은 물체 크기의 0.3~1.2 배
    lam = (0.3 + 0.9 * torch.rand(n_wave, 1, generator=gen, device=dev,
                                  dtype=dt)) * ext
    k = k_dir * (2.0 * math.pi / lam)
    ph = 2.0 * math.pi * torch.rand(n_wave, generator=gen, device=dev, dtype=dt)
    amp = torch.randn(n_wave, 3, generator=gen, device=dev, dtype=dt)
    amp = amp / math.sqrt(n_wave) * sigma
    ang = x @ k.T + ph                                   # [N,W]
    u = torch.sin(ang) @ amp                             # [N,3]
    # grad u = sum_j cos(ang_j) amp_j (x) k_j
    c = torch.cos(ang)                                   # [N,W]
    gu = torch.einsum("nw,wi,wj->nij", c, amp, k)        # [N,3,3]
    return u, gu

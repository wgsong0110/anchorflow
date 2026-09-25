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
        # 중력의 **값**은 기준점에 따른 상수 오프셋이라 보고에 쓸모가 없다.
        # xtil 기준으로 재면 기울기는 그대로(-m g)이고 값은 해석이 된다.
        e_g = -(mass[free] * ((x2[free] - xtil[free]) * g).sum(-1)).sum()
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


def rebuild_F(x_all, cfg, dt, k=16, chunk=4096):
    """교사 위치에서 **탄성** 변형구배를 되살린다. [T,N,3,3]

    궤적에 든 F 는 전부 단위행렬이다 (생성기의 F 읽기가 예외로 빠져 항등으로
    대체됐다). 그대로 쓰면 Psi 가 항등적으로 0 이라 증분 포텐셜이 관성+중력만
    남고, 그러면 자유낙하가 정답이 되어 버린다.

    다시 뽑는 대신 위치에서 복원한다: 프레임 사이의 국소 최소제곱으로 한 스텝
    사상의 야코비안을 구하고, F <- return_map(J F) 를 t=0 의 F=I 에서부터 재생한다.
    MPM 이 하는 것과 같은 절차를 **프레임 해상도로** 되짚는 것이라, 서브스텝마다
    사영하는 원본과 완전히 같지는 않지만 소성 이력이 살아 있는 F 를 준다.
    """
    dev = x_all.device
    T, N = x_all.shape[0], x_all.shape[1]
    X0 = x_all[0]
    # t=0 배치에서 이웃을 한 번만 잡는다 (재질 이웃은 변하지 않는다)
    idx = torch.empty(N, k, dtype=torch.long, device=dev)
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        dd = torch.cdist(X0[s:e], X0)
        idx[s:e] = dd.topk(k + 1, largest=False).indices[:, 1:]
    F = torch.eye(3, device=dev).repeat(N, 1, 1)
    out = torch.empty(T, N, 3, 3, device=dev, dtype=torch.float16)
    out[0] = F.half()
    for t in range(T - 1):
        d0 = x_all[t][idx] - x_all[t].unsqueeze(1)             # [N,k,3]
        d1 = x_all[t + 1][idx] - x_all[t + 1].unsqueeze(1)
        w = 1.0 / (d0.norm(dim=-1, keepdim=True) ** 2 + 1e-12)
        w = w / w.sum(1, keepdim=True)
        A = torch.einsum("nkc,nki,nkj->nij", w, d1, d0)
        B = torch.einsum("nkc,nki,nkj->nij", w, d0, d0)
        B = B + 1e-10 * torch.eye(3, device=dev)
        J = A @ torch.linalg.inv(B)
        F_tr = J @ F
        _, dlog = psi_of(F_tr, cfg, dt)
        F = plastic_step(F_tr, dlog)
        out[t + 1] = F.half()
    return out


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


# ------------------------------------------------------------------ 격자 목적함수
# i-PG 는 미지수를 **격자 증분** Δu_I 로 두고 푼다. 학생은 가우시안을 옮기므로,
# 그 변위를 MPM 격자로 되돌려(P2G, 질량가중) Δu_I 를 역산한 뒤 같은 식을 잰다.
# 가중치는 PG 와 같은 이차 B-스플라인(3^3 스텐실)이다.

def _bspline(xr):
    """격자 단위 좌표 xr 에 대해 (base, w[3], dw[3]). 모두 [N,3,3] 축별."""
    base = (xr - 0.5).floor()
    fx = xr - base                                   # [N,3], 0.5~1.5
    w = torch.stack([0.5 * (1.5 - fx) ** 2,
                     0.75 - (fx - 1.0) ** 2,
                     0.5 * (fx - 0.5) ** 2], -1)     # [N,3,3] (축, 노드)
    dw = torch.stack([fx - 1.5,
                      -2.0 * (fx - 1.0),
                      fx - 0.5], -1)
    return base.long(), w, dw


_OFF3 = None


def _offsets(device):
    global _OFF3
    if _OFF3 is None or _OFF3.device != device:
        r = torch.arange(3, device=device)
        _OFF3 = torch.stack(torch.meshgrid(r, r, r, indexing="ij"),
                            -1).reshape(-1, 3)       # [27,3]
    return _OFF3


def p2g_increment(x, du, vel, mass, n_grid, grid_lim):
    """가우시안 변위 du 를 격자로 되돌린다.

    반환: (m_I [M], du_I [M,3], v_I [M,3], 색인정보) -- 모두 **점유 노드만**.
    질량가중 평균이라 Δu_I = Σ m_p w du_p / Σ m_p w 이고, 이것이 i-PG 의 미지수다.
    """
    dev = x.device
    dx = float(grid_lim) / float(n_grid)
    xr = x / dx
    base, w, _ = _bspline(xr)
    off = _offsets(dev)                              # [27,3]
    idx3 = base.unsqueeze(1) + off.unsqueeze(0)      # [N,27,3]
    idx3 = idx3.clamp(0, n_grid - 1)
    flat = ((idx3[..., 0] * n_grid + idx3[..., 1]) * n_grid
            + idx3[..., 2])                          # [N,27]
    ww = (w[:, 0, :].unsqueeze(-1).unsqueeze(-1)
          * w[:, 1, :].unsqueeze(1).unsqueeze(-1)
          * w[:, 2, :].unsqueeze(1).unsqueeze(1)).reshape(x.shape[0], 27)
    mw = ww * mass.unsqueeze(-1)                     # [N,27]
    # 점유 노드만 모은다 (100^3 전부 들고 있으면 낭비다)
    uniq, inv = torch.unique(flat.reshape(-1), return_inverse=True)
    M = uniq.numel()
    m_I = torch.zeros(M, device=dev, dtype=x.dtype).index_add_(
        0, inv, mw.reshape(-1))
    du_I = torch.zeros(M, 3, device=dev, dtype=x.dtype).index_add_(
        0, inv, (mw.reshape(-1, 1) * du.repeat_interleave(27, 0)))
    v_I = torch.zeros(M, 3, device=dev, dtype=x.dtype).index_add_(
        0, inv, (mw.reshape(-1, 1) * vel.repeat_interleave(27, 0)))
    den = m_I.clamp_min(1e-20).unsqueeze(-1)
    return m_I, du_I / den, v_I / den, (flat, ww, inv, uniq, dx)


def g2p_grad(x, du_I, info, n_grid):
    """격자 증분에서 입자 위치의 변위기울기 ∇Δu [N,3,3] 를 뽑는다 (MPM 과 같은 길)."""
    flat, ww, inv, uniq, dx = info
    dev = x.device
    xr = x / dx
    base, w, dw = _bspline(xr)
    off = _offsets(dev)
    # 축별 가중치/도함수를 27 스텐실로 펼친다
    i0, i1, i2 = off[:, 0], off[:, 1], off[:, 2]
    wx, wy, wz = w[:, 0][:, i0], w[:, 1][:, i1], w[:, 2][:, i2]     # [N,27]
    dxw = dw[:, 0][:, i0] / dx
    dyw = dw[:, 1][:, i1] / dx
    dzw = dw[:, 2][:, i2] / dx
    gw = torch.stack([dxw * wy * wz, wx * dyw * wz, wx * wy * dzw], -1)
    u_nodes = du_I[inv].reshape(x.shape[0], 27, 3)                  # [N,27,3]
    return torch.einsum("nkd,nkc->ndc", u_nodes, gw)                # ∇Δu


def grid_ip_energy(x, du, vel, F, mass, vol, cfg, h, n_grid, grid_lim,
                   g=None, norm=None, free=None):
    """i-PG 의 목적함수를 **격자 증분** 기준으로 잰다.

        E = Σ_I m_I/(2h²)‖Δu_I − h v_I‖² + Σ_p V_p Ψ(F_p) − Σ_I m_I g·Δu_I
        F_p = (I + ∇Δu) F_p^n

    관성·중력은 격자에서, 탄성은 입자에서 잰다 -- MPM 이 힘을 만드는 자리와 같다.
    """
    m_I, du_I, v_I, info = p2g_increment(x, du, vel, mass, n_grid, grid_lim)
    d = du_I - h * v_I
    e_in = (0.5 * m_I / (h * h) * (d * d).sum(-1)).sum()
    e_g = torch.zeros((), device=x.device, dtype=x.dtype)
    if g is not None:
        e_g = -(m_I * (du_I * g).sum(-1)).sum()
    gu = g2p_grad(x, du_I, info, n_grid)
    F_tr = (torch.eye(3, device=x.device, dtype=F.dtype) + gu) @ F
    psi, dlog = psi_of(F_tr, cfg, h)
    e_el = (vol * psi).sum()
    tot = e_in + e_el + e_g
    if norm is not None:
        tot = tot / norm
    return tot, dlog, F_tr, (float(e_in), float(e_el), float(e_g))

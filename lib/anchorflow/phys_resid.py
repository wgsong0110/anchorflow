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
import os

from typing import NamedTuple

import torch

__all__ = ["lame", "psi_of", "plastic_step", "Plast", "ip_energy", "residual",
           "smooth_noise", "mat_name", "bc_node_mask", "bc_energy",
           "grid_ip_sub", "grid_ip_pts", "jac_neighbors", "grad_from_points",
           "p2g_ls", "g2p_from_nodes"]


def lame(E, nu):
    E, nu = float(E), float(nu)
    return E / (2.0 * (1.0 + nu)), E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))


def mat_name(cfg):
    return str(cfg.get("material", "jelly"))


# 겹친 고유값을 깨는 아주 작은 이방성 흔들림. 1 계 미분은 겹쳐도 멀쩡하지만
# **2 계** 미분은 1/(l_i - l_j) 를 타서 터진다 (잔차 보상이 그 경로를 쓴다).
_JIT = None


# 고유분해 정밀도. 실측(exe/test_sig_dtype.py, 251001 입자, A6000):
#   f32 는 f64 대비 3.2 배 빠르고(43ms -> 14ms) 비유한 값이 없다.
#   탄성에너지 합 상대오차 ~1e-07, 기울기 상대오차 최대 2.4e-04 -- fp32 학습의
#   자연 잡음 수준이다. sigma 상대오차가 큰 입자는 sigma~0 이고 절대차 3e-04 로
#   psi_of 의 clamp_min(0.01) 아래라 쓰이지 않는다.
# AF_SIG_F64=1 로 되돌린다.
_SIG_DT = torch.float64 if os.environ.get("AF_SIG_F64") else torch.float32


def _sig(F):
    """특이값 [N,3] (오름차순). C = F^T F 의 고유값으로 구한다."""
    global _JIT
    C = (F.transpose(-1, -2) @ F).to(_SIG_DT)
    if _JIT is None or _JIT.device != C.device or _JIT.dtype != _SIG_DT:
        _JIT = torch.diag(torch.tensor([0.0, 1e-9, 2e-9], dtype=_SIG_DT,
                                       device=C.device))
    tr = C.diagonal(dim1=-2, dim2=-1).sum(-1).reshape(
        *C.shape[:-2], 1, 1).clamp_min(1e-12)
    return torch.linalg.eigvalsh(C + tr * _JIT).clamp_min(1e-12).sqrt().to(
        F.dtype)


def _sig_vec(F):
    """(특이값 [N,3] 오름차순, C=F^T F 의 고유기저 V [N,3,3]).

    소성 사영은 주응력 공간에서만 대각이라 V 가 필요하고, 에너지는 값만 쓴다.
    예전에는 에너지에서 eigvalsh, 소성에서 eigh 를 **따로** 돌아 같은 분해를
    두 번 했다 -- 한 번에 받아 넘긴다.
    """
    global _JIT
    C = (F.transpose(-1, -2) @ F).to(_SIG_DT)
    if _JIT is None or _JIT.device != C.device or _JIT.dtype != _SIG_DT:
        _JIT = torch.diag(torch.tensor([0.0, 1e-9, 2e-9], dtype=_SIG_DT,
                                       device=C.device))
    tr = C.diagonal(dim1=-2, dim2=-1).sum(-1).reshape(
        *C.shape[:-2], 1, 1).clamp_min(1e-12)
    w, V = torch.linalg.eigh(C + tr * _JIT)
    return w.clamp_min(1e-12).sqrt().to(F.dtype), V


class Plast(NamedTuple):
    """소성 사영에 필요한 것: 주응력 공간 보정과 그 기저."""
    dlog: torch.Tensor
    V: torch.Tensor


def _psi_fcr(sig, mu, lam):
    J = sig.prod(-1)
    return mu * ((sig - 1.0) ** 2).sum(-1) + 0.5 * lam * (J - 1.0) ** 2


def _psi_hencky(eps, mu, lam):
    return mu * (eps ** 2).sum(-1) + 0.5 * lam * (eps.sum(-1) ** 2)


def _vm_project(eps, mu, lam, ys):
    """PG von_mises_return_mapping 의 주응력 공간 판본.

    노름을 sqrt(|.|^2 + eps^2) 로 무르게 잡고 분기를 **곱셈**으로 둔다.
    변형이 없는 입자는 ehat 이 정확히 0 이라 그냥 norm 을 쓰면 미분이 정의되지
    않고, where 로 가려도 선택 안 된 가지의 NaN 이 역전파로 새어 나온다.
    """
    tr = eps.sum(-1, keepdim=True)
    tau = 2.0 * mu * eps + lam * tr
    dev = tau - tau.sum(-1, keepdim=True) / 3.0
    # 엡실론은 **양의 스케일**에 맞춰야 한다. 1e-24 처럼 작게 두면 변형이 없는
    # 입자에서 sn ~ 1e-12 이 되어 2 계 미분이 1/sn^3 ~ 1e36 으로 float32 를
    # 넘긴다 (잔차 손실이 그 경로를 쓴다).
    _es = 1e-4                                  # 변형률 기준 바닥
    dn = torch.sqrt((dev ** 2).sum(-1, keepdim=True) + (2.0 * mu * _es) ** 2)
    over = (dn > ys).to(eps.dtype)
    ehat = eps - tr / 3.0
    n = torch.sqrt((ehat ** 2).sum(-1, keepdim=True) + _es ** 2)
    dg = (n - ys / (2.0 * mu)).clamp_min(0.0)
    return eps - (over * dg / n) * ehat


def _visco_project(eps, mu, lam, ys, eta, dt):
    """PG viscoplasticity_return_mapping_with_StVK 의 주응력 공간 판본."""
    tr = eps.sum(-1, keepdim=True)
    ehat = eps - tr / 3.0
    s = 2.0 * mu * ehat
    # 같은 이유로 노름을 무르게 잡고 분기를 곱셈으로 둔다
    _es = 1e-4                                  # 변형률 기준 바닥 (위와 같은 이유)
    sn = torch.sqrt((s ** 2).sum(-1, keepdim=True) + (2.0 * mu * _es) ** 2)
    y = sn - math.sqrt(2.0 / 3.0) * ys
    b = (2.0 * eps).exp()                      # sig^2
    mu_hat = mu * b.sum(-1, keepdim=True) / 3.0
    s_new = sn - y / (1.0 + eta / (2.0 * mu_hat * dt).clamp_min(1e-12))
    eps_new = (s_new / sn) * s / (2.0 * mu) + tr / 3.0
    w = (y > 0).to(eps.dtype)
    return w * eps_new + (1.0 - w) * eps


def psi_of(F_trial, cfg, dt):
    """(에너지밀도 [N], 주응력 공간의 소성 보정 dlog [N,3] 또는 None)."""
    mu, lam = lame(cfg["E"], cfg["nu"])
    m = mat_name(cfg)
    if m in ("jelly", "elastic_damage", "watermelon"):
        # 탄성 전용이면 고유벡터가 필요 없다 (값만 쓰는 쪽이 싸다)
        return _psi_fcr(_sig(F_trial).clamp_min(0.01), mu, lam), None
    sig, V = _sig_vec(F_trial)
    sig = sig.clamp_min(0.01)                  # PG 도 0.01 로 자른다
    eps = sig.log()
    ys = float(cfg.get("yield_stress", 0.0))
    if m in ("metal", "plasticine"):
        eps2 = _vm_project(eps, mu, lam, ys)
    elif m == "foam":
        eps2 = _visco_project(eps, mu, lam, ys,
                              float(cfg.get("plastic_viscosity", 0.0)), dt)
    else:
        raise ValueError(f"아직 옮기지 않은 재질: {m}")
    return _psi_hencky(eps2, mu, lam), Plast(eps2 - eps, V)


@torch.no_grad()
def plastic_step(F_trial, pl):
    """소성 사영을 배치에 반영한다. F_e = F_trial V diag(exp(dlog)) V^T.

    보정은 C 의 고유기저에서 대각이라 F_trial 오른쪽에 곱하면 된다. 사영 자체는
    비매끄러워 기울기를 타면 튀므로 **여기서만** 떼어낸다 (다중 스텝에서 상태를
    이어 나르는 용도). 에너지의 기울기는 psi_of 를 통해 그대로 흐른다.

    기저 V 는 psi_of 가 이미 구한 것을 그대로 받는다 (재분해 없음). 그리고
    **항복한 입자만** 곱한다 -- dlog 가 정확히 0 인 입자는 P = V V^T = I 라
    수학적으로 항등이고, 건너뛰면 float64 직교성 오차(~1e-16)조차 안 생긴다.
    `_vm_project` 가 항복 안 한 입자에서 `over` 또는 `dg` 를 정확히 0 으로 두므로
    `eps2 - eps` 가 비트 단위로 0 이다.
    """
    if pl is None:
        return F_trial
    dlog, V = (pl.dlog, pl.V) if isinstance(pl, Plast) else (pl, None)
    nz = dlog.abs().sum(-1) > 0
    if not bool(nz.any()):
        return F_trial
    if V is None:                              # 옛 호출 형태 (기저를 안 받은 경우)
        _, V = _sig_vec(F_trial)
    out = F_trial.clone()
    Fi = F_trial[nz].to(_SIG_DT)
    Vi = V[nz].to(_SIG_DT)
    P = (Vi @ torch.diag_embed(dlog[nz].to(_SIG_DT).exp())
         @ Vi.transpose(-1, -2))
    out[nz] = (Fi @ P).to(F_trial.dtype)
    return out


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
    return tot, dlog, (e_in.detach(), e_el.detach(), e_g.detach())


def pts_ip_energy(x, du, vel, F, jac, mass, vol, cfg, h, n_grid, grid_lim,
                  g=None, norm=None, free=None, elastic_mask=None):
    """증분 포텐셜을 **입자에서 바로** 잰다 -- MPM 격자를 전혀 거치지 않는다.

        E = Σ_p m_p/(2h²)‖Δu_p − h v_p − h² g‖² + Σ_p V_p Ψ(F_p) + E_c
        F_p = J_p F_p^n           (J = 변형장 grad_x Phi 의 야코비안)

    격자판(grid_ip_energy)은 Δu 를 P2G 로 노드에 모으고 ∇Δu 를 G2P 미분으로
    되받아 왔다. 그 왕복은 (i) 노드 평균이 선형장조차 편향되게 만들고,
    (ii) 셀 내부 재배열·스키닝 가중치를 통째로 잃어 실제로 입자를 옮긴 사상과
    다른 F 를 만든다. 변형장의 야코비안이 손에 있으면 왕복이 필요 없다.

    n_grid, grid_lim 은 접촉항의 길이 단위(dx)에만 쓰인다 -- 전달이 아니다.
    """
    F_tr = (jac.to(F.dtype) @ F) if jac is not None else F
    psi, dlog = psi_of(F_tr, cfg, h)
    if elastic_mask is not None:
        # det 가 문턱 아래인 사면체는 제대로 된 셀이 아니다 -- 거기서 나온
        # Psi 는 발산하거나 뜻이 없으므로 **탄성항에서만** 뺀다 (관성·중력·
        # 접촉은 그대로 받는다). 되돌리는 일은 복구 손실이 맡는다.
        psi = psi * elastic_mask.to(psi.dtype)
    e_el = (vol * psi).sum()
    # 중력은 관성항의 목표에 미리 넣는다: x̃ = x + h v + h² g. 따로 −m g·Δu 로
    # 두면 서브스텝으로 나눌 때 각 구간이 자기 변위만 보게 되어 중력 일이
    # 1/K 로 줄어든다 (격자판에서 겪은 것과 같은 함정이다).
    _gh = (h * h) * g if g is not None else 0.0
    d = du - h * vel - _gh
    w = (torch.ones_like(mass) if free is None else free.to(mass.dtype))
    e_in = (0.5 * w * mass / (h * h) * (d * d).sum(-1)).sum()
    e_g = torch.zeros((), device=x.device, dtype=x.dtype)
    if g is not None:
        # 보고용: 이 스텝에서 중력이 한 일 (기울기에는 이미 관성항으로 들어갔다)
        e_g = -(w * mass * (du * g).sum(-1)).sum().detach()
    e_bc = bc_energy(x, du, mass, cfg, h, grid_lim, n_grid)
    tot = e_in + e_el + e_bc
    if norm is not None:
        tot = tot / norm
    # 보고용 항은 **텐서로** 돌려준다. float() 는 GPU 동기화라 스텝마다
    # 배치x4 회 파이프라인을 세운다 -- 출력하는 자리에서만 변환한다.
    return tot, dlog, F_tr, (e_in.detach(), e_el.detach(),
                             e_g.detach(), e_bc.detach())


def bc_project_nodes(npos, dp, cfg, h, grid_lim, n_grid):
    """바닥·경계를 **격자점 변위**에 하드로 사영한다 -> (dp_사영, 활성마스크).

    입자 단계에서 사영하면 변형장 자체는 바닥을 모른 채 아무 값이나 내고 그
    뒤에 결과만 고쳐진다 (손잡이에서 겪은 것과 같은 구조다). 노드에 걸면 바닥이
    변형장의 일부가 되어, 그 노드를 꼭짓점으로 갖는 모든 셀이 바닥을 본다.

    바닥은 **한쪽 부등식**이라 활성 집합이 매 스텝 달라진다 -- 사영은 미분
    가능한 자리에서 기울기를 그대로 흘려 보낸다 (손잡이처럼 끊지 않는다).
    """
    dx = float(grid_lim) / float(n_grid)
    act = torch.zeros(npos.shape[0], dtype=torch.bool, device=npos.device)
    dp_p = dp
    for bc in (cfg.get("boundary_conditions") or []):
        t = bc.get("type")
        if t == "surface_collider":
            pt = torch.as_tensor(bc["point"], device=npos.device,
                                 dtype=npos.dtype)
            nr = torch.as_tensor(bc["normal"], device=npos.device,
                                 dtype=npos.dtype)
            nr = nr / nr.norm().clamp_min(1e-12)
            # **i-PG 와 같은 판정**: 격자점 자체가 면 아래인 노드만 건다
            # (collide 커널은 grid_x*dx 로 노드 위치를 재서 dot < 0 을 본다).
            # 변위 뒤 위치로 재거나 "면에서 dx 안" 까지 접선을 묶으면 실제
            # 접촉면보다 두꺼운 띠가 얼어붙는다 -- 로프에서 두께의 2/3 가
            # 묶여 비 0.23 -> 0.61 로 나빠졌다.
            # **위치를 면 위로 사영한다.** 변위를 clamp_min 으로 자르면 잘린
            # 쪽에서 기울기가 정확히 0 이라, 관성 예측자가 음수로 시작시킨 노드는
            # 에너지가 아무리 위로 밀어도 신호를 못 받고 영영 갇힌다 (실측:
            # 4821 번 전부 '올라가려던 노드 0 개'). 사영은 미분 가능해 기울기가
            # 그대로 흐르고, 면 아래 노드를 면 위로 끌어올린다.
            sd = ((npos + dp_p - pt) * nr).sum(-1)
            pen = sd < 0.0
            dp_p = dp_p - torch.where(pen.unsqueeze(-1),
                                      sd.unsqueeze(-1) * nr,
                                      torch.zeros_like(dp_p))
            if str(bc.get("surface", "sticky")) == "sticky":
                # 붙는다: 면에 닿은 노드는 접선 변위도 0
                dn = (dp_p * nr).sum(-1, keepdim=True)
                dp_p = torch.where(pen.unsqueeze(-1), dn * nr, dp_p)
            act = act | pen
            under = pen
            if os.environ.get("AF_BC_DIAG"):
                with torch.no_grad():
                    _nu = int(under.sum())
                    _up = int((under & (dn.reshape(-1) > 0)).sum())
                    print(f"      [바닥] 면 아래 노드 {_nu}  그중 올라가려던 "
                          f"것 {_up}  (바뀐 노드는 이 {_up} 개뿐이다)",
                          flush=True)
        elif t == "bounding_box":
            b = float(cfg.get("bound", 3)) * dx
            lo, hi = b, float(grid_lim) - b
            tgt = (npos + dp_p).clamp(min=lo, max=hi)
            out = ((npos + dp_p) < lo) | ((npos + dp_p) > hi)
            dp_p = torch.where(out, tgt - npos, dp_p)
            act = act | out.any(-1)
    return dp_p, act


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


def p2g_increment(x, du, vel, mass, n_grid, grid_lim, free=None):
    """가우시안 변위 du 를 격자로 되돌린다.

    반환: (m_I [M], du_I [M,3], v_I [M,3], 색인정보, 자유질량비 [M]) -- 모두
    **점유 노드만**. 질량가중 평균이라 Δu_I = Σ m_p w du_p / Σ m_p w 이고,
    이것이 i-PG 의 미지수다.

    free 가 주어지면 노드마다 **자유 입자가 실은 질량의 비율**도 함께 낸다.
    손잡이(Dirichlet) 입자가 지배하는 노드는 그 운동을 교사가 박아 두었으므로
    관성·중력 잔차를 매기면 안 된다 -- 망이 정하는 양이 아니라서 기울기가 가짜다.
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
    if free is None:
        frac = None
    else:
        mwf = (ww * (mass * free.to(mass.dtype)).unsqueeze(-1)).reshape(-1)
        m_free = torch.zeros(M, device=dev, dtype=x.dtype).index_add_(0, inv, mwf)
        frac = m_free / m_I.clamp_min(1e-20)
    return m_I, du_I / den, v_I / den, (flat, ww, inv, uniq, dx), frac


def p2g_ls(x, du, mass, n_grid, grid_lim, ridge=1e-6):
    """노드마다 **국소 최소제곱**으로 (값, 기울기) 를 함께 푼다.

    질량가중 평균은 선형장에서 편향된다: du_I = u(x̄_I) 이고 x̄_I 는 주변 입자의
    무게중심이라 노드 위치 x_I 와 다르다. 그러면 Σ_I x_I ∇w_ip = I 라는 항등식이
    깨져, 답을 아는 균일 변형조차 45~53% 틀리게 나온다 (실측). 여기서는

        min_{u_I, G_I} Σ_p m_p w_ip | u_I + G_I (x_p - x_I) - u_p |^2

    을 풀어 u_I 를 편향 없이 얻고 G_I 도 직접 얻는다. 4x4 정규방정식이라 노드
    수만큼의 작은 해를 한 번에 푼다.

    반환 (m_I, u_I [M,3], G_I [M,3,3], 색인정보, 노드좌표 [M,3]).
    """
    dev = x.device
    dx = float(grid_lim) / float(n_grid)
    xr = x / dx
    base, w, _dw = _bspline(xr)
    off = _offsets(dev)
    idx3 = (base.unsqueeze(1) + off.unsqueeze(0)).clamp(0, n_grid - 1)
    flat = ((idx3[..., 0] * n_grid + idx3[..., 1]) * n_grid + idx3[..., 2])
    ww = (w[:, 0, :].unsqueeze(-1).unsqueeze(-1)
          * w[:, 1, :].unsqueeze(1).unsqueeze(-1)
          * w[:, 2, :].unsqueeze(1).unsqueeze(1)).reshape(x.shape[0], 27)
    mw = (ww * mass.unsqueeze(-1)).reshape(-1)                      # [N*27]
    uniq, inv = torch.unique(flat.reshape(-1), return_inverse=True)
    M = uniq.numel()
    # 노드 좌표
    ki = uniq % n_grid
    kj = (uniq // n_grid) % n_grid
    kk = uniq // (n_grid * n_grid)
    xI = torch.stack([kk, kj, ki], -1).to(x.dtype) * dx             # [M,3]
    dvec = x.repeat_interleave(27, 0) - xI[inv]                     # [N*27,3]
    up = du.repeat_interleave(27, 0)                                # [N*27,3]
    # 4x4 정규방정식: [[S, S d^T], [S d, S d d^T]] [u; G^T] = [S u; S d u^T]
    S0 = torch.zeros(M, device=dev, dtype=x.dtype).index_add_(0, inv, mw)
    S1 = torch.zeros(M, 3, device=dev, dtype=x.dtype).index_add_(
        0, inv, mw.unsqueeze(-1) * dvec)
    S2 = torch.zeros(M, 3, 3, device=dev, dtype=x.dtype).index_add_(
        0, inv, mw.reshape(-1, 1, 1) * dvec.unsqueeze(-1) * dvec.unsqueeze(-2))
    b0 = torch.zeros(M, 3, device=dev, dtype=x.dtype).index_add_(
        0, inv, mw.unsqueeze(-1) * up)
    b1 = torch.zeros(M, 3, 3, device=dev, dtype=x.dtype).index_add_(
        0, inv, mw.reshape(-1, 1, 1) * dvec.unsqueeze(-1) * up.unsqueeze(-2))
    A = torch.zeros(M, 4, 4, device=dev, dtype=x.dtype)
    A[:, 0, 0] = S0
    A[:, 0, 1:] = S1
    A[:, 1:, 0] = S1
    A[:, 1:, 1:] = S2
    B = torch.cat([b0.unsqueeze(1), b1], 1)                         # [M,4,3]
    # 입자가 적은 노드는 정칙이 아니다 -> 리지로 값만 살리고 기울기는 0 으로 간다
    tr = A.diagonal(dim1=-2, dim2=-1).sum(-1).clamp_min(1e-30)
    A = A + (ridge * tr).reshape(M, 1, 1) * torch.eye(4, device=dev,
                                                      dtype=x.dtype)
    sol = torch.linalg.solve(A, B)                                  # [M,4,3]
    u_I = sol[:, 0, :]
    G_I = sol[:, 1:, :].transpose(1, 2)                             # ∂u_d/∂x_c
    return S0, u_I, G_I, (flat, ww, inv, uniq, dx), xI


def g2p_from_nodes(x, G_I, info, n_grid):
    """노드 기울기를 입자로 보간한다 (B-spline 가중)."""
    flat, ww, inv, uniq, dx = info
    g_nodes = G_I[inv].reshape(x.shape[0], 27, 3, 3)
    return torch.einsum("nk,nkdc->ndc", ww, g_nodes)


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


def bc_node_mask(uniq, n_grid, dx, cfg, dtype, dev):
    """PG 경계조건이 **변위를 규정하는** 격자 노드 [M] 불리언.

    PG 의 `surface_collider`(sticky) 는 면 안쪽 노드의 속도를 0 으로 박고,
    `bounding_box` 는 경계 3 셀 띠를 막는다. 증분 포텐셜에서 이것은 그 노드의
    Δu 가 0 으로 규정된 Dirichlet 조건이다. 이 항이 빠져 있으면 중력만 아래로
    끌고 막는 것이 없어 **바닥을 뚫고 내려가는 것이 목적함수상 이득**이 된다.
    """
    k = uniq % n_grid
    j = (uniq // n_grid) % n_grid
    i = uniq // (n_grid * n_grid)
    pos = torch.stack([i, j, k], -1).to(dtype) * dx
    m = torch.zeros(uniq.shape[0], dtype=torch.bool, device=dev)
    if os.environ.get("AF_NO_BC"):              # 비교용으로만 끈다
        return m
    for bc in (cfg.get("boundary_conditions") or []):
        t = bc.get("type")
        if t == "surface_collider":
            pt = torch.as_tensor(bc["point"], device=dev, dtype=dtype)
            nr = torch.as_tensor(bc["normal"], device=dev, dtype=dtype)
            nr = nr / nr.norm().clamp_min(1e-12)
            m |= (((pos - pt) * nr).sum(-1) < 0)
        elif t == "bounding_box":
            b = int(cfg.get("bound", 3))
            m |= ((i < b) | (j < b) | (k < b) | (i >= n_grid - b)
                  | (j >= n_grid - b) | (k >= n_grid - b))
    return m


def bc_energy(x, du, mass, cfg, h, grid_lim, n_grid, stiff=None):
    """PG 경계조건을 증분 포텐셜의 **접촉항**으로 옮긴다.

    PG 는 격자 속도를 사영해서 (`surface_collider` sticky 는 면 안쪽 노드의 속도를
    0 으로, `bounding_box` 는 경계 3 셀 띠를 막는다) 바닥을 만든다. 격자 쪽에서
    같은 일을 하려 하면 P2G 가 **점유 노드만** 모으기 때문에 입자가 실제로 면
    아래로 넘어가기 전까지 걸릴 노드가 없어 아무 것도 막지 못한다. 그래서 입자
    쪽에 벌점을 둔다:

        E_c = Σ_p k m_p/(2h²) [ relu(-sd(x_p+Δu_p))²                 (비관통)
                              + 1[접촉] ‖Δu_p − (Δu_p·n)n‖² ]        (sticky)

    관성항과 같은 m/(2h²) 스케일이라 k 가 무차원이고, 평형 관통 깊이는 g h²/k 로
    k=1000, h=1/60 에서 2.7e-6 이다 (실측 관통 0.97%). sticky 항은 면에 닿은 입자의
    접선 운동까지 묶어 PG 의 "속도를 0 으로" 와 맞춘다.
    """
    if os.environ.get("AF_NO_BC"):
        return torch.zeros((), device=x.device, dtype=x.dtype)
    # 실측: k=10 이면 손잡이가 끌 때 관성항이 접촉항을 이겨 입자-프레임의 5.8%
    # 가 바닥 밑으로 내려간다. k=1000 이면 0.97% 로 떨어지고 목적함수 값은 거의
    # 바뀌지 않는다 (비 2.286 -> 2.274). 그래서 기본을 1000 으로 둔다.
    k = float(os.environ.get("AF_BC_STIFF", 1000.0) if stiff is None else stiff)
    dx = float(grid_lim) / float(n_grid)
    x2 = x + du
    c = 0.5 * k * mass / (h * h)
    e = torch.zeros((), device=x.device, dtype=x.dtype)
    for bc in (cfg.get("boundary_conditions") or []):
        t = bc.get("type")
        if t == "surface_collider":
            pt = torch.as_tensor(bc["point"], device=x.device, dtype=x.dtype)
            nr = torch.as_tensor(bc["normal"], device=x.device, dtype=x.dtype)
            nr = nr / nr.norm().clamp_min(1e-12)
            sd = ((x2 - pt) * nr).sum(-1)
            e = e + (c * sd.clamp_max(0.0) ** 2).sum()
            if str(bc.get("surface", "sticky")) == "sticky":
                # 닿아 있는 입자는 접선 방향도 묶인다 (속도를 0 으로 박는 것과 같다)
                touch = (((x - pt) * nr).sum(-1) < dx).to(x.dtype).detach()
                du_t = du - (du * nr).sum(-1, keepdim=True) * nr
                e = e + (c * touch * (du_t * du_t).sum(-1)).sum()
        elif t == "bounding_box":
            b = float(cfg.get("bound", 3)) * dx
            lo, hi = b, float(grid_lim) - b
            e = e + (c.unsqueeze(-1) * ((x2 - lo).clamp_max(0.0) ** 2
                                        + (hi - x2).clamp_max(0.0) ** 2)).sum()
    return e


def bc_project(x, du, cfg, h, grid_lim, n_grid):
    """바닥·경계를 **하드로 사영**한다 -> (du_사영, 활성마스크).

    PG/i-PG 는 격자 속도를 사영해 막는다(하드). 벌점(`bc_energy`)은 관성항과
    겨루므로 새고(실측 관통 0.97%), 강성을 올려도 완전히 막히지는 않는다. 그래서
    손잡이와 같은 틀로 하드 구속으로 바꾼다.

    바닥은 손잡이와 달리 **한쪽 부등식** 구속이라 활성 집합이 상태에 따라 매 스텝
    달라진다 -- 관통하는 입자만 사영하고, sticky 접선 고정은 닿은 입자만 건다.
    그래서 반환하는 마스크는 그 스텝에서만 유효하다.

      비관통:  sd(x+du) < 0 인 입자를 면 위로 올린다 (법선 성분만 제거)
      sticky:  닿은 입자는 접선 변위까지 0 (속도를 0 으로 박는 것과 같다)
    """
    x2 = x + du
    dx = float(grid_lim) / float(n_grid)
    act = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
    du_p = du
    for bc in (cfg.get("boundary_conditions") or []):
        t = bc.get("type")
        if t == "surface_collider":
            pt = torch.as_tensor(bc["point"], device=x.device, dtype=x.dtype)
            nr = torch.as_tensor(bc["normal"], device=x.device, dtype=x.dtype)
            nr = nr / nr.norm().clamp_min(1e-12)
            sd = ((x + du_p - pt) * nr).sum(-1)            # 부호거리
            pen = sd < 0.0
            # 법선 성분만 걷어내 면 위로 올린다
            du_p = du_p - torch.where(pen.unsqueeze(-1),
                                      sd.unsqueeze(-1) * nr,
                                      torch.zeros_like(du_p))
            act = act | pen
            if str(bc.get("surface", "sticky")) == "sticky":
                touch = ((x - pt) * nr).sum(-1) < dx
                dn = (du_p * nr).sum(-1, keepdim=True) * nr
                du_p = torch.where(touch.unsqueeze(-1), dn, du_p)
                act = act | touch
        elif t == "bounding_box":
            b = float(cfg.get("bound", 3)) * dx
            lo, hi = b, float(grid_lim) - b
            tgt = (x + du_p).clamp(min=lo, max=hi)
            out = ((x + du_p) < lo) | ((x + du_p) > hi)
            du_p = torch.where(out, tgt - x, du_p)
            act = act | out.any(-1)
    return du_p, act


def jac_neighbors(x0, k=16, chunk=2048):
    """입자 이웃을 한 번 잡아 둔다 (변형기울기를 격자 왕복 없이 뽑기 위해).

    반환 (이웃색인 [N,k], 가중 [N,k,1], B^{-1} [N,3,3]). B 는 sum w d0 d0^T 로
    최소제곱의 정규방정식 행렬이라 이웃이 고정이면 한 번만 만들면 된다.
    """
    N = x0.shape[0]
    idx = torch.empty(N, k, dtype=torch.long, device=x0.device)
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        idx[s:e] = torch.cdist(x0[s:e], x0).topk(
            k + 1, largest=False).indices[:, 1:]
    d0 = x0[idx] - x0.unsqueeze(1)
    w = 1.0 / (d0.norm(dim=-1, keepdim=True) ** 2 + 1e-12)
    w = w / w.sum(1, keepdim=True)
    B = torch.einsum("nkc,nki,nkj->nij", w, d0, d0)
    B = B + 1e-10 * torch.eye(3, device=x0.device)
    return idx, w, torch.linalg.inv(B)


def grad_from_points(x0, du, nb):
    """입자 이웃의 최소제곱으로 변위기울기 ∇Δu [N,3,3] 를 뽑는다.

    격자 왕복(P2G -> G2P)으로 뽑으면 실측으로 변형의 95~97% 가 날아간다
    (|(I+G)-J| / |J-I| 중앙 0.95). 질량가중 평균이 국소 기울기를 평탄화하기
    때문이다. 탄성항은 이 함수로 만든 기울기를 써야 실제 변형을 본다.
    """
    idx, w, Bi = nb
    d0 = x0[idx] - x0.unsqueeze(1)
    dd = du[idx] - du.unsqueeze(1)
    A = torch.einsum("nkc,nki,nkj->nij", w, dd, d0)
    return A @ Bi


def grid_ip_pts(x, du, vel, F, mass, vol, cfg, h, n_grid, grid_lim,
                g=None, norm=None, free=None, nb=None):
    """증분 포텐셜. 관성·중력은 격자에서, **탄성은 입자 이웃 기울기**로 잰다.

    grid_ip_energy 는 탄성도 격자 왕복으로 잰다 -- 그러면 변형기울기가 뭉개져
    탄성항이 사실상 정보를 잃는다 (F=I 로 바꿔도 최적해가 거의 같았다). 여기서는
    ∇Δu 를 입자 이웃 최소제곱으로 직접 뽑아 그 경로를 피한다.
    """
    m_I, du_I, v_I, info, frac = p2g_increment(
        x, du, vel, mass, n_grid, grid_lim, free=free)
    w_free = torch.ones_like(m_I) if frac is None else (frac > 0.5).to(m_I.dtype)
    d = du_I - h * v_I
    e_in = (0.5 * w_free * m_I / (h * h) * (d * d).sum(-1)).sum()
    e_g = torch.zeros((), device=x.device, dtype=x.dtype)
    if g is not None:
        e_g = -(w_free * m_I * (du_I * g).sum(-1)).sum()
    if nb is None:
        nb = jac_neighbors(x.detach())
    gu = grad_from_points(x, du, nb)
    F_tr = (torch.eye(3, device=x.device, dtype=F.dtype) + gu) @ F
    psi, dlog = psi_of(F_tr, cfg, h)
    e_el = (vol * psi).sum()
    e_bc = bc_energy(x, du, mass, cfg, h, grid_lim, n_grid)
    tot = e_in + e_el + e_g + e_bc
    if norm is not None:
        tot = tot / norm
    # 보고용 항은 **텐서로** 돌려준다. float() 는 GPU 동기화라 스텝마다
    # 배치x4 회 파이프라인을 세운다 -- 출력하는 자리에서만 변환한다.
    return tot, dlog, F_tr, (e_in.detach(), e_el.detach(),
                             e_g.detach(), e_bc.detach())


def grid_ip_sub(x, du, vel, F, mass, vol, cfg, h, n_grid, grid_lim,
                g=None, norm=None, free=None, K=1):
    """프레임 변위를 **K 개 서브스텝**으로 나눠 증분 포텐셜의 합을 잰다.

    이것이 없으면 프레임 h 로 증분 포텐셜을 쓰게 되는데, 관성 계수가 m/(2h^2)
    이라 h 를 PG 의 서브스텝(5e-5)에서 프레임(1/60)으로 키우면 관성항이 9e-6 배로
    줄어 탄성항에 압도된다. 실측으로 그때 최적해는 교사에서 0.317% 벗어나고
    (정지가 0.466%) h 를 5e-5 로 낮추면 0.001% 로 일치한다 -- 즉 프레임 h 로는
    목적함수의 최소점이 교사가 아니다.

    변위를 등분해 x_k = x + (k/K) du 를 지나가는 경로로 보고, 각 구간을 h/K 로
    평가한 포텐셜을 더한다. 속도는 구간마다 갱신하고 F 도 같이 전진시킨다.
    """
    hs = h / float(max(K, 1))
    tot = torch.zeros((), device=x.device, dtype=x.dtype)
    xk, vk, Fk = x, vel, F
    dk = du / float(max(K, 1))
    info_sum = [0.0, 0.0, 0.0, 0.0]
    dlog_last = None
    F_last = F
    for _k in range(max(K, 1)):
        e, dlog, F_tr, info = grid_ip_energy(
            xk, dk, vk, Fk, mass, vol, cfg, hs, n_grid, grid_lim,
            g=g, norm=norm, free=free)
        tot = tot + e
        for _i in range(4):
            info_sum[_i] += info[_i]
        dlog_last, F_last = dlog, F_tr
        if _k + 1 < max(K, 1):
            xk = xk + dk
            vk = dk / hs
            Fk = plastic_step(F_tr, dlog)
    return tot, dlog_last, F_last, tuple(info_sum)


def grid_ip_energy(x, du, vel, F, mass, vol, cfg, h, n_grid, grid_lim,
                   g=None, norm=None, free=None, jac=None):
    """i-PG 의 목적함수를 **격자 증분** 기준으로 잰다.

        E = Σ_I m_I/(2h²)‖Δu_I − h v_I‖² + Σ_p V_p Ψ(F_p) − Σ_I m_I g·Δu_I
        F_p = (I + ∇Δu) F_p^n

    관성·중력은 격자에서, 탄성은 입자에서 잰다 -- MPM 이 힘을 만드는 자리와 같다.
    """
    m_I, du_I, v_I, info, frac = p2g_increment(
        x, du, vel, mass, n_grid, grid_lim, free=free)
    # 손잡이가 절반 넘게 실린 노드는 Dirichlet 으로 보고 관성·중력에서 뺀다.
    # 그 노드의 규정된 변위는 탄성항의 ∇Δu 를 통해 이웃 잔차에 그대로 들어간다.
    w_free = torch.ones_like(m_I) if frac is None else (frac > 0.5).to(m_I.dtype)
    # 중력은 **관성항의 목표에 미리 넣는다**: x̃ = x + h v + h² g.
    # 예전에는 `-m g·Δu` 로 따로 뒀는데, 한 스텝만 풀 때는 기울기가 같아 무해해도
    # 프레임을 K 서브스텝으로 나누면 각 구간이 자기 변위만 보게 되어 중력 일이
    # 1/K 로 줄어든다 (실측: K=104 에서 한 프레임 누적이 100 배 작았다).
    _gh = (h * h) * g if g is not None else 0.0
    d = du_I - h * v_I - _gh
    e_in = (0.5 * w_free * m_I / (h * h) * (d * d).sum(-1)).sum()
    e_g = torch.zeros((), device=x.device, dtype=x.dtype)
    if g is not None:
        # 보고용: 이 스텝에서 중력이 한 일 (기울기에는 이미 관성항으로 들어갔다)
        e_g = -(w_free * m_I * (du_I * g).sum(-1)).sum().detach()
    if jac is None:
        gu = g2p_grad(x, du_I, info, n_grid)
        F_tr = (torch.eye(3, device=x.device, dtype=F.dtype) + gu) @ F
    else:
        # 입자를 실제로 옮긴 **변형장 자신의** 야코비안으로 민다. 격자 B-스플라인
        # 공간미분(g2p_grad)은 셀 내부 재배열이나 스키닝 가중치를 통째로 무시해
        # 실제 사상과 다른 F 를 만든다.
        F_tr = jac.to(F.dtype) @ F
    psi, dlog = psi_of(F_tr, cfg, h)
    e_el = (vol * psi).sum()
    e_bc = bc_energy(x, du, mass, cfg, h, grid_lim, n_grid)
    tot = e_in + e_el + e_g + e_bc
    if norm is not None:
        tot = tot / norm
    # 보고용 항은 **텐서로** 돌려준다. float() 는 GPU 동기화라 스텝마다
    # 배치x4 회 파이프라인을 세운다 -- 출력하는 자리에서만 변환한다.
    return tot, dlog, F_tr, (e_in.detach(), e_el.detach(),
                             e_g.detach(), e_bc.detach())

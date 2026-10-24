"""셀 안 로컬 회전 사상 (--local_rot bubble, 사용자 10-24).

노드 출력 = 변위 U_k (3) + 회전 벡터 θ_k (3). 입자 p 가 든 사면체 (꼭짓점 X_k = 이 스텝 시작 노드 위치) 에서
  λ = barycentric(x),  c = (1/4) Σ X_k,  b(λ) = 256 λ0 λ1 λ2 λ3   (중심 1, 면 위 0)
  θ̃_k = cap_c(θ_k),  θ_p = Σ λ_k θ̃_k,  φ = b(λ) θ_p
  1) 로컬:   y = c + R(φ) (x − c),          R = 로드리게스 exp
  2) 글로벌: x' = Φ_c(y) = Σ λ_k(y) (X_k + U_k)       (셀 아핀 사상, 지금 선형 경로와 같은 것)
  -> x' − x = Σ λ_k(x) U_k + J_Φ (y − x),  J_Φ = I + Σ U_k ⊗ ∇λ_k   (Φ_c 가 아핀이라 정확)
     F_p' = J_Φ J_loc F_p,  J_loc = ∂y/∂x = R(φ) + ∂(R(φ)v)/∂φ · ∂φ/∂x   (v = x − c; 해석식, 이웃 추정 없음)
     ∂φ/∂x = θ_p ⊗ ∇b + b Σ_k θ̃_k ⊗ ∇λ_k,   ∇b = 256 Σ_k (Π_{j≠k} λ_j) ∇λ_k
θ = 0 이면 y − x 와 J_loc − I 가 정확히 0 (모든 항이 φ 또는 θ 에 곱해진다) -> 선형 경로와 비트 단위로 같다.

셀 밖으로 안 나가게 하는 방법: **회전각 상한** (셀마다, 꼭짓점 값에 매끄러운 tanh 상한)
  cap_c(θ) = θmax_c tanh(|θ|/θmax_c) θ/|θ|,   θmax_c = κ / (r_c (16 g_c + S_c))
  r_c = max_k |X_k − c|,  g_c = max_k |∇λ_k|,  S_c = Σ_k |∇λ_k|,  κ < 1 (기본 0.9)
증명: |θ̃_k| ≤ θmax_c -> |θ_p| ≤ θmax_c, 0 ≤ b ≤ 1, |∇b| ≤ 256 g_c e3(λ) ≤ 16 g_c (e3 ≤ 4/4³),
  ‖∂φ/∂x‖ ≤ θmax_c (16 g_c + S_c),  ‖∂(R v)/∂φ‖ = ‖R [v]× J_r(φ)‖ ≤ |v| ≤ r_c
  -> K = J_loc − R 에 대해 ‖Rᵀ K‖ ≤ κ < 1 -> J_loc = R (I + RᵀK) 의 특이값 ≥ 1 − κ, det J_loc ≥ (1−κ)³ > 0
  **셀 안 모든 점에서** (입자 위치만이 아니라). 면 위에서 b = 0 이라 y = x (경계 고정) 이고, 경계가 항등인
  국소 미분동형은 사면체를 자기 자신으로 보내는 전단사다 (차수 논증) -> 어떤 입자도 셀 밖으로 못 나간다.
  셀마다 꼭짓점 값을 따로 잘라도 공유 면 위에서는 b = 0 이라 양쪽 모두 y = x -> 이웃 셀과 연속.
장벽(로그 det) 대신 상한을 고른 이유: 출력만 최적화 표준 설정은 --inv_barrier 0 · --det_w 0 (장벽 없음) 이고
L-BFGS strong-Wolfe 선 탐색이 장벽 밖(무한대) 시험점을 다루지 않는다. 상한은 목적함수(증분 포텐셜)를 그대로 두고
입자 위치가 아니라 셀 전체에서 det > 0 을 보장한다.
"""
import os

import torch

KAPPA = 0.9
# 시험 전용: AF_LOCAL_ROT_ZERO=1 이면 θ 를 0 으로 (통합 시험에서 선형 실행과 대조)
_ZERO = bool(os.environ.get("AF_LOCAL_ROT_ZERO"))


def _skew(w):
    z = torch.zeros_like(w[..., 0])
    return torch.stack([torch.stack([z, -w[..., 2], w[..., 1]], -1),
                        torch.stack([w[..., 2], z, -w[..., 0]], -1),
                        torch.stack([-w[..., 1], w[..., 0], z], -1)], -2)


def _rod_coefs(t2):
    """A = sin t/t, B = (1−cos t)/t², C = A'(t)/t, D = B'(t)/t (t² 의 함수, t = 0 근처는 급수)."""
    sm = t2 < 0.25
    ts2 = torch.where(sm, torch.ones_like(t2), t2)
    t = ts2.sqrt()
    s, c = torch.sin(t), torch.cos(t)
    A_ = s / t
    B_ = (1 - c) / ts2
    C_ = (t * c - s) / (ts2 * t)
    D_ = (t * s - 2 * (1 - c)) / (ts2 * ts2)
    u = t2
    A = torch.where(sm, 1 - u / 6 + u * u / 120 - u * u * u / 5040, A_)
    B = torch.where(sm, 0.5 - u / 24 + u * u / 720 - u * u * u / 40320, B_)
    C = torch.where(sm, -1.0 / 3 + u / 30 - u * u / 840 + u * u * u / 45360, C_)
    D = torch.where(sm, -1.0 / 12 + u / 180 - u * u / 6720 + u * u * u / 453600, D_)
    return A, B, C, D


def _tanh_over(u2):
    """tanh(u)/u (u² 의 함수)."""
    sm = u2 < 1e-2
    us2 = torch.where(sm, torch.ones_like(u2), u2)
    us = us2.sqrt()
    return torch.where(sm, 1 - u2 / 3 + 2 * u2 * u2 / 15 - 17 * u2 * u2 * u2 / 315, torch.tanh(us) / us)


def cell_geom(npos, rows):
    Xv = npos[rows]                                                   # [N,4,3]
    Minv = torch.linalg.inv((Xv[:, 1:] - Xv[:, :1]).transpose(1, 2))
    dlam = torch.cat([-Minv.sum(1, keepdim=True), Minv], 1)          # [N,4,3] ∇λ_k
    c = Xv.mean(1)
    r_c = (Xv - c[:, None]).norm(dim=-1).amax(1)
    gn = dlam.norm(dim=-1)
    return Xv, dlam, c, r_c, gn.amax(1), gn.sum(1)


def theta_max(r_c, g_c, S_c, kappa=KAPPA):
    return kappa / (r_c * (16.0 * g_c + S_c))


def local_rot(x, npos, rows, lam, theta, du_lin, Du_lin, kappa=KAPPA, with_stats=False, cap=True):
    """x [N,3], npos [M,3], rows [N,4], lam [N,4] (선형 경로와 같은 barycentric), theta [M,3] 노드 회전 벡터,
    du_lin [N,3] = Σλ U, Du_lin [N,3,3] = J_Φ − I (선형 경로 값).
    -> (du, Du[, stats]) : du = x' − x, Du = J_Φ J_loc − I.
    stats [10] (기울기 없음): det J_loc 최소, 셀 밖 입자 수 (λ(y) < −1e-5 이면서 λ(x) ≥ −1e-5), λ(y) 최소, λ(x) 최소,
      |φ| 최대, |φ| 평균, |θ_p|/θmax 최대, ‖K‖_F 최대 (K = J_loc − R, 상계 ‖K‖₂ ≤ κ), θmax 중앙, 입자 수"""
    if _ZERO:
        theta = theta * 0.0
    Xv, dlam, c, r_c, g_c, S_c = cell_geom(npos, rows)
    tm = theta_max(r_c, g_c, S_c, kappa)                              # [N]
    thk = theta[rows]                                                 # [N,4,3]
    if cap:
        u2 = (thk * thk).sum(-1) / (tm * tm)[:, None]
        thk = _tanh_over(u2)[..., None] * thk                       # θmax tanh(|θ|/θmax) θ/|θ|
    thp = (lam[..., None] * thk).sum(1)                               # [N,3]
    l0, l1, l2, l3 = lam.unbind(1)
    b = 256.0 * l0 * l1 * l2 * l3
    P = torch.stack([l1 * l2 * l3, l0 * l2 * l3, l0 * l1 * l3, l0 * l1 * l2], 1)
    gb = 256.0 * (P[..., None] * dlam).sum(1)                         # [N,3]
    phi = b[:, None] * thp
    dphi = thp[:, :, None] * gb[:, None, :] + b[:, None, None] * (thk[..., :, None] * dlam[..., None, :]).sum(1)
    v = x - c
    t2 = (phi * phi).sum(-1)
    A, B, C, D = _rod_coefs(t2)
    pv = torch.cross(phi, v, dim=-1)
    ppv = torch.cross(phi, pv, dim=-1)
    ymx = A[:, None] * pv + B[:, None] * ppv                          # y − x = (R − I) v
    Sp, Sv = _skew(phi), _skew(v)
    M = (C[:, None, None] * pv[:, :, None] * phi[:, None, :] + D[:, None, None] * ppv[:, :, None] * phi[:, None, :]
         - A[:, None, None] * Sv - B[:, None, None] * (_skew(pv) + Sp @ Sv))      # ∂(R v)/∂φ
    RmI = A[:, None, None] * Sp + B[:, None, None] * (Sp @ Sp)
    K = M @ dphi
    JmI = RmI + K                                                     # J_loc − I
    JPhi = Du_lin + torch.eye(3, device=x.device, dtype=x.dtype)
    du = du_lin + (JPhi @ ymx[..., None]).squeeze(-1)
    Du = Du_lin + JPhi @ JmI
    if not with_stats:
        return du, Du
    with torch.no_grad():
        I3 = torch.eye(3, device=x.device, dtype=x.dtype)
        dJ = torch.linalg.det(I3 + JmI.detach())
        lam_y = lam.detach() + (dlam.detach() @ ymx.detach()[..., None]).squeeze(-1)
        lyx, lxx = lam_y.amin(1), lam.detach().amin(1)
        n_out = ((lyx < -1e-5) & (lxx >= -1e-5)).sum()
        pn = phi.detach().norm(dim=-1)
        st = torch.stack([dJ.min(), n_out.to(x.dtype), lyx.min(), lxx.min(), pn.max(), pn.mean(),
                          (thp.detach().norm(dim=-1) / tm).max(),
                          K.detach().flatten(1).norm(dim=-1).max(), tm.median(),
                          torch.full((), float(x.shape[0]), device=x.device, dtype=x.dtype)])
    return du, Du, st


STAT_KEYS = ("detJloc_min", "out_of_cell", "lam_y_min", "lam_x_min", "phi_max", "phi_mean",
             "thp_over_max", "K_fro_max", "theta_max_med", "n")
LAST = [None]          # advance 가 jvp 밖에서 넣는다 (마지막 평가의 stats 텐서)

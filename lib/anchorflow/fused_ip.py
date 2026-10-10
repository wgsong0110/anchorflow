"""ours (Lattice, Gregory 가중치) 의 증분 포텐셜 목적을 하나의 함수로 -- torch.compile 로 통합 커널을 만든다.

rep_ip 의 REP.yJ + energy (단일 물체, 탄성 jelly, 접촉 없음) 와 **같은 식**:
  dy, J  = Lattice.yJ (Gregory·방사형 가중치, ρ 매개 포함)
  F      = J Fe
  ψ      = μ|F − R|² + λ/2 (det F − 1)²,  R = 극분해 (Newton, 기울기는 포락선 정리로 R 을 뗀다)
           -- psi_of(jelly) 의 μΣ(σ−1)² + λ/2(Πσ−1)² 와 같다 (σ ≥ 0.01, det F > 0 일 때)
  E·NORM = [½/h² Σ m|x − x̃|² − Σ m g·(x − x̃) + Σ V ψ + 바닥 벌점] · NORM
σ < 0.01 이나 det F ≤ 0 인 입자가 있으면 bad > 0 을 돌려주고, 부르는 쪽이 그 평가만 예전 경로로 한다.
"""
import torch

from . import repmaps as rm


def cof3(A):
    """여인수 행렬 (A⁻ᵀ = cof / det)."""
    a, b, c = A[:, 0, 0], A[:, 0, 1], A[:, 0, 2]
    d, e, f = A[:, 1, 0], A[:, 1, 1], A[:, 1, 2]
    g, h, i = A[:, 2, 0], A[:, 2, 1], A[:, 2, 2]
    return torch.stack([e * i - f * h, f * g - d * i, d * h - e * g,
                        c * h - b * i, a * i - c * g, b * g - a * h,
                        b * f - c * e, c * d - a * f, a * e - b * d], -1).reshape(-1, 3, 3)


def polar_newton(F, iters=8):
    """크기 조정 Newton 극분해 R <- (γR + γ⁻¹R⁻ᵀ)/2 (기울기 없음, 해석적 역행렬이라 한 커널로 합쳐진다)."""
    R = F.detach()
    for _ in range(iters):
        d = rm.det3(R)
        ds = torch.where(d.abs() > 1e-20, d, torch.full_like(d, 1e-20))
        gm = ds.abs().pow(-1.0 / 3.0)[:, None, None]
        R = 0.5 * (gm * R + cof3(R) / (ds[:, None, None] * gm))
    return R


def weights_greg(rho_raw, rows, r, dr, w, dw, hl, aa):
    """Gregory·방사형 가중치 W [P,4] 와 그 공간 기울기 dW [P,4,3] (Lattice.yJ 와 같은 식)."""
    rho = hl * (0.05 + 0.95 * torch.sigmoid(rho_raw)[rows])
    ins = r < rho
    inner = 1.0 - aa * (torch.minimum(r, rho) / rho) ** 2
    q = 1.0 + (r - rho).clamp_min(0.0) / (0.5 * hl)
    outer = (1.0 - aa) / q
    psi_raw = torch.where(ins, inner, outer)
    psi = psi_raw.clamp_min(1e-6)
    dpsi = torch.where(ins, -2.0 * aa * r / (rho * rho), -(1.0 - aa) / (0.5 * hl) / (q * q)) * (psi_raw > 1e-6)
    g = w * psi
    dg = dw * psi[..., None] + (w * dpsi)[..., None] * dr
    G = g.sum(1, keepdim=True).clamp_min(1e-12)
    W = g / G
    dW = dg / G[..., None] - W[..., None] * dg.sum(1, keepdim=True) / G[..., None]
    return W, dW


def diag_u(M, rows, W, dW, sq, Fp, F, wv, ce, sla):
    """가우스-뉴턴 계량의 u 블록 대각 [M,3]: 관성 Σ sq² W² + ce·Σ wv² (|a|² + sla² (cof(F) a)_c²), a = Fpᵀ dW."""
    a = (Fp.transpose(1, 2)[:, None] @ dW[..., None]).squeeze(-1)          # [P,4,3]
    cof = cof3(F)                                                           # [P,3,3]
    ca = (cof[:, None] @ a[..., None]).squeeze(-1)                         # [P,4,3]
    el = (wv * wv)[:, None, None] * ((a * a).sum(-1, keepdim=True) + sla * sla * ca * ca)
    inr = ((sq * sq)[:, None] * W * W)[..., None].expand(-1, -1, 3)
    D = torch.zeros(M, 3, device=W.device, dtype=W.dtype)
    D.index_add_(0, rows.reshape(-1), (ce * el + inr).reshape(-1, 3))
    return D


def yJ_greg(u, rho_raw, rows, r, dr, w, dw, hl, aa):
    """repmaps.Lattice.yJ (greg) 와 같은 식."""
    U = u[rows]
    rho = hl * (0.05 + 0.95 * torch.sigmoid(rho_raw)[rows])
    ins = r < rho
    inner = 1.0 - aa * (torch.minimum(r, rho) / rho) ** 2
    q = 1.0 + (r - rho).clamp_min(0.0) / (0.5 * hl)
    outer = (1.0 - aa) / q
    psi_raw = torch.where(ins, inner, outer)
    psi = psi_raw.clamp_min(1e-6)
    dpsi = torch.where(ins, -2.0 * aa * r / (rho * rho), -(1.0 - aa) / (0.5 * hl) / (q * q)) * (psi_raw > 1e-6)
    g = w * psi
    dg = dw * psi[..., None] + (w * dpsi)[..., None] * dr
    G = g.sum(1, keepdim=True).clamp_min(1e-12)
    W = g / G
    dW = dg / G[..., None] - W[..., None] * dg.sum(1, keepdim=True) / G[..., None]
    return (W[..., None] * U).sum(1), rm.eye_plus(rm.outer_sum(U, dW))


def objective(theta, M3: int, rows, r, dr, w, dw, hl: float, aa: float, X_, xtil, Fe, VOL, MASS, g,
              zf: float, use_floor: bool, kfl: float, hdt: float, mu: float, la: float, norm: float):
    u = theta[:M3].reshape(-1, 3); rho_raw = theta[M3:]
    dy, J = yJ_greg(u, rho_raw, rows, r, dr, w, dw, hl, aa)
    x = X_ + dy
    F = rm.mm3(J, Fe)
    with torch.no_grad():
        R = polar_newton(F)
        dF = rm.det3(F)
        # 극분해 회전의 좌우 신장 S = Rᵀ F 의 대각이 σ 의 근사 -- 0.01 아래면 psi_of 의 자르기가 걸린다
        S = R.transpose(1, 2) @ F
        bad = ((dF <= 1e-6) | (S.diagonal(dim1=1, dim2=2).amin(1) < 0.0105)).sum()
    psi = mu * ((F - R) ** 2).sum((1, 2)) + 0.5 * la * (rm.det3(F) - 1.0) ** 2
    d = x - xtil
    e = 0.5 / (hdt * hdt) * (MASS * (d * d).sum(1)).sum() - (MASS * (d * g).sum(1)).sum() + (VOL * psi).sum()
    if use_floor:
        e = e + kfl * 0.5 / (hdt * hdt) * (MASS * (zf - x[:, 2]).clamp_min(0) ** 2).sum()
    return e * norm, bad



def res_inertia(x, M3: int, rows, r, dr, w, dw, hl: float, aa: float, sq):
    """관성 항의 가우스-뉴턴 잔차 √(NORM m/h²)·dy (θ 의 함수, ρ 포함)."""
    dy, _ = yJ_greg(x[:M3].reshape(-1, 3), x[M3:], rows, r, dr, w, dw, hl, aa)
    return (sq[:, None] * dy).reshape(-1)


OBJ = torch.compile(objective, dynamic=True)

"""학습 모델의 셀 안 로컬 변환: 출력만 최적화(rep_ip ours, repmaps.Lattice)와 **같은** Gregory·방사형 스키닝.

노드마다 변위 u (글로벌 변환) 와 반경 ρ_raw (로컬 변환의 방사형 감쇠 반경) 를 둔다. 입자 가중치는
  w_k = λ_k² / Σλ²  (Gregory),  ψ_k = 방사형 감쇠(r_k, ρ_k),  W_k = w_k ψ_k / Σ w ψ
변위 = Σ W_k u_k, 변위 기울기 ∇u = Σ u_k ⊗ ∇W_k (F = I + ∇u 는 호출 쪽에서). 식은 repmaps.Lattice.rebind/yJ 를 그대로 옮겼다 (a = 0.5,
ρ = h (0.05 + 0.95 σ(ρ_raw)) -- ρ_raw = 0 에서 시작하는 것도 같다). λ 는 그 사면체 네 꼭짓점 위치로 다시 계산한다.
"""
import torch

A_IN = 0.5


def gregory_skin(x, npos, rows, U, rho_raw, h):
    """x [N,3] 입자, npos [M,3] 노드 위치, rows [N,4] 입자의 사면체 꼭짓점, U [M,3], rho_raw [M], h 노드 간격.
    -> (du [N,3], Du [N,3,3]) -- Du 는 변위의 공간 기울기 ∇u (단위행렬 **없음**, g2p_jac_pre 와 같은 규약.
    deform_model.advance 가 F = I + Du 로 만든다)"""
    Xv = npos[rows]                                                  # [N,4,3]
    Minv = torch.linalg.inv((Xv[:, 1:] - Xv[:, :1]).transpose(1, 2))
    l3 = (Minv @ (x - Xv[:, 0])[..., None]).squeeze(-1)
    lam = torch.cat([1.0 - l3.sum(1, keepdim=True), l3], 1)        # [N,4]
    dlam = torch.cat([-Minv.sum(1, keepdim=True), Minv], 1)         # [N,4,3]
    d = x[:, None] - Xv
    r = d.norm(dim=-1)
    dr = d / r[..., None].clamp_min(1e-30)
    S = (lam * lam).sum(1, keepdim=True)
    w = lam * lam / S.clamp_min(1e-12)
    dS = (2 * lam[..., None] * dlam).sum(1, keepdim=True)
    dw = 2 * lam[..., None] * dlam / S[..., None] - (lam * lam)[..., None] * dS / (S * S)[..., None]
    rho = h * (0.05 + 0.95 * torch.sigmoid(rho_raw)[rows])
    ins = r < rho
    inner = 1.0 - A_IN * (torch.minimum(r, rho) / rho) ** 2
    q = 1.0 + (r - rho).clamp_min(0.0) / (0.5 * h)
    outer = (1.0 - A_IN) / q
    psi_raw = torch.where(ins, inner, outer)
    psi = psi_raw.clamp_min(1e-6)
    dpsi = torch.where(ins, -2.0 * A_IN * r / (rho * rho), -(1.0 - A_IN) / (0.5 * h) / (q * q)) * (psi_raw > 1e-6)
    g = w * psi
    dg = dw * psi[..., None] + (w * dpsi)[..., None] * dr
    G = g.sum(1, keepdim=True).clamp_min(1e-12)
    W = g / G
    dW = dg / G[..., None] - W[..., None] * dg.sum(1, keepdim=True) / G[..., None]
    Ur = U[rows]                                                     # [N,4,3]
    du = (W[..., None] * Ur).sum(1)
    Du = (Ur[..., :, None] * dW[..., None, :]).sum(1)
    return du, Du

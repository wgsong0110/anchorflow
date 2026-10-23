"""학습 모델의 셀 안 로컬 변형 (--gregory).

사용자 정의 (10-12, 10-23): **로컬 변형이 각 입자의 셀 안 위치(barycentric 좌표)를 조정하고, 글로벌 변형이
셀 자체(꼭짓점)를 옮기는 두 단계.** 격자가 고정(U = 0)이어도 로컬 변형만으로 셀 안 입자가 움직여야 한다.

노드마다 변위 U (글로벌) 와 반경 ρ_raw (로컬; 꼭짓점 방사형 함수의 꺾임 반경) 를 둔다.
  Gregory 가중치  w_k = λ_k² / Σλ²,  ψ_k = 방사형 감쇠(r_k, ρ_k),  W_k(ρ) = w_k ψ_k / Σ w ψ
  ρ = h (0.05 + 0.95 σ(ρ_raw)),  기준 반경 ρ_ref = ρ(ρ_raw = 0) (망 출력·변수의 시작값)
  1) 로컬:   λ'_k = λ_k + W_k(ρ) − W_k(ρ_ref)            (Σλ' = 1; ρ = ρ_ref 이면 λ' = λ)
  2) 글로벌: x' = Σ_k λ'_k Y_k,  Y_k = X_k + U_k         (셀 꼭짓점 현재 위치로 그 셀의 아핀 사상)
  -> du = x' − x = Σ λ'_k U_k + Σ (W_k(ρ) − W_k(ρ_ref)) (X_k − x)
     ∇u = Σ U_k ⊗ ∇λ'_k + Σ (X_k − x) ⊗ ∇(W_k(ρ) − W_k(ρ_ref))      (실제 야코비안, F = I + ∇u 는 호출 쪽)
성질: U = 0, ρ = ρ_ref -> du = 0;  ρ = ρ_ref -> 선형 barycentric (로컬 변형 없음) 과 정확히 같다;
      맞은편 면(λ_k = 0)에서 w_k 와 ∇w_k 가 0 이라 그 면 위 λ' 는 그 면의 꼭짓점(공유)으로만 정해져 셀 사이 연속.

gregory_skin_v0 은 10-12 ~ 10-23 의 옛 구현 (du = Σ W_k U_k, repmaps.Lattice 와 같다). U = 0 이면 ρ 가 어떻든
입자가 움직이지 않아 로컬 변형이 실제로는 없었다 (버그). 옛 덤프 진단(greg_split_el.py)·Lattice 대조용으로만 남긴다.
"""
import torch

A_IN = 0.5


def _geom(x, npos, rows):
    Xv = npos[rows]                                                  # [N,4,3]
    Minv = torch.linalg.inv((Xv[:, 1:] - Xv[:, :1]).transpose(1, 2))
    l3 = (Minv @ (x - Xv[:, 0])[..., None]).squeeze(-1)
    lam = torch.cat([1.0 - l3.sum(1, keepdim=True), l3], 1)        # [N,4]
    dlam = torch.cat([-Minv.sum(1, keepdim=True), Minv], 1)         # [N,4,3]
    d = x[:, None] - Xv                                              # x − X_k
    r = d.norm(dim=-1)
    dr = d / r[..., None].clamp_min(1e-30)
    S = (lam * lam).sum(1, keepdim=True)
    w = lam * lam / S.clamp_min(1e-12)
    dS = (2 * lam[..., None] * dlam).sum(1, keepdim=True)
    dw = 2 * lam[..., None] * dlam / S[..., None] - (lam * lam)[..., None] * dS / (S * S)[..., None]
    return lam, dlam, d, r, dr, w, dw


def _greg_W(r, dr, w, dw, rho_raw_v, h):
    """rho_raw_v [N,4] (각 입자의 네 꼭짓점 반경) -> W [N,4], ∇W [N,4,3]"""
    rho = h * (0.05 + 0.95 * torch.sigmoid(rho_raw_v))
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
    return W, dW


def gregory_skin(x, npos, rows, U, rho_raw, h, return_parts=False):
    """x [N,3] 입자, npos [M,3] 노드 위치, rows [N,4] 입자의 사면체 꼭짓점, U [M,3], rho_raw [M], h 노드 간격.
    -> (du [N,3], Du [N,3,3]) -- Du 는 변위의 공간 기울기 ∇u (단위행렬 **없음**, g2p_jac_pre 와 같은 규약.
    deform_model.advance 가 F = I + Du 로 만든다).
    return_parts=True 면 (du, Du, lam, lam') 도 준다 (진단용)."""
    lam, dlam, d, r, dr, w, dw = _geom(x, npos, rows)
    W, dW = _greg_W(r, dr, w, dw, rho_raw[rows], h)
    W0, dW0 = _greg_W(r, dr, w, dw, torch.zeros_like(r), h)
    dl = W - W0                                                      # 로컬 단계의 barycentric 좌표 변화
    ddl = dW - dW0
    lam2 = lam + dl
    dlam2 = dlam + ddl
    Ur = U[rows]                                                     # [N,4,3]
    # x' − x = Σ λ'_k (X_k + U_k) − x = Σ λ'_k U_k + Σ dl_k (X_k − x)   (Σ dl = 0, Σ λ X = x)
    du = (lam2[..., None] * Ur).sum(1) - (dl[..., None] * d).sum(1)
    Du = (Ur[..., :, None] * dlam2[..., None, :]).sum(1) - (d[..., :, None] * ddl[..., None, :]).sum(1)
    if return_parts:
        return du, Du, lam, lam2
    return du, Du


def gregory_skin_v0(x, npos, rows, U, rho_raw, h):
    """옛 구현 (버그): du = Σ W_k U_k, ∇u = Σ U_k ⊗ ∇W_k. repmaps.Lattice.yJ 와 같다."""
    lam, dlam, d, r, dr, w, dw = _geom(x, npos, rows)
    W, dW = _greg_W(r, dr, w, dw, rho_raw[rows], h)
    Ur = U[rows]
    du = (W[..., None] * Ur).sum(1)
    Du = (Ur[..., :, None] * dW[..., None, :]).sum(1)
    return du, Du

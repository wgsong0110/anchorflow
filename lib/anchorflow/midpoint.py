"""암시적 중점법(implicit midpoint) 적분기 -- 출력만 최적화의 `--integrator midpoint`.

유도 (질량 M, 탄성 포텐셜 Ψ, 외력 가속 g, 스텝 h)
-------------------------------------------------
중점법:  x_{n+1} = x_n + h (v_n + v_{n+1})/2,   v_{n+1} = v_n + h a(y),
         y = (x_n + x_{n+1})/2,                  a(y) = g − M⁻¹∇Ψ(y).

(1) 첫 식에서 x_{n+1} = 2y − x_n 이므로  h (v_n + v_{n+1})/2 = 2 (y − x_n)
    →  v_{n+1} = (4/h)(y − x_n) − v_n.
(2) 이를 둘째 식에 넣으면  (4/h)(y − x_n) − 2 v_n = h a(y)
    →  y − x_n = (h/2) v_n + (h/2)² (g − M⁻¹∇Ψ(y)).
(3) 이것은 스텝 h' = h/2 짜리 후퇴 오일러의 정류 조건 그대로다. 즉 u = y − x_n 은
        min_u  Σ_p m_p/(2h'²) ‖u_p − h' v_p − h'² g‖² + Σ_p V_p Ψ((I+∇u)_p F_p)
    (기존 증분 포텐셜에서 h → h/2, 관성 목표 h' v_n + h'² g; 중력은 기존과 같이
    관성 목표 안에 넣는다) 의 최소점이다. 기울기 = M(u − h'v − h'²g)/h'² + ∇Ψ = 0
    ⇔ u = h'v + h'²(g − M⁻¹∇Ψ) 로 (2) 와 같다.
(4) 끝 상태:  x_{n+1} = x_n + 2u,  v_{n+1} = (4/h) u − v_n.
    입자 위치는 노드 변위의 선형 보간(bary)이므로 입자 2u = 노드 2u_mid 의 보간이고,
    변형 사상 x ↦ x + 2u(x) 의 야코비안은 I + 2∇u = 2J_mid − I  (J_mid = I + ∇u).
    따라서  F_{n+1} = (2 J_mid − I) F_n.

에너지: 선형계(Ψ 2차)에서 중점법은 ½vᵀMv + Ψ 를 정확히 보존한다 (진폭비 1).
후퇴 오일러는 고유진동수 ω 모드의 진폭을 스텝마다 1/√(1+(ωh)²) 배 (에너지는
1/(1+(ωh)²) 배) 줄인다.

주의: J_mid 가 안 뒤집혀도 (det J_mid > 0) 2J_mid − I 는 뒤집힐 수 있다 (예:
J_mid 의 특이값 < 1/2). "셀 안 뒤집힘 ⇒ 입자 안 뒤집힘" 보장이 깨지므로 호출 쪽이
외삽 사상의 det ≤ 0 셀·입자 수를 기록한다 (막지 않는다).

경계조건: 속도가 정해진 입자(모든 꼭짓점 노드가 구속된 입자)는 위치가 노드
구속으로 이미 맞으므로 (중점 u = h' v_bc → 끝 2u = h v_bc), 속도만 (4/h)u − v_n
대신 정해진 속도 2u/h (= v_bc) 로 둔다. 그러지 않으면 화분처럼 v_bc = 0 인
입자가 v_{n+1} = −v_n 으로 스텝마다 부호가 뒤집힌다 (첫 프레임 균일 초기속도).
"""
import torch


def mid_to_end(x, x_mid, v, h, pinned=None):
    """중점 상태 -> 끝 상태 (x_{n+1}, v_{n+1}).

    x, x_mid, v: [N,3]; h: 프레임(스텝) 길이; pinned: [N] bool (속도가 정해진 입자).
    """
    u = x_mid - x
    x_end = x + 2.0 * u
    v_end = (4.0 / float(h)) * u - v
    if pinned is not None:
        v_end = torch.where(pinned.unsqueeze(-1), (2.0 / float(h)) * u, v_end)
    return x_end, v_end


def extrap_jac(J):
    """중점 사상 야코비안 J = I + ∇u -> 끝 사상 야코비안 I + 2∇u = 2J − I."""
    I3 = torch.eye(3, device=J.device, dtype=J.dtype)
    return 2.0 * J - I3

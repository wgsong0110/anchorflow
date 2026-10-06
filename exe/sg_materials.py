"""Spring-Gaus 에 **소성·점소성·파괴**를 넣는다 (그쪽에는 탄성밖에 없다).

그쪽 모델은 스프링-질량이라 구성식이 하나뿐이다. 세 물성을 스프링 수준의
표준 확장으로 더한다 -- 전부 **쉬는 길이 l0** 와 **스프링 끊김**으로 표현된다.

  탄소성 : |ε| 가 항복 변형률을 넘으면 그 초과분만큼 l0 를 그 자리에서 옮긴다
           (율에 무관). 소성 흐름이 즉시 완료되는 극한.
  점소성 : 초과분을 한 번에 옮기지 않고 완화시간 tau 로 끌고 간다.
           한 서브스텝에 옮기는 비율 = dt/(dt+tau)  (맥스웰 요소)
  파괴   : 변형률이 임계값을 넘은 스프링을 **영구히 끊는다** (K=0).

항복 변형률과 완화시간은 MPM 쪽에서 쓴 물성 상수에서 그대로 끌어온다.
E=2e6, nu=0.3 -> mu=E/(2(1+nu))=7.69e5.
  탄소성: plasticine yield_stress 1e4  -> eps_y = ys/E = 5.0e-3
  점소성: foam yield_stress 5e3        -> eps_y = 2.5e-3
          plastic_viscosity 10          -> tau = eta/(2 mu) = 6.5e-6 초
파괴의 임계 변형률만은 대응되는 MPM 상수가 없다(그쪽은 Cam-Clay 항복면 +
Borden 손상이라 1 차원 스프링으로 옮길 값이 없다). 그래서 **유일한 자유
손잡이**로 두고 표에 값을 명시한다 (기본 0.10 = 10% 늘어나면 끊김).
"""
from __future__ import annotations

import types

import torch

E_MPM, NU_MPM = 2e6, 0.3
MU_MPM = E_MPM / (2.0 * (1.0 + NU_MPM))

MAT = {
    "elastoplastic": dict(kind="plastic", eps_y=1e4 / E_MPM),
    "viscoplastic": dict(kind="visco", eps_y=5e3 / E_MPM,
                         tau=10.0 / (2.0 * MU_MPM)),
    "fracture": dict(kind="break", eps_break=0.10),
}


def attach(sim, material, eps_break=0.10):
    """시뮬레이터에 물성을 붙인다 (탄성이면 아무것도 안 한다).

    `sim.step` 을 감싸 서브스텝마다 쉬는 길이와 끊김 마스크를 갱신한다.
    원래 힘 계산(compute_force)은 손대지 않는다 -- 그쪽 코드 그대로 쓰고,
    바뀐 l0 와 마스크만 흘려 넣는다.
    """
    if material == "elastic":
        return sim
    cfg = dict(MAT[material])
    if material == "fracture":
        cfg["eps_break"] = eps_break
    kind = cfg["kind"]
    sim._l0_ref = sim.origin_len.detach().clone()
    sim._k_mask = torch.ones_like(sim.origin_len)
    orig_step = sim.step

    def step(self, xyz, v, K, m, rebound_k, fric_k, damp, dt):
        xyz, v = orig_step(xyz=xyz, v=v, K=K * self._k_mask, m=m,
                           rebound_k=rebound_k, fric_k=fric_k, damp=damp,
                           dt=dt)
        with torch.no_grad():
            cur = torch.norm(xyz[self.knn_index] - xyz.unsqueeze(1), dim=2)
            st = (cur - self.origin_len) / (self.origin_len + self.eps)
            if kind == "break":
                self._k_mask *= (st.abs() <= cfg["eps_break"]).to(st.dtype)
            else:
                over = (st.abs() - cfg["eps_y"]).clamp_min(0.0) * torch.sign(st)
                rate = 1.0 if kind == "plastic" else dt / (dt + cfg["tau"])
                self.origin_len += rate * over * self.origin_len
        return xyz, v

    sim.step = types.MethodType(step, sim)
    sim._af_material = material
    sim._af_cfg = cfg
    return sim


def reset(sim):
    """롤아웃마다 소성 상태(쉬는 길이·끊김)를 초기로 되돌린다."""
    if hasattr(sim, "_l0_ref"):
        with torch.no_grad():
            sim.origin_len.copy_(sim._l0_ref)
            sim._k_mask.fill_(1.0)

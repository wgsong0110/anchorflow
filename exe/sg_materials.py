"""Spring-Gaus 에 **소성·점소성·파괴**를 넣는다 (그쪽에는 탄성밖에 없다).

그쪽 모델은 스프링-질량이라 구성식이 하나뿐이다. 세 물성을 스프링 수준의
표준 확장으로 더한다 -- 전부 **쉬는 길이 l0** 와 **스프링 끊김**으로 표현된다.

  탄소성 : |ε| 가 항복 변형률을 넘으면 그 초과분만큼 l0 를 그 자리에서 옮긴다
           (율에 무관). 소성 흐름이 즉시 완료되는 극한.
  점소성 : 초과분을 한 번에 옮기지 않고 완화시간 tau 로 끌고 간다.
           한 서브스텝에 옮기는 비율 = dt/(dt+tau)  (맥스웰 요소)
  파괴   : 변형률이 임계값을 넘은 스프링을 **영구히 끊는다** (K=0).

항복응력과 점성은 MPM 쪽 물성 상수를 그대로 쓰고(plasticine ys 1e4,
foam ys 5e3 · eta 10), 변형률로 바꾸는 분모만은 **그 스프링망의 유효 영률**
로 잰다 -- MPM 의 E 를 그대로 쓰면 파라미터화가 다른 두 모델을 섞는 셈이다.
등방 스프링망의 변형에너지가 U = (1/10) eps^2 sum k l0^2 이므로
  E_eff = sum k l0^2 / (10 V),   eps_y = ys / E_eff
실측: lego E_eff 1.98e6, mic 1.53e6 (MPM 의 2e6 과 거의 같다).

⚠ 변형률이 항복에 닿지 않는 장면이 있다 (lego 는 최대 0.17% 로 항복
0.5% 에 못 미친다 -- `--phase diag` 로 실측). 그 칸은 탄성과 같은 궤적이
나오는 게 맞고, 조용히 지나가지 않도록 attach 가 유효 영률과 항복
변형률을, 롤아웃이 끊긴 스프링 비율을 찍는다.
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
    "elastoplastic": dict(kind="plastic", ys=1e4),
    "viscoplastic": dict(kind="visco", ys=5e3, eta=10.0),
    "fracture": dict(kind="break", eps_break=0.10),
}


def eff_modulus(sim):
    """피팅된 스프링망의 유효 영률 (등방 가정, 방향 쌍 중복을 1/2 로)."""
    with torch.no_grad():
        l0 = sim.origin_len
        gk = sim.global_k
        kv = 10.0 ** (gk.reshape(-1, 1) if gk.dim() == 1 else gk)
        if kv.shape != l0.shape:
            kv = kv.expand_as(l0)
        k_s = kv / (l0 + sim.eps)              # forward 의 K (힘 = dl * K)
        bb = sim.init_xyz.max(0).values - sim.init_xyz.min(0).values
        vol = float(bb[0] * bb[1] * bb[2])
        return float(0.5 * (k_s * l0 ** 2).sum() / (5.0 * vol)), vol


def attach(sim, material, eps_break=0.10):
    """시뮬레이터에 물성을 붙인다 (탄성이면 아무것도 안 한다).

    `sim.step` 을 감싸 서브스텝마다 쉬는 길이와 끊김 마스크를 갱신한다.
    원래 힘 계산(compute_force)은 손대지 않는다 -- 그쪽 코드 그대로 쓰고,
    바뀐 l0 와 마스크만 흘려 넣는다.
    """
    if material == "elastic":
        return sim
    cfg = dict(MAT[material])
    E_eff, vol = eff_modulus(sim)
    cfg["E_eff"], cfg["volume"] = E_eff, vol
    if material == "fracture":
        cfg["eps_break"] = eps_break
    else:
        cfg["eps_y"] = cfg["ys"] / E_eff
        if material == "viscoplastic":
            cfg["tau"] = cfg["eta"] / (2.0 * E_eff / (2.0 * (1.0 + NU_MPM)))
    kind = cfg["kind"]
    print(f"[확장] {material}  E_eff {E_eff:.4e} (MPM {E_MPM:.1e})  "
          + (f"eps_break {cfg['eps_break']:.4f}" if kind == "break"
             else f"eps_y {cfg['eps_y']:.5f}"
                  + (f"  tau {cfg['tau']:.3e}s" if "tau" in cfg else "")),
          flush=True)
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

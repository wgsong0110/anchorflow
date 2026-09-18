"""PhysGaussian 에 "탄성 + 인장 손상" 재질을 더한다 (수정은 한 번만 먹는다).

왜 필요한가. 지금 있는 두 갈래가 둘 다 우리가 원하는 그림이 아니다.

  PhysGaussian plasticine  항복응력 연화뿐이라 소성으로 늘어나기만 하고 안 끊긴다
  GaussianFluent CD-MPM    Cam-Clay 계열이라 흙처럼 전체가 흘러내린다

원하는 것은 **일정 변형까지는 탄성으로 버티다가 그 너머에서 손상이 쌓여 끊기는**
것이다. 그래서 소성 되돌림이 아예 없는 탄성(jelly 와 같은 fixed corotated)에
인장 손상만 얹는다. 입자마다

    s1 = max singular value of F        # 그 자리에서 가장 크게 늘어난 배율
    s1 > xi  이면  d <- min(1, d + softening * (s1 - xi))
    stress <- (1 - d) * kirchoff_FCR(F)

d 가 1 이 되면 그 입자는 응력을 전혀 내지 않으므로 그 자리에서 재질이 끊긴다.
소성 흐름이 없어 응력이 **가장 가는 단면에 몰리고**, 그래서 손상이 거기서 먼저
쌓인다 -- 흙처럼 전체가 무너지지 않는 이유가 이것이다.

구조체를 건드리지 않으려고 이미 있는 자리를 빌린다.

    문턱      model.xi          (탄성 재질에서는 안 쓰던 값)
    쌓는 속도 model.softening   (von Mises 손상용이라 여기서는 안 쓴다)
    손상 상태 state.particle_Jp (소성 상태 변수, 탄성 재질에서는 안 쓴다)

config 에서는 `"material": "elastic_damage", "xi": 1.1, "softening": 0.05` 처럼 쓴다.
"""
from __future__ import annotations

import argparse
import os

MARK = "        elif model.material == 8:  # [anchorflow] 탄성 + 인장 손상"

OLD_DISPATCH = """        elif model.material == 5:
            state.particle_F[p] = von_mises_return_mapping_with_damage(
                state.particle_F_trial[p], model, p
            )
        else:  # elastic
            state.particle_F[p] = state.particle_F_trial[p]"""
NEW_DISPATCH = """        elif model.material == 5:
            state.particle_F[p] = von_mises_return_mapping_with_damage(
                state.particle_F_trial[p], model, p
            )
""" + MARK + """
            # 소성 되돌림을 하지 않는다. 변형구배를 그대로 둬야 응력이 가는 단면에
            # 몰리고, 그래야 손상이 거기서 먼저 쌓인다.
            state.particle_F[p] = state.particle_F_trial[p]
        else:  # elastic
            state.particle_F[p] = state.particle_F_trial[p]"""

OLD_STRESS = """        if model.material == 0 or model.material == 5:
            stress = kirchoff_stress_FCR(
                state.particle_F[p], U, V, J, model.mu[p], model.lam[p]
            )"""
NEW_STRESS = """        if model.material == 0 or model.material == 5 or model.material == 8:
            stress = kirchoff_stress_FCR(
                state.particle_F[p], U, V, J, model.mu[p], model.lam[p]
            )
        if model.material == 8:
            # 가장 크게 늘어난 배율이 문턱을 넘은 만큼만 손상을 쌓는다. 한 번 쌓인
            # 손상은 줄지 않으므로 파괴는 비가역이다.
            s1 = wp.max(sig[0], wp.max(sig[1], sig[2]))
            d = state.particle_Jp[p]
            if s1 > model.xi:
                d = wp.min(1.0, d + model.softening * (s1 - model.xi))
                state.particle_Jp[p] = d
            stress = (1.0 - d) * stress"""

OLD_MAT = """            elif kwargs["material"] == "plasticine":
                self.mpm_model.material = 5"""
NEW_MAT = """            elif kwargs["material"] == "plasticine":
                self.mpm_model.material = 5
            elif kwargs["material"] == "elastic_damage":
                # [anchorflow] 탄성 + 인장 손상. 문턱은 xi, 쌓는 속도는 softening,
                # 입자별 손상은 particle_Jp 에 든다.
                self.mpm_model.material = 8"""

ap = argparse.ArgumentParser()
ap.add_argument("--pg", required=True)
a = ap.parse_args()

pu = os.path.join(a.pg, "mpm_solver_warp", "mpm_utils.py")
ps = os.path.join(a.pg, "mpm_solver_warp", "mpm_solver_warp.py")
s = open(pu).read()
if MARK in s:
    raise SystemExit("이미 들어가 있다: " + pu)
for old in (OLD_DISPATCH, OLD_STRESS):
    if old not in s:
        raise SystemExit("붙일 자리를 못 찾았다:\n" + old[:80])
s = s.replace(OLD_DISPATCH, NEW_DISPATCH, 1).replace(OLD_STRESS, NEW_STRESS, 1)
open(pu, "w").write(s)

t = open(ps).read()
if OLD_MAT not in t:
    raise SystemExit("재질 이름 표를 못 찾았다")
open(ps, "w").write(t.replace(OLD_MAT, NEW_MAT, 1))
print("고쳤다:", pu, "와", ps, flush=True)
print("EDAMAGE_OK", flush=True)

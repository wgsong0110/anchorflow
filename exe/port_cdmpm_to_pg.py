"""GaussianFluent 의 CD-MPM(파괴) 재질을 PhysGaussian 으로 **그대로** 옮긴다.

PhysGaussian 에는 파괴가 되는 구성식이 아예 없다 (jelly/metal/sand/foam/snow/
plasticine 뿐). 그래서 두 솔버로 "같은 파괴" 를 보여줄 수가 없었다.

옮기는 방식이 중요하다 -- 다시 구현하지 않고 **GF 소스에서 함수 본문을 그대로
읽어다 붙인다**. 그래야 두 쪽이 같은 코드를 돌린다는 것이 보장된다.

옮기는 것:
  NonAssociativeCamClay_return_mapping   되돌림 (비연관 Cam-Clay + 손상)
  kirchoff_stress_neoHookeanBoarden      그 재질의 응력
  MPMModelStruct 의 kappa / beta / M     그 둘이 쓰는 필드
  set_parameters_dict 의 alpha_0 / beta / M   초기 상태와 인장 강도

PhysGaussian 의 material 은 스칼라, GF 는 입자별 배열이라 디스패치 문법이 다르다.
그 한 줄만 바꿔 붙인다 (model.material[p] -> model.material).
"""
from __future__ import annotations

import argparse
import os
import re

MARK = "# [anchorflow] GaussianFluent 의 CD-MPM 재질을 그대로 옮겨왔다"


def grab(src, name):
    """`@wp.func` 데코레이터부터 다음 데코레이터 직전까지를 통째로 떼어낸다."""
    i = src.index(f"def {name}(")
    j = src.rindex("@wp.func", 0, i)
    k = src.find("\n@wp.", i)
    if k < 0:
        k = len(src)
    return src[j:k].rstrip() + "\n"


ap = argparse.ArgumentParser()
ap.add_argument("--gf", required=True)
ap.add_argument("--pg", required=True)
a = ap.parse_args()

gf_u = open(os.path.join(a.gf, "mpm_solver_warp", "mpm_utils.py")).read()
pu = os.path.join(a.pg, "mpm_solver_warp", "mpm_utils.py")
pw = os.path.join(a.pg, "mpm_solver_warp", "warp_utils.py")
ps = os.path.join(a.pg, "mpm_solver_warp", "mpm_solver_warp.py")
s = open(pu).read()
if MARK in s:
    raise SystemExit("이미 옮겨져 있다")

fns = (grab(gf_u, "NonAssociativeCamClay_return_mapping")
       + "\n\n" + grab(gf_u, "kirchoff_stress_neoHookeanBoarden"))
fns = fns.replace("model.material[p]", "model.material")
anchor = "@wp.kernel\ndef compute_mu_lam_from_E_nu("
s = s.replace(anchor, MARK + "\n" + fns + "\n\n" + anchor, 1)

# 디스패치 두 곳
OLD_D = """        elif model.material == 5:
            state.particle_F[p] = von_mises_return_mapping_with_damage(
                state.particle_F_trial[p], model, p
            )"""
s = s.replace(OLD_D, OLD_D + """
        elif model.material == 7:      # [anchorflow] CD-MPM
            state.particle_F[p] = NonAssociativeCamClay_return_mapping(
                state.particle_F_trial[p], state, model, p
            )""", 1)
OLD_S = """        if model.material == 1:
            stress = kirchoff_stress_StVK("""
s = s.replace(OLD_S, """        if model.material == 7:      # [anchorflow] CD-MPM
            stress = kirchoff_stress_neoHookeanBoarden(
                state.particle_F[p], U, V, J, sig, model.mu[p], model.lam[p],
                model.kappa[p], state
            )
""" + OLD_S, 1)
# kappa 계산 한 줄 (GF 와 같은 식)
s = s.replace("""    model.lam[p] = (
        model.E[p] * model.nu[p] / ((1.0 + model.nu[p]) * (1.0 - 2.0 * model.nu[p]))
    )""", """    model.lam[p] = (
        model.E[p] * model.nu[p] / ((1.0 + model.nu[p]) * (1.0 - 2.0 * model.nu[p]))
    )
    model.kappa[p] = 2.0 * model.mu[p] / 3.0 + model.lam[p]   # [anchorflow]""", 1)
open(pu, "w").write(s)

w = open(pw).read()
if "kappa" not in w:
    w = w.replace("""    material: int""",
                  """    material: int
    kappa: wp.array(dtype=float)       # [anchorflow] CD-MPM
    beta: wp.array(dtype=float)        # [anchorflow] CD-MPM
    M: float                           # [anchorflow] CD-MPM""", 1)
    open(pw, "w").write(w)

t = open(ps).read()
if "[anchorflow] CD-MPM" not in t:
    t = t.replace("""        self.mpm_model.gravitational_accelaration = wp.vec3(0.0, 0.0, 0.0)""",
                  """        self.mpm_model.gravitational_accelaration = wp.vec3(0.0, 0.0, 0.0)
        # [anchorflow] CD-MPM 용 필드
        self.mpm_model.kappa = wp.zeros(shape=n_particles, dtype=float,
                                        device=device)
        self.mpm_model.beta = wp.full(shape=n_particles, value=1.0,
                                      dtype=wp.float32, device=device)
        self.mpm_model.M = 1.0""", 1)
    t = t.replace("""            elif kwargs["material"] == "plasticine":
                self.mpm_model.material = 5""",
                  """            elif kwargs["material"] == "plasticine":
                self.mpm_model.material = 5
            elif kwargs["material"] == "watermelon":   # [anchorflow] CD-MPM
                self.mpm_model.material = 7""", 1)
    t = t.replace("""        if "hardening" in kwargs:""",
                  """        if "alpha_0" in kwargs:            # [anchorflow] CD-MPM 초기 상태
            self.mpm_state.particle_Jp = wp.full(
                shape=self.n_particles, value=kwargs["alpha_0"],
                dtype=wp.float32, device=device)
        if "beta" in kwargs:               # [anchorflow] CD-MPM 인장 강도
            self.mpm_model.beta = wp.full(
                shape=self.n_particles, value=kwargs["beta"],
                dtype=wp.float32, device=device)
        if "hardening" in kwargs:""", 1)
    t = t.replace("""            self.mpm_model.alpha = wp.sqrt(2.0 / 3.0) * 2.0 * sin_phi / (3.0 - sin_phi)""",
                  """            self.mpm_model.alpha = wp.sqrt(2.0 / 3.0) * 2.0 * sin_phi / (3.0 - sin_phi)
            self.mpm_model.M = (self.mpm_model.alpha * 3.0
                                / wp.sqrt(2.0 / 3.0))   # [anchorflow] CD-MPM""", 1)
    open(ps, "w").write(t)
# decode_param 이 CD-MPM 키를 걸러내면 옮겨봐야 값이 안 들어간다.
# GF 는 alpha_0 기본 -0.04, beta 기본 2 를 넣는데 PhysGaussian 은 둘 다 없어서
# 초기 강도 p0 = kappa*(1e-5 + sinh(xi*max(-logJp,0))) 가 사실상 0 이 된다.
pd = os.path.join(a.pg, "utils", "decode_param.py")
dsrc = open(pd).read()
if "alpha_0" not in dsrc:
    OLD_H = """    if "hardening" in sim_params.keys():
        material_params["hardening"] = sim_params["hardening"]"""
    if OLD_H not in dsrc:
        raise SystemExit("decode_param 에서 hardening 지점을 못 찾았다")
    dsrc = dsrc.replace(OLD_H, """    # [anchorflow] CD-MPM 키. GF 의 기본값과 같게 둔다
    material_params["alpha_0"] = sim_params.get("alpha_0", -0.04)
    material_params["beta"] = sim_params.get("beta", 2.0)

""" + OLD_H, 1)
    open(pd, "w").write(dsrc)
    print("decode_param 도 열었다:", pd, flush=True)

print("옮겼다:", pu, pw, ps, flush=True)
print("PORT_OK", flush=True)

"""GF 의 warp MPM 에 **눈(snow) 재질을 실제로 넣는다**.

`material_2_num` 에 snow 가 4 번으로 있지만, `compute_stress_from_F_trial` 에는
4 번 갈래가 **되돌림에도 응력에도 없다** -- 되돌림은 `else: F = F_trial` 로 빠지고
응력은 어떤 `if` 에도 안 걸려 **0 으로 남는다**. 즉 번호만 있고 모델이 없다.
그대로 돌리면 응력 없는 먼지가 되어 그냥 흩어진다.

Stomakhin et al. 2013 의 눈을 넣는다. 둘뿐이다.
  되돌림  특이값을 [1-theta_c, 1+theta_s] 로 자르고, 잘라낸 만큼 Jp 에 쌓는다
  응력    고정 코로테이션인데 mu, lam 에 **exp(xi(1-Jp))** 를 곱한다
          -- 눌려서 Jp 가 작아질수록 단단해지는 것이 눈의 특징이다

theta_c=2.5e-2, theta_s=7.5e-3 은 논문 값을 그대로 쓰고, 경화 계수는 config 에
이미 있는 `xi` 를 쓴다. **`alpha_0` 를 1.0 으로 줘야 한다** -- GF 는 particle_Jp 를
alpha_0 로 채우는데 눈의 Jp 는 부피비라 1 에서 시작해야 한다.

  python exe/patch_gf_snow.py --gf <GaussianFluent>
"""
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("--gf", required=True)
a = ap.parse_args()

MARK = "# [anchorflow] 눈(snow) 재질"
FUNCS = '''

''' + MARK + ''' -- Stomakhin et al. 2013
@wp.func
def snow_return_mapping(F_trial: wp.mat33, state: MPMStateStruct,
                        model: MPMModelStruct, p: int):
    U = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    V = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    sig = wp.vec3(0.0)
    wp.svd3(F_trial, U, sig, V)

    theta_c = 2.5e-2
    theta_s = 7.5e-3
    J_old = sig[0] * sig[1] * sig[2]
    s0 = wp.clamp(sig[0], 1.0 - theta_c, 1.0 + theta_s)
    s1 = wp.clamp(sig[1], 1.0 - theta_c, 1.0 + theta_s)
    s2 = wp.clamp(sig[2], 1.0 - theta_c, 1.0 + theta_s)
    J_new = s0 * s1 * s2
    # 잘라낸 부피비를 소성 쪽에 쌓는다
    state.particle_Jp[p] = state.particle_Jp[p] * J_old / wp.max(J_new, 1e-8)
    sig_new = wp.mat33(s0, 0.0, 0.0, 0.0, s1, 0.0, 0.0, 0.0, s2)
    return U * sig_new * wp.transpose(V)


@wp.func
def kirchoff_stress_snow(F: wp.mat33, U: wp.mat33, V: wp.mat33, J: float,
                         mu: float, lam: float, xi: float, Jp: float):
    # 눌릴수록(Jp 가 작아질수록) 단단해진다
    h = wp.exp(xi * (1.0 - Jp))
    mu_h = mu * h
    lam_h = lam * h
    R = U * wp.transpose(V)
    id = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    return 2.0 * mu_h * (F - R) * wp.transpose(F) + id * lam_h * J * (J - 1.0)
'''

p = os.path.join(a.gf, "mpm_solver_warp", "mpm_utils.py")
s = open(p).read()
if MARK in s:
    print("이미 되어 있다"); raise SystemExit(0)

OLD_RM = """        else:  # elastic
            state.particle_F[p] = state.particle_F_trial[p]"""
NEW_RM = """        elif model.material[p] == 4:  # [anchorflow] snow
            state.particle_F[p] = snow_return_mapping(
                state.particle_F_trial[p], state, model, p
            )
        else:  # elastic
            state.particle_F[p] = state.particle_F_trial[p]"""
if OLD_RM not in s:
    raise SystemExit("되돌림 디스패치를 못 찾았다")
s = s.replace(OLD_RM, NEW_RM, 1)

OLD_ST = """        if model.material[p] == 2:
            stress = kirchoff_stress_drucker_prager(
                state.particle_F[p], U, V, sig, model.mu[p], model.lam[p]
            )"""
NEW_ST = OLD_ST + """
        if model.material[p] == 4:  # [anchorflow] snow
            stress = kirchoff_stress_snow(
                state.particle_F[p], U, V, J, model.mu[p], model.lam[p],
                model.xi, state.particle_Jp[p]
            )"""
if OLD_ST not in s:
    raise SystemExit("응력 디스패치를 못 찾았다")
s = s.replace(OLD_ST, NEW_ST, 1)

# 함수 정의는 디스패치보다 앞에 있어야 한다 -- 파일 맨 앞쪽 stress 함수들 뒤에 붙인다
anchor = "@wp.func\ndef von_mises_return_mapping("
if anchor not in s:
    raise SystemExit("함수 삽입 자리를 못 찾았다")
s = s.replace(anchor, FUNCS.strip("\n") + "\n\n\n" + anchor, 1)

open(p, "w").write(s)
print(f"고쳤다: {p}")
print("GFSNOW_OK")

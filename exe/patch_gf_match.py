"""GaussianFluent 를 PhysGaussian 과 **맞출 수 있게** 두 곳을 config 로 연다.

물성을 똑같이 줘도 두 솔버의 궤적이 갈렸고, 원인을 짚어 보니 물성이 아니라
적분 설정이었다.

  1. GF 는 config 의 `substep_dt` 를 **무시하고** 음속에서 다시 계산한다
     (`cfl 0.6 * dx / c`). 같은 1e-4 를 줘도 GF 는 1.03e-3 으로 프레임당 9 스텝,
     PhysGaussian 은 100 스텝을 밟는다 -- 시간 해상도가 11 배 다르다.
  2. GF 는 `gs_simulation.py` 에서 모든 입자의 초기 속도를 **[0,0,-6.0] 으로
     하드코딩**한다. config 에도 없고 주석도 없다 -- 중력을 0 으로 줘도 공이
     초속 6 으로 내려간다. 이것 하나가 궤적 차이의 대부분이었다.
  3. GF 는 FLIP/PIC 전달(`p2g_flip_pic_with_stress` + `g2p_flip`)을 쓰고
     PhysGaussian 은 APIC(`p2g_apic_with_stress` + `g2p`)를 쓴다. GF 코드에
     APIC 갈래가 이미 있는데 `flip_pic=True` 로 고정되어 안 닿는다.

그래서 config 에 `use_config_dt: true` 가 있으면 1 을 끄고, `flip_pic_ratio` 가
0 이면 APIC 갈래를 타게 한다. 두 키가 없으면 원래 동작 그대로다.
"""
from __future__ import annotations

import argparse
import os

MARK = "    # [anchorflow] config 로 적분 설정을 맞출 수 있게 연다"
OLD = """    cfl = 0.6
    substep_dt = cfl * dx / evaluate_sound_speed_linear_elasticity_analysis(E, nu, rho)
"""
NEW = MARK + """
    cfl = 0.6
    _auto_dt = cfl * dx / evaluate_sound_speed_linear_elasticity_analysis(E, nu, rho)
    import json as _json
    _cfg = _json.load(open(args.config))
    if _cfg.get("use_config_dt", False):
        print(f"[anchorflow] config 의 substep_dt {substep_dt:g} 를 쓴다 "
              f"(GF 자동값 {_auto_dt:g})", flush=True)
    else:
        substep_dt = _auto_dt
    _flip_pic = float(_cfg.get("flip_pic_ratio", 0.7)) > 0.0
"""

OLD_V = ("    mpm_solver.import_particle_v_from_torch(torch.zeros("
         "mpm_init_pos.shape[0], 3, device='cuda').add_(torch.tensor("
         "[0.0, 0.0, -6.0], device='cuda')))")
NEW_V = ("    # [anchorflow] 하드코딩된 초기 속도를 config 로 뺀다 (없으면 -6)\n"
         "    import json as _json2\n"
         "    _v0 = _json2.load(open(args.config)).get('init_velocity', "
         "[0.0, 0.0, -6.0])\n"
         "    mpm_solver.import_particle_v_from_torch(torch.zeros("
         "mpm_init_pos.shape[0], 3, device='cuda').add_(torch.tensor("
         "_v0, device='cuda', dtype=torch.float32)))")

OLD2 = ("                mpm_solver.p2g2p(step, substep_dt, device=device, "
        "flip_pic_ratio=material_params['flip_pic_ratio'])")
NEW2 = ("                mpm_solver.p2g2p(step, substep_dt, device=device, "
        "flip_pic_ratio=material_params['flip_pic_ratio'], "
        "flip_pic=_flip_pic)")

ap = argparse.ArgumentParser()
ap.add_argument("--gf", required=True)
a = ap.parse_args()

p = os.path.join(a.gf, "gs_simulation.py")
s = open(p).read()
if MARK in s:
    raise SystemExit("이미 열려 있다: " + p)
for old in (OLD, OLD2, OLD_V):
    if old not in s:
        raise SystemExit("붙일 자리를 못 찾았다:\n" + old[:90])
s = s.replace(OLD, NEW, 1).replace(OLD2, NEW2, 1).replace(OLD_V, NEW_V, 1)
open(p, "w").write(s)
print("고쳤다:", p, flush=True)
print("GFMATCH_OK", flush=True)

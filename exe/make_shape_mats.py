"""여러 형상에 대해 **같은 물리 씬 + 물성만 다른** config 를 만든다.

각 형상의 공식 config 에서 전처리(회전·스케일·sim_area·불투명도 문턱)와 카메라는
그대로 가져오고, 경계조건·중력·시간은 wolf 공식 config 의 것(중력 -9.8 + 바닥
collider)으로 통일한다. 바닥 높이는 형상마다 다르므로 실행 중에 입자 최저점으로
맞춘다 (exe/patch_pg_floor.py).

  python exe/make_shape_mats.py --cfgdir <PG/config> --out <디렉토리> --n_grid 100
"""
from __future__ import annotations

import argparse
import copy
import json
import os

ap = argparse.ArgumentParser()
ap.add_argument("--cfgdir", required=True, help="PhysGaussian 의 config 디렉토리")
ap.add_argument("--out", required=True)
ap.add_argument("--shapes", default="ficus,bread,plane")
ap.add_argument("--n_grid", type=int, default=100)
ap.add_argument("--substep_dt", type=float, default=2e-5)
ap.add_argument("--frame_dt", type=float, default=1.0 / 60.0)
ap.add_argument("--frame_num", type=int, default=180)
a = ap.parse_args()

SRC = {"ficus": "ficus_config.json", "bread": "tear_bread_config.json",
       "plane": "plane_config.json", "vasedeck": "vasedeck_config.json",
       "pillow2sofa": "pillow2sofa_config.json", "wolf": "wolf_config.json"}

MAT_KEYS = ("material", "E", "nu", "density", "yield_stress", "hardening", "xi",
            "friction_angle", "softening", "plastic_viscosity")
CASES = {
    "elD": dict(material="jelly", E=2e6, nu=0.4, density=200),   # ficus 공식 값
    "clayC": dict(material="metal", E=2e6, nu=0.3, density=1000, yield_stress=4e4),
    "viscoplastic": dict(material="foam", E=1e6, nu=0.3, density=1000,
                         yield_stress=2e3, plastic_viscosity=100.0),
}

wolf = json.load(open(os.path.join(a.cfgdir, SRC["wolf"])))
os.makedirs(a.out, exist_ok=True)
for sh in a.shapes.split(","):
    base = json.load(open(os.path.join(a.cfgdir, SRC[sh])))
    for name, mat in CASES.items():
        c = copy.deepcopy(base)
        for k in MAT_KEYS:
            c.pop(k, None)
        c.update(mat)
        c["g"] = wolf["g"]
        c["boundary_conditions"] = copy.deepcopy(wolf["boundary_conditions"])
        c["boundary_conditions"] = [b for b in c["boundary_conditions"]
                                    if b.get("type") != "release_particles_sequentially"]
        c["particle_filling"] = copy.deepcopy(wolf["particle_filling"])
        c["n_grid"] = a.n_grid
        c["substep_dt"] = a.substep_dt
        c["frame_dt"] = a.frame_dt
        c["frame_num"] = a.frame_num
        # 감쇠는 넣지 않는다 -- 0.999 는 서브스텝마다 곱해져 1초 만에 속도를
        # 0.027 배로 깎아 탄성 반발을 없앤다
        c.pop("grid_v_damping_scale", None)
        c["default_camera_index"] = 0
        c["move_camera"] = False
        c.pop("init_velocity", None)
        dst = os.path.join(a.out, f"{sh}_{name}.json")
        json.dump(c, open(dst, "w"), indent=4, ensure_ascii=False)
    print(f"{sh:12} <- {SRC[sh]}  물성 {len(CASES)}개")

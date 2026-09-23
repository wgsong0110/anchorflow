"""wolf 공식 config 에서 **재질 블록만** 갈아끼운 네 개의 config 를 만든다.

씬(중력 -9.8, 바닥 sticky collider z=0.48, release_particles_sequentially)과 전처리는
PhysGaussian 이 배포한 wolf_config.json 그대로 두고, 물성만 바꾼다.

물성 출처 -- PhysGaussian 이 수치를 공개한 곳은 배포된 config 파일뿐이다
(논문 Tab.2 는 구성모델 이름만, Tab.3 은 기호 정의만 싣는다):
  탄성체  jelly (fixed corotated) : ficus_config.json
  금속    metal (von Mises)       : plane_config.json
  과립    sand  (Drucker-Prager)  : wolf_config.json
  점소성  foam  (Herschel-Bulkley): **공개된 config 없음** -- jam·cake 는 미공개다.
          아래 값은 내가 고른 것이라 공식이 아니다.

n_grid 는 "격자당 입자 8~16개" 요청에 맞춰 넘겨받은 값으로 덮어쓴다.
"""
from __future__ import annotations

import argparse
import copy
import json
import os

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True, help="PhysGaussian 의 wolf_config.json")
ap.add_argument("--out", required=True)
ap.add_argument("--n_grid", type=int, default=100)
ap.add_argument("--substep_dt", type=float, default=2e-5)
ap.add_argument("--frame_dt", type=float, default=1.0 / 60.0)
ap.add_argument("--frame_num", type=int, default=180)
ap.add_argument("--no_release", type=int, default=1,
                help="release_particles_sequentially 를 뺀다 (중력+바닥만 남김)")
a = ap.parse_args()

# 재질 키는 전부 지우고 각 case 의 것만 넣는다 (wolf 의 friction_angle 이 남지 않게)
MAT_KEYS = ("material", "E", "nu", "density", "yield_stress", "hardening", "xi",
            "friction_angle", "softening", "plastic_viscosity")

CASES = {
    "elastic": dict(material="jelly", E=2e6, nu=0.4, density=200),           # ficus 공식
    # plane 공식(E 1e5, yield 100)은 중력 0 인 씬용이라 yield/(rho g L)=0.01 -- 자중에
    # 그대로 주저앉는다. 자립하는 탄소성이 되도록 E 와 항복응력을 올렸다 (비공식).
    # 자중(rho g L = 9800)은 버티되 손으로는 찰흙처럼 눌리는 탄소성:
    # 항복/자중 = 4.1 로 낮추고 E 도 낮춰 유연하게 한다 (비공식).
    "metal": dict(material="metal", E=1e7, nu=0.3, density=1000,
                  yield_stress=4e4),
    "sand": dict(material="sand", E=5e7, nu=0.3, density=2000,
                 friction_angle=30),                                         # wolf 공식
    "viscoplastic": dict(material="foam", E=1e6, nu=0.3, density=1000,
                         yield_stress=2000.0, plastic_viscosity=100.0),      # 비공식
}

base = json.load(open(a.src))
os.makedirs(a.out, exist_ok=True)
for name, mat in CASES.items():
    c = copy.deepcopy(base)
    for k in MAT_KEYS:
        c.pop(k, None)
    c.update(mat)
    if a.no_release:
        # 이 BC 가 입자를 층층이 풀어 주는 바람에 몸통이 통째로 넘어간다.
        # 순수하게 자중만 보려면 빼야 한다 (카메라는 원래 고정이다).
        c["boundary_conditions"] = [b for b in c["boundary_conditions"]
                                    if b.get("type") != "release_particles_sequentially"]
    c["default_camera_index"] = 0        # cameras.json[0] 고정
    c["move_camera"] = False
    c["n_grid"] = a.n_grid
    c["substep_dt"] = a.substep_dt
    c["frame_dt"] = a.frame_dt
    c["frame_num"] = a.frame_num
    dst = os.path.join(a.out, f"wolf_{name}.json")
    json.dump(c, open(dst, "w"), indent=4, ensure_ascii=False)
    rho, g, L = mat["density"], abs(base["g"][2]), 1.0
    ys = mat.get("yield_stress")
    extra = f", 항복/rho g L = {ys / (rho * g * L):.3f}" if ys else ""
    print(f"{name:13} {mat['material']:8} E {mat['E']:.3g} nu {mat['nu']} "
          f"rho {rho}{extra}  -> {dst}")

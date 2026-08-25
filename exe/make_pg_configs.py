"""PhysGaussian 의 씬 설정을 이 솔버의 단위로 옮긴다.

DreamPhysics 의 `compute_mu_lam_from_E_nu` 는 E 에 무조건 1e7 을 곱한다
(mpm_utils.py:270). PhysGaussian 원본에는 없는 개조라서, PhysGaussian 이 배포한
config 의 E 를 그대로 넣으면 실효 강성이 1e7 배가 되고 MPM 은 정지 상태 첫
서브스텝에서 |v| 가 100 단위로 튄 뒤 격자를 벗어나 죽는다.

그래서 여기서 E 를 미리 나눠 둔다. 최상위 E 와 additional_material_params 의 영역별
E 를 모두 나눠야 한다 -- 영역 쪽을 빼먹으면 그 영역만 1e7 배로 남아 똑같이 터진다.

씬 설정을 손으로 고쳐 두지 않고 생성하는 이유는, PhysGaussian 이 배포한 값과 우리가
쓰는 값의 차이가 이 나눗셈 하나뿐이라는 게 파일에 남게 하기 위해서다.
"""
from __future__ import annotations

import argparse
import json
import os

# mpm_utils.py 의 compute_mu_lam_from_E_nu 가 곱하는 값
SOLVER_E_GAIN = 1e7

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="/home/dkta/work/PhysGaussian/config",
                help="PhysGaussian 저장소의 config 디렉토리")
ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "cfg", "pg"))
args = ap.parse_args()

os.makedirs(args.out, exist_ok=True)
for name in sorted(os.listdir(args.src)):
    if not name.endswith("_config.json"):
        continue
    cfg = json.load(open(os.path.join(args.src, name)))
    before = cfg["E"]
    cfg["E"] = cfg["E"] / SOLVER_E_GAIN
    regions = cfg.get("additional_material_params", [])
    for reg in regions:
        reg["E"] = reg["E"] / SOLVER_E_GAIN
    cfg["_note"] = ("PhysGaussian 원본에서 E 를 %g 로 나눔 -- 솔버가 곱하는 만큼"
                    % SOLVER_E_GAIN)
    dst = os.path.join(args.out, name.replace("_config.json", ".json"))
    json.dump(cfg, open(dst, "w"), indent=4, ensure_ascii=False)
    print(f"{name:26} E {before:g} -> {cfg['E']:g}"
          f"{'  영역 %d 개도 함께' % len(regions) if regions else ''}")

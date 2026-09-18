"""찰흙이 끊어지게 손상(damage) 항을 살린다 (수정은 한 번만 먹는다).

PhysGaussian 의 plasticine(material 5) 은 이미 `von_mises_return_mapping_with_damage`
를 타고, 그 안에 항복응력을 깎는 연화(softening) 와 완전 파손 처리가 들어 있다 --

    yield <- yield - softening * (소성 증분)
    yield <= 0 이면  mu = lam = 0     # 응력이 0 이 되어 응집이 사라진다

문제는 두 가지였다.

1. `softening` 기본값이 0.1 이다. 항복응력 30 을 0 까지 깎으려면 누적 소성변형이
   300 이어야 하는데, 당기는 3 초 동안 목에 쌓이는 것은 1 남짓이다 -- 사실상 꺼져 있다
2. `utils/decode_param.py` 가 config 키를 화이트리스트로 걸러서 `softening` 을
   넣어도 솔버까지 가지 않는다 (`yield_stress`, `hardening`, `xi` 등만 통과한다)

그래서 여기서는 2 번을 뚫어 config 로 `softening` 을 넘길 수 있게만 한다. 손상의
수식 자체는 저자 코드 그대로다.
"""
from __future__ import annotations

import argparse
import os

MARK = "    # [anchorflow] softening 을 config 로 넘길 수 있게 뚫었다"
OLD = ('    if "hardening" in sim_params.keys():\n'
       '        material_params["hardening"] = sim_params["hardening"]\n')
NEW = (MARK + "\n"
       '    if "softening" in sim_params.keys():\n'
       '        material_params["softening"] = sim_params["softening"]\n'
       "\n" + OLD)

ap = argparse.ArgumentParser()
ap.add_argument("--pg", required=True)
a = ap.parse_args()

p = os.path.join(a.pg, "utils", "decode_param.py")
s = open(p).read()
if MARK in s:
    raise SystemExit("이미 뚫려 있다: " + p)
if OLD not in s:
    raise SystemExit("hardening 통과 지점을 못 찾았다")
open(p, "w").write(s.replace(OLD, NEW, 1))
print("뚫었다:", p, flush=True)
print("DAMAGE_OK", flush=True)

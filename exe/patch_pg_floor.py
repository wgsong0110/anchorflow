"""바닥 collider 높이를 입자 최저점에 맞춘다 (형상마다 크기가 다르므로).

  python exe/patch_pg_floor.py --pg <PG>
환경변수: AF_FLOOR_AUTO=1, AF_FLOOR_M (여유, 기본 0.02)
"""
from __future__ import annotations

import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("--pg", required=True)
a = ap.parse_args()
p = os.path.join(a.pg, "gs_simulation.py")
s = open(p).read()
if "AF_FLOOR_AUTO" in s:
    print("이미 패치됨")
    raise SystemExit(0)

# _af_os 가 없으면(다른 패치를 안 걸었으면) 여기서 만든다
if "import os as _af_os" not in s:
    s = s.replace("from particle_filling.filling import *",
                  "from particle_filling.filling import *\nimport os as _af_os", 1)

A = "    set_boundary_conditions(mpm_solver, bc_params, time_params)"
B = '''    if _af_os.environ.get("AF_FLOOR_AUTO"):
        _zmin = float(mpm_init_pos[:, 2].min())
        _mg = float(_af_os.environ.get("AF_FLOOR_M", 0.02))
        for _b in bc_params:
            if _b.get("type") == "surface_collider":
                _b["point"] = [_b["point"][0], _b["point"][1], _zmin - _mg]
                print(f"[바닥] z = {_b['point'][2]:.4f} (입자 최저 {_zmin:.4f})",
                      flush=True)
''' + A
assert A in s
open(p, "w").write(s.replace(A, B, 1))
print(f"패치 완료: {p}")

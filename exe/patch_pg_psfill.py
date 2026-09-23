"""PhysGaussian 의 `fill_particles` 를 포아송 디스크 내부 채움으로 갈아끼운다.

PG 원본 채움은 칸당 한 점이라 분포가 표면 밀도를 따라가고, 3DGS 껍데기에 난 구멍
때문에 속이 제대로 안 메워진다 (lib/anchorflow/psfill.py 참고).

AF_PSFILL=1 일 때만 동작하므로 패치해 둬도 원본 동작을 되돌릴 수 있다.
격자당 입자 수(ppc)도 찍어 준다 -- n_grid 를 고를 때 이 값을 본다.

  python exe/patch_pg_psfill.py --pg /workspace/PG
"""
from __future__ import annotations

import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("--pg", required=True)
a = ap.parse_args()

p = os.path.join(a.pg, "gs_simulation.py")
s = open(p).read()
if "AF_PSFILL" in s:
    print("이미 패치됨")
    raise SystemExit(0)

ANCHOR = "from particle_filling.filling import *"
OVERRIDE = ANCHOR + '''

# ---- anchorflow: 포아송 디스크 내부 채움 (AF_PSFILL=1 일 때만) ----
import os as _af_os
if _af_os.environ.get("AF_PSFILL"):
    import torch as _af_t
    from anchorflow.psfill import poisson_fill as _af_pf
    from anchorflow.psfill import poisson_fill_mesh as _af_pfm

    import numpy as _af_np

    def fill_particles(pos, opacity=None, cov=None, **_kw):
        _S = pos.detach().cpu().numpy().astype("float64")
        _ck = _af_os.environ.get("AF_FILL_CACHE")
        if _ck and _af_os.path.exists(_ck):
            _L = _af_np.load(_ck)
            print(f"[채움] 캐시에서 {_L.shape[0]} 개", flush=True)
            return _af_t.cat([pos.detach().cpu(), _af_t.from_numpy(_L).float()], 0)
        _sp = float(_af_os.environ.get("AF_FILL_SPACING", 0.012))
        _gd = int(_af_os.environ.get("AF_FILL_GRID", 160))
        if _af_os.environ.get("AF_FILL_MESH"):
            _L = _af_pfm(_S, spacing=_sp, grid=_gd,
                         sigma=float(_af_os.environ.get("AF_FILL_SIGMA", 2.0)),
                         level=float(_af_os.environ.get("AF_FILL_LEVEL", 0.5)),
                         mesh_out=_af_os.environ.get("AF_MESH_OUT"))
        else:
            _L = _af_pf(_S, spacing=_sp, grid=_gd,
                        close=float(_af_os.environ.get("AF_FILL_CLOSE", 0.03)))
        if _ck:
            _af_np.save(_ck, _L)
        print(f"[채움] 표면 {pos.shape[0]} + 내부 {_L.shape[0]} "
              f"= {pos.shape[0] + _L.shape[0]}", flush=True)
        return _af_t.cat([pos.detach().cpu(), _af_t.from_numpy(_L).float()], 0)
'''
assert ANCHOR in s
s = s.replace(ANCHOR, OVERRIDE, 1)

PPC_ANCHOR = "    mpm_solver = MPM_Simulator_WARP(10)"
PPC = '''    _af_dx = material_params["grid_lim"] / material_params["n_grid"]
    _af_occ = torch.unique(torch.floor(mpm_init_pos / _af_dx).long(), dim=0).shape[0]
    print(f"[격자] n_grid {material_params['n_grid']} dx {_af_dx:.5f} | "
          f"입자 {mpm_init_pos.shape[0]} | 점유칸 {_af_occ} | "
          f"칸당 입자 {mpm_init_pos.shape[0] / _af_occ:.2f}", flush=True)
''' + PPC_ANCHOR
assert PPC_ANCHOR in s
s = s.replace(PPC_ANCHOR, PPC, 1)

# 최신 diff-gaussian-rasterization 은 (image, radii, depth) 세 개를 돌려준다
RA = "            rendering, raddi = rasterize("
if RA in s:
    s = s.replace(RA, "            _af_out = rasterize(", 1)
    i = s.index("_af_out = rasterize(")
    j = s.index("\n            )\n", i) + len("\n            )\n")
    s = s[:j] + "            rendering = _af_out[0]\n" + s[j:]

open(p, "w").write(s)
print(f"패치 완료: {p}")

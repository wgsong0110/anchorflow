"""PG 자체 내부 채우기 결과를 캐시해서 다시 계산하지 않게 한다.

`fill_particles` 는 taichi 레이캐스팅이라 씬마다 수십 초에서 분 단위로 걸린다.
결과는 씬·설정이 같으면 늘 같으므로 한 번 만들어 .npy 로 두고 재사용한다.
반환 순서(가우시안 먼저, 채운 입자 나중)는 원본과 같아야 뒤따르는
`init_filled_particles` 가 맞물린다.

  python exe/patch_pg_fillcache.py --pg <PG>
환경변수: AF_PGFILL_NPY (있으면 읽고, 없으면 계산 후 저장)
"""
from __future__ import annotations

import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("--pg", required=True)
a = ap.parse_args()

p = os.path.join(a.pg, "gs_simulation.py")
s = open(p).read()
if "AF_PGFILL_NPY" in s:
    print("이미 패치됨")
    raise SystemExit(0)

A = "from particle_filling.filling import *"
B = A + '''
import os as _pf_os
import numpy as _pf_np
import torch as _pf_t

_pf_orig_fill = fill_particles


def fill_particles(*args, **kwargs):
    _ck = _pf_os.environ.get("AF_PGFILL_NPY")
    if _ck and _pf_os.path.exists(_ck):
        _L = _pf_np.ascontiguousarray(_pf_np.load(_ck))
        print(f"[PG채움] 캐시에서 {_L.shape[0]} 개", flush=True)
        return _pf_t.from_numpy(_L).float().contiguous()
    _r = _pf_orig_fill(*args, **kwargs)
    if _ck:
        _pf_np.save(_ck, _r.detach().cpu().numpy())
        print(f"[PG채움] 캐시 저장 {_ck}", flush=True)
    return _r
'''
assert A in s
open(p, "w").write(s.replace(A, B, 1))
print(f"패치 완료: {p}")

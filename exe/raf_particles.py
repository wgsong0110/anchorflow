"""RAF 방식으로 3DGS 를 **통일 입자셋**으로 바꿔 h5 로 남긴다.

RAF(Scene-Level Heterogeneous Physics, CVPR-F 2026)의 기여는 표현 추상화다 --
3DGS·메시·유체를 하나의 물리 입자셋으로 옮기고, 결과로 각 자산을 되돌려 그린다.
**그 위의 솔버는 갈아 끼울 수 있다.** 그래서 여기서는 입자셋만 RAF 방식으로 만들고
물리는 교사와 같은 warp MPM 으로 푼다 (같은 물리라야 세 데이터셋을 섞어 쓸 수 있다).

RAF 의 `gaussian.py: load_3dgs_ply` 와 같은 자리를 따른다: 위치와 불투명도를 읽고,
불투명도 문턱으로 거르고, 꼬리 1% 를 떼어 중심·크기를 잡는다.

  python exe/raf_particles.py --ply <3dgs.ply> --out cloud.h5 [--n 200000]
"""
import argparse
import os

import h5py
import numpy as np
from plyfile import PlyData

ap = argparse.ArgumentParser()
ap.add_argument("--ply", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=200000, help="입자 수 (솎는다)")
ap.add_argument("--opacity_min", type=float, default=0.02)
ap.add_argument("--center", type=float, nargs=3, default=[1.0, 1.0, 1.0],
                help="놓을 자리 (warp MPM 은 [0, grid_lim] 안이어야 한다)")
ap.add_argument("--size", type=float, default=1.0, help="최대 변 길이")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

p = PlyData.read(a.ply)["vertex"]
x = np.stack([np.asarray(p[k]) for k in ("x", "y", "z")], 1).astype(np.float64)
n0 = len(x)
if "opacity" in p.data.dtype.names:
    x = x[1.0 / (1.0 + np.exp(-np.asarray(p["opacity"]))) > a.opacity_min]
x = x[np.isfinite(x).all(1)]
lo, hi = np.percentile(x, 0.5, 0), np.percentile(x, 99.5, 0)
x = x[((x > lo) & (x < hi)).all(1)]
if a.n and len(x) > a.n:
    i = np.random.default_rng(a.seed).choice(len(x), a.n, replace=False)
    x = x[np.sort(i)]
c = 0.5 * (x.min(0) + x.max(0))
x = (x - c) * (a.size / max(x.max(0) - x.min(0))) + np.asarray(a.center)
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
with h5py.File(a.out, "w") as h:
    h.create_dataset("x", data=x.T.astype(np.float32))
print(f"[RAF 입자셋] {os.path.basename(a.ply)}  {n0} -> {len(x)} 입자, "
      f"범위 {np.round(x.min(0), 3)}~{np.round(x.max(0), 3)}")
print("RAF_PARTICLES_OK")

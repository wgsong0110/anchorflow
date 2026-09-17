"""PhysGaussian 의 입자 채우기가 왜 안 끝나는지 가린다.

`fill_particles` 의 첫 커널 `densify_grids` 에서 진행이 멈춘다 ("after dense
grids" 가 끝내 안 찍힌다). 후보는 둘이다.

1. 일거리가 너무 많다 -- 가우시안마다 (2r+1)^3 셀을 도는데 r 이 크다
2. `ti.sym_eig` 가 안 끝난다 -- 타이치의 3x3 대칭 고유분해는 `dsyevq3` 의
   `while True` 안에서 `assert nIter <= 30` 로만 빠져나오는데, 릴리스 빌드는
   assert 를 버린다. 수렴 못 하는 행렬 하나면 커널이 영원히 돈다

둘 다 전처리를 거친 공분산에서 판정할 수 있으므로 토치로 재현해서 센다.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--pg", required=True, help="PhysGaussian 체크아웃")
ap.add_argument("--config", required=True)
ap.add_argument("--ply", required=True)
ap.add_argument("--debug_ti", action="store_true",
                help="타이치 assert 를 켠다. dsyevq3 의 'Timeout' 이 뜨면 고유분해가 "
                     "수렴 못 하는 것이 확정이다 (릴리스 빌드는 이 assert 를 버려서 "
                     "그냥 영원히 돈다)")
ap.add_argument("--timeout", type=float, default=300,
                help="이 초를 넘기면 스스로 죽는다. 도는 CUDA 커널은 밖에서 못 멈추니 "
                     "프로세스를 내려 컨텍스트째 반납한다")
ap.add_argument("--run", action="store_true",
                help="채우기 단계를 하나씩 직접 돌려 어디서 멈추는지 시간으로 본다")
a = ap.parse_args()

sys.path.insert(0, a.pg)
sys.path.insert(0, os.path.join(a.pg, "gaussian-splatting"))
os.chdir(a.pg)

import torch                                                      # noqa: E402
from scene.gaussian_model import GaussianModel                    # noqa: E402
from utils.decode_param import decode_param_json                  # noqa: E402
from utils.transformation_utils import (apply_cov_rotations,       # noqa: E402
                                        apply_rotations,
                                        generate_rotation_matrices,
                                        shift2center111,
                                        transform2origin)

mp, _bc, _tp, pp, _cp = decode_param_json(a.config)
g = GaussianModel(3)
g.load_ply(a.ply)
pos = g.get_xyz.detach()
cov = g.get_covariance().detach()
op = g.get_opacity.detach()

keep = (op[:, 0] > pp["opacity_threshold"])
pos, cov, op = pos[keep], cov[keep], op[keep]
R = generate_rotation_matrices(torch.tensor(pp["rotation_degree"]),
                               pp["rotation_axis"])
pos = apply_rotations(pos, R)
sa = pp.get("sim_area")
if sa is not None:
    m = torch.ones(pos.shape[0], dtype=torch.bool, device=pos.device)
    for i in range(3):
        m &= (pos[:, i] > sa[2 * i]) & (pos[:, i] < sa[2 * i + 1])
    pos, cov, op = pos[m], cov[m], op[m]
pos, scale_origin, _ = transform2origin(pos, pp["scale"])
pos = shift2center111(pos)
cov = apply_cov_rotations(cov, R)
cov = scale_origin * scale_origin * cov

# fill_particles 안에서 하는 것과 똑같이: 경계로 자르고 격자 간격을 **다시** 잡는다
fp = pp["particle_filling"]
b = fp["boundary"]
m = torch.ones(pos.shape[0], dtype=torch.bool, device=pos.device)
mx = 0.0
for i in range(3):
    m &= (pos[:, i] > b[2 * i]) & (pos[:, i] < b[2 * i + 1])
    mx = max(mx, b[2 * i + 1] - b[2 * i])
pos, cov, op = pos[m], cov[m], op[m]
dx = mx / fp["n_grid"]
print(f"[전처리] scale_origin {scale_origin:.4f}  경계안 {pos.shape[0]}  "
      f"격자 간격 {dx:.5f} (설정값 {mp['grid_lim'] / fp['n_grid']:.5f} 아님)",
      flush=True)

C = torch.stack([
    torch.stack([cov[:, 0], cov[:, 1], cov[:, 2]], -1),
    torch.stack([cov[:, 1], cov[:, 3], cov[:, 4]], -1),
    torch.stack([cov[:, 2], cov[:, 4], cov[:, 5]], -1)], -2).double()

bad = ~torch.isfinite(C).all(-1).all(-1)
print(f"[후보 2] 공분산에 NaN/Inf 인 가우시안 {int(bad.sum())}", flush=True)
ev = torch.linalg.eigvalsh(C[~bad])
lo, hi = ev.min(-1).values, ev.max(-1).values
cond = hi / lo.abs().clamp(min=1e-300)
print(f"[후보 2] 최소 고유값 <=0 {int((lo <= 0).sum())}, "
      f"<1e-20 {int((lo.abs() < 1e-20).sum())}, "
      f"조건수 중앙 {cond.median():.3e} p99 {cond.quantile(0.99):.3e} "
      f"최대 {cond.max():.3e}", flush=True)
# dsyevq3 의 수렴 판정은 비대각 성분이 대각 크기에 비해 무시될 만한가이다.
# 대각이 0 에 붙으면 (abs(e)+g == g) 가 영원히 거짓이 된다.
diag = torch.diagonal(C[~bad], dim1=-2, dim2=-1).abs().sum(-1)
print(f"[후보 2] 대각합 < 1e-30 인 것 {int((diag < 1e-30).sum())}, "
      f"최소 {diag.min():.3e}", flush=True)

r = torch.ceil(hi.clamp(min=0).sqrt() / dx)
print(f"[후보 1] 이웃 반경 r 중앙 {r.median():.0f} p99 {r.quantile(0.99):.0f} "
      f"최대 {r.max():.0f},  셀 방문 합 {((2 * r + 1) ** 3).sum():.3e}",
      flush=True)
if a.run:
    # 단계마다 동기화하고 시간을 찍는다. 어느 커널이 안 끝나는지 그것만 보면 된다.
    import threading
    import time

    def _bail():
        time.sleep(a.timeout)
        print(f"[시간초과] {a.timeout:.0f}s 안에 안 끝났다", flush=True)
        os._exit(3)

    threading.Thread(target=_bail, daemon=True).start()

    import taichi as ti
    from particle_filling.filling import (densify_grids, fill_dense_grids,
                                          internal_filling)
    ti.init(arch=ti.cuda, device_memory_GB=8.0, debug=a.debug_ti)
    n = fp["n_grid"]
    ti_pos = ti.Vector.field(n=3, dtype=float, shape=pos.shape[0])
    ti_op = ti.field(dtype=float, shape=pos.shape[0])
    ti_cov = ti.Vector.field(n=6, dtype=float, shape=pos.shape[0])
    ori = torch.tensor([b[0], b[2], b[4]], device=pos.device)
    ti_pos.from_torch((pos - ori).reshape(-1, 3))
    ti_op.from_torch(op.reshape(-1))
    ti_cov.from_torch(cov.reshape(-1, 6))
    grid = ti.field(dtype=int, shape=(n, n, n))
    dens = ti.field(dtype=float, shape=(n, n, n))
    parts = ti.Vector.field(n=3, dtype=float, shape=fp["max_particles_num"])

    t = time.time(); densify_grids(ti_pos, ti_op, ti_cov, grid, dens, dx)
    ti.sync(); print(f"[단계] densify_grids {time.time() - t:.2f}s", flush=True)
    t = time.time()
    fn = fill_dense_grids(grid, dens, dx, fp["density_threshold"], parts, 0,
                          fp["max_partciels_per_cell"])
    print(f"[단계] fill_dense_grids {time.time() - t:.2f}s -> {fn}", flush=True)
    if fp["smooth"]:
        import mcubes
        t = time.time()
        df = dens.to_numpy()
        import numpy as np
        dens.from_numpy(mcubes.smooth(df, method="constrained",
                                      max_iters=500).astype(np.float32))
        print(f"[단계] mcubes.smooth {time.time() - t:.2f}s", flush=True)
    t = time.time()
    fn = internal_filling(grid, dens, dx, parts, fn,
                          fp["max_partciels_per_cell"],
                          exclude_dir=fp["search_exclude_direction"],
                          ray_cast_dir=fp["ray_cast_direction"],
                          threshold=fp["search_threshold"])
    print(f"[단계] internal_filling {time.time() - t:.2f}s -> {fn}", flush=True)

print("PROBE_OK", flush=True)

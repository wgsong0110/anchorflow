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
pos, cov = pos[keep], cov[keep]
R = generate_rotation_matrices(torch.tensor(pp["rotation_degree"]),
                               pp["rotation_axis"])
pos = apply_rotations(pos, R)
sa = pp.get("sim_area")
if sa is not None:
    m = torch.ones(pos.shape[0], dtype=torch.bool, device=pos.device)
    for i in range(3):
        m &= (pos[:, i] > sa[2 * i]) & (pos[:, i] < sa[2 * i + 1])
    pos, cov = pos[m], cov[m]
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
pos, cov = pos[m], cov[m]
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
print("PROBE_OK", flush=True)

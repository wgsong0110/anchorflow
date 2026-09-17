"""PhysGaussian 의 입자 채우기가 끝나지 않는 버그를 고친다 (수정은 한 번만 먹는다).

증상: `fill_particles` 의 첫 커널 `densify_grids` 에서 GPU 가 100% 인 채 몇 시간이
지나도 "after dense grids" 가 안 찍힌다. wolf(모래)·bread(찰흙) 둘 다 그랬다.

원인: 커널 안의 `ti.sym_eig` 가 **float32 에서 무한대를 돌려준다**. 고유값이 거의
같은(등방에 가까운) 공분산에서 타이치 1.6 의 QL 반복 `dsyevq3` 가 발산하는데,
그 함수의 유일한 탈출구인 `assert nIter <= 30` 은 릴리스 빌드에서 지워진다.
wolf 의 경계 안 가우시안 139018 개 중 **68 개**가 여기 걸렸다. 예:

    공분산 대각 4.9295e-06 세 개, 비대각 ~1e-10  (거의 등방)
    타이치(f32) 고유값 [4.93e-06, inf, inf]      토치(f64) 세 개 다 4.93e-06

그 다음 줄이 `r = ceil(sqrt(max sig)/grid_dx)` 라 r 이 정수로 넘치고,
`for dx in range(-r, r+1)` 삼중 루프가 사실상 끝나지 않는다.

고치는 방법: 고유분해를 커널 밖 **토치 float64** 로 옮긴다. 커널이 하려던 계산과
같은 것을 그대로 준다 --

    sig, Q = eigh(C);  sig <- max(sig, 1e-8)
    cov_inv = Q diag(1/sig) Q^T          (커널이 밀도 계산에 쓰는 행렬)
    r = ceil(sqrt(max sig) / grid_dx)    (커널이 도는 이웃 반경)

즉 채우기 결과를 바꾸는 것이 아니라, 발산하는 풀이기를 멀쩡한 풀이기로 바꾼다.
저자 코드의 정의는 그대로다.
"""
from __future__ import annotations

import argparse
import os
import re

MARK = "# [anchorflow] sym_eig 가 f32 에서 발산해 고유분해를 토치로 옮겼다"

NEW_KERNEL = '''@ti.kernel
def densify_grids(
    init_particles: ti.template(),
    opacity: ti.template(),
    cov_inv: ti.template(),
    radius: ti.template(),
    grid: ti.template(),
    grid_density: ti.template(),
    grid_dx: float,
):
    ''' + MARK + '''
    for pi in range(init_particles.shape[0]):
        pos = init_particles[pi]
        i = ti.floor(pos[0] / grid_dx, dtype=int)
        j = ti.floor(pos[1] / grid_dx, dtype=int)
        k = ti.floor(pos[2] / grid_dx, dtype=int)
        ti.atomic_add(grid[i, j, k], 1)
        cov = ti.Matrix(
            [
                [cov_inv[pi][0], cov_inv[pi][1], cov_inv[pi][2]],
                [cov_inv[pi][1], cov_inv[pi][3], cov_inv[pi][4]],
                [cov_inv[pi][2], cov_inv[pi][4], cov_inv[pi][5]],
            ]
        )
        r = radius[pi]
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                for dz in range(-r, r + 1):
                    if (
                        i + dx >= 0
                        and i + dx < grid_density.shape[0]
                        and j + dy >= 0
                        and j + dy < grid_density.shape[1]
                        and k + dz >= 0
                        and k + dz < grid_density.shape[2]
                    ):
                        density = compute_density(
                            ti.Vector([i + dx, j + dy, k + dz]),
                            pos,
                            opacity[pi],
                            cov,
                            grid_dx,
                        )
                        ti.atomic_add(grid_density[i + dx, j + dy, k + dz], density)
'''

NEW_CALL = '''    ''' + MARK + '''
    _C = torch.stack(
        [
            torch.stack([cov[:, 0], cov[:, 1], cov[:, 2]], -1),
            torch.stack([cov[:, 1], cov[:, 3], cov[:, 4]], -1),
            torch.stack([cov[:, 2], cov[:, 4], cov[:, 5]], -1),
        ],
        -2,
    ).double()
    _sig, _Q = torch.linalg.eigh(_C)
    _sig = _sig.clamp(min=1e-8)
    _inv = _Q @ torch.diag_embed(1.0 / _sig) @ _Q.transpose(-1, -2)
    _cov_inv = torch.stack(
        [_inv[:, 0, 0], _inv[:, 0, 1], _inv[:, 0, 2],
         _inv[:, 1, 1], _inv[:, 1, 2], _inv[:, 2, 2]], -1
    ).float()
    _rad = torch.ceil(_sig.max(-1).values.sqrt() / grid_dx).int()
    ti_cov_inv = ti.Vector.field(n=6, dtype=float, shape=cov.shape[0])
    ti_radius = ti.field(dtype=ti.i32, shape=cov.shape[0])
    ti_cov_inv.from_torch(_cov_inv.reshape(-1, 6))
    ti_radius.from_torch(_rad.reshape(-1))
    print("이웃 반경 r: 중앙 %d 최대 %d, 셀 방문 %.3e"
          % (int(_rad.median()), int(_rad.max()),
             float(((2 * _rad.double() + 1) ** 3).sum())))
    densify_grids(ti_pos, ti_opacity, ti_cov_inv, ti_radius, grid, grid_density, grid_dx)
'''

ap = argparse.ArgumentParser()
ap.add_argument("--pg", required=True, help="PhysGaussian 체크아웃")
a = ap.parse_args()

p = os.path.join(a.pg, "particle_filling", "filling.py")
s = open(p).read()
if MARK in s:
    raise SystemExit("이미 고쳐져 있다: " + p)

m = re.search(r"@ti\.kernel\ndef densify_grids\(.*?\n\n\n", s, re.S)
if m is None:
    raise SystemExit("densify_grids 를 못 찾았다")
s = s[:m.start()] + NEW_KERNEL + "\n\n" + s[m.end():]

old_call = ("    # compute density_field\n"
            "    densify_grids(ti_pos, ti_opacity, ti_cov, grid, grid_density, grid_dx)\n")
if old_call not in s:
    raise SystemExit("densify_grids 호출부를 못 찾았다")
s = s.replace(old_call, "    # compute density_field\n" + NEW_CALL)

if "import torch" not in s.split("def ")[0]:
    s = s.replace("import taichi as ti", "import taichi as ti\nimport torch", 1)

open(p, "w").write(s)
print("고쳤다:", p, flush=True)
print("FIX_OK", flush=True)

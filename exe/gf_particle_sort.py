"""GF 솔버의 입자를 **격자 칸 순서로 다시 줄 세운다**. 물리는 그대로다.

왜 하는가. 수박 씬(입자 138 만, 격자 300^3)을 단계별로 재보니 p2g 가 70% 다.
입자 순서를 무작위로 섞으면 p2g 가 4.4 배 느려지고, 반대로 칸 순서로 세우면
1.7 배 빨라진다. 즉 이 커널을 붙잡고 있는 것은 계산량이 아니라 **격자 메모리
접근의 지역성**이다. 격자는 x 우선 선형으로 놓여 있으므로, 같은 순서로 입자를
세우면 이웃한 스레드가 같은 캐시 줄을 본다 (모턴 순서보다 이쪽이 빨랐다).

입자 순서를 바꾸면 가우시안과의 짝이 어긋나므로 `orig_index` 를 같이 들고 다니며
내보낼 때 되돌린다. 원자적 덧셈의 순서가 바뀌어 비트 단위로는 달라지지만,
그 차이는 같은 코드를 두 번 돌렸을 때의 차이보다 작다.
"""
import numpy as np
import torch
import warp as wp


def _cell_key(x, dx, n_grid):
    c = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0) / dx
    c = c.floor().long().clamp_(0, n_grid - 1)
    return (c[:, 0] * n_grid + c[:, 1]) * n_grid + c[:, 2]


def _permute_all(solver, perm):
    """입자 축이 있는 배열을 전부 같은 순서로 옮긴다."""
    n = solver.n_particles
    moved = 0
    for obj in (solver.mpm_state, solver.mpm_model):
        for name in dir(obj):
            if name.startswith("_"):
                continue
            arr = getattr(obj, name, None)
            if not isinstance(arr, wp.array) or arr.shape[0] not in (n, 6 * n):
                continue
            t = wp.to_torch(arr)
            if t.shape[0] == n:
                t.copy_(t[perm].contiguous())
            else:                      # (6n,) 으로 눕혀 둔 공분산
                t.copy_(t.view(n, 6)[perm].reshape(-1).contiguous())
            moved += 1
    return moved


def sort_by_cell(solver, n_grid=None, grid_lim=None):
    """칸 순서로 다시 세우고, 옮긴 배열 수를 돌려준다."""
    n = solver.n_particles
    n_grid = n_grid or solver.mpm_model.n_grid
    grid_lim = grid_lim or solver.mpm_model.grid_lim
    dx = grid_lim / n_grid
    x = wp.to_torch(solver.mpm_state.particle_x)
    key = _cell_key(x, dx, n_grid)
    perm = torch.argsort(key, stable=True)
    if not hasattr(solver, "orig_index"):
        solver.orig_index = torch.arange(n, device=x.device)
    solver.orig_index = solver.orig_index[perm].contiguous()
    return _permute_all(solver, perm)


def original_order(solver):
    """지금 순서를 **처음 순서**로 되돌리는 색인. 내보낼 때 쓴다."""
    if not hasattr(solver, "orig_index"):
        return None
    return torch.argsort(solver.orig_index)

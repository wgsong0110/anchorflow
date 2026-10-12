# [anchorflow] Newmark a^n 전달: 입자 가속도를 격자로 (질량 가중) / 격자 가속도를 입자로 -- PG g2p 와 같은 2차 B-스플라인
import warp as wp
from mpm_solver_warp.mpm_utils import *  # noqa


@wp.kernel
def af_p2g_acc(state: MPMStateStruct, model: MPMModelStruct, pa: wp.array(dtype=wp.vec3), ga: wp.array(dtype=wp.vec3, ndim=3)):
    p = wp.tid()
    if state.particle_selection[p] == 0:
        grid_pos = state.particle_x[p] * model.inv_dx
        bx = wp.int(grid_pos[0] - 0.5)
        by = wp.int(grid_pos[1] - 0.5)
        bz = wp.int(grid_pos[2] - 0.5)
        fx = grid_pos - wp.vec3(wp.float(bx), wp.float(by), wp.float(bz))
        wa = wp.vec3(1.5) - fx
        wb = fx - wp.vec3(1.0)
        wc = fx - wp.vec3(0.5)
        w = wp.mat33(wp.cw_mul(wa, wa) * 0.5, wp.vec3(0.0, 0.0, 0.0) - wp.cw_mul(wb, wb) + wp.vec3(0.75), wp.cw_mul(wc, wc) * 0.5)
        m = state.particle_mass[p]
        for i in range(0, 3):
            for j in range(0, 3):
                for k in range(0, 3):
                    weight = w[0, i] * w[1, j] * w[2, k]
                    wp.atomic_add(ga, bx + i, by + j, bz + k, pa[p] * (weight * m))


@wp.kernel
def af_g2p_acc(state: MPMStateStruct, model: MPMModelStruct, ga: wp.array(dtype=wp.vec3, ndim=3), pa: wp.array(dtype=wp.vec3)):
    p = wp.tid()
    if state.particle_selection[p] == 0:
        grid_pos = state.particle_x[p] * model.inv_dx
        bx = wp.int(grid_pos[0] - 0.5)
        by = wp.int(grid_pos[1] - 0.5)
        bz = wp.int(grid_pos[2] - 0.5)
        fx = grid_pos - wp.vec3(wp.float(bx), wp.float(by), wp.float(bz))
        wa = wp.vec3(1.5) - fx
        wb = fx - wp.vec3(1.0)
        wc = fx - wp.vec3(0.5)
        w = wp.mat33(wp.cw_mul(wa, wa) * 0.5, wp.vec3(0.0, 0.0, 0.0) - wp.cw_mul(wb, wb) + wp.vec3(0.75), wp.cw_mul(wc, wc) * 0.5)
        acc = wp.vec3(0.0, 0.0, 0.0)
        for i in range(0, 3):
            for j in range(0, 3):
                for k in range(0, 3):
                    weight = w[0, i] * w[1, j] * w[2, k]
                    acc = acc + ga[bx + i, by + j, bz + k] * weight
        pa[p] = acc

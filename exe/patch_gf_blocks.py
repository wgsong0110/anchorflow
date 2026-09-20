"""GF 솔버가 **입자가 닿은 블록만** 비우고 정규화하게 고친다. 물리는 그대로다.

수박 씬을 재보면 격자는 300^3 = 2700 만 칸인데 입자가 닿는 칸은 몇 % 뿐이다.
그런데 `zero_grid` 와 `grid_normalization_and_gravity` 는 서브스텝마다(프레임당
458 번) 2700 만 칸을 전부 훑는다. 실측으로 비우기 0.94 ms, 정규화 0.27 ms 로
서브스텝 7.03 ms 의 17% 다.

**왜 답이 안 바뀌는가.** g2p 가 읽는 칸은 p2g 가 질량을 넣은 칸뿐이고, 그것은
입자 스텐실이 덮는 칸, 즉 여기서 표시하는 블록 안이다. 블록 밖 칸은 값이 낡아
있어도 아무도 읽지 않는다. 표시는 p2g **전에** 입자 위치로 하므로 p2g 가 쓸
칸은 전부 덮인다.

  python exe/patch_gf_blocks.py --gf <GaussianFluent> [--block 4]
"""
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("--gf", required=True)
ap.add_argument("--block", type=int, default=4)
a = ap.parse_args()

MARK = "# [anchorflow] 블록 표시"

KERNELS = '''

''' + MARK + ''' -- 입자가 닿은 블록만 비우고 정규화한다
@wp.kernel
def af_clear_blocks(blocks: wp.array(dtype=int)):
    b = wp.tid()
    blocks[b] = 0


@wp.kernel
def af_mark_blocks(state: MPMStateStruct, model: MPMModelStruct,
                   blocks: wp.array(dtype=int), bs: int, nb: int):
    p = wp.tid()
    if state.particle_selection[p] == 0:
        gp = state.particle_x[p] * model.inv_dx
        b0x = wp.int(gp[0] - 0.5)
        b0y = wp.int(gp[1] - 0.5)
        b0z = wp.int(gp[2] - 0.5)
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    # 스텐실은 세 칸이라 축마다 블록 두 개를 걸칠 수 있다
                    bx = (b0x + i * 2) / bs
                    by = (b0y + j * 2) / bs
                    bz = (b0z + k * 2) / bs
                    if bx >= 0 and bx < nb and by >= 0 and by < nb and bz >= 0 and bz < nb:
                        blocks[(bx * nb + by) * nb + bz] = 1


@wp.kernel
def af_zero_blocks(state: MPMStateStruct, model: MPMModelStruct,
                   blocks: wp.array(dtype=int), bs: int, nb: int):
    bx, by, bz = wp.tid()
    if blocks[(bx * nb + by) * nb + bz] == 1:
        for i in range(bs):
            for j in range(bs):
                for k in range(bs):
                    gx = bx * bs + i
                    gy = by * bs + j
                    gz = bz * bs + k
                    if gx < model.grid_dim_x and gy < model.grid_dim_y and gz < model.grid_dim_z:
                        state.grid_m[gx, gy, gz] = 0.0
                        state.grid_v_in[gx, gy, gz] = wp.vec3(0.0, 0.0, 0.0)
                        state.grid_v_out[gx, gy, gz] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def af_norm_blocks(state: MPMStateStruct, model: MPMModelStruct, dt: float,
                   blocks: wp.array(dtype=int), bs: int, nb: int):
    bx, by, bz = wp.tid()
    if blocks[(bx * nb + by) * nb + bz] == 1:
        for i in range(bs):
            for j in range(bs):
                for k in range(bs):
                    gx = bx * bs + i
                    gy = by * bs + j
                    gz = bz * bs + k
                    if gx < model.grid_dim_x and gy < model.grid_dim_y and gz < model.grid_dim_z:
                        if state.grid_m[gx, gy, gz] > 1e-15:
                            v_out = (state.grid_v_in[gx, gy, gz]
                                     + state.grid_v_out[gx, gy, gz]) * (
                                1.0 / state.grid_m[gx, gy, gz])
                            v_out = v_out + dt * model.gravitational_accelaration
                            state.grid_v_out[gx, gy, gz] = v_out
'''

p = os.path.join(a.gf, "mpm_solver_warp", "mpm_utils.py")
s = open(p).read()
if MARK not in s:
    open(p, "w").write(s + KERNELS)
    print(f"커널 추가: {p}")

q = os.path.join(a.gf, "mpm_solver_warp", "mpm_solver_warp.py")
s = open(q).read()
if MARK in s:
    print("이미 되어 있다"); raise SystemExit(0)

# 1) 블록 배열을 initialize 에서 만든다
OLD = "        self.time = 0.0"
NEW = f'''        self.time = 0.0
        {MARK}: 입자가 닿은 블록만 비우고 정규화한다 (bs^3 칸 단위)
        self.af_bs = {a.block}
        self.af_nb = (n_grid + self.af_bs - 1) // self.af_bs
        self.af_blocks = wp.zeros(shape=self.af_nb ** 3, dtype=int, device=device)'''
if OLD not in s:
    raise SystemExit("initialize 안의 self.time 을 못 찾았다")
s = s.replace(OLD, NEW, 1)

# 2) zero_grid 를 블록판으로 바꾼다
OLD2 = """        wp.launch(
            kernel=zero_grid,
            dim=(grid_size),
            inputs=[self.mpm_state, self.mpm_model],
            device=device,
        )"""
NEW2 = f"""        {MARK}: 전체 격자 대신 닿은 블록만
        wp.launch(kernel=af_clear_blocks, dim=self.af_nb ** 3,
                  inputs=[self.af_blocks], device=device)
        wp.launch(kernel=af_mark_blocks, dim=self.n_particles,
                  inputs=[self.mpm_state, self.mpm_model, self.af_blocks,
                          self.af_bs, self.af_nb], device=device)
        wp.launch(kernel=af_zero_blocks,
                  dim=(self.af_nb, self.af_nb, self.af_nb),
                  inputs=[self.mpm_state, self.mpm_model, self.af_blocks,
                          self.af_bs, self.af_nb], device=device)"""
if OLD2 not in s:
    raise SystemExit("zero_grid 호출을 못 찾았다")
s = s.replace(OLD2, NEW2, 1)

# 3) 정규화도 블록판으로
OLD3 = """            wp.launch(
                kernel=grid_normalization_and_gravity,
                dim=(grid_size),
                inputs=[self.mpm_state, self.mpm_model, dt],
                device=device,
            )"""
NEW3 = """            wp.launch(kernel=af_norm_blocks,
                      dim=(self.af_nb, self.af_nb, self.af_nb),
                      inputs=[self.mpm_state, self.mpm_model, dt,
                              self.af_blocks, self.af_bs, self.af_nb],
                      device=device)"""
if OLD3 not in s:
    raise SystemExit("grid_normalization 호출을 못 찾았다")
s = s.replace(OLD3, NEW3, 1)

open(q, "w").write(s)
print(f"고쳤다: {q}")
print("GFBLOCKS_OK")

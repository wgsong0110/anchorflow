# i-PG 재현본: Newmark a^n 을 0 으로 고정하던 것(BUG 4 FIX)을 입자 가속도 저장 + P2G/G2P 전달로 (논문 Eq.8-9, 14)
import re, shutil, os
R = "/home/dkta/work/i-physgaussian"
P = f"{R}/implicit_mpm_solver.py"
if not os.path.exists(P + ".bak_before_an"):
    shutil.copy(P, P + ".bak_before_an")
s = open(P).read()
if "from af_newmark import" in s:
    print("이미 적용됨"); raise SystemExit(0)
open(f"{R}/af_newmark.py", "w").write('''# [anchorflow] Newmark a^n 전달: 입자 가속도를 격자로 (질량 가중) / 격자 가속도를 입자로 -- PG g2p 와 같은 2차 B-스플라인
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
''')
old = '''        a_n = np.zeros_like(v_grid_n)
'''
new = '''        a_n = np.zeros_like(v_grid_n)
        # [anchorflow] 위 "BUG 4 FIX"(a^n=0) 를 되돌린다: a^n=0 이면 Newmark(beta 0.25, gamma 0.5) 속도 갱신
        # v^{n+1} = v^n + dt(0.5 a^n + 0.5 a^{n+1}) 에서 일정 가속도가 절반만 들어가 자유낙하가 정확히 절반이 됐다.
        # 격자는 매 스텝 새로 만들지만 가속도는 **입자에 저장**해 두었다가 P2G 로 옮기면 된다 (첫 스텝만 0).
        if getattr(self, "_particle_a", None) is not None:
            from af_newmark import af_p2g_acc
            _ga = wp.zeros(shape=grid_size, dtype=wp.vec3, device=device)
            wp.launch(kernel=af_p2g_acc, dim=self.n_particles,
                      inputs=[self.mpm_state, self.mpm_model, self._particle_a, _ga], device=device)
            _ga_np = _ga.numpy()
            for d in range(3):
                a_n[..., d][mask] = _ga_np[..., d][mask] / _af_m
'''
assert s.count(old) == 1, "a_n 줄"; s = s.replace(old, new)
old = '''        du_k = cp.asarray(dt * v_grid_n, dtype=cp.float64)
'''
new = '''        du_k = cp.asarray(dt * v_grid_n + 0.5 * dt * dt * a_n, dtype=cp.float64)   # [anchorflow] Eq.14 (a^n 복원)
'''
assert s.count(old) == 1, "du_k 줄"; s = s.replace(old, new)
old = '''        # Restore and final G2P with converged velocity
        self._restore_state()
'''
new = '''        # Restore and final G2P with converged velocity
        self._restore_state()
        # [anchorflow] a^{n+1} 을 입자로 (x^n 에서의 가중치 -- g2p 가 x 를 옮기기 전에)
        from af_newmark import af_g2p_acc
        if getattr(self, "_particle_a", None) is None or self._particle_a.shape[0] != self.n_particles:
            self._particle_a = wp.zeros(shape=self.n_particles, dtype=wp.vec3, device=device)
        _ga2 = wp.array(cp.asnumpy(a_final.astype(cp.float32)), dtype=wp.vec3, device=device)
        wp.launch(kernel=af_g2p_acc, dim=self.n_particles,
                  inputs=[self.mpm_state, self.mpm_model, _ga2, self._particle_a], device=device)
'''
assert s.count(old) == 1, "restore 줄"; s = s.replace(old, new)
open(P, "w").write(s)
print("적용 완료")

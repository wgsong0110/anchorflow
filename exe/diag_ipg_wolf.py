# [final_fix_mpmbl] i-PG wolf 첫 서브스텝 Newton 진단: 잔차가 어느 격자점에 몰리는지 (바닥·붙잡힌 입자·자유), 전방차분 야코비안이 잔차와 맞는지.
# 사용: python exe/diag_ipg_wolf.py <pareto_mpmbl.py 인자들 그대로>   (pareto_mpmbl 의 준비부만 실행, 굴리기 없음)
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "pareto_mpmbl.py")).read()
cut = src.index("# ------------------------------------------------------------------------------------------------굴리기")
sys.argv = [os.path.join(HERE, "pareto_mpmbl.py")] + sys.argv[1:]
g = {"__name__": "__ipgdiag__", "__file__": sys.argv[0]}
exec(compile(src[:cut], sys.argv[0], "exec"), g)
sol, dt, wp, np = g["sol"], g["dt"], g["wp"], g["np"]
import cupy as cp  # noqa: E402

grid_size = (sol.mpm_model.grid_dim_x, sol.mpm_model.grid_dim_y, sol.mpm_model.grid_dim_z)
dev = sol.device
for k in range(len(sol.pre_p2g_operations)):
    wp.launch(kernel=sol.pre_p2g_operations[k], dim=sol.n_particles, inputs=[sol.time, dt, sol.mpm_state, sol.impulse_params[k]], device=dev)
for k in range(len(sol.particle_velocity_modifiers)):
    wp.launch(kernel=sol.particle_velocity_modifiers[k], dim=sol.n_particles,
              inputs=[sol.time, sol.mpm_state, sol.particle_velocity_modifier_params[k]], device=dev)
print(f"[준비] 입자 속도 수정기 {len(sol.particle_velocity_modifiers)} 개, 격자 후처리 {len(sol.grid_postprocess)} 개, dt {dt:.3e}", flush=True)
# 처음 상태의 F·응력
_F = wp.to_torch(sol.mpm_state.particle_F).reshape(-1, 3, 3); _Ft = wp.to_torch(sol.mpm_state.particle_F_trial).reshape(-1, 3, 3)
_I = g["torch"].eye(3, device=_F.device)
print(f"[F0] |F-I| 최대 {float((_F - _I).abs().max()):.3e}, |F_trial-I| 최대 {float((_Ft - _I).abs().max()):.3e}, "
      f"|stress| 최대 {float(wp.to_torch(sol.mpm_state.particle_stress).abs().max()):.3e}", flush=True)
_sel = wp.to_torch(sol.mpm_state.particle_selection)
print(f"[선택] selection 0 (계산) {int((_sel == 0).sum())} / {_sel.numel()}", flush=True)
from mpm_solver_warp import mpm_utils as _mu  # noqa: E402
wp.launch(kernel=_mu.compute_stress_from_F_trial,
          dim=sol.n_particles, inputs=[sol.mpm_state, sol.mpm_model, dt], device=dev)
_s = wp.to_torch(sol.mpm_state.particle_stress).reshape(-1, 3, 3); _F2 = wp.to_torch(sol.mpm_state.particle_F).reshape(-1, 3, 3)
print(f"[F=F_trial 응력] |stress| 최대 {float(_s.abs().max()):.3e} 중앙 {float(_s.abs().amax((1, 2)).median()):.3e}, 되돌림 뒤 |F-I| 최대 {float((_F2 - _I).abs().max()):.3e}", flush=True)
sol._save_state()
mom, mass = sol._p2g_momentum_only(dt)
mask = mass > 1e-8 * mass.max()
vn = np.zeros_like(mom)
vn[mask] = mom[mask] / mass[mask][:, None]
# BC 판정 (원 코드와 같음: 시작 속도에 후처리를 걸어 바뀐 점)
sol._write_grid_v(vn.astype(np.float32))
for k in range(len(sol.grid_postprocess)):
    wp.launch(kernel=sol.grid_postprocess[k], dim=grid_size, inputs=[sol.time, dt, sol.mpm_state, sol.mpm_model, sol.collider_params[k]], device=dev)
vbc = sol.mpm_state.grid_v_out.numpy()
bc_old = np.abs(vn - vbc).max(-1) > 1e-10
# 시험 속도 (모든 점 1) 로 판정하면 정지 상태에서도 바닥·상자 점이 잡힌다
one = np.ones_like(vn, dtype=np.float32)
sol._write_grid_v(one)
for k in range(len(sol.grid_postprocess)):
    wp.launch(kernel=sol.grid_postprocess[k], dim=grid_size, inputs=[sol.time, dt, sol.mpm_state, sol.mpm_model, sol.collider_params[k]], device=dev)
v1 = sol.mpm_state.grid_v_out.numpy()
bc_new = np.abs(one - v1).max(-1) > 1e-10
print(f"[BC] 질량 있는 점 {int(mask.sum())}, 원 판정 BC {int((bc_old & mask).sum())}, 시험속도 판정 BC {int((bc_new & mask).sum())}", flush=True)

g0 = np.array(sol.mpm_model.gravitational_accelaration, dtype=np.float64)
a_n = np.zeros_like(vn, dtype=np.float64)
a_n[mask] = g0
du0 = dt * vn + 0.5 * dt * dt * a_n
vng = cp.asarray(vn, dtype=cp.float64); ang = cp.asarray(a_n)
free = cp.asarray(mask & ~bc_old)


def R_of(du):
    a_new = (du - dt * vng - dt * dt * 0.25 * ang) / (0.25 * dt * dt)
    v_new = vng + dt * (0.5 * ang + 0.5 * a_new)
    fi, fe, m, kd = sol._af_eval_forces_gpu(v_new.astype(cp.float32), dt)
    R = cp.asarray(fe) + cp.asarray(fi) - cp.asarray(m)[..., None] * a_new
    R[~free] = 0
    return R, cp.asarray(fi), cp.asarray(m)


du = cp.asarray(du0)
R0, fi0, m0 = R_of(du)
rn = cp.linalg.norm(R0, axis=-1).get()
print(f"[잔차] |R| {float(np.sqrt((rn ** 2).sum())):.3e}, |m g| 합 {float(np.linalg.norm(mass[mask & ~bc_old]) * 9.8):.3e}, "
      f"|f_int| {float(cp.linalg.norm(fi0[free])):.3e}", flush=True)
idx = np.argsort(rn.ravel())[::-1][:12]
dx = float(sol.mpm_model.dx)
for i in idx:
    ijk = np.unravel_index(i, rn.shape)
    print(f"   점 {ijk} z {ijk[2] * dx:.3f} |R| {rn[ijk]:.3e} m {mass[ijk]:.3e} bc_new {bool(bc_new[ijk])} R {R0[ijk].get()} f_int {fi0[ijk].get()}", flush=True)
# 바닥 근처 (z<0.5) 와 BC 새 판정 점에 몰린 잔차 몫
zz = (np.arange(rn.shape[2]) * dx)[None, None, :] * np.ones(rn.shape)
for nm, sel in [("새 BC 점", bc_new), ("z<0.50", zz < 0.50), ("z<0.52", zz < 0.52)]:
    print(f"[몫] {nm}: |R|^2 몫 {float((rn[sel] ** 2).sum() / max((rn ** 2).sum(), 1e-30)):.3f}", flush=True)
# 전방차분 일관성: 무작위 방향 p 에 대해 (R(du+e p)-R)/e 가 e 에 따라 일정한가
rng = np.random.default_rng(0)
p = cp.asarray(rng.standard_normal(vn.shape)) * free[..., None]
p = p * dt * 1e-3 / float(cp.abs(p).max())
for e in [1e-1, 1e-2, 1e-3, 1e-4]:
    Rp, _, _ = R_of(du + e * p * 1e3)
    print(f"[FD] 크기 {e:.0e}: |dR|/크기 {float(cp.linalg.norm(Rp - R0)) / e:.4e}", flush=True)
# 관성만 (f_int 무시) 했을 때 잔차가 0 이 되는 du 를 넣어 남는 잔차 = f_int 와 BC 기여
a_star = ang * 0 + cp.asarray(g0)
du_star = dt * vng + dt * dt * (0.25 * ang + 0.25 * a_star)
Rs, fis, _ = R_of(du_star)
print(f"[a=g 대입] |R| {float(cp.linalg.norm(Rs)):.3e}, |f_int| {float(cp.linalg.norm(fis[free])):.3e}", flush=True)
sol._restore_state()

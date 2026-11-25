"""MPM baselines: PhysGaussian (PG), i-PhysGaussian (i-PG) and OmniPhysGS.

Subcommands (old script -> new):
  run              exe/stab_mpm.py        substep-stability runner, PG / i-PG (run from the i-PhysGaussian clone)
  omni             exe/stab_omni.py       substep-stability runner, OmniPhysGS (official MPMModel.p2g2p)
  pareto           exe/pareto_mpmbl.py    speed-accuracy runner, i-PG / OmniPhysGS (/ PG for checks)
  gt-timing        exe/gt_sens.py         ground-truth PG MPM re-run (sensitivity, frame timing; was baselines/gt_timing.py)
  patch-<name>     exe/patch_ipg_<name>.py  idempotent source patches applied once to the i-PhysGaussian clone,
                   <name> = bodyacc, fillcache, gravity, handle, initv, noisefloor, particles, scen
newton_gmres_fast  exe/ipg_fast.py        same Newton-GMRES as i-PG, free nodes compressed (enabled by AF_IPG_FAST=1)

  python exe/baseline_mpm.py run --method ipg --h5 init.h5 --config cfg.json --substeps 64 --out r.npz

(was baselines/mpm_baselines.py + baselines/gt_timing.py)
"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ==================================================================================================== i-PG fast Newton-GMRES (old exe/ipg_fast.py)
# i-PG Newton-GMRES 빠른 구현 (AF_IPG_FAST=1): 수식·허용오차·정체 판정·정밀도 그대로, 계산 자리만 줄임.
#
# i-physgaussian implicit_mpm_solver.ImplicitMPMSolver.p2g2p_newton_gmres (AF_IPG_GPU=1, AF_IPG_FUSED!=1 경로, patch-noisefloor 적용판)
# 와 같은 알고리즘을 줄 하나하나 옮겼다. 바뀐 것은 하나뿐:
#   원 코드는 Newton·GMRES 벡터를 격자 전체 (n_grid^3 x 3 = 200^3 x 3 = 2400 만 float64) 로 들고 다니며 비자유 노드를 매번 0 으로 지운다.
#   비자유 성분은 원 코드에서도 항상 정확히 0 (R·탐색 방향·Newton 증분 모두 0 으로 지움, 줄 탐색·갱신도 0 더하기) 이고 du 의 비자유 성분은
#   걸음 내내 초기값 그대로다 -> 자유 노드 성분만 모아 (압축) 같은 원소별 연산을 같은 순서로 한다. 격자 전체 v^{n+1} 은 걸음 처음 du0 에서
#   만든 float32 바탕에 자유 성분만 덮어써서 원 코드와 비트 단위로 같은 값을 warp 에 넘긴다.
#   -> 원소별 값은 비트 단위로 같고, 합 (노름·내적·GMRES Arnoldi) 만 0 항이 빠져 더하는 순서가 달라진다 (float64 반올림 수준).
# 걸음 준비 (v^n·a^n·경계 노드 판별, numpy) 와 마지막 G2P 는 원 코드 그대로.
#
#   쓰는 법: AF_IPG_FAST=1 이면 run·pareto 가 ImplicitMPMSolver.p2g2p_newton_gmres 를 이 함수로 바꿔 낀다.


def newton_gmres_fast(self, step: int, dt: float) -> dict:
    import numpy as np
    import warp as wp
    import cupy as cp
    from cupyx.scipy.sparse.linalg import gmres as cp_gmres
    from cupyx.scipy.sparse.linalg import LinearOperator as CpLinearOperator
    import implicit_mpm_solver as ims
    from implicit_mpm_solver import mpm, zero_stress, update_x_F_from_grid_v, clamp_particle_positions
    assert os.environ.get("AF_IPG_FUSED") != "1" and os.environ.get("AF_IPG_GPU") == "1", "AF_IPG_FAST 는 AF_IPG_GPU=1 (통합 커널 아님) 경로만"

    device = self.device
    grid_size = (self.mpm_model.grid_dim_x, self.mpm_model.grid_dim_y, self.mpm_model.grid_dim_z)

    # ---- Step 1: Pre-P2G impulses ----
    for k in range(len(self.pre_p2g_operations)):
        wp.launch(kernel=self.pre_p2g_operations[k], dim=self.n_particles,
                  inputs=[self.time, dt, self.mpm_state, self.impulse_params[k]], device=device)
    for k in range(len(self.particle_velocity_modifiers)):
        wp.launch(kernel=self.particle_velocity_modifiers[k], dim=self.n_particles,
                  inputs=[self.time, self.mpm_state, self.particle_velocity_modifier_params[k]], device=device)

    # ---- Step 2: Save state ----
    self._save_state()
    # 재질 상태 되돌림 (AF_IPG_MATRESTORE=1): PG plasticine (material 5, von_mises_return_mapping_with_damage) 의
    # 반환 사상은 model.yield_stress[p] 를 소성 변형만큼 깎고 0 이하면 model.mu·lam 을 0 으로 **바꿔 쓴다**. _save_state/_restore_state 는
    # 입자 x·v·C·F_trial 만 되돌려서, 걸음마다 수백 번 하는 잔차 평가 (Newton·GMRES 행렬곱·줄 탐색) 가 그때마다 연화를 누적한다
    # (PG 는 서브스텝당 1 회). -> 걸음 처음 값을 저장해 매 평가 전에 되돌리고, 마지막 확정 평가 1 회에만 연화가 남게 한다.
    _matr = os.environ.get("AF_IPG_MATRESTORE") == "1"
    if _matr:
        _md0 = self.mpm_model
        _mat_bufs = []
        for _k in ("yield_stress", "mu", "lam"):
            _a = getattr(_md0, _k); _b = wp.empty_like(_a); wp.copy(_b, _a); _mat_bufs.append((_a, _b))

        def _mat_restore():
            for _dst, _src in _mat_bufs:
                wp.copy(_dst, _src)
    else:
        def _mat_restore():
            pass

    # ---- Step 3: Setup (원 코드 그대로, numpy) ----
    beta_nm = 0.25
    gamma_nm = 0.5
    momentum_n, mass_n = self._p2g_momentum_only(dt)
    _af_rel = float(os.environ.get('AF_MASS_THRESH', '1e-08'))
    mass_thresh = (max(1e-30, _af_rel * float(mass_n.max())) if mass_n.max() > 0 else 1e-30)
    mask = mass_n > mass_thresh
    v_grid_n = np.zeros_like(momentum_n)
    _af_m = np.maximum(mass_n[mask], 1e-30)
    for d in range(3):
        v_grid_n[..., d][mask] = momentum_n[..., d][mask] / _af_m
    a_n = np.zeros_like(v_grid_n)
    if getattr(self, "_particle_a", None) is not None:
        from af_newmark import af_p2g_acc
        _ga = wp.zeros(shape=grid_size, dtype=wp.vec3, device=device)
        wp.launch(kernel=af_p2g_acc, dim=self.n_particles,
                  inputs=[self.mpm_state, self.mpm_model, self._particle_a, _ga], device=device)
        _ga_np = _ga.numpy()
        for d in range(3):
            a_n[..., d][mask] = _ga_np[..., d][mask] / _af_m
    else:
        _g0 = self.mpm_model.gravitational_accelaration
        for d in range(3):
            a_n[..., d][mask] = float(_g0[d])
    bc_mask = np.zeros(mass_n.shape, dtype=bool)
    if len(self.grid_postprocess) > 0:
        test_v = v_grid_n.copy()
        self._write_grid_v(test_v.astype(np.float32))
        for k_bc in range(len(self.grid_postprocess)):
            wp.launch(kernel=self.grid_postprocess[k_bc], dim=grid_size,
                      inputs=[self.time, dt, self.mpm_state, self.mpm_model, self.collider_params[k_bc]], device=device)
        test_v_bc = self.mpm_state.grid_v_out.numpy()
        diff_bc = np.abs(test_v - test_v_bc).max(axis=-1)
        bc_mask = diff_bc > 1e-10
    free_mask = mask & (~bc_mask)

    # ---- Step 4: Initial guess (Eq.14) ----
    du_full = cp.asarray(dt * v_grid_n + 0.5 * dt * dt * a_n, dtype=cp.float64)
    v_n_gpu = cp.asarray(v_grid_n, dtype=cp.float64)
    a_n_gpu = cp.asarray(a_n, dtype=cp.float64)
    mass_gpu = cp.asarray(mass_n, dtype=cp.float64)
    free_mask_gpu = cp.asarray(free_mask)

    # 자유 노드 압축 색인 (노드 순서 그대로, 성분 3 개)
    fi = cp.nonzero(free_mask_gpu.ravel())[0]
    idx3 = (fi[:, None] * 3 + cp.arange(3)[None]).ravel()
    du_k = du_full.ravel()[idx3]
    vn_c = v_n_gpu.ravel()[idx3]
    an_c = a_n_gpu.ravel()[idx3]
    # 격자 전체 v^{n+1} float32 바탕 (비자유 성분은 du 가 걸음 내내 du0 이라 이 값 그대로) -- 원 식 그대로
    _a0 = (du_full - dt * v_n_gpu - dt*dt*(0.5 - beta_nm) * a_n_gpu) / (beta_nm * dt * dt)
    _v0 = v_n_gpu + dt * ((1.0 - gamma_nm) * a_n_gpu + gamma_nm * _a0)
    base32 = _v0.astype(cp.float32).ravel()
    del _a0, _v0
    V = self._af_views()
    gin_flat = V["gin"].reshape(-1)
    gm_flat = V["gm"].reshape(-1)
    gout_flat = V["gout"].reshape(-1)
    g_ = self.mpm_model.gravitational_accelaration
    gvec = cp.asarray([float(g_[0]), float(g_[1]), float(g_[2])], dtype=cp.float64)
    stress_buf = getattr(self, "_af_stress_buf", None)
    if stress_buf is None or stress_buf.shape != self.mpm_state.particle_stress.shape:
        stress_buf = wp.empty_like(self.mpm_state.particle_stress); self._af_stress_buf = stress_buf
    if getattr(self, "_af_mat_cache", None) is None or getattr(self, "_af_rho_cache", None) is None:
        self._eval_forces_and_kdiag(cp.asnumpy(base32.reshape(V["gout"].shape)), dt)   # 캐시 (한 번) -- 원 경로와 같은 μ·λ·밀도 평균
    mu_avg, lam_avg = self._af_mat_cache
    _dx = float(self.mpm_model.dx)
    _kd_coef = (lam_avg + 2.0 * mu_avg) / (self._af_rho_cache * _dx * _dx)   # 원 식과 같은 순서
    st, md = self.mpm_state, self.mpm_model

    def eval_forces_c(v_new_c):
        """_af_eval_forces_gpu 와 같은 계산, 자유 성분만 돌려줌 (f_int·f_ext float64 [nc], mass float64 [nf], k_diag float32 [nf])."""
        self._restore_state()
        _mat_restore()
        cp.cuda.get_current_stream().synchronize()
        gout_flat[...] = base32
        gout_flat[idx3] = v_new_c.astype(cp.float32)
        cp.cuda.get_current_stream().synchronize()
        for k in range(len(self.grid_postprocess)):
            wp.launch(kernel=self.grid_postprocess[k], dim=grid_size,
                      inputs=[self.time, dt, st, md, self.collider_params[k]], device=device)
        wp.launch(kernel=update_x_F_from_grid_v, dim=self.n_particles,
                  inputs=[st.particle_x, st.particle_F_trial, st.grid_v_out, md.dx, md.inv_dx, dt, st.particle_selection, md.n_grid],
                  device=device)
        wp.launch(kernel=mpm.compute_stress_from_F_trial, dim=self.n_particles, inputs=[st, md, dt], device=device)
        wp.copy(stress_buf, st.particle_stress)
        wp.launch(kernel=zero_stress, dim=self.n_particles, inputs=[st.particle_stress, st.particle_selection], device=device)
        wp.launch(kernel=mpm.zero_grid, dim=grid_size, inputs=[st, md], device=device)
        wp.launch(kernel=mpm.p2g_apic_with_stress, dim=self.n_particles, inputs=[st, md, dt], device=device)
        wp.synchronize_device(device)
        momentum_c = gin_flat[idx3]
        mass32_c = gm_flat[fi]
        wp.copy(st.particle_stress, stress_buf)
        wp.launch(kernel=mpm.zero_grid, dim=grid_size, inputs=[st, md], device=device)
        wp.launch(kernel=mpm.p2g_apic_with_stress, dim=self.n_particles, inputs=[st, md, dt], device=device)
        wp.synchronize_device(device)
        f_int_c = ((gin_flat[idx3] - momentum_c) / cp.float32(dt)).astype(cp.float64)
        mass_c = mass32_c.astype(cp.float64)
        f_ext_c = (mass_c[:, None] * gvec).ravel()
        k_diag_c = _kd_coef * mass32_c
        return f_int_c, f_ext_c, mass_c, k_diag_c

    def eval_residual(du_c):
        a_new = (du_c - dt * vn_c - dt*dt*(0.5 - beta_nm) * an_c) / (beta_nm * dt * dt)
        v_new = vn_c + dt * ((1.0 - gamma_nm) * an_c + gamma_nm * a_new)
        f_int_c, f_ext_c, mass_c, k_diag_c = eval_forces_c(v_new)
        mass_expanded = cp.repeat(mass_c, 3)
        R = f_ext_c + f_int_c - mass_expanded * a_new
        return R, v_new, a_new, k_diag_c

    # ---- Step 5: Newton outer loop (원 코드 그대로) ----
    nc = int(du_k.size)
    converged = False
    total_evals = 0
    final_residual = float('inf')
    EW_eta_max = 0.9
    EW_eta_min = 1e-4
    min_newton_iters = int(os.environ.get("AF_NEWTON_MIN", "5"))
    _newton_tol = float(os.environ.get("AF_NEWTON_TOL", self.implicit_tolerance))
    _newton_max = int(os.environ.get("AF_NEWTON_MAX", self.newton_max_iters))
    _fd_mode = os.environ.get("AF_NEWTON_FD", "fwd")
    _fd_eps = float(os.environ.get("AF_FD_EPS", "1e-4"))
    prev_res_norm = None
    prev_prev_res_norm = None
    eta_k = EW_eta_max
    k_diag_latest = None
    # 전처리기 질량 항 (걸음 내내 같음)
    _mass_flat_c = cp.maximum(mass_gpu.ravel()[fi], mass_thresh)
    _inertia_c = cp.repeat(_mass_flat_c, 3) / (beta_nm * dt * dt)

    for newton_it in range(_newton_max):
        R_k, v_new_k, a_new_k, k_diag_latest = eval_residual(du_k)
        total_evals += 1
        res_norm = float(cp.sqrt(cp.sum(R_k ** 2)))
        if step == 0:
            _rel = res_norm / res_0 if newton_it > 0 else 1.0
            print(f"  [Newton {newton_it}] res={res_norm:.3e} rel={_rel:.3e} eta={eta_k:.2e}", flush=True)
        if newton_it == 0:
            res_0 = max(res_norm, 1e-14)
        rel_res = res_norm / res_0
        if newton_it == 0:
            _af_atol_c = float(os.environ.get("AF_NEWTON_ATOL_FEXT", "0") or 0)
            _af_atol = (_af_atol_c * 9.8 * float(cp.sqrt(cp.sum((mass_gpu * free_mask_gpu) ** 2)))) if _af_atol_c > 0 else 0.0
        if rel_res < _newton_tol or res_norm < 1e-10 or res_norm < _af_atol:
            converged = True
            final_residual = res_norm
            break
        if newton_it >= min_newton_iters and prev_res_norm is not None:
            if prev_res_norm > 1e-14:
                contraction = res_norm / prev_res_norm
                if contraction > 0.995:
                    if step == 0:
                        print(f"  [Newton] stagnation iter {newton_it}, c={contraction:.3f}", flush=True)
                    break
        final_residual = res_norm
        if prev_res_norm is not None and prev_prev_res_norm is not None and prev_prev_res_norm > 1e-14:
            eta_ew = abs(res_norm - prev_res_norm) / prev_prev_res_norm
            eta_sg = 0.9 * (res_norm / prev_res_norm) ** 1.5 if prev_res_norm > 1e-14 else EW_eta_max
            eta_k = min(EW_eta_max, max(EW_eta_min, max(eta_ew, eta_sg)))
        elif prev_res_norm is not None:
            eta_k = EW_eta_max
        prev_prev_res_norm = prev_res_norm
        prev_res_norm = res_norm

        R_k_flat = R_k.ravel()

        def matvec(p_flat, _du_k=du_k, _R_base=R_k):
            nonlocal total_evals
            norm_inf = float(cp.max(cp.abs(p_flat)))
            if norm_inf < 1e-14:
                return p_flat.copy()
            eps_fd = _fd_eps / norm_inf
            R_plus, _, _, _ = eval_residual(_du_k + eps_fd * p_flat)
            total_evals += 1
            if _fd_mode == "fwd":
                return (R_plus.ravel() - _R_base.ravel()) / eps_fd
            R_minus, _, _, _ = eval_residual(_du_k - eps_fd * p_flat)
            total_evals += 1
            return (R_plus.ravel() - R_minus.ravel()) / (2.0 * eps_fd)

        A = CpLinearOperator((nc, nc), matvec=matvec, dtype=cp.float64)
        _k_diag_rep = cp.repeat(cp.asarray(k_diag_latest.ravel(), dtype=cp.float64), 3)
        _diag_w = _inertia_c + _k_diag_rep
        _diag_w = cp.maximum(_diag_w, 1e-8)
        M_prec = CpLinearOperator((nc, nc), matvec=lambda x: x / _diag_w, dtype=cp.float64)
        _gk = dict(M=M_prec, restart=self.gmres_max_iters, maxiter=self.gmres_max_cycles, atol=0)
        try:
            delta_flat, info = cp_gmres(A, -R_k_flat, rtol=float(eta_k), **_gk)
        except TypeError:
            delta_flat, info = cp_gmres(A, -R_k_flat, tol=float(eta_k), **_gk)
        delta_du = delta_flat

        phi_0 = 0.5 * float(cp.sum(R_k_flat ** 2))
        J_delta = matvec(delta_flat, du_k)
        dphi_0 = float(cp.dot(R_k_flat, cp.asarray(J_delta, dtype=cp.float64)))
        if dphi_0 >= 0:
            delta_du = -R_k.copy()
            dphi_0 = -2.0 * phi_0
        alpha = 1.0
        c1 = 1e-4
        for ls_it in range(10):
            du_trial = du_k + alpha * delta_du
            R_trial, _, _, _ = eval_residual(du_trial)
            total_evals += 1
            phi_trial = 0.5 * float(cp.sum(R_trial.ravel() ** 2))
            if phi_trial <= phi_0 + c1 * alpha * dphi_0:
                break
            alpha *= 0.5
        else:
            print(f"  [LS 실패] step={step} newton={newton_it} reject={os.environ.get('AF_LS_REJECT') == '1'}", flush=True)
            if os.environ.get("AF_LS_REJECT") == "1":
                final_residual = res_norm
                break
        du_k = du_k + alpha * delta_du

    # ---- Step 6: Final state (원 코드 그대로, 격자 전체 du 로 되돌림) ----
    du_full.ravel()[idx3] = du_k
    a_final = (du_full - dt * v_n_gpu - dt*dt*(0.5 - beta_nm) * a_n_gpu) / (beta_nm * dt * dt)
    v_final = v_n_gpu + dt * ((1.0 - gamma_nm) * a_n_gpu + gamma_nm * a_final)
    v_final_np = cp.asnumpy(v_final.astype(cp.float32))
    self._restore_state()
    _mat_restore()
    if _matr:
        _ys = wp.to_torch(self.mpm_model.yield_stress)
        _matr_msg = f" ys_min={float(_ys.min()):.2f} ys_le0={int((_ys <= 0).sum())}"
    else:
        _matr_msg = ""
    from af_newmark import af_g2p_acc
    if getattr(self, "_particle_a", None) is None or self._particle_a.shape[0] != self.n_particles:
        self._particle_a = wp.zeros(shape=self.n_particles, dtype=wp.vec3, device=device)
    _a_np = cp.asnumpy(a_final.astype(cp.float32))
    _abc_msg = ""
    if os.environ.get("AF_IPG_BCA0") == "1" and len(self.grid_postprocess) > 0:
        # 경계조건 (sticky 바닥·상자) 이 속도를 정하는 노드는 v^{n+1} 이 BC 값으로 덮이는데 a^{n+1} 은 Newton 이 안 건드린
        # 예측값 (= a^n, 정지 바닥이면 중력 g 가 그대로) 이 남아 af_g2p_acc 로 입자에 실린다 -> 다음 걸음 v^{n+1} = v^n + dt/2 (a^n + a^{n+1}) 에
        # 반력 없는 아래 방향 가속이 계속 들어감. BC 가 속도를 정하는 노드 (탐침 속도 +-1 이 바뀌는 노드) 의 a^{n+1} = 0 (정지 노드의 가속도).
        _cm = np.zeros(mass_n.shape, dtype=bool)
        for _sg in (1.0, -1.0):
            _pv = np.full(v_grid_n.shape, _sg, dtype=np.float32)
            self._write_grid_v(_pv)
            for k_bc in range(len(self.grid_postprocess)):
                wp.launch(kernel=self.grid_postprocess[k_bc], dim=grid_size,
                          inputs=[self.time, dt, self.mpm_state, self.mpm_model, self.collider_params[k_bc]], device=device)
            _cm |= np.abs(self.mpm_state.grid_v_out.numpy() - _pv).max(axis=-1) > 1e-10
        _cmm = _cm & mask
        _abc_msg = (f" bcA0 nodes={int(_cmm.sum())} az_bc_mean={float(_a_np[..., 2][_cmm].mean()) if _cmm.any() else 0.0:.3f}")
        _a_np[_cm] = 0.0
    _ga2 = wp.array(_a_np, dtype=wp.vec3, device=device)
    wp.launch(kernel=af_g2p_acc, dim=self.n_particles,
              inputs=[self.mpm_state, self.mpm_model, _ga2, self._particle_a], device=device)
    wp.launch(kernel=mpm.compute_stress_from_F_trial, dim=self.n_particles,
              inputs=[self.mpm_state, self.mpm_model, dt], device=device)
    if self.mpm_model.grid_v_damping_scale < 1.0:
        v_final_np = v_final_np * self.mpm_model.grid_v_damping_scale
    self._write_grid_v(v_final_np)
    for k_bc in range(len(self.grid_postprocess)):
        wp.launch(kernel=self.grid_postprocess[k_bc], dim=grid_size,
                  inputs=[self.time, dt, self.mpm_state, self.mpm_model, self.collider_params[k_bc]], device=device)
        if self.modify_bc[k_bc] is not None:
            self.modify_bc[k_bc](self.time, dt, self.collider_params[k_bc])
    wp.launch(kernel=mpm.g2p, dim=self.n_particles, inputs=[self.mpm_state, self.mpm_model, dt], device=device)
    wp.launch(kernel=clamp_particle_positions, dim=self.n_particles,
              inputs=[self.mpm_state.particle_x, self.mpm_state.particle_v, self.mpm_state.particle_selection,
                      self.mpm_model.inv_dx, self.mpm_model.n_grid], device=device)
    self.time += dt
    self.iteration_stats['total_iters'] += newton_it + 1
    self.iteration_stats['total_steps'] += 1
    if converged:
        self.iteration_stats['converged'] += 1
    particle_v_np = self.mpm_state.particle_v.numpy()
    max_v = float(np.max(np.abs(particle_v_np)))
    gv_max = float(np.max(np.abs(v_final_np))) if np.all(np.isfinite(v_final_np)) else float('inf')
    print(f"[Newton-GMRES] step={step} newton={newton_it+1} converged={converged} "
          f"res={final_residual:.3e} evals={total_evals} "
          f"gv_max={gv_max:.2f} max_v={max_v:.3f}{_abc_msg}{_matr_msg}", flush=True)
    return {'converged': converged, 'iterations': newton_it + 1, 'final_residual': final_residual, 'max_velocity': max_v}


# ==================================================================================================== i-PhysGaussian clone patches (old exe/patch_ipg_*.py)


def patch_bodyacc(a):
    # i-PG 에 **체적 가속도** 경계조건(`body_acceleration`)을 더한다.
    #
    # 그쪽 `particle_impulse` 는 상자 안 입자마다 **같은 힘**을 주고 각자 질량으로 나눈다
    # (`v += F/m·dt`). 입자 질량이 제각각이라(표면 가우시안 입자는 아주 가볍다) 가벼운
    # 입자만 크게 튄다 -- 시연에서 얇은 판·가장자리만 날아가고 몸통은 버텼다. 물체 한
    # 부분에 고르게 힘을 주려면 힘이 질량에 비례해야 하고, 그러면 가속도가 같다:
    #     v += a·dt         (a 는 모든 입자에 같다)
    #
    # 쓰는 법 (config 의 boundary_conditions):
    #     {"type": "body_acceleration", "acc": [ax, ay, az],
    #      "point": [...], "size": [...],        # 처음 위치로 한 번 고른다 (반폭 상자)
    #      "start_time": t0, "end_time": t1}
    #
    # impulse_scale 을 곱하지 않는다 -- 가속도는 매 스텝 실제 dt 로 적분되므로 암시
    # 적분의 큰 스텝에서도 시간에 대해 그대로다. 여러 번 돌려도 안전하다.
    #
    #   python exe/baseline_mpm.py patch-bodyacc --ipg /workspace/i-physgaussian
    # 0) 가속도 커널은 **모듈 최상위**에 둔다. 메서드 안에서 정의한 클로저 커널은
    #    모듈이 이미 적재된 뒤라 warp 가 "Failed to find forward kernel" 로 못 찾는
    #    일이 있었다 (PG 에서는 매번, i-PG 에서는 가끔).
    KERNEL = """
@wp.kernel
def af_apply_acc(
    time: float, dt: float, state: MPMStateStruct, param: Impulse_modifier
):
    p = wp.tid()
    if time >= param.start_time and time < param.end_time:
        if param.mask[p] == 1:
            state.particle_v[p] = state.particle_v[p] + param.force * dt


"""

    # 1) 솔버 메서드
    sp = os.path.join(a.ipg, "mpm_solver_warp", "mpm_solver_warp.py")
    s = open(sp).read()
    # 예전 판(클로저 커널)을 걷어낸다
    old_closure = """
        @wp.kernel
        def apply_acc(
            time: float, dt: float, state: MPMStateStruct, param: Impulse_modifier
        ):
            p = wp.tid()
            if time >= param.start_time and time < param.end_time:
                if param.mask[p] == 1:
                    state.particle_v[p] = state.particle_v[p] + param.force * dt

        self.pre_p2g_operations.append(apply_acc)
"""
    if old_closure in s:
        s = s.replace(old_closure, "\n        self.pre_p2g_operations.append(af_apply_acc)\n", 1)
        print("[패치] 클로저 커널을 걷어냈다")
    if "def af_apply_acc" not in s:
        cls = "class MPM_Simulator_WARP"
        assert cls in s, "클래스 위치를 못 찾았다"
        s = s.replace(cls, KERNEL + cls, 1)
        print("[패치] 최상위 커널을 넣었다")
    open(sp, "w").write(s)
    if "def add_acceleration_on_particles" in s:
        print("[패치] 솔버 메서드는 이미 있다")
    else:
        anchor = "    def enforce_particle_velocity_translation("
        assert anchor in s, "삽입 위치를 못 찾았다"
        meth = '''    def add_acceleration_on_particles(
        self,
        acc,
        point=[1, 1, 1],
        size=[1, 1, 1],
        start_time=0.0,
        end_time=999.0,
        device="cuda:0",
    ):
        """[fusedgs] 상자 안 입자(처음 위치로 고름)에 **같은 가속도** 를 준다."""
        param = Impulse_modifier()
        param.start_time = start_time
        param.end_time = end_time
        param.point = wp.vec3(point[0], point[1], point[2])
        param.size = wp.vec3(size[0], size[1], size[2])
        param.mask = wp.zeros(shape=self.n_particles, dtype=int, device=device)
        param.force = wp.vec3(acc[0], acc[1], acc[2])   # 여기서는 가속도
        wp.launch(
            kernel=selection_add_impulse_on_particles,
            dim=self.n_particles,
            inputs=[self.mpm_state, param],
            device=device,
        )
        self.impulse_params.append(param)
        self.pre_p2g_operations.append(af_apply_acc)

'''
        s = s.replace(anchor, meth + anchor, 1)
        open(sp, "w").write(s)
        print(f"[패치] {sp}")

    # 2) 설정 해석
    dp = os.path.join(a.ipg, "utils", "decode_param.py")
    s = open(dp).read()
    if '"body_acceleration"' in s:
        print("[패치] decode_param 에 이미 있다")
    else:
        anchor = '''        elif bc["type"] == "bounding_box":'''
        assert anchor in s, "decode_param 삽입 위치를 못 찾았다"
        br = '''        elif bc["type"] == "body_acceleration":
            # [fusedgs] 질량에 비례하는 힘 = 같은 가속도 (impulse_scale 미적용)
            mpm_solver.add_acceleration_on_particles(
                acc=bc["acc"],
                point=bc.get("point", [1, 1, 1]),
                size=bc.get("size", [1, 1, 1]),
                start_time=bc.get("start_time", 0.0),
                end_time=bc.get("end_time", 1e3),
            )
'''
        s = s.replace(anchor, br + anchor, 1)
        open(dp, "w").write(s)
        print(f"[패치] {dp}")


def patch_fillcache(a):
    # i-PG 의 입자 채우기를 **캐시**하게 만든다 (읽기 + 쓰기).
    #
    # 채우기는 i-PG 실행에서 가장 느리고 가장 잘 깨지는 단계다. 한 번 성공하면 npy 로
    # 남겨 두고 다음 실행은 즉시 읽어 쓴다 -- 그래야 중단돼도 손실 없이 재시작된다.
    #
    # `patch-scen` 가 넣은 래퍼는 읽기만 했다. 여기서 없으면 만들어 두는 쪽을
    # 더한다. 여러 번 돌려도 안전하다.
    #
    #   python exe/baseline_mpm.py patch-fillcache --ipg /workspace/i-physgaussian
    p = os.path.join(a.ipg, "gs_simulation.py")
    s = open(p).read()
    if "AF_FILL_DUMP" in s:
        print("[패치] 이미 들어 있다")
        return

    old = """        return _af_orig_fill(*args, **kwargs)
"""
    new = """        _r = _af_orig_fill(*args, **kwargs)
        # 성공한 채우기를 남겨 둔다 -- 다음 실행은 위 분기에서 즉시 읽는다.
        if _ck:
            try:
                _af_np.save(_ck, _r.detach().cpu().numpy())
                print(f'[PG채움] 캐시 저장 {_ck} ({_r.shape[0]} 개) AF_FILL_DUMP',
                      flush=True)
            except Exception as _e:
                print(f'[PG채움] 캐시 저장 실패: {_e}', flush=True)
        return _r
"""
    assert s.count(old) == 1, "patch_ipg_scen 의 채우기 래퍼를 못 찾았다"
    open(p, "w").write(s.replace(old, new, 1))
    print(f"[패치] 채우기 캐시 쓰기 추가 -> {p}")


def patch_gravity(a):
    # i-PG implicit 솔버가 **가벼운 격자점을 통째로 버려 중력이 사라지는** 버그를 고친다.
    #
    # 증상: 몸체에서 떨어져 나간 입자가 중력을 전혀 받지 않는다. 실측(교사 궤적,
    # dv_z/dt 분위) --
    #
    #     PG   : 1% 분위 -9.84,  -9.8+-2 인 비율 2.52%,  v_z 최소 -1.672
    #     i-PG : 1% 분위 -6.17,  -9.8+-2 인 비율 0.38%,  v_z 최소 -0.875
    #
    # 원인:
    #
    #     mass_thresh = max(1e-10, 1e-2 * mass_n.max())    # **최대 질량의 1%**
    #     mask = mass_n > mass_thresh
    #     free_mask = mask & (~bc_mask)
    #     ...
    #     R[~free_mask] = 0.0
    #
    # 0 으로 나누기를 막으려는 장치인데 문턱이 **상대값**이다. 몸체 내부 격자점은
    # 여러 입자가 겹쳐 질량이 크고 이탈 입자는 혼자 싣기 때문에 두 자릿수 작다 --
    # 그래서 1% 문턱이 정확히 이탈 입자의 격자점만 골라 잘라낸다. 잘린 격자점은
    # `R=0` 이 되어 운동방정식이 사라지고 속도가 `v^n` 에 얼어붙는다. `f_ext = m*g`
    # 도 함께 지워지므로 **중력이 통째로 없어진다**.
    #
    # 고치는 방법: 문턱을 수치 보호 목적에 맞는 **절대 기준**으로 내린다. 질량이
    # 0 에 가까운 격자점만 빼고(0 나눗셈 방지), 실제 입자가 있는 가벼운 격자점은
    # 방정식을 살려 둔다. 나눗셈 자리에는 하한을 둬서 보호를 유지한다.
    #
    # 여러 번 돌려도 안전하다.
    #
    #   python exe/baseline_mpm.py patch-gravity --ipg /workspace/i-physgaussian
    p = os.path.join(a.ipg, "implicit_mpm_solver.py")
    s = open(p).read()
    if "AF_MASS_THRESH" in s:
        print("[패치] 이미 들어 있다")
        return

    old = ("        mass_thresh = max(1e-10, 1e-2 * float(mass_n.max()))"
           " if mass_n.max() > 0 else 1e-10\n"
           "        mask = mass_n > mass_thresh\n")
    assert old in s, "mass_thresh 줄을 못 찾았다"
    new = (
        "        # AF_MASS_THRESH: 문턱은 **0 나눗셈 방지용** 이라 절대값이어야 한다.\n"
        "        # 원래 1e-2*max 였는데, 몸체 내부 격자점(여러 입자 겹침)보다 두\n"
        "        # 자릿수 가벼운 **이탈 입자의 격자점**이 거기 걸려 잘려 나갔다.\n"
        "        # 잘리면 R=0 이 되어 운동방정식이 사라지고 f_ext=m*g 까지 지워져\n"
        "        # 중력을 전혀 받지 못한다 (실측: 이탈 입자 dv_z/dt ~ 0, PG 는 -9.8).\n"
        "        _af_rel = float(_af_os_g.environ.get('AF_MASS_THRESH', '%g'))\n"
        "        mass_thresh = (max(1e-30, _af_rel * float(mass_n.max()))\n"
        "                       if mass_n.max() > 0 else 1e-30)\n"
        "        mask = mass_n > mass_thresh\n" % a.rel)
    s = s.replace(old, new, 1)

    # 나눗셈 보호: 질량으로 나누는 자리에 하한을 둔다 (문턱을 내린 대신)
    old2 = ("        v_grid_n = np.zeros_like(momentum_n)\n"
            "        for d in range(3):\n"
            "            v_grid_n[..., d][mask] = momentum_n[..., d][mask]"
            " / mass_n[mask]\n")
    assert old2 in s, "v_grid_n 나눗셈을 못 찾았다"
    new2 = ("        v_grid_n = np.zeros_like(momentum_n)\n"
            "        # 문턱을 내렸으므로 나눗셈 자리에서 보호한다\n"
            "        _af_m = np.maximum(mass_n[mask], 1e-30)\n"
            "        for d in range(3):\n"
            "            v_grid_n[..., d][mask] = momentum_n[..., d][mask] / _af_m\n")
    s = s.replace(old2, new2, 1)

    if "import os as _af_os_g" not in s:
        s = s.replace("import numpy as np\n", "import numpy as np\nimport os as _af_os_g\n", 1)
    open(p, "w").write(s)
    print(f"[패치] 질량 문턱을 1e-2*max -> {a.rel:g}*max 로 내리고 나눗셈 보호 추가")
    print(f"        -> {p}")


def patch_handle(a):
    # i-PG 의 손잡이를 **하드 Dirichlet 격자 BC** 로 바꾼다.
    #
    # 전에는 스텝 전에 입자 속도를 감쇠 가중으로 섞었다(`patch-scen`). implicit
    # 솔버에서는 그게 지켜지지 않는다 -- `bc_mask` 는 **`grid_postprocess` 가 바꾼
    # 노드**에서만 잡히므로(implicit_mpm_solver.py 의 "Identify Dirichlet (BC) nodes"),
    # 입자 쪽에서만 건드린 값은 Newton 반복이 그대로 풀어 버린다.
    #
    # 그래서 손잡이를 구면 Dirichlet 콜라이더로 등록한다:
    #
    #     |x_node - c_j| < R  ->  grid_v_out[node] = v_cmd_j        (하드)
    #
    # c_j 는 그 손잡이 입자의 **현재 위치**(PG 규약과 같다), v_cmd_j 는 시나리오 명령
    # 속도다. 둘 다 프레임마다 호스트에서 콜라이더 파라미터에 써 넣는다.
    #
    # 여러 번 돌려도 안전하다.
    #
    #   python exe/baseline_mpm.py patch-handle --ipg /workspace/i-physgaussian
    # ---------------------------------------------------------------- 1) 솔버 API
    sp = os.path.join(a.ipg, "mpm_solver_warp", "mpm_solver_warp.py")
    s = open(sp).read()
    if "add_handle_sphere" in s:
        print("[패치] 솔버에 이미 들어 있다")
    else:
        anchor = "    def add_bounding_box(self"
        assert anchor in s, "add_bounding_box 를 못 찾았다"
        new = '''    def add_handle_sphere(self, point, velocity, radius,
                          start_time=0.0, end_time=999.0):
        """움직이는 구면 **하드 Dirichlet**. 반경 안 격자 속도를 명령값으로 박는다.

        grid_postprocess 에 들어가므로 implicit 솔버의 bc_mask 가 이 노드를 잡고,
        Newton 잔차·탐색방향에서 0 이 되어 실제로 구속된다. 프레임마다 호스트에서
        `collider_params[k].point / .velocity` 를 갱신해 손잡이를 움직인다.
        """
        collider_param = Dirichlet_collider()
        collider_param.start_time = start_time
        collider_param.end_time = end_time
        collider_param.point = wp.vec3(float(point[0]), float(point[1]),
                                       float(point[2]))
        collider_param.velocity = wp.vec3(float(velocity[0]), float(velocity[1]),
                                          float(velocity[2]))
        collider_param.radius = float(radius)
        collider_param.surface_type = 0
        collider_param.friction = 0.0
        self.collider_params.append(collider_param)

        @wp.kernel
        def collide(
            time: float,
            dt: float,
            state: MPMStateStruct,
            model: MPMModelStruct,
            param: Dirichlet_collider,
        ):
            grid_x, grid_y, grid_z = wp.tid()
            if time >= param.start_time and time < param.end_time:
                offset = wp.vec3(
                    float(grid_x) * model.dx - param.point[0],
                    float(grid_y) * model.dx - param.point[1],
                    float(grid_z) * model.dx - param.point[2],
                )
                if wp.length(offset) < param.radius:
                    state.grid_v_out[grid_x, grid_y, grid_z] = wp.vec3(
                        param.velocity[0], param.velocity[1], param.velocity[2]
                    )

        self.grid_postprocess.append(collide)
        self.modify_bc.append(None)

'''
        s = s.replace(anchor, new + anchor, 1)
        open(sp, "w").write(s)
        print(f"[패치] add_handle_sphere 추가 -> {sp}")

    # ------------------------------------------------------- 2) gs_simulation 연결
    gp = os.path.join(a.ipg, "gs_simulation.py")
    g = open(gp).read()
    if "AF_H_HARD" in g:
        print("[패치] gs_simulation 에 이미 들어 있다")
        return

    # 예전 소프트 혼합 블록을 찾아 하드 BC 등록·갱신으로 교체한다.
    old_key = "    _af_scen = None\n"
    assert old_key in g, "patch_ipg_scen 의 블록을 못 찾았다 (먼저 그것을 적용할 것)"
    # 소프트 혼합(프레임 루프 안의 _vr 조작)을 제거한다
    import re
    g2 = re.sub(
        r"\n        if _af_scen is not None:\n"
        r"(?:            .*\n)+?(?=        [^ ])",
        "\n", g, count=1)
    assert g2 != g, "프레임 루프 안 소프트 혼합 블록을 못 찾았다"
    g = g2
    # 손잡이 소속·가중 계산도 필요 없다. 등록 + 프레임 갱신으로 바꾼다.
    old_reg = g[g.index(old_key):g.index("    for frame in tqdm(range(frame_num)):")]
    new_reg = '''    _af_scen = None
    if _af_os.environ.get('AF_H_SCEN'):
        # 하드 Dirichlet: 손잡이 입자 위치를 중심으로 반경 R 격자 속도를 박는다.
        _af_scen = _af_np.load(_af_os.environ['AF_H_SCEN'])
        _af_R = float(_af_os.environ.get('AF_H_R', 0.15))
        _af_hid = _af_t.as_tensor(_af_scen['hid'], dtype=_af_t.long)
        _af_vel = _af_scen['vel']
        _x0 = mpm_solver.export_particle_x_to_torch()
        _af_k0 = len(mpm_solver.collider_params)
        for _j in range(_af_hid.numel()):
            _c = _x0[_af_hid[_j]].tolist()
            mpm_solver.add_handle_sphere(_c, [0.0, 0.0, 0.0], _af_R)
        print(f'[손잡이-하드] {_af_hid.numel()} 개, R={_af_R}, '
              f'콜라이더 {_af_k0}..{_af_k0 + _af_hid.numel() - 1}, '
              f'시나리오 {_af_vel.shape[0]} 프레임  AF_H_HARD', flush=True)
'''
    g = g.replace(old_reg, new_reg, 1)
    # 프레임 루프 머리에 파라미터 갱신을 넣는다
    old_loop = "    for frame in tqdm(range(frame_num)):\n"
    new_loop = old_loop + '''        if _af_scen is not None:
            # 손잡이 중심은 **그 입자의 현재 위치**(PG 규약), 속도는 명령값
            _xc = mpm_solver.export_particle_x_to_torch()
            _vc = _af_vel[min(frame, _af_vel.shape[0] - 1)]
            for _j in range(_af_hid.numel()):
                _p = _xc[_af_hid[_j]].tolist()
                _cp = mpm_solver.collider_params[_af_k0 + _j]
                _cp.point = wp.vec3(float(_p[0]), float(_p[1]), float(_p[2]))
                _cp.velocity = wp.vec3(float(_vc[_j][0]), float(_vc[_j][1]),
                                       float(_vc[_j][2]))
'''
    assert old_loop in g
    g = g.replace(old_loop, new_loop, 1)
    if "import warp as wp" not in g:
        g = g.replace("import torch", "import torch\nimport warp as wp", 1)
    open(gp, "w").write(g)
    print(f"[패치] gs_simulation 에 하드 손잡이 등록·갱신 추가 -> {gp}")


def patch_initv(a):
    # i-PG 러너가 config 의 `init_velocity` 를 읽게 한다 (PG 러너에 넣은 것과 같은 패치).
    #
    # PhysGaussian 쪽 gs_simulation.py 에는 이 패치가 들어가 있었는데 i-PG 쪽에는 없어서,
    # 파괴 시연에서 초기 하강속도가 **조용히 무시**됐다 (궤적의 첫 프레임 vz 가 0, 중력으로만
    # 천천히 떨어졌다). finalize_mu_lam 바로 뒤, 모든 입자 속도를 그 값으로 둔다.
    # 여러 번 돌려도 안전하다.
    #
    #   python exe/baseline_mpm.py patch-initv --ipg /workspace/i-physgaussian
    p = os.path.join(a.ipg, "gs_simulation.py")
    s = open(p).read()
    if "[fusedgs] init v" in s:
        print("[패치] 이미 있다")
    else:
        anchor = "    mpm_solver.finalize_mu_lam()\n"
        assert s.count(anchor) == 1, "finalize_mu_lam 위치가 하나가 아니다"
        add = anchor + '''    import json as _j3
    _v0 = _j3.load(open(args.config)).get("init_velocity", None)
    if _v0 is not None:
        mpm_solver.import_particle_v_from_torch(
            torch.zeros(mpm_init_pos.shape[0], 3, device="cuda").add_(
                torch.tensor(_v0, device="cuda", dtype=torch.float32)))
        print("[fusedgs] init v", _v0, flush=True)
'''
        s = s.replace(anchor, add, 1)
        open(p, "w").write(s)
        print(f"[패치] {p}")


def patch_noisefloor(a):
    # i-PG Newton-GMRES 에 float32 응력 잡음 바닥을 다루는 두 선택지를 넣는다 (환경 변수로만 켜짐, 기본 = 원래 동작 그대로).
    #
    # 진단 (exe/diag_ipg_wolf.py, wolf E 5e7 모래): F_trial = I 정확히여도 warp svd3 반올림으로 모든 입자 Kirchhoff 응력이 11.3 Pa
    # -> 첫 서브스텝 잔차 |R0| = |f_int| = 0.24 (전체 |m g| 4.46 의 5 %) 가 전부 이 잡음이다. 상대 기준 |R| < 1e-4 |R0| 은 잡음 아래라
    # 닿을 수 없고, Armijo 줄 탐색이 10 번 모두 실패한 뒤에도 마지막 alpha 로 갱신해 (원 코드) k>=5 에서 첫 프레임에 발산한다.
    #
    #   AF_NEWTON_ATOL_FEXT=c : |R| < c * |f_ext(자유점)| 이면 수렴 (절대 기준을 중력 하중에 맞춤). 0/없음 = 끔.
    #   AF_LS_REJECT=1        : 줄 탐색이 10 번 모두 Armijo 를 못 맞추면 그 갱신을 버리고 Newton 을 멈춤 (원 코드는 alpha=1/512 로 갱신).
    #
    # 여러 번 돌려도 안전하다.  python exe/baseline_mpm.py patch-noisefloor --ipg /workspace/i-physgaussian
    p = os.path.join(a.ipg, "implicit_mpm_solver.py")
    s = open(p).read()
    # (3) AF_FD_EPS=e : 전방차분 섭동의 최대 성분 (변위 단위, 원 코드 1e-4 고정). 원 값은 dt 와 무관해 N100 에서 속도 섭동 0.5 m/s, 변형 1e-2
    #     -> 모래 인장 절단 꺾임 (F=I) 을 가로지르는 할선이 되어 GMRES 방향이 하강 방향이 아님. 기본 = 1e-4 (원 동작).
    old3 = """                eps_fd = 1e-4 / norm_inf"""
    new3 = """                eps_fd = float(_os.environ.get("AF_FD_EPS", "1e-4")) / norm_inf   # [fusedgs noisefloor]"""
    if old3 in s:
        s = s.replace(old3, new3); open(p, "w").write(s); print("[적용] AF_FD_EPS")
    if "AF_NEWTON_ATOL_FEXT" in s:
        if "[LS 실패]" not in s:
            o4 = """            else:                                                # [fusedgs noisefloor] 줄 탐색 실패 -> 갱신 버림
                if _os.environ.get("AF_LS_REJECT") == "1":"""
            n4 = """            else:                                                # [fusedgs noisefloor] 줄 탐색 실패 -> 갱신 버림
                print(f"  [LS 실패] step={step} newton={newton_it} reject={_os.environ.get('AF_LS_REJECT') == '1'}", flush=True)
                if _os.environ.get("AF_LS_REJECT") == "1":"""
            assert s.count(o4) == 1
            s = s.replace(o4, n4); open(p, "w").write(s); print("[적용] LS 실패 출력")
        print("[이미 적용] patch_ipg_noisefloor")
        return

    old1 = """            if rel_res < _newton_tol or res_norm < 1e-10:
                converged = True"""
    new1 = """            if newton_it == 0:                                   # [fusedgs noisefloor] 절대 기준 (중력 하중 대비)
                _af_atol_c = float(_os.environ.get("AF_NEWTON_ATOL_FEXT", "0") or 0)
                _af_atol = (_af_atol_c * 9.8 * float(cp.sqrt(cp.sum((mass_gpu * free_mask_gpu) ** 2)))) if _af_atol_c > 0 else 0.0
            if rel_res < _newton_tol or res_norm < 1e-10 or res_norm < _af_atol:
                converged = True"""
    assert s.count(old1) == 1, "old1"
    s = s.replace(old1, new1)

    old2 = """                if phi_trial <= phi_0 + c1 * alpha * dphi_0:
                    break
                alpha *= 0.5

            du_k = du_k + alpha * delta_du
"""
    new2 = """                if phi_trial <= phi_0 + c1 * alpha * dphi_0:
                    break
                alpha *= 0.5
            else:                                                # [fusedgs noisefloor] 줄 탐색 실패 -> 갱신 버림
                print(f"  [LS 실패] step={step} newton={newton_it} reject={_os.environ.get('AF_LS_REJECT') == '1'}", flush=True)
                if _os.environ.get("AF_LS_REJECT") == "1":
                    final_residual = res_norm
                    break

            du_k = du_k + alpha * delta_du
"""
    assert s.count(old2) == 1, "old2"
    s = s.replace(old2, new2)
    open(p, "w").write(s)
    print("[적용] patch_ipg_noisefloor")


def patch_particles(a):
    # PG/i-PG 가 **주어진 입자 집합**을 그대로 시뮬레이션하게 만든다. 멱등이다.
    #
    # gs_simulation 은 입자를 3DGS 모델의 가우시안 중심에서 가져온다 (채우기를 끄면
    # 가우시안 중심 그 자체다). 합성한 장면 -- 예를 들어 공중의 공 -- 을 MPM 으로
    # 돌려 보려면 그 자리에 우리 입자를 넣어야 한다.
    #
    # AF_PARTICLES_NPY 가 있으면 좌표 변환(transform2origin / shift2center111) **뒤에**
    # 입자를 갈아끼운다. 즉 npy 는 시뮬레이션 좌표계(도메인 [0, grid_lim]) 그대로다.
    # 가우시안 속성(shs/opacity/cov)은 렌더용이라 첫 행을 복제해 길이만 맞춘다 --
    # 우리가 쓰는 것은 --output_h5 로 나오는 입자 궤적뿐이다.
    #
    #   python exe/baseline_mpm.py patch-particles --ipg /workspace/i-physgaussian
    p = os.path.join(a.ipg, "gs_simulation.py")
    s = open(p).read()
    if "AF_PARTICLES_NPY" in s:
        print("[패치] 이미 들어 있다")
        return

    old = """    # fill particles if needed
    gs_num = transformed_pos.shape[0]
"""
    new = """    # --- AF: 입자 집합을 통째로 갈아끼운다 (합성 장면용) ----------------
    _afp = os.environ.get("AF_PARTICLES_NPY")
    if _afp:
        import numpy as _afnp
        _P = _afnp.load(_afp)
        _P = torch.tensor(_P, dtype=transformed_pos.dtype,
                          device=transformed_pos.device)
        print(f"[AF입자] {_afp} -> {_P.shape[0]} 개로 갈아끼운다 "
              f"(가우시안 {transformed_pos.shape[0]} 개 대신) AF_PARTICLES_NPY",
              flush=True)
        _n = _P.shape[0]
        transformed_pos = _P
        init_cov = init_cov[:1].repeat(_n, 1)
        init_opacity = init_opacity[:1].repeat(_n, 1)
        init_shs = init_shs[:1].repeat(
            *([_n] + [1] * (init_shs.dim() - 1)))
    # fill particles if needed
    gs_num = transformed_pos.shape[0]
"""
    assert s.count(old) == 1, "채우기 앞 지점을 못 찾았다"
    if "\nimport os" not in s and "\nimport os\n" not in s:
        s = "import os\n" + s
    open(p, "w").write(s.replace(old, new, 1))
    print(f"[패치] 입자 갈아끼우기 추가 -> {p}")


def patch_scen(a):
    # i-PhysGaussian 클론에 **같은 채우기**와 **시나리오 구동 손잡이**를 넣는다.
    #
    # 벤치가 성립하려면 입자 집합이 PG 와 글자 그대로 같아야 한다 (시나리오의 제어
    # 입자 색인이 그 집합을 가리킨다). 손잡이는 PG 와 같은 규약으로 적용한다:
    #
    #     v_p <- (1 - w_p) v_p + w_p v_cmd,   w = (1 - q^2)^2,  q = |x_p - c| / R
    #
    # 여러 번 돌려도 안전하다.
    #
    #   python exe/baseline_mpm.py patch-scen --ipg /root/work/i-physgaussian
    p = os.path.join(a.ipg, "gs_simulation.py")
    s = open(p).read()
    if "AF_H_SCEN" in s:
        print("[패치] 이미 들어 있다")
        return

    # 1) 채우기 캐시 -- PG 와 같은 입자 집합을 쓴다
    old = "    from particle_filling.filling import *\n"
    new = ("    from particle_filling.filling import *\n"
           "    import os as _af_os, numpy as _af_np, torch as _af_t\n"
           "    _af_orig_fill = fill_particles\n"
           "\n"
           "    def fill_particles(*args, **kwargs):\n"
           "        _ck = _af_os.environ.get('AF_PGFILL_NPY')\n"
           "        if _ck and _af_os.path.exists(_ck):\n"
           "            _L = _af_np.ascontiguousarray(_af_np.load(_ck))\n"
           "            print(f'[PG채움] 캐시에서 {_L.shape[0]} 개', flush=True)\n"
           "            return _af_t.from_numpy(_L).float().contiguous()\n"
           "        return _af_orig_fill(*args, **kwargs)\n")
    assert old in s, "채우기 임포트를 못 찾았다"
    s = s.replace(old, new, 1)

    # 2) 시나리오 적재 + 손잡이 소속 계산 (정지 자세 기준, PG 와 같은 규약)
    old = "    for frame in tqdm(range(frame_num)):\n"
    new = ("    _af_scen = None\n"
           "    if _af_os.environ.get('AF_H_SCEN'):\n"
           "        _af_scen = _af_np.load(_af_os.environ['AF_H_SCEN'])\n"
           "        _af_R = float(_af_os.environ.get('AF_H_R', 0.15))\n"
           "        _x0 = mpm_solver.export_particle_x_to_torch()\n"
           "        _af_hid = _af_t.as_tensor(_af_scen['hid'], dtype=_af_t.long,\n"
           "                                  device=_x0.device)\n"
           "        _af_mem, _af_w = [], []\n"
           "        for _j in range(_af_hid.numel()):\n"
           "            _dd = (_x0 - _x0[_af_hid[_j]]).norm(dim=1)\n"
           "            _m = _af_t.nonzero(_dd < _af_R, as_tuple=False).flatten()\n"
           "            _q = (_dd[_m] / _af_R).clamp(0, 1)\n"
           "            _af_mem.append(_m)\n"
           "            _af_w.append(((1.0 - _q * _q) ** 2).unsqueeze(-1))\n"
           "            print(f'[손잡이-{_j}] 입자 {int(_af_hid[_j])} 소속 "
           "{_m.numel()} 개', flush=True)\n"
           "        _af_vel = _af_scen['vel']\n"
           "        print(f'[손잡이] 시나리오 {_af_vel.shape[0]} 프레임', flush=True)\n"
           "    for frame in tqdm(range(frame_num)):\n"
           "        if _af_scen is not None:\n"
           "            _vr = mpm_solver.export_particle_v_to_torch()\n"
           "            _vc = _af_t.as_tensor(\n"
           "                _af_vel[min(frame, _af_vel.shape[0] - 1)],\n"
           "                dtype=_af_t.float32, device=_vr.device)\n"
           "            for _j in range(len(_af_mem)):\n"
           "                _mm, _ww = _af_mem[_j], _af_w[_j]\n"
           "                _vr[_mm] = (1.0 - _ww) * _vr[_mm] + _ww * _vc[_j]\n")
    assert old in s, "프레임 루프를 못 찾았다"
    s = s.replace(old, new, 1)
    open(p, "w").write(s)
    print("[패치] i-PG 에 채우기 캐시 + AF_H_SCEN 손잡이 추가 완료")


PATCHES = {"bodyacc": patch_bodyacc, "fillcache": patch_fillcache, "gravity": patch_gravity, "handle": patch_handle, "initv": patch_initv, "noisefloor": patch_noisefloor, "particles": patch_particles, "scen": patch_scen}

COMMANDS = ("run", "omni", "pareto", "gt-timing") + tuple("patch-" + k for k in PATCHES)
if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        sys.exit("usage: python exe/baseline_mpm.py {" + "|".join(COMMANDS) + "} [options]")
    CMD = sys.argv.pop(1)
else:
    CMD = None

if CMD is not None and CMD.startswith("patch-"):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ipg", required=True)
    if CMD == "patch-gravity":
        ap.add_argument("--rel", type=float, default=1e-8,
                        help="최대 질량 대비 절대 문턱 비율 (기본 1e-8). 1e-2 가 기본값"
                             "이었고 그것이 이탈 입자를 잘라냈다")
    PATCHES[CMD[len("patch-"):]](ap.parse_args())

if CMD == "run":
    # 서브스텝 안정성 실험: PG(공식 명시 MPM) / i-PG(암시 MPM, newton_gmres) 를 프레임당 N 서브스텝으로.
    #
    # 입력: 초기 상태 h5 (x, v) + config (탄성 jelly). 출력 npz: 프레임별 측정(stabstat) + 고정 부분표본 위치 + 시간.
    #
    #   cd i-physgaussian && python <root>/exe/baseline_mpm.py run --method pg --h5 init.h5 --config cfg.json --substeps 64 --out r.npz
    import argparse
    import json
    import os
    import sys
    import time

    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["pg", "ipg"], required=True)
    ap.add_argument("--h5", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--substeps", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pg", default=os.environ.get('FUSEDGS_WORK', '/workspace') + '/i-physgaussian')
    ap.add_argument("--ckpt_every", type=int, default=5, help="이어 돌리기 체크포인트 간격 (프레임). <out>.ckpt.pt, 0 이면 끔")
    ap.add_argument("--safe", type=int, default=0,
                    help="1 이면 발산 방지 보정판 (evaluation.py safe_*): 발산 조짐에서만 걸리고 정상 실행엔 영향 없음")
    ap.add_argument("--frames", type=int, default=0, help="0 이면 설정의 frame_num (검사용으로 짧게)")
    ap.add_argument("--dump", default="", help="영상용: 프레임마다 전체 입자 h5 (x, F) 를 이 폴더에")
    ap.add_argument("--dump_meta", default="",
                    help="lb_gen 궤적 meta.json: 덤프 F 를 F_sim·R 로 (lb_gen 과 같은 규약, 0 프레임 F=R), meta 를 덤프 폴더에 복사 (lb_render 용)")
    a = ap.parse_args()
    import evaluation as ss                                          # noqa: E402
    sys.path.insert(0, a.pg); os.chdir(a.pg)
    import h5py                                                      # noqa: E402
    import numpy as np                                               # noqa: E402
    import torch                                                     # noqa: E402
    import warp as wp                                                # noqa: E402
    from tqdm import tqdm                                            # noqa: E402
    from mpm_solver_warp.mpm_solver_warp import MPM_Simulator_WARP   # noqa: E402

    cfg = json.load(open(a.config))
    n_grid, GL = int(cfg["n_grid"]), float(cfg["grid_lim"]); dx = GL / n_grid
    frame_dt, NF = float(cfg["frame_dt"]), int(cfg["frame_num"])
    if a.frames:
        NF = int(a.frames)
    dt = frame_dt / a.substeps
    G = [float(q) for q in cfg["g"]]
    floor = ([b for b in cfg["boundary_conditions"] if b["type"] == "surface_collider"] or [None])[0]
    FZ = float(floor["point"][2]) if floor is not None else 0.0      # 바닥이 없는 씬(plane·pillow2sofa)은 관통 지표를 안 쓴다
    OFFICIAL_BC = any(b["type"] not in ("bounding_box", "surface_collider") for b in cfg["boundary_conditions"])
    # 새 형상 장면 (레고 L000 세팅 + PG plasticine/metal, config 에 stab_official) 은 경계조건이 상자·바닥뿐이어도 재질·감쇠·경계조건을 공식 함수로 (jelly 고정 판을 쓰지 않음)
    OFFICIAL_BC = OFFICIAL_BC or bool(cfg.get("stab_official", False))
    with h5py.File(a.h5) as h:
        x = np.array(h["x"]); x = np.ascontiguousarray((x.T if x.shape[0] == 3 else x).astype(np.float32))
        v = np.array(h["v"]) if "v" in h else np.zeros_like(x.T)
        v = np.ascontiguousarray((v.T if v.shape[0] == 3 else v).astype(np.float32))
    N = x.shape[0]
    X = torch.as_tensor(x).cuda()
    cell = (X / dx).floor().long(); key = (cell[:, 0] * n_grid + cell[:, 1]) * n_grid + cell[:, 2]
    _, inv, cnt = torch.unique(key, return_inverse=True, return_counts=True)
    VOL = (dx ** 3 / cnt[inv].float()).contiguous()
    MASS = VOL.double().cpu().numpy() * float(cfg["density"])
    SUB = ss.subset(N)
    wp.init()
    if a.method == "ipg":
        from implicit_mpm_solver import ImplicitMPMSolver
        if os.environ.get("AF_IPG_FAST") == "1":                   # 같은 Newton-GMRES 를 자유 노드만 압축 (newton_gmres_fast)
            ImplicitMPMSolver.p2g2p_newton_gmres = newton_gmres_fast
            print("[AF_IPG_FAST] newton_gmres_fast", flush=True)
        sol = ImplicitMPMSolver(n_particles=10, n_grid=n_grid, grid_lim=GL)
    else:
        sol = MPM_Simulator_WARP(10)
    sol.load_initial_data_from_torch(X, VOL, None, n_grid=n_grid, grid_lim=GL)
    if OFFICIAL_BC:
        # 공식 씬 (plane·pillow2sofa 등): 재질·감쇠·경계조건을 **공식 gs_simulation 과 같은 함수**로 넣는다
        from utils.decode_param import set_boundary_conditions
        _mp = {k: cfg[k] for k in ("material", "E", "nu", "density", "yield_stress", "hardening", "xi", "friction_angle",
                                    "grid_v_damping_scale", "softening", "beta", "mass") if k in cfg}
        _mp.update({"g": G, "n_grid": n_grid, "grid_lim": GL})
        sol.set_parameters_dict(_mp)
        set_boundary_conditions(sol, cfg["boundary_conditions"], {"substep_dt": dt, "frame_dt": frame_dt, "frame_num": NF})
        print(f"[공식 설정] 재질 {_mp.get('material')}, 경계조건 {[b['type'] for b in cfg['boundary_conditions']]}", flush=True)
    else:
        sol.set_parameters_dict({"material": "jelly", "E": float(cfg["E"]), "nu": float(cfg["nu"]), "density": float(cfg["density"]),
                                 "g": G, "n_grid": n_grid, "grid_lim": GL, "grid_v_damping_scale": 1.1})
        sol.add_bounding_box()
        sol.add_surface_collider(tuple(floor["point"]), tuple(floor["normal"]), floor["surface"], float(floor.get("friction", 0.0)))
    sol.finalize_mu_lam()
    sol.import_particle_v_from_torch(torch.as_tensor(v).cuda())
    print(f"[{a.method}] 입자 {N}, dx {dx:.4f}, N {a.substeps} (dt {dt:.3e}), {NF} 프레임", flush=True)
    RD = np.eye(3, dtype=np.float32)                               # 덤프 F 에 오른쪽으로 곱할 초기 회전 (lb_gen: F_sim·R)
    if a.dump:
        os.makedirs(a.dump, exist_ok=True)
        if a.dump_meta:
            import shutil
            RD = np.array(json.load(open(a.dump_meta))["R"], dtype=np.float32)
            shutil.copyfile(a.dump_meta, f"{a.dump}/meta.json")
            print(f"[덤프] F = F_sim·R (meta {a.dump_meta})", flush=True)
        with h5py.File(f"{a.dump}/sim_{0:010d}.h5", "w") as h:
            h.create_dataset("x", data=x); h.create_dataset("F", data=np.tile(RD.reshape(1, 9), (N, 1)))
    _VOLN = VOL.double().cpu().numpy(); _EK = dict(vol=_VOLN, E=float(cfg["E"]), nu=float(cfg["nu"]))
    rows = [ss.frame_stats(x, v, np.tile(np.eye(3), (N, 1, 1)), MASS, G, FZ, **_EK)]
    XS = [x[SUB]]; T = []
    step = 0
    # ---- 이어 돌리기: 입자 상태 전부 + i-PG 의 직전 속도(_prev_particle_v) + 누적 기록
    CKPT = a.out + ".ckpt.pt"
    _PF = [k for k in ("particle_x", "particle_v", "particle_F", "particle_F_trial", "particle_C", "particle_R", "particle_cov",
                       "particle_stress", "particle_Jp", "particle_selection") if getattr(sol.mpm_state, k, None) is not None]


    def _save_ckpt(f):
        st = {k: wp.to_torch(getattr(sol.mpm_state, k)).clone().cpu() for k in _PF}
        pv = getattr(sol, "_prev_particle_v", None)
        st["_prev_particle_v"] = (wp.to_torch(pv).clone().cpu() if isinstance(pv, wp.array) else
                                  (pv.clone().cpu() if torch.is_tensor(pv) else None))
        _pa = getattr(sol, "_particle_a", None)                   # i-PG Newmark 입자 가속도 (다음 스텝 a^n)
        st["_particle_a"] = wp.to_torch(_pa).clone().cpu() if isinstance(_pa, wp.array) else None
        torch.save(dict(f=f, step=step, state=st, rows=rows, XS=XS, T=T, safe_n=list(SAFE_N)), CKPT + ".tmp")
        os.replace(CKPT + ".tmp", CKPT)


    # ---- 보정판: 정상 실행에서는 걸리지 않는 조건에서만 개입한다 -------------------------------------------
    SAFE_N = [0, 0, 0]                                           # [클램프·되돌림 입자 수, 반으로 나눠 다시 푼 횟수, 관성 대체 횟수]
    if a.safe:
        _SP = ss.safe_params(x, FZ, G, float(cfg["grid_lim"]), int(cfg["n_grid"]))
        print(f"[보정판] 상자 [{_SP['lo']:.4f}, {_SP['hi']:.4f}], 속력 상한 {_SP['vcap']:.2f}", flush=True)


    def _snap():
        st = {k: wp.to_torch(getattr(sol.mpm_state, k)).clone() for k in _PF}
        pv = getattr(sol, "_prev_particle_v", None)
        st["_pv"] = wp.to_torch(pv).clone() if isinstance(pv, wp.array) else (pv.clone() if torch.is_tensor(pv) else None)
        pa = getattr(sol, "_particle_a", None)
        st["_pa"] = wp.to_torch(pa).clone() if isinstance(pa, wp.array) else None
        return st


    def _restore(st):
        for k in _PF:
            wp.to_torch(getattr(sol.mpm_state, k)).copy_(st[k])
        pv = getattr(sol, "_prev_particle_v", None)
        if st["_pv"] is not None:
            if isinstance(pv, wp.array):
                wp.to_torch(pv).copy_(st["_pv"])
            else:
                sol._prev_particle_v = st["_pv"].clone()
        if st["_pa"] is not None and isinstance(getattr(sol, "_particle_a", None), wp.array):
            wp.to_torch(sol._particle_a).copy_(st["_pa"])


    def _pre_clamp(h):
        """서브스텝 전: 이번 스텝에 영역 상자 밖으로 나갈 입자의 속도만 자른다 (정상 실행은 상자 안이라 0 건)."""
        X_ = wp.to_torch(sol.mpm_state.particle_x); V_ = wp.to_torch(sol.mpm_state.particle_v)
        nx = X_ + h * V_
        out = ((nx < _SP["lo"]) | (nx > _SP["hi"])).any(1) | ~torch.isfinite(nx).all(1)
        k = int(out.sum())
        if k:
            tgt = nx[out].clamp(_SP["lo"], _SP["hi"])
            tgt = torch.where(torch.isfinite(tgt), tgt, X_[out])
            V_[out] = (tgt - X_[out]) / h
        return k


    def _one(h, depth=0):
        """한 서브스텝 (보정판). 예외·비유한이면 되돌려 반으로 두 번, 깊이 6 (1/64) 까지. 끝내 실패하면 관성만으로 전진."""
        if a.safe:
            wp.synchronize(); torch.cuda.synchronize()            # warp·torch 스트림이 달라도 순서를 보장
        pre = _snap() if a.safe else None
        if a.safe:
            SAFE_N[0] += _pre_clamp(h)
            torch.cuda.synchronize()
        try:
            if a.method == "ipg":
                sol.p2g2p_newton_gmres(step, h)
            else:
                sol.p2g2p(step, h)
            ok = True
        except Exception as e:                                   # i-PG: GMRES 안 SVD 미수렴 등
            if not a.safe:
                raise
            ok = False
        if a.safe and ok:
            wp.synchronize()
            # 비유한이 생긴 입자는 입자 상태 **전부**(F_trial·C 등 포함)를 직전 값으로, 속도 0. 하나라도 남기면 다음 스텝에 다시 퍼진다
            bad = None
            for k in ("particle_x", "particle_v", "particle_F", "particle_F_trial", "particle_C"):
                if k in _PF:
                    t_ = wp.to_torch(getattr(sol.mpm_state, k)).reshape(N, -1)
                    b_ = ~torch.isfinite(t_).all(1)
                    bad = b_ if bad is None else (bad | b_)
            nb = int(bad.sum())
            if nb:
                for k in _PF:
                    t_ = wp.to_torch(getattr(sol.mpm_state, k))
                    if t_.dtype.is_floating_point:                   # 공분산처럼 입자당 여러 값이 1 차원으로 펼쳐진 필드도 있다
                        t_.view(N, -1)[bad] = pre[k].view(N, -1)[bad]
                wp.to_torch(sol.mpm_state.particle_v)[bad] = 0.0
                for nm, key in (("_prev_particle_v", "_pv"), ("_particle_a", "_pa")):
                    q = getattr(sol, nm, None)
                    if pre[key] is not None and q is not None:
                        (wp.to_torch(q) if isinstance(q, wp.array) else q).view(N, -1)[bad] = pre[key].view(N, -1)[bad]
            X_ = wp.to_torch(sol.mpm_state.particle_x); V_ = wp.to_torch(sol.mpm_state.particle_v)
            SAFE_N[0] += nb + ss.safe_fix(X_, V_, None, pre["particle_x"], _SP)
            for _fk in ("particle_F", "particle_F_trial"):          # ④ F 특이값 상한 100
                if _fk in _PF:
                    SAFE_N[0] += ss.clamp_F(wp.to_torch(getattr(sol.mpm_state, _fk)))
            torch.cuda.synchronize()
        if ok or not a.safe:
            return
        _restore(pre)
        if depth < 6:
            SAFE_N[1] += 1
            _one(h / 2, depth + 1); _one(h / 2, depth + 1)
            return
        SAFE_N[2] += 1                                           # 관성 대체: x += h v, v += h g (영역 안으로)
        X_ = wp.to_torch(sol.mpm_state.particle_x); V_ = wp.to_torch(sol.mpm_state.particle_v)
        V_ += h * torch.as_tensor(G, device=V_.device, dtype=V_.dtype)
        X_.add_(h * V_).clamp_(_SP["lo"], _SP["hi"])


    F_START = 1
    if a.ckpt_every > 0 and os.path.exists(CKPT):
        ck = torch.load(CKPT, weights_only=False)
        for k in _PF:
            wp.to_torch(getattr(sol.mpm_state, k)).copy_(ck["state"][k].cuda())
        if ck["state"]["_prev_particle_v"] is not None:
            pv = getattr(sol, "_prev_particle_v", None)
            if isinstance(pv, wp.array):
                wp.to_torch(pv).copy_(ck["state"]["_prev_particle_v"].cuda())
            else:
                sol._prev_particle_v = (wp.from_torch(ck["state"]["_prev_particle_v"].cuda().contiguous(), dtype=wp.vec3)
                                        if a.method == "ipg" and ck["state"]["_prev_particle_v"].dim() == 2 else ck["state"]["_prev_particle_v"].cuda())
        if ck["state"].get("_particle_a") is not None:
            sol._particle_a = wp.from_torch(ck["state"]["_particle_a"].cuda().contiguous(), dtype=wp.vec3)
        elif a.method == "ipg" and ck["f"] > 0:
            print("[이어 돌리기] 경고: 체크포인트에 i-PG 입자 가속도가 없다 (a^n 패치 전 체크포인트)", flush=True)
        rows, XS, T, step, F_START = ck["rows"], ck["XS"], ck["T"], ck["step"], ck["f"] + 1
        SAFE_N[:] = list(ck.get("safe_n") or [0, 0, 0])
        print(f"[이어 돌리기] {CKPT} 프레임 {ck['f']} 부터 (step {step})", flush=True)
    # ---- 발산 감시 (보정판 아닐 때): 서브스텝의 P2G 앞에서 입자가 유한하고 격자 배열 안 (P2G 기저 노드 0 .. n_grid-3) 인지 본다.
    #      벗어난 채 P2G 를 돌면 warp 커널이 배열 밖을 써서 CUDA 700 으로 죽는다 -> 그 전에 멈추고 지금까지의 프레임만 저장
    #      (프레임 수가 모자란 npz = evaluation.py load 가 발산으로 읽음). 읽기만 하므로 안정한 실행의 결과는 그대로.
    _XLO, _XHI = (-0.5 + 1e-3) * dx, (n_grid - 1.5 - 1e-3) * dx
    _ENV_ERR = ("out of memory", "ModuleNotFoundError", "ImportError", "Failed to find forward kernel", "Failed to load CUDA module", "통합 커널은 탄성")
    DIV, DEAD = "", False

    def _left_grid():
        torch.cuda.synchronize()
        Xv = wp.to_torch(sol.mpm_state.particle_x)
        mn, mx = torch.aminmax(Xv)                               # NaN 은 그대로 전파 -> 아래 비교가 거짓
        if bool((mn > _XLO) & (mx < _XHI)):
            return 0
        return int((~((Xv > _XLO) & (Xv < _XHI)).all(1)).sum())

    for f in tqdm(range(F_START, NF + 1), desc=f"{a.method} N{a.substeps}"):
        t0 = time.time()
        try:
            for s in range(a.substeps):
                if not a.safe:
                    _nb = _left_grid()
                    if _nb:
                        DIV = f"[diverged] frame {f} substep {s}: {_nb} particles left the grid"
                        break
                _one(dt)
                step += 1
            if not DIV:
                torch.cuda.synchronize()
        except Exception as e:                                   # i-PG: GMRES 안 SVD 미수렴 등 풀이 실패 = 발산 (실행 환경 오류·메모리 부족은 그대로 올림)
            _msg = f"{type(e).__name__}: {e}"
            if a.safe or any(k in _msg for k in _ENV_ERR):
                raise
            DIV = f"[diverged] frame {f}: solver failed ({' '.join(_msg.split())[:160]})"
            DEAD = "CUDA" in _msg
        if DIV:
            print("\n" + DIV, flush=True); break
        T.append(time.time() - t0)
        xt = sol.export_particle_x_to_torch().cpu().numpy()
        vt = wp.to_torch(sol.mpm_state.particle_v).cpu().numpy()
        Ft = sol.export_particle_F_to_torch().cpu().numpy()
        rows.append(ss.frame_stats(xt, vt, Ft, MASS, G, FZ, **_EK))
        rows[-1]["safe_n"], rows[-1]["safe_split"], rows[-1]["safe_inert"] = (float(q) for q in SAFE_N)
        if a.dump:
            with h5py.File(f"{a.dump}/sim_{f:010d}.h5", "w") as h:
                h.create_dataset("x", data=xt.astype(np.float32))
                h.create_dataset("F", data=(Ft.reshape(N, 3, 3).astype(np.float32) @ RD).reshape(N, 9))
        XS.append(xt[SUB])
        if rows[-1]["nan"] > 0.5 and not a.safe:                     # 대부분 발산했으면 더 돌 이유가 없다
            print(f"  발산 프레임 {f}", flush=True); break
        if a.ckpt_every > 0 and f % a.ckpt_every == 0 and f < NF:
            _save_ckpt(f)
    np.savez_compressed(a.out, sub=SUB, x=np.stack(XS).astype(np.float32), t=np.array(T),
                        stats=np.array([[r.get(k, np.nan) for k in ss.KEYS + ss.SAFE_KEYS] for r in rows]),
                        keys=np.array(ss.KEYS + ss.SAFE_KEYS), substeps=a.substeps, dt=dt, safe=a.safe)
    print(f"[저장] {a.out}  프레임당 {(f'{np.mean(T):.3f}s' if T else '-')}" + (f"  ({len(XS) - 1}/{NF} frames, diverged)" if DIV else ""), flush=True)
    if DEAD:                                                     # CUDA 문맥이 깨진 뒤: 정리 단계에서 다시 죽지 않게 바로 끝냄
        sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if CMD == "omni":
    # OmniPhysGS (공식 저장소 wgsxm/OmniPhysGS 의 MPMModel p2g2p, 수정 없음) 의 서브스텝 안정성 실행기. stab_mpm.py 와 같은 입출력 npz.
    # 공식 코드는 영역 [0,1]^3 (num_grids 격자, 입자 부피 = prod(size)/N 균일) 이라 우리 장면 (영역 [0,2], 격자 100) 을 길이 s=0.5 배로 줄여 넣는다:
    #  x'=s x, v'=s v, g'=s g, E'=s^2 E (같은 밀도·시간) -> 궤적이 정확히 s 배 (num_grids=100 = 우리 격자 dx 0.02 의 s 배), 입자 부피 = size 의 곱 / N 을 우리 평균 입자 부피 x s^3 로.
    # 재질: 학습 단계 (신경 구성모델 선택) 는 쓰지 않고 정답과 같은 고정 공회전 탄성 (공식 CorotatedElasticity, E 1e6 nu 0.3) + 항등 소성으로 고정, MPM 솔버 부분만 쓴다.
    # config stab_official 장면은 PG 재질을 공식 물리 구성모델 목록으로 (metal = StVK + VonMises, plasticine = Corotated + VonMises).
    # 바닥 = 공식 surface_collider (sticky, 격자 노드), 상자 = 공식 clip_bound.
    #   python exe/baseline_mpm.py omni --repo <OmniPhysGS> --h5 init.h5 --config config.json --substeps N --out r.npz [--dump DIR] [--frames F]
    import argparse, json, os, sys, time
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=os.environ.get('FUSEDGS_WORK', '/workspace') + '/OmniPhysGS'); ap.add_argument("--h5", required=True); ap.add_argument("--config", required=True)
    ap.add_argument("--substeps", type=int, required=True); ap.add_argument("--out", required=True); ap.add_argument("--frames", type=int, default=0)
    ap.add_argument("--dump", default=""); ap.add_argument("--ckpt_every", type=int, default=0)
    a = ap.parse_args()
    import evaluation as ss
    sys.path.insert(0, a.repo); os.chdir(a.repo)
    import h5py, numpy as np, torch
    import warp as wp
    wp.init()
    from tqdm import tqdm
    from src.mpm_core import MPMModel, set_boundary_conditions
    from src.constitutive_models.physical_constitutive_models import CorotatedElasticity, StVKElasticity, VonMisesPlasticity
    cfg = json.load(open(a.config)); S = 0.5
    GL = float(cfg["grid_lim"]); assert abs(GL * S - 1.0) < 1e-9
    n_grid = int(cfg["n_grid"]); dx = GL / n_grid
    NF = a.frames or int(cfg["frame_num"]); frame_dt = float(cfg["frame_dt"]); dt = frame_dt / a.substeps
    G = [float(q) for q in cfg["g"]]
    fl = [b for b in cfg["boundary_conditions"] if b["type"] == "surface_collider"][0]; FZ = float(fl["point"][2])
    with h5py.File(a.h5) as h:
        x = np.array(h["x"]); x = np.ascontiguousarray((x.T if x.shape[0] == 3 else x).astype(np.float32))
    N = x.shape[0]
    X = torch.as_tensor(x).cuda()
    cell = (X / dx).floor().long(); key = (cell[:, 0] * n_grid + cell[:, 1]) * n_grid + cell[:, 2]
    _, inv, cnt = torch.unique(key, return_inverse=True, return_counts=True)
    VOL = (dx ** 3 / cnt[inv].float()).contiguous(); MASS = VOL.double().cpu().numpy() * float(cfg["density"])
    SUB = ss.subset(N)
    E, nu = float(cfg["E"]), float(cfg["nu"])
    vol_p = float(VOL.mean()) * S ** 3
    # 새 형상 장면 (config stab_official): PG 재질을 OmniPhysGS 공식 물리 구성모델 목록으로 -- metal = StVK + VonMises, plasticine = Corotated + VonMises
    # (OmniPhysGS 목록에 PG plasticine 의 연화 softening 이 없어 뺌), 항복 응력 σy' = s^2 σy (E 와 같은 단위 변환); 격자 감쇠 = config grid_v_damping_scale (PG 와 같이 걸음마다 격자 운동량에 곱함)
    MAT = cfg.get("material", "jelly") if cfg.get("stab_official") else "jelly"
    DAMP = float(cfg.get("grid_v_damping_scale", 1.0)) if cfg.get("stab_official") else 1.0
    DAMP = DAMP if DAMP < 1.0 else 1.0
    sim = {"num_grids": n_grid, "dt": dt, "gravity": [g * S for g in G], "clip_bound": 0.5, "damping": DAMP}
    mat = {"center": [0.5] * 3, "size": [vol_p * N, 1.0, 1.0], "rho": float(cfg["density"])}
    model = MPMModel(sim, mat, init_pos=X * S, enable_train=False, device=torch.device("cuda"))
    set_boundary_conditions(model, [{"type": "surface_collider", "point": [p * S for p in fl["point"]], "normal": fl["normal"], "surface": fl["surface"],
                                      "friction": float(fl.get("friction", 0.0)), "start_time": 0.0, "end_time": 1000.0}])
    elas = (StVKElasticity() if MAT == "metal" else CorotatedElasticity()).cuda()
    elas.log_E = torch.tensor([E * S * S], device="cuda").log(); elas.nu = torch.tensor([nu], device="cuda")
    plas = None
    if MAT in ("metal", "plasticine"):
        plas = VonMisesPlasticity().cuda()
        plas.log_E = torch.tensor([E * S * S], device="cuda").log(); plas.nu = torch.tensor([nu], device="cuda")
        plas.sigma_y = torch.tensor([float(cfg["yield_stress"]) * S * S], device="cuda")
    elif MAT != "jelly":
        raise SystemExit(f"[omni] 재질 {MAT} 대응 없음")
    print(f"[omni] 재질 {MAT}: 탄성 {type(elas).__name__}, 소성 {type(plas).__name__ if plas is not None else '항등'}, 감쇠 {DAMP}", flush=True)
    xs = X * S; v = torch.zeros_like(xs); C = torch.zeros(N, 3, 3, device="cuda"); F = torch.eye(3, device="cuda").repeat(N, 1, 1)
    _EK = dict(vol=VOL.double().cpu().numpy(), E=E, nu=nu)
    rows = [ss.frame_stats(x, np.zeros_like(x), np.tile(np.eye(3), (N, 1, 1)), MASS, G, FZ, **_EK)]; XS = [x[SUB]]; T = []
    if a.dump:
        os.makedirs(a.dump, exist_ok=True)
        with h5py.File(f"{a.dump}/sim_{0:010d}.h5", "w") as hh:
            hh.create_dataset("x", data=x); hh.create_dataset("F", data=np.tile(np.eye(3, dtype=np.float32).reshape(1, 9), (N, 1)))
    print(f"[omni] 입자 {N}, N {a.substeps} (dt {dt:.3e}), {NF} 프레임, 입자 부피 {vol_p:.3e}", flush=True)
    for f in tqdm(range(1, NF + 1), desc=f"omni N{a.substeps}"):
        t0 = time.time(); bad = False
        with torch.no_grad():
            try:
                for s in range(a.substeps):
                    stress = elas(F)
                    xs, v, C, F = model(xs, v, C, F, stress)
                    if plas is not None:                                 # 공식 main.py 순서: 탄성 응력 -> MPM 걸음 -> 소성 되돌림
                        F = plas(F)
            except Exception as e:                                   # 공식 구성모델의 유한성 단언 -> 발산
                print("  발산 (예외)", repr(e)[:120], flush=True); bad = True
        torch.cuda.synchronize(); T.append(time.time() - t0)
        xt = (xs / S).cpu().numpy(); vt = (v / S).cpu().numpy(); Ft = F.cpu().numpy()
        if bad:
            xt = np.full_like(xt, np.nan)
        rows.append(ss.frame_stats(xt, vt, Ft, MASS, G, FZ, **_EK))
        rows[-1]["safe_n"] = rows[-1]["safe_split"] = rows[-1]["safe_inert"] = 0.0
        if a.dump and not bad:
            with h5py.File(f"{a.dump}/sim_{f:010d}.h5", "w") as hh:
                hh.create_dataset("x", data=xt.astype(np.float32)); hh.create_dataset("F", data=Ft.reshape(N, 9).astype(np.float32))
        XS.append(xt[SUB])
        if rows[-1]["nan"] > 0.5:
            print(f"  발산 프레임 {f}", flush=True); break
    np.savez_compressed(a.out, sub=SUB, x=np.stack(XS).astype(np.float32), t=np.array(T),
                        stats=np.array([[r.get(k, np.nan) for k in ss.KEYS + ss.SAFE_KEYS] for r in rows]),
                        keys=np.array(ss.KEYS + ss.SAFE_KEYS), substeps=a.substeps, dt=dt, safe=0)
    print(f"[저장] {a.out}  프레임당 {np.mean(T):.3f}s", flush=True)


if CMD == "pareto":
    # 파레토 MPM 베이스라인 실행기: i-PhysGaussian (i-PG) · OmniPhysGS 를 파레토 장면 (레고 L000·pillow2sofa·bread·wolf) 의 정답 MPM 과 같은 입력으로.
    #
    # 틀 = exe/pareto_mpm.py (정답 MPM 파레토 행 실행기) 그대로 -- 같은 초기 상태 (정답 0 프레임 h5 x·v), 같은 입자 부피 (dx0^3 / 칸 입자 수),
    # 같은 재질·경계조건 경로 (--pg_cfg: PG decode_param.set_boundary_conditions, 레고: jelly + bounding_box + 바닥 sticky), 같은 오차
    # (프레임마다 입자 평균 |x - x_gt| / EXT, EXT = 자료 .pt 0 프레임 바운딩박스 대각선), 같은 시간 (--time: 서브스텝 루프 앞뒤 동기화 벽시계, 첫 2 프레임 뺀 평균).
    # 바뀌는 것은 솔버만:
    #   ipg  = i-physgaussian ImplicitMPMSolver.p2g2p_newton_gmres (안정성 실험 stab_mpm.py --method ipg 와 같은 호출). AF_IPG_FUSED 는 환경 변수로 (탄성 장면만 1).
    #   omni = OmniPhysGS 공식 MPMModel.p2g2p (stab_omni.py 와 같은 길이 s=0.5 축소: x'=s x, v'=s v, g'=s g, E'=s^2 E, 항복 응력 s^2, 입자 부피 = 평균 x s^3),
    #          구성모델 = 공식 물리 구성모델 목록: jelly = Corotated + 항등, plasticine = Corotated + VonMises, sand = StVK + DruckerPrager (마찰각 = 정답 경로 값, 점착 0),
    #          경계조건 = 공식 set_boundary_conditions 목록 (cuboid·enforce_particle_translation·surface_collider; 좌표·속도 s 배), bounding_box = 공식 clip_bound,
    #          release_particles_sequentially (공식은 NotImplementedError) = PG 구현과 같은 50 겹 enforce_particle_translation (속도 0) 으로 펼침.
    #   pg   = 점검용 (정답 MPM 경로 그대로; --mat_extra 로 재질 키 확인)
    # 출력 DIR/err.json (+ --dump 이면 pred.pt {pred [T,N,3] 프레임 1..T, F [T,N,3,3] = F_sim}; evaluation.py phys·evaluation.py roll-to-h5 입력 형식)
    #
    #   python exe/baseline_mpm.py pareto --method ipg --pg <i-physgaussian> --gt H5DIR --config CFG --pt DATA.pt --frames 300 --nsub 10 --pg_cfg --time --out DIR [--dump]
    import argparse
    import json
    import os
    import sys
    import time

    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["ipg", "omni", "pg"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gt", default=os.environ.get('FUSEDGS_WORK', '/workspace') + '/lbdata/test_low/L000')
    ap.add_argument("--pt", default=os.environ.get('FUSEDGS_WORK', '/workspace') + '/lbpt/test_low/L000.pt')
    ap.add_argument("--config", default=os.environ.get('FUSEDGS_WORK', '/workspace') + '/lbdata/config.json')
    ap.add_argument("--pg", default=os.environ.get('FUSEDGS_WORK', '/workspace') + '/i-physgaussian')
    ap.add_argument("--omni", default=os.environ.get('FUSEDGS_WORK', '/workspace') + '/OmniPhysGS')
    ap.add_argument("--frames", type=int, default=80)
    ap.add_argument("--nsub", type=int, required=True)
    ap.add_argument("--pg_cfg", action="store_true")
    ap.add_argument("--mat_extra", default="", help="점검용: PG 재질 키 추가 (쉼표, 예 yield_stress,friction_angle) -- 정답 경로 (pareto_mpm) 는 안 넘김")
    ap.add_argument("--omni_yield", type=float, default=None, help="omni plasticine 항복 응력 (원 단위; 기본 = 정답 경로 실제 값)")
    ap.add_argument("--omni_fric", type=float, default=None, help="omni sand 마찰각 (도; 기본 = 정답 경로 실제 값)")
    ap.add_argument("--omni_elas", default="", help="omni 탄성 구성모델 강제 (corotated|stvk; 기본 = 재질 대응표: metal·sand StVK, 나머지 Corotated)")
    ap.add_argument("--time", action="store_true")
    ap.add_argument("--dump", action="store_true")
    ap.add_argument("--ckpt_every", type=int, default=0, help="이어 돌리기: K 프레임마다 입자·솔버 상태 + 지금까지 기록을 out/ckpt.pt 에 (ipg·pg 만), 0 = 끔")
    ap.add_argument("--resume", action="store_true", help="out/ckpt.pt 에서 이어 돌림")
    a = ap.parse_args()
    import h5py                                                      # noqa: E402
    import numpy as np                                               # noqa: E402
    import torch                                                     # noqa: E402

    cfg = json.load(open(a.config))
    cfg.setdefault("n_grid", 50)
    cfg.setdefault("grid_lim", 2.0)
    meta = json.load(open(os.path.join(a.gt, "meta.json"))) if os.path.exists(os.path.join(a.gt, "meta.json")) else {}
    GL = float(cfg["grid_lim"]); n_grid = int(cfg["n_grid"]); dx0 = GL / n_grid
    frame_dt = float(cfg["frame_dt"]); nsub = a.nsub; dt = frame_dt / nsub
    G = [float(q) for q in cfg["g"]]
    _fl = [b for b in cfg["boundary_conditions"] if b["type"] == "surface_collider"]
    floor = dict(_fl[0]) if _fl else None
    d = torch.load(a.pt, map_location="cpu", weights_only=False, mmap=True)
    x0d = d["x"][0].float()
    EXT = float((x0d.max(0).values - x0d.min(0).values).norm())
    NPT = int(x0d.shape[0])
    del d, x0d


    def h5x(t, key="x"):
        with h5py.File(os.path.join(a.gt, f"sim_{t:010d}.h5")) as h:
            r = np.array(h[key])
        return r.T.copy() if (r.ndim == 2 and r.shape[0] == 3 and r.shape[1] != 3) else r


    x0 = h5x(0).astype(np.float32)
    try:
        v0 = h5x(0, "v").astype(np.float32)
    except KeyError:
        v0 = np.zeros_like(x0)
    if v0.shape != x0.shape:
        v0 = v0.T.copy()
    N = x0.shape[0]
    assert N == NPT, f"정답 h5 입자 {N} != 자료 .pt 입자 {NPT}"
    L = min(a.frames, int(meta.get("frames", cfg.get("frame_num", a.frames))))
    X = torch.as_tensor(x0).cuda().contiguous()
    cell = (X / dx0).floor().long()
    key = (cell[:, 0] * n_grid + cell[:, 1]) * n_grid + cell[:, 2]
    _, inv, cnt = torch.unique(key, return_inverse=True, return_counts=True)
    VOL = (dx0 ** 3 / cnt[inv].float()).contiguous()
    MAT = cfg.get("material", "jelly") if a.pg_cfg else "jelly"
    print(f"[{a.method}] 입자 {N}, n_grid {n_grid}, 서브스텝 {dt:.4e} x {nsub}, 재질 {MAT}, 바닥 {floor}, EXT {EXT:.6f}, 프레임 {L}, "
          f"AF_IPG_FUSED={os.environ.get('AF_IPG_FUSED')}", flush=True)
    print(f"[명령] {' '.join(sys.argv)}", flush=True)
    import warp as wp                                                # noqa: E402
    wp.init()

    # ------------------------------------------------------------------------------------------------ 솔버 세우기
    if a.method in ("ipg", "pg"):
        sys.path.insert(0, a.pg); os.chdir(a.pg)
        if a.method == "ipg":
            from implicit_mpm_solver import ImplicitMPMSolver     # noqa: E402
            sol = ImplicitMPMSolver(n_particles=10, n_grid=n_grid, grid_lim=GL)
            if os.environ.get("AF_IPG_FAST") == "1":              # 같은 알고리즘, 자유 노드 압축 (newton_gmres_fast)
                ImplicitMPMSolver.p2g2p_newton_gmres = newton_gmres_fast
                print("[AF_IPG_FAST] newton_gmres_fast", flush=True)
        else:
            from mpm_solver_warp.mpm_solver_warp import MPM_Simulator_WARP   # noqa: E402
            sol = MPM_Simulator_WARP(10)
        sol.load_initial_data_from_torch(X, VOL, None, n_grid=n_grid, grid_lim=GL)
        if a.pg_cfg:                                             # pareto_mpm --pg_cfg 와 같은 순서·같은 키 (재질 -> 경계조건 -> mu·lam)
            from utils.decode_param import set_boundary_conditions   # noqa: E402
            ks = ["material", "E", "nu", "density", "g", "grid_v_damping_scale", "additional_material_params"] + [k for k in a.mat_extra.split(",") if k]
            _mp = {k: cfg[k] for k in ks if k in cfg}
            _mp.update({"n_grid": n_grid, "grid_lim": GL, "rpic_damping": 0.0})
            sol.set_parameters_dict(_mp)
            set_boundary_conditions(sol, cfg.get("boundary_conditions", []), {"substep_dt": dt, "frame_dt": frame_dt, "frame_num": L})
            if not np.any(v0) and cfg.get("init_velocity_bc"):
                set_boundary_conditions(sol, cfg["init_velocity_bc"], {"substep_dt": dt, "frame_dt": frame_dt, "frame_num": L})
            sol.finalize_mu_lam()
        else:
            sol.set_parameters_dict({"material": "jelly", "E": float(cfg["E"]), "nu": float(cfg["nu"]), "density": float(cfg["density"]),
                                     "g": G, "n_grid": n_grid, "grid_lim": GL, "grid_v_damping_scale": 1.1, "rpic_damping": 0.0})
            sol.add_bounding_box()
            sol.add_surface_collider(tuple(floor["point"]), tuple(floor["normal"]), floor["surface"], float(floor.get("friction", 0.0)))
            sol.finalize_mu_lam()
        sol.import_particle_v_from_torch(torch.as_tensor(v0).cuda().contiguous())
        _mm = sol.mpm_model
        try:
            print(f"[재질 확인] material {int(_mm.material)}, friction_angle {float(getattr(_mm, 'friction_angle', float('nan')))}, "
                  f"yield_stress {float(wp.to_torch(_mm.yield_stress)[0])}, softening {getattr(_mm, 'softening', None)}, "
                  f"damping {getattr(_mm, 'grid_v_damping_scale', None)}", flush=True)
        except Exception as _e:
            print("[재질 확인] 실패", repr(_e)[:120], flush=True)

        def substep(step):
            if a.method == "ipg":
                sol.p2g2p_newton_gmres(step, dt)
            else:
                sol.p2g2p(step, dt)

        def state_x():
            return sol.export_particle_x_to_torch()

        def state_F():
            return sol.export_particle_F_to_torch().reshape(-1, 3, 3)
    else:
        sys.path.insert(0, a.omni); os.chdir(a.omni)
        from src.mpm_core import MPMModel, set_boundary_conditions as omni_bc   # noqa: E402
        from src.constitutive_models.physical_constitutive_models import (CorotatedElasticity, StVKElasticity,   # noqa: E402
                                                                           VonMisesPlasticity, DruckerPragerPlasticity)
        S = 0.5
        assert abs(GL * S - 1.0) < 1e-9
        E, nu = float(cfg["E"]), float(cfg["nu"])
        # 정답 경로 (pareto_mpm) 의 실제 재질 값: PG 모델 기본값 (항복 응력 0, 마찰각 25) 은 키를 안 넘기면 그대로 -- 점검 판 (--method pg --mat_extra) 으로 확인한 값을 인자로
        sig_y = a.omni_yield if a.omni_yield is not None else 0.0
        fric = a.omni_fric if a.omni_fric is not None else 25.0
        DAMP = float(cfg.get("grid_v_damping_scale", 1.0)) if a.pg_cfg else 1.0
        DAMP = DAMP if DAMP < 1.0 else 1.0
        vol_p = float(VOL.mean()) * S ** 3
        sim = {"num_grids": n_grid, "dt": dt, "gravity": [g * S for g in G], "clip_bound": 0.5, "damping": DAMP}
        mat = {"center": [0.5] * 3, "size": [vol_p * N, 1.0, 1.0], "rho": float(cfg["density"])}
        model = MPMModel(sim, mat, init_pos=X * S, enable_train=False, device=torch.device("cuda"))
        bcs = []
        for b in cfg["boundary_conditions"]:
            t = b["type"]
            if t == "bounding_box":
                continue                                              # 공식 clip_bound
            if t == "surface_collider":
                bcs.append(dict(type=t, point=[p * S for p in b["point"]], normal=b["normal"], surface=b["surface"], friction=float(b.get("friction", 0.0)),
                                start_time=float(b.get("start_time", 0.0)), end_time=float(b.get("end_time", 1000.0))))
            elif t == "cuboid":
                bcs.append(dict(type=t, point=[p * S for p in b["point"]], size=[q * S for q in b["size"]], velocity=[q * S for q in b["velocity"]],
                                start_time=float(b.get("start_time", 0.0)), end_time=float(b.get("end_time", 1000.0)), reset=bool(b.get("reset", False))))
            elif t == "enforce_particle_translation":
                bcs.append(dict(type=t, point=[p * S for p in b["point"]], size=[q * S for q in b["size"]], velocity=[q * S for q in b["velocity"]],
                                start_time=float(b["start_time"]), end_time=float(b["end_time"])))
            elif t == "release_particles_sequentially":                  # PG mpm_solver_warp.release_particles_sequentially 와 같은 펼침
                nl = 50; point = [0, 0, 0]; size = [0, 0, 0]; axis = -1
                for i in range(3):
                    if b["normal"][i] == 0:
                        point[i] = 1; size[i] = 1
                    else:
                        axis = i; point[i] = b["end_position"]
                half = abs(b["start_position"] - b["end_position"]) / nl; etp = b["end_time"] / nl
                for i in range(nl):
                    sz = list(size); sz[axis] = half * (nl - i)
                    bcs.append(dict(type="enforce_particle_translation", point=[p * S for p in point], size=[q * S for q in sz], velocity=[0.0, 0.0, 0.0],
                                    start_time=float(b["start_time"]), end_time=float(etp * (i + 1))))
            else:
                raise SystemExit(f"[omni] 경계조건 {t} 대응 없음")
        if not a.pg_cfg:                                                 # 레고: 정답 경로와 같이 바닥만 (재질 jelly)
            bcs = [q for q in bcs if q["type"] == "surface_collider"]
        omni_bc(model, bcs)
        _st = (a.omni_elas == "stvk") if a.omni_elas else (MAT in ("metal", "sand"))
        elas = (StVKElasticity() if _st else CorotatedElasticity()).cuda()
        elas.log_E = torch.tensor([E * S * S], device="cuda").log(); elas.nu = torch.tensor([nu], device="cuda")
        plas = None
        if MAT in ("metal", "plasticine"):
            plas = VonMisesPlasticity().cuda()
            plas.log_E = torch.tensor([E * S * S], device="cuda").log(); plas.nu = torch.tensor([nu], device="cuda")
            plas.sigma_y = torch.tensor([sig_y * S * S], device="cuda")
        elif MAT == "sand":
            plas = DruckerPragerPlasticity().cuda()
            plas.log_E = torch.tensor([E * S * S], device="cuda").log(); plas.nu = torch.tensor([nu], device="cuda")
            plas.friction_angle = torch.tensor([fric], device="cuda"); plas.cohesion = torch.tensor([0.0], device="cuda")
        elif MAT != "jelly":
            raise SystemExit(f"[omni] 재질 {MAT} 대응 없음")
        print(f"[omni] 재질 {MAT}: 탄성 {type(elas).__name__}, 소성 {type(plas).__name__ if plas is not None else '항등'}, 감쇠 {DAMP}, "
              f"항복 {sig_y if MAT in ('metal', 'plasticine') else '-'}, 마찰각 {fric if MAT == 'sand' else '-'}, 경계조건 {[q['type'] for q in bcs]} ({len(bcs)} 개), 입자 부피 {vol_p:.3e}", flush=True)
        ST = dict(x=X * S, v=torch.as_tensor(v0).cuda() * S, C=torch.zeros(N, 3, 3, device="cuda"), F=torch.eye(3, device="cuda").repeat(N, 1, 1))

        def substep(step):
            with torch.no_grad():
                stress = elas(ST["F"])
                ST["x"], ST["v"], ST["C"], ST["F"] = model(ST["x"], ST["v"], ST["C"], ST["F"], stress)
                if plas is not None:
                    ST["F"] = plas(ST["F"])

        def state_x():
            return ST["x"] / S

        def state_F():
            return ST["F"]

    # ------------------------------------------------------------------------------------------------굴리기
    os.makedirs(a.out, exist_ok=True)
    PX = torch.empty(L, N, 3, dtype=torch.float32) if a.dump else None
    PF = torch.empty(L, N, 3, 3, dtype=torch.float32) if a.dump else None
    errs, MS, ZMIN = [], [], []
    DIV = None
    step = 0
    t0 = time.time()
    F0 = 1
    CK = os.path.join(a.out, "ckpt.pt")


    def _st_arrays():                                                    # 입자 크기 wp 배열 전부 (x·v·C·F·F_trial·응력·Jp·선택 ...)
        out = {}
        for k in dir(sol.mpm_state):
            v = getattr(sol.mpm_state, k, None)
            if isinstance(v, wp.array) and v.shape and v.shape[0] in (N, 6 * N):
                out[k] = v
        return out


    def ckpt_save(f):
        S = {k: wp.to_torch(v).detach().cpu().clone() for k, v in _st_arrays().items()}
        pa = getattr(sol, "_particle_a", None)
        torch.save(dict(frame=f, step=step, time=float(sol.time), state=S, particle_a=None if pa is None else wp.to_torch(pa).cpu().clone(),
                        errs=errs, MS=MS, ZMIN=ZMIN, PX=PX[:f].clone() if PX is not None else None, PF=PF[:f].clone() if PF is not None else None,
                        nsub=nsub, method=a.method, argv=sys.argv), CK + ".tmp")
        os.replace(CK + ".tmp", CK)
        print(f"  [체크포인트] 프레임 {f} 걸음 {step}", flush=True)


    if a.resume and a.method in ("ipg", "pg"):
        if os.path.exists(CK):
            c = torch.load(CK, map_location="cpu", weights_only=False)
            assert c["nsub"] == nsub and c["method"] == a.method, "체크포인트 설정 다름"
            arr = _st_arrays()
            for k, t in c["state"].items():
                wp.copy(arr[k], wp.from_torch(t.cuda().contiguous(), dtype=arr[k].dtype))
            if c["particle_a"] is not None:
                sol._particle_a = wp.from_torch(c["particle_a"].cuda().contiguous(), dtype=wp.vec3)
            sol.time = c["time"]; step = c["step"]; F0 = c["frame"] + 1
            errs, MS, ZMIN = list(c["errs"]), list(c["MS"]), list(c["ZMIN"])
            if PX is not None:
                PX[:c["frame"]] = c["PX"]; PF[:c["frame"]] = c["PF"]
            print(f"[이어 돌리기] 프레임 {c['frame']} 걸음 {step} 시각 {sol.time:.6f} 에서 (상태 {len(c['state'])} 개)", flush=True)
    for f in range(F0, L + 1):
        torch.cuda.synchronize(); wp.synchronize(); _tf = time.perf_counter()
        try:
            for s in range(nsub):
                substep(step)
                step += 1
        except Exception as e:                                           # i-PG GMRES 안 SVD 미수렴·OmniPhysGS 유한성 단언 등 -> 발산
            DIV = dict(frame=f, substep=s + 1, reason=f"예외 {repr(e)[:200]}")
        wp.synchronize(); torch.cuda.synchronize(); MS.append(1000.0 * (time.perf_counter() - _tf))
        if DIV is not None:
            errs.append(float("nan")); print(f"[발산] 프레임 {f}: {DIV['reason']}", flush=True); break
        xt = state_x()
        gt = torch.as_tensor(h5x(f)).cuda()
        e = float((xt - gt).norm(dim=-1).mean()) / EXT
        if not torch.isfinite(xt).all():
            e = float("nan")
        errs.append(e)
        if a.dump:
            PX[f - 1] = xt.float().cpu(); PF[f - 1] = state_F().float().reshape(N, 3, 3).cpu()
        print(f"  프레임 {f}/{L} 오차 {100 * e:.3f}% (누적 평균 {100 * np.nanmean(errs):.3f}%) {MS[-1]:.1f} ms 최저 z {float(xt[:, 2].min()):.5f} {time.time() - t0:.0f}s", flush=True)
        ZMIN.append(float(xt[:, 2].min()))
        if a.ckpt_every and a.method in ("ipg", "pg") and f % a.ckpt_every == 0 and f < L:
            ckpt_save(f)
        if floor is not None and ZMIN[-1] < 2 * dx0 and DIV is None:      # 바닥을 뚫고 영역 상자 바닥 (clip) 까지 빠짐 = 폭주 (clip 이 붙잡아 NaN 이 안 됨)
            DIV = dict(frame=f, substep=nsub, reason=f"영역 바닥까지 빠짐 (최저 z {ZMIN[-1]:.4f}, 바닥 {float(floor['point'][2])})")
            print(f"[발산] 프레임 {f}: {DIV['reason']}", flush=True); break
        if not np.isfinite(e) or e > 2.0:
            DIV = dict(frame=f, substep=nsub, reason="비유한" if not np.isfinite(e) else f"오차 {100 * e:.1f}% > 200%")
            print(f"[발산] 프레임 {f}", flush=True); break
    summ = dict(args=vars(a), method=a.method, nsub=nsub, dt=dt, n_grid=n_grid, material=MAT, EXT=EXT, n_particles=N, frames_total=L, frames_done=len(errs),
                err_mean=float(np.mean(errs)) if errs else None, err_pct=(100 * float(np.mean(errs))) if (errs and DIV is None) else None, div=DIV,
                err=errs, zmin=ZMIN, sec=time.time() - t0, ms=MS, ms_mean=(float(np.mean(MS[2:])) if len(MS) > 2 else None), ms_max=(float(np.max(MS[2:])) if len(MS) > 2 else None),
                af_ipg_fused=os.environ.get("AF_IPG_FUSED"), af_ipg_gpu=os.environ.get("AF_IPG_GPU"), af_ipg_fast=os.environ.get("AF_IPG_FAST"), gpu=torch.cuda.get_device_name(0))
    json.dump(summ, open(os.path.join(a.out, "err.json"), "w"), ensure_ascii=False, indent=1)
    if a.dump:
        nd = len(errs) if DIV is None else max(len(errs) - 1, 0)
        torch.save(dict(pred=PX[:nd].clone() if nd < L else PX, F=PF[:nd].clone() if nd < L else PF, part=nd < L, diverged=DIV,
                        method=a.method, nsub=nsub), os.path.join(a.out, "pred.pt"))
    print(f"[결과] {a.method} N {nsub}: 평균 오차 {summ['err_pct']}% ({len(errs)}/{L} 프레임), 프레임당 {summ['ms_mean']} ms, {summ['sec']:.0f}s", flush=True)


if CMD == "gt-timing":
    # 정답 MPM (PG 공식 warp 솔버, lb_gen.py 와 같은 경로) 의 민감도 -- lego L000 을 조건 하나씩 바꿔 다시 굴려 정답과의 차를 잰다.
    #
    # 정답 L000 = lb_gen.py (i-physgaussian MPM_Simulator_WARP, jelly = 고정 FCR, APIC rpic_damping 0, n_grid 100 · grid_lim 2 (dx 0.02),
    # substep_dt 2.0008e-5 (833/프레임), 바닥 surface_collider sticky (격자 노드 z < 0.48 -> 격자 속도 0), bounding_box, 감쇠 없음 (scale 1.1)).
    # 초기 상태는 정답 0 프레임 h5 (x, v) 와 같은 값, 입자 부피 = dx0³ / (dx0 = 0.02 셀의 입자 수) -- **격자를 바꿔도 질량·부피는 정답 그대로** (격자 효과만 분리).
    # 오차 = 프레임마다 입자 평균 |x - x_gt| / EXT 의 80 프레임 평균 (fem_ex_sim 과 같은 지표, EXT = L000.pt _ext).
    #
    #   cd i-physgaussian && python <root>/exe/baseline_mpm.py gt-timing --out DIR [--nsub 833] [--rpic 0|-1] [--n_grid 100] [--floor_z 0.48] [--surface sticky|slip]
    # 출력 DIR/: x.pt (float32 [80, N, 3]), err.json (프레임별 오차, 요약)
    #
    # (old exe/gt_sens.py)
    import argparse
    import json
    import os
    import sys
    import time

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--gt", default=os.environ.get('FUSEDGS_WORK', '/workspace') + '/lbdata/test_low/L000', help="정답 h5 폴더 (sim_XXXXXXXXXX.h5, meta.json)")
    ap.add_argument("--pt", default=os.environ.get('FUSEDGS_WORK', '/workspace') + '/lbpt/test_low/L000.pt', help="EXT (_ext) 를 읽을 곳")
    ap.add_argument("--config", default=os.environ.get('FUSEDGS_WORK', '/workspace') + '/lbdata/config.json')
    ap.add_argument("--pg", default=os.environ.get('FUSEDGS_WORK', '/workspace') + '/i-physgaussian')
    ap.add_argument("--frames", type=int, default=80)
    ap.add_argument("--nsub", type=int, default=0, help="프레임당 서브스텝 (0 = config 그대로 833)")
    ap.add_argument("--rpic", type=float, default=0.0, help="PG rpic_damping: 0 = APIC (정답), -1 = PIC")
    ap.add_argument("--n_grid", type=int, default=0, help="0 = config 그대로 (100)")
    ap.add_argument("--floor_z", type=float, default=None, help="바닥 평면 높이 (정답 0.48)")
    ap.add_argument("--surface", default=None, help="바닥 종류 (정답 sticky)")
    ap.add_argument("--no_dump", action="store_true")
    ap.add_argument("--time", action="store_true", help="exe/pareto_mpm.py 와 같은 프레임 시간 (서브스텝 루프 앞뒤 동기화 벽시계); 이때 --sub_chk 격자 안 확인은 프레임 끝 한 번 (시간 밖)")
    ap.add_argument("--dump_pred", action="store_true", help="안정 서브스텝 표 (doc/pareto.md 절 S): 발산 없이 끝나면 pred.pt {pred [L,N,3], F [L,N,3,3]} (lb_roll_to_h5 -> 3DGS 렌더용; 풀이 불변)")
    ap.add_argument("--pg_cfg", action="store_true", help="절 AK.15: PhysGaussian gs_simulation 형식 정답 (ficus): config 재질 (additional_material_params 포함)·감쇠·경계조건 (decode_param.set_boundary_conditions) 그대로, 바닥 충돌 없음")
    ap.add_argument("--flip", type=float, default=0.0, help="절 AK.14 진단: 서브스텝마다 FLIP 섞기 v = v_PIC + α (v_old − v_s), v_s = 같은 서브스텝 시작 상태의 운동량만 P2G (APIC 같은 식, 힘·중력·경계 없음) -> G2P 속도 (0 = 원래 정답 솔버)")
    ap.add_argument("--sub_chk", action="store_true", help="큰 dt 실험 (절 BD): 서브스텝마다 입자 위치가 유한하고 격자 배열 안 (P2G 기저 노드 0 .. n_grid−1, x/dx ∈ (−0.5, n_grid−1.5)) 인지 확인 -- 벗어나면 그 프레임을 발산으로 적고 멈춤 (벗어난 채 다음 P2G 를 돌면 warp 커널이 배열 밖을 써서 CUDA 700/717 로 죽음). 프레임 오차 > 2 도 발산 (우리 실행기와 같은 규칙)")
    ap.add_argument("--rpic_cfg", action="store_true", help="rpic_damping 을 config 값 그대로 (공식 설정; 키가 없으면 0)")
    ap.add_argument("--diag", action="store_true", help="절 AK.14: 프레임마다 몸 좌표 높이 (0 프레임 대비 Kabsch)·운동 에너지·입자 탄성 Σ V ψ(F) -> err.json diag")
    a = ap.parse_args()
    sys.path.insert(0, a.pg)
    os.chdir(a.pg)
    import h5py                                                      # noqa: E402
    import numpy as np                                               # noqa: E402
    import torch                                                     # noqa: E402
    import warp as wp                                                # noqa: E402
    from mpm_solver_warp.mpm_solver_warp import MPM_Simulator_WARP   # noqa: E402

    if not os.path.exists(os.path.join(a.gt, "sim_0000000000.h5")) and os.path.exists(os.path.join(a.gt, "simulation_ply", "sim_0000000000.h5")):
        a.gt = os.path.join(a.gt, "simulation_ply")                                      # PG gs_simulation 출력은 <output_path>/simulation_ply/sim_*.h5 (wolf·bread 원본 정답)
    cfg = json.load(open(a.config))
    cfg.setdefault("n_grid", 50)                                                         # PG decode_param 기본값 (PG 원본 config 는 키가 없을 수 있음; tools.py pg-h5-to-pt 와 같음)
    cfg.setdefault("grid_lim", 2.0)
    meta = json.load(open(os.path.join(a.gt, "meta.json"))) if os.path.exists(os.path.join(a.gt, "meta.json")) else {}   # PG 원본 h5 폴더는 meta 없음
    if a.rpic_cfg:
        a.rpic = float(cfg.get("rpic_damping", 0.0))
    GL = float(cfg["grid_lim"])
    n_grid0 = int(cfg["n_grid"])
    dx0 = GL / n_grid0
    n_grid = a.n_grid or n_grid0
    frame_dt = float(cfg["frame_dt"])
    if a.pg_cfg and not a.nsub:                                                          # 원래 서브스텝: PG gs_simulation 과 똑같이 int(frame_dt / substep_dt) 걸음 x substep_dt
        nsub = int(frame_dt / float(cfg["substep_dt"]))                                   # (wolf 0.04 / 2e-5 = 1999.99.. -> 1999 걸음; round 로 2000 을 돌리면 정답과 어긋남)
        dt = float(cfg["substep_dt"])
    else:
        nsub = a.nsub or int(round(frame_dt / float(cfg["substep_dt"])))
        dt = frame_dt / nsub
    G = [float(q) for q in cfg["g"]]
    _fl = [b for b in cfg["boundary_conditions"] if b["type"] == "surface_collider"]
    floor = dict(_fl[0]) if _fl else None
    if a.floor_z is not None and floor is not None:
        floor["point"] = [floor["point"][0], floor["point"][1], float(a.floor_z)]
    if a.surface is not None and floor is not None:
        floor["surface"] = a.surface
    d = torch.load(a.pt, map_location="cpu", weights_only=False, mmap=True)
    x0d = d["x"][0].float()
    EXT = float((x0d.max(0).values - x0d.min(0).values).norm())       # deform_model.extent (바운딩박스 대각선) 와 같다 -- fem_ex_sim 의 EXT
    del d, x0d


    def h5x(t, key="x"):
        with h5py.File(os.path.join(a.gt, f"sim_{t:010d}.h5")) as h:
            r = np.array(h[key])
        return r.T.copy() if (r.ndim == 2 and r.shape[0] == 3 and r.shape[1] != 3) else r


    x0 = h5x(0).astype(np.float32)
    try:
        v0 = h5x(0, "v").astype(np.float32)
    except KeyError:                                                                      # PG gs_simulation h5 는 속도가 없다 (초기 속도는 경계조건으로)
        v0 = np.zeros_like(x0)
    if v0.shape[0] == 3 and x0.shape[0] != 3:
        v0 = v0.T.copy()
    if x0.shape[0] == 3 and x0.shape[1] != 3:
        x0, v0 = x0.T.copy(), (v0.T.copy() if v0.shape[0] == 3 else v0)
    N = x0.shape[0]
    L = min(a.frames, int(meta.get("frames", cfg.get("frame_num", a.frames))))
    X = torch.as_tensor(x0).cuda().contiguous()
    cell = (X / dx0).floor().long()
    key = (cell[:, 0] * n_grid0 + cell[:, 1]) * n_grid0 + cell[:, 2]
    _, inv, cnt = torch.unique(key, return_inverse=True, return_counts=True)
    VOL = (dx0 ** 3 / cnt[inv].float()).contiguous()
    if a.pg_cfg and cfg.get("material") == "sand":                                       # PG get_particle_volume(unifrom=material == "sand"): 모래는 평균 부피 균일
        VOL = VOL.mean().repeat(N).contiguous()
    print(f"[정답 민감도] 입자 {N}, n_grid {n_grid} (dx {GL / n_grid:.5f}), 서브스텝 {dt:.4e} x {nsub}, rpic {a.rpic}, 바닥 {floor}, EXT {EXT:.6f}", flush=True)
    print(f"[명령] {' '.join(sys.argv)}", flush=True)
    wp.init()


    @wp.kernel
    def ak_p2g_mom(x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3), C: wp.array(dtype=wp.mat33), mass: wp.array(dtype=float), sel: wp.array(dtype=int),
                   inv_dx: float, dx: float, gv: wp.array3d(dtype=wp.vec3), gm: wp.array3d(dtype=float)):
        p = wp.tid()
        if sel[p] == 0:
            gp = x[p] * inv_dx
            bx = wp.int(gp[0] - 0.5)
            by = wp.int(gp[1] - 0.5)
            bz = wp.int(gp[2] - 0.5)
            fx = gp - wp.vec3(wp.float(bx), wp.float(by), wp.float(bz))
            wa = wp.vec3(1.5) - fx
            wb = fx - wp.vec3(1.0)
            wc = fx - wp.vec3(0.5)
            w = wp.mat33(wp.cw_mul(wa, wa) * 0.5, wp.vec3(0.0, 0.0, 0.0) - wp.cw_mul(wb, wb) + wp.vec3(0.75), wp.cw_mul(wc, wc) * 0.5)
            for i in range(0, 3):
                for j in range(0, 3):
                    for k in range(0, 3):
                        dpos = (wp.vec3(wp.float(i), wp.float(j), wp.float(k)) - fx) * dx
                        weight = w[0, i] * w[1, j] * w[2, k]
                        wp.atomic_add(gv, bx + i, by + j, bz + k, weight * mass[p] * (v[p] + C[p] * dpos))
                        wp.atomic_add(gm, bx + i, by + j, bz + k, weight * mass[p])


    @wp.kernel
    def ak_g2p_v(x: wp.array(dtype=wp.vec3), sel: wp.array(dtype=int), inv_dx: float, gv: wp.array3d(dtype=wp.vec3), gm: wp.array3d(dtype=float),
                 vs: wp.array(dtype=wp.vec3)):
        p = wp.tid()
        if sel[p] == 0:
            gp = x[p] * inv_dx
            bx = wp.int(gp[0] - 0.5)
            by = wp.int(gp[1] - 0.5)
            bz = wp.int(gp[2] - 0.5)
            fx = gp - wp.vec3(wp.float(bx), wp.float(by), wp.float(bz))
            wa = wp.vec3(1.5) - fx
            wb = fx - wp.vec3(1.0)
            wc = fx - wp.vec3(0.5)
            w = wp.mat33(wp.cw_mul(wa, wa) * 0.5, wp.vec3(0.0, 0.0, 0.0) - wp.cw_mul(wb, wb) + wp.vec3(0.75), wp.cw_mul(wc, wc) * 0.5)
            nv = wp.vec3(0.0, 0.0, 0.0)
            for i in range(0, 3):
                for j in range(0, 3):
                    for k in range(0, 3):
                        weight = w[0, i] * w[1, j] * w[2, k]
                        m_ = gm[bx + i, by + j, bz + k]
                        if m_ > 1e-15:
                            nv = nv + gv[bx + i, by + j, bz + k] * (weight / m_)
            vs[p] = nv

    sol = MPM_Simulator_WARP(10)
    sol.load_initial_data_from_torch(X, VOL, None, n_grid=n_grid, grid_lim=GL)
    if a.pg_cfg:                                                                         # gs_simulation 과 같은 순서 (재질 -> 경계조건 -> mu·lam)
        from utils.decode_param import set_boundary_conditions                           # noqa: E402
        _mp = {k: cfg[k] for k in ("material", "E", "nu", "density", "g", "grid_v_damping_scale", "additional_material_params") if k in cfg}
        for k in ("yield_stress", "hardening", "xi", "plastic_viscosity", "softening", "friction_angle", "bulk_modulus"):   # PG 재질 매개 (wolf friction_angle 30 · bread plasticine 항목) -- 키가 있으면 config 그대로
            if k in cfg:
                _mp[k] = cfg[k]
        _mp.update({"n_grid": n_grid, "grid_lim": GL, "rpic_damping": float(a.rpic)})
        sol.set_parameters_dict(_mp)
        set_boundary_conditions(sol, cfg.get("boundary_conditions", []), {"substep_dt": dt, "frame_dt": frame_dt, "frame_num": L})
        if not np.any(v0) and cfg.get("init_velocity_bc"):
            set_boundary_conditions(sol, cfg["init_velocity_bc"], {"substep_dt": dt, "frame_dt": frame_dt, "frame_num": L})
        sol.finalize_mu_lam()
    else:
        sol.set_parameters_dict({"material": "jelly", "E": float(cfg["E"]), "nu": float(cfg["nu"]), "density": float(cfg["density"]),
                                 "g": G, "n_grid": n_grid, "grid_lim": GL, "grid_v_damping_scale": 1.1, "rpic_damping": float(a.rpic)})
        sol.add_bounding_box()
        sol.add_surface_collider(tuple(floor["point"]), tuple(floor["normal"]), floor["surface"], float(floor.get("friction", 0.0)))
        sol.finalize_mu_lam()
    sol.import_particle_v_from_torch(torch.as_tensor(v0).cuda().contiguous())
    os.makedirs(a.out, exist_ok=True)
    XD = None if a.no_dump else torch.empty(L, N, 3, dtype=torch.float32)
    FD = torch.empty(L, N, 3, 3, dtype=torch.float32) if a.dump_pred else None
    errs = []
    t0 = time.time()
    step = 0
    if a.flip > 0:
        _st, _md = sol.mpm_state, sol.mpm_model
        _gv = wp.zeros(shape=(n_grid, n_grid, n_grid), dtype=wp.vec3, device="cuda:0")
        _gm = wp.zeros(shape=(n_grid, n_grid, n_grid), dtype=float, device="cuda:0")
        _vs = wp.zeros(shape=N, dtype=wp.vec3, device="cuda:0")
        print(f"[절 AK.14] FLIP 섞기 α {a.flip}", flush=True)
    DIAG = []
    if a.diag:
        import math as _m
        _x0 = torch.as_tensor(x0).cuda().double()
        _q0 = _x0 - _x0.mean(0)
        _mass = (VOL.double() * float(cfg["density"]))
        _E, _nu = float(cfg["E"]), float(cfg["nu"])
        _mu, _lam = _E / (2 * (1 + _nu)), _E * _nu / ((1 + _nu) * (1 - 2 * _nu))

        def _diag(f):
            xx = sol.export_particle_x_to_torch().double()
            vv = wp.to_torch(sol.mpm_state.particle_v).double()
            FF = wp.to_torch(sol.mpm_state.particle_F_trial if hasattr(sol.mpm_state, "particle_F_trial") else sol.mpm_state.particle_F).double().reshape(-1, 3, 3)
            c = xx.mean(0)
            U_, S_, Vh_ = torch.linalg.svd(_q0.T @ (xx - c))
            Dg = torch.eye(3, device="cuda", dtype=torch.float64)
            Dg[2, 2] = torch.sign(torch.det(Vh_.T @ U_.T))
            Rm = Vh_.T @ Dg @ U_.T
            b = (xx - c) @ Rm
            res = ((xx - c) - _q0 @ Rm.T).norm(dim=-1)
            sv = torch.linalg.svdvals(FF)
            J = torch.linalg.det(FF)
            psi = _mu * ((sv - 1) ** 2).sum(-1) + 0.5 * _lam * (J - 1) ** 2
            DIAG.append(dict(f=f, height=float(b[:, 2].max() - b[:, 2].min()), res=100 * float(res.pow(2).mean().sqrt()) / EXT, comz=float(c[2]),
                             ke=float(0.5 * (_mass * (vv ** 2).sum(-1)).sum()), ep=float((VOL.double() * psi).sum())))
    DIV = None
    _gsz = GL / n_grid


    def _out_of_grid():
        xx = wp.to_torch(sol.mpm_state.particle_x)
        if not bool(torch.isfinite(xx).all()):
            return "비유한"
        lo, hi = torch.aminmax(xx / _gsz)
        if float(lo) <= -0.5 or float(hi) >= n_grid - 1.5:
            return f"격자 밖 (x/dx 범위 {float(lo):.3g} .. {float(hi):.3g}, 배열 0..{n_grid - 1})"
        return None


    MS = []
    for f in range(1, L + 1):
        if a.time:
            torch.cuda.synchronize(); wp.synchronize(); _tf = time.perf_counter()
        for s in range(nsub):
            if a.flip > 0:                                                                # v_s: 서브스텝 시작 상태의 운동량만 왕복
                _gv.zero_()
                _gm.zero_()
                wp.launch(ak_p2g_mom, dim=N, inputs=[_st.particle_x, _st.particle_v, _st.particle_C, _st.particle_mass, _st.particle_selection,
                                                     _md.inv_dx, _md.dx, _gv, _gm], device="cuda:0")
                wp.launch(ak_g2p_v, dim=N, inputs=[_st.particle_x, _st.particle_selection, _md.inv_dx, _gv, _gm, _vs], device="cuda:0")
                v_old = wp.to_torch(_st.particle_v).clone()
                v_s = wp.to_torch(_vs).clone()
            sol.p2g2p(step, dt)
            if a.flip > 0:
                vt = wp.to_torch(_st.particle_v)
                vt.add_(a.flip * (v_old - v_s))
            step += 1
            if a.sub_chk and not a.time:
                _r = _out_of_grid()
                if _r:
                    DIV = dict(frame=f, substep=s + 1, reason=_r)
                    break
        if a.time:
            try:
                wp.synchronize(); torch.cuda.synchronize(); MS.append(1000.0 * (time.perf_counter() - _tf))
            except RuntimeError as _ex:                                                   # 서브스텝 격자 확인이 없는 시간 판의 격자 밖 쓰기 (CUDA 700) -> 그 프레임 발산, 그때까지 시간은 남김
                DIV = dict(frame=f, substep=nsub, reason=f"CUDA 오류 (시간 판, 격자 밖 쓰기 추정): {str(_ex)[:80]}")
            if a.sub_chk and DIV is None:
                _r = _out_of_grid()
                if _r:
                    DIV = dict(frame=f, substep=nsub, reason=_r)
        if DIV is not None:
            errs.append(float("nan"))
            print(f"[발산] 프레임 {f} 서브스텝 {DIV['substep']}/{nsub}: {DIV['reason']}", flush=True)
            break
        if a.diag:
            _diag(f)
        xt = sol.export_particle_x_to_torch()
        gt = torch.as_tensor(h5x(f)).cuda()
        e = float((xt - gt).norm(dim=-1).mean()) / EXT
        if not torch.isfinite(xt).all():
            e = float("nan")
        errs.append(e)
        if XD is not None:
            XD[f - 1] = xt.cpu()
        if FD is not None:
            FD[f - 1] = wp.to_torch(sol.mpm_state.particle_F_trial if hasattr(sol.mpm_state, "particle_F_trial") else sol.mpm_state.particle_F).float().reshape(-1, 3, 3).cpu()
        print(f"  프레임 {f}/{L} 오차 {100 * e:.3f}% (누적 평균 {100 * np.nanmean(errs):.3f}%) 최저 {float(xt[:, 2].min()):.5f} {time.time() - t0:.0f}s", flush=True)
        if not np.isfinite(e) or (a.sub_chk and e > 2.0):
            DIV = dict(frame=f, substep=nsub, reason="비유한" if not np.isfinite(e) else f"오차 {100 * e:.1f}% > 200%")
            print(f"[발산] 프레임 {f}", flush=True)
            break
    if DIV is not None and a.time:                                                         # 발산한 시간 판 -- 발산 프레임 전까지 시간 (warp 문맥이 깨졌을 수 있어 아래 내보내기 없이 바로 씀)
        json.dump(dict(div=DIV, frames_done=len(MS), ms=MS, ms_mean=(float(np.mean(MS[2:])) if len(MS) > 2 else None), ms_max=(float(np.max(MS[2:])) if len(MS) > 2 else None)),
                  open(os.path.join(a.out, "time_part.json"), "w"), ensure_ascii=False, indent=1)
    summ = dict(args=vars(a), nsub=nsub, dt=dt, n_grid=n_grid, floor=floor, EXT=EXT, frames_done=len(errs), err_mean=float(np.mean(errs)), div=DIV, err_mean_done=float(np.nanmean(errs)) if np.isfinite(errs).any() else None,
                err=errs, sec=time.time() - t0, diag=DIAG, ms=MS, ms_mean=(float(np.mean(MS[2:])) if len(MS) > 2 else None), ms_max=(float(np.max(MS[2:])) if len(MS) > 2 else None))
    json.dump(summ, open(os.path.join(a.out, "err.json"), "w"), ensure_ascii=False, indent=1)
    if XD is not None:
        torch.save(XD, os.path.join(a.out, "x.pt"))
    if FD is not None and DIV is None and XD is not None:
        torch.save({"pred": XD, "F": FD}, os.path.join(a.out, "pred.pt"))
    print(f"[정답 민감도 결과] {os.path.basename(a.out)}: 평균 오차 {100 * summ['err_mean']:.3f}% ({len(errs)}/{L} 프레임), {summ['sec']:.0f}s", flush=True)

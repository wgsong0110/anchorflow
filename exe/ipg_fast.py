"""[final_fix_mpmbl] i-PG Newton-GMRES 빠른 구현 (AF_IPG_FAST=1): 수식·허용오차·정체 판정·정밀도 그대로, 계산 자리만 줄임.

i-physgaussian implicit_mpm_solver.ImplicitMPMSolver.p2g2p_newton_gmres (AF_IPG_GPU=1, AF_IPG_FUSED!=1 경로, patch_ipg_noisefloor 적용판)
와 같은 알고리즘을 줄 하나하나 옮겼다. 바뀐 것은 하나뿐:
  원 코드는 Newton·GMRES 벡터를 격자 전체 (n_grid^3 x 3 = 200^3 x 3 = 2400 만 float64) 로 들고 다니며 비자유 노드를 매번 0 으로 지운다.
  비자유 성분은 원 코드에서도 항상 정확히 0 (R·탐색 방향·Newton 증분 모두 0 으로 지움, 줄 탐색·갱신도 0 더하기) 이고 du 의 비자유 성분은
  걸음 내내 초기값 그대로다 -> 자유 노드 성분만 모아 (압축) 같은 원소별 연산을 같은 순서로 한다. 격자 전체 v^{n+1} 은 걸음 처음 du0 에서
  만든 float32 바탕에 자유 성분만 덮어써서 원 코드와 비트 단위로 같은 값을 warp 에 넘긴다.
  -> 원소별 값은 비트 단위로 같고, 합 (노름·내적·GMRES Arnoldi) 만 0 항이 빠져 더하는 순서가 달라진다 (float64 반올림 수준).
걸음 준비 (v^n·a^n·경계 노드 판별, numpy) 와 마지막 G2P 는 원 코드 그대로.

  쓰는 법: AF_IPG_FAST=1 이면 exe/pareto_mpmbl.py 가 ImplicitMPMSolver.p2g2p_newton_gmres 를 이 함수로 바꿔 낀다.
"""
import os

import numpy as np
import warp as wp


def newton_gmres_fast(self, step: int, dt: float) -> dict:
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
    # [stabfinal ship] 재질 상태 되돌림 (AF_IPG_MATRESTORE=1): PG plasticine (material 5, von_mises_return_mapping_with_damage) 의
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

    # [fast] 자유 노드 압축 색인 (노드 순서 그대로, 성분 3 개)
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

    _diag = os.environ.get("AF_IPG_DIAG") == "1"
    _fz = float(os.environ.get("AF_DIAG_FLOORZ", "0.48"))
    if _diag:
        _pz = wp.to_torch(self.mpm_state.particle_x)[:, 2]
        print(f"  [diag] 걸음 {step} 바닥 아래 입자 {int((_pz < _fz).sum())} 최저 z {float(_pz.min()):.4f} BC 노드 {int(bc_mask.sum())} "
              f"질량 노드 {int(mask.sum())}", flush=True)
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
        if _diag:                                                   # [진단, AF_IPG_DIAG=1] 결과 불변 -- 출력만
            _r = cp.sqrt((R_k.reshape(-1, 3) ** 2).sum(1)); _tot = float((_r ** 2).sum()) + 1e-30
            _iz = fi % grid_size[2]; _z = _iz.astype(cp.float64) * _dx
            _nf = _z < _fz + 3 * _dx
            _top = int(cp.argmax(_r)); _ti = int(fi[_top])
            _ijk = np.unravel_index(_ti, grid_size)
            print(f"    [diag] it {newton_it} |R| {res_norm:.3e} 바닥근처(z<{_fz + 3 * _dx:.3f}) 몫 {float((_r[_nf] ** 2).sum()) / _tot:.3f} "
                  f"최대 |R| {float(_r[_top]):.3e} @ {_ijk} z {_ijk[2] * _dx:.3f} m {float(mass_gpu.ravel()[_ti]):.2e} "
                  f"GMRES info {info} dphi0 {dphi_0:.2e} alpha {alpha:.4f} LS {'ok' if phi_trial <= phi_0 + c1 * alpha * dphi_0 else 'FAIL'} "
                  f"|delta|inf {float(cp.abs(delta_du).max()):.2e} BC {int(bc_mask.sum())} 자유 {int(fi.size)}", flush=True)
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
        # [stabfinal ship] 경계조건 (sticky 바닥·상자) 이 속도를 정하는 노드는 v^{n+1} 이 BC 값으로 덮이는데 a^{n+1} 은 Newton 이 안 건드린
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

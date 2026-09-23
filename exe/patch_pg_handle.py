"""PhysGaussian 에 **힘으로 끄는 손잡이(handle)** 를 넣는다.

control point 방식(입자 속도를 덮어쓰는 Dirichlet)과 달리, 손잡이 입자 주변 반경 R
안의 입자들에게 힘을 계속 가해서 손잡이가 목표 위치로 끌려가게 한다. 입자는 고정되지
않으므로 물질이 저항하면 목표에 다 못 간다.

- 힘은 **프레임 시작에 한 번** 정하고 그 프레임의 모든 서브스텝에서 상수로 쓴다.
- 손잡이는 표면·내부 구분 없이 전체 입자에서 무작위로 뽑는다.
- AF_H_ROUNDS 회에 걸쳐 **매번 손잡이를 새로 잡고 새 목표로 옮긴다**.
  각 회차는 AF_H_RF 프레임이고, 방향은 씨앗 고정 난수라 재현된다.
- 힘은 회차 내내 **방향도 크기도 고정**이다 (목표 추종·속도 서보는 진동만 만든다):
    a_k = a_max * (M_avg / M_k) * d_k      (회차 내내 고정, sum_k M_k a_k = 0)
    v_p += w_p a dt        w_p 는 회차 시작 시점 거리로 정해진 falloff

  python exe/patch_pg_handle.py --pg <PG>
환경변수: AF_HANDLE, AF_H_R, AF_H_D, AF_H_VMAX, AF_H_KV, AF_H_KA, AF_H_AMAX,
          AF_H_N, AF_H_ROUNDS, AF_H_RF, AF_H_SEED, AF_H_MARK
"""
from __future__ import annotations

import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("--pg", required=True)
a = ap.parse_args()

p = os.path.join(a.pg, "gs_simulation.py")
s = open(p).read()
if "AF_HANDLE" in s:
    print("이미 패치됨")
    raise SystemExit(0)

# _af_os 가 없으면(다른 패치를 안 걸었으면) 여기서 만든다
if "import os as _af_os" not in s:
    s = s.replace("from particle_filling.filling import *",
                  "from particle_filling.filling import *\nimport os as _af_os", 1)

# ---------------------------------------------------------------- 1) 설정
A1 = '    substep_dt = time_params["substep_dt"]'
B1 = '''    # ---- anchorflow: 힘으로 끄는 손잡이 ----
    _af_on = bool(_af_os.environ.get("AF_HANDLE"))
    if _af_on:
        import numpy as _af_np
        import warp as wp
        _af_R = float(_af_os.environ.get("AF_H_R", 0.20))
        _af_D = float(_af_os.environ.get("AF_H_D", 0.45))

        _af_amax = float(_af_os.environ.get("AF_H_AMAX", 2e3))
        # AF_H_FORCE 를 주면 가속도 대신 **손잡이당 힘**을 고정한다.
        # 밀도가 다른 재질을 같은 힘으로 비교하려면 이쪽을 쓴다.
        _af_force = float(_af_os.environ.get("AF_H_FORCE", 0.0))
        # AF_H_VMAX > 0 이면 **속도 지정 + 가속도 상한** 방식.
        #   a = clamp(k_a (v_max d_k - v_h), a_max)   (프레임 시작에 한 번 계산)
        _af_vmax = float(_af_os.environ.get("AF_H_VMAX", 0.0))
        _af_ka = float(_af_os.environ.get("AF_H_KA", 2e4))
        _af_tol = float(_af_os.environ.get("AF_H_TOL", 0.05))
        _af_kv = float(_af_os.environ.get("AF_H_KV", 0.0))
        _af_kin = bool(_af_os.environ.get("AF_H_KIN"))
        _af_N = int(_af_os.environ.get("AF_H_N", 4))
        _af_rounds = int(_af_os.environ.get("AF_H_ROUNDS", 3))
        _af_rf = int(_af_os.environ.get("AF_H_RF", 60))
        _af_on_f = int(_af_os.environ.get("AF_H_ON", _af_rf))  # 회차당 힘 주는 프레임
        _af_rng = _af_np.random.default_rng(int(_af_os.environ.get("AF_H_SEED", 0)))
        # 회차별 방향. 4개일 때는 **서로 마주보는 두 쌍**(+u,-u,+v,-v)을 쓴다.
        # 임의 방향 4개면 합력이 남아 몸이 통째로 끌려가기만 하고 늘어나지 않는다.
        # 전부 살짝 위로 기울여 바닥(sticky)에 박히지 않게 한다.
        _af_dirs = []
        for _r in range(_af_rounds):
            if _af_N == 4:
                _th = _af_rng.uniform(0, 2 * _af_np.pi)
                _u = _af_np.array([_af_np.cos(_th), _af_np.sin(_th), 0.0])
                _v = _af_np.array([-_af_np.sin(_th), _af_np.cos(_th), 0.0])
                # 수평으로 두면 ±u, ±v 넷 모두 바닥을 파고들지 않는다
                _q = [_u, -_u, _v, -_v]
            else:
                _q = []
                while len(_q) < _af_N:
                    _d = _af_rng.normal(size=3)
                    _q.append(_d / _af_np.linalg.norm(_d))
            # 기울이지 않는다 -- 네 방향의 합이 정확히 0 이어야 알짜힘이 0 이다.
            _af_dirs.append([_d / _af_np.linalg.norm(_d) for _d in _q])
        _af_dirs = _af_np.asarray(_af_dirs).reshape(_af_rounds, _af_N, 3)
        # 목표 방향. "out" 은 잡은 방향 그대로 바깥으로 -- 자유표면에서 잡아당기면
        # 덩어리만 뜯겨 나오고 몸통은 안 변한다. "twist" 는 90도 돌린 방향이라
        # 접선으로 끌어 몸 전체가 비틀리며 변형된다.
        _af_mode = _af_os.environ.get("AF_H_MODE", "twist")
        if _af_mode == "rand":
            # 목표점을 손잡이마다 **독립적인 임의 방향**으로 잡는다.
            # 바닥(-z) 으로 파고들지 않게 z 성분만 살짝 제한한다.
            _td = []
            while len(_td) < _af_rounds * _af_N:
                _v3 = _af_rng.normal(size=3)
                _n3 = _af_np.linalg.norm(_v3)
                if _n3 < 1e-6:
                    continue
                _v3 = _v3 / _n3
                if _v3[2] < -0.3:
                    continue
                _td.append(_v3)
            _af_tdir = _af_np.asarray(_td).reshape(_af_rounds, _af_N, 3)
        elif _af_mode == "twist" and _af_N == 4:
            _af_tdir = _af_dirs[:, [2, 3, 1, 0], :].copy()   # u->v, -u->-v, v->-u, -v->u
        else:
            _af_tdir = _af_dirs.copy()
        _af_hid_t = None
        _af_Rbox = [_af_R]   # 회차마다 실제로 쓴 반경
        _af_hist = {"hid": [], "hpos": [], "hvel": [], "R": []}

        def _af_pick(_r):
            """회차 _r 의 손잡이를 지금 입자 위치에서 새로 잡는다."""
            _x = mpm_solver.export_particle_x_to_torch()
            _ids, _tg, _mem, _w = [], [], [], []
            # 제어점은 표면이 아니라 **내부까지 포함한 전체 입자에서 무작위**로 뽑되,
            # 손잡이 공이 서로 겹치지 않도록 중심 간 거리를 2R 이상으로 강제한다.
            # 형상이 가늘면 요청한 반경으로 4 개가 안 들어간다 -> 될 때까지 줄인다
            _Rtry = _af_R
            while True:
                _pick, _tries = [], 0
                while len(_pick) < _af_N and _tries < 40000:
                    _tries += 1
                    _c = int(_af_rng.integers(0, _x.shape[0]))
                    if _pick:
                        if float((_x[_pick] - _x[_c]).norm(dim=1).min()) < 2.0 * _Rtry:
                            continue
                    _pick.append(_c)
                if len(_pick) == _af_N:
                    break
                _Rtry *= 0.8
                if _Rtry < 0.02:
                    raise RuntimeError("손잡이를 겹치지 않게 잡을 수 없다")
            _af_Rbox[0] = _Rtry
            if _Rtry != _af_R:
                print(f"  [반경축소] {_af_R} -> {_Rtry:.4f}", flush=True)
            print(f"  [겹침없음] 시도 {_tries}회, 중심간 최소거리 "
                  f"{float(torch.cdist(_x[_pick], _x[_pick]).masked_fill(torch.eye(_af_N, dtype=torch.bool, device=_x.device), 9e9).min()):.3f} "
                  f"(>= 2R = {2 * _Rtry:.3f})", flush=True)
            for _j in range(_af_N):
                _d = torch.tensor(_af_dirs[_r, _j], dtype=torch.float32,
                                  device=_x.device)
                _i = int(_pick[_j])
                _ids.append(_i)
                _td = torch.tensor(_af_tdir[_r, _j], dtype=torch.float32,
                                   device=_x.device)
                _tg.append(_x[_i] + _af_D * _td)
                _dd = (_x - _x[_i]).norm(dim=1)
                _m = torch.nonzero(_dd < _Rtry, as_tuple=False).flatten()
                _q = (_dd[_m] / _Rtry).clamp(0, 1)
                _mem.append(_m)
                _w.append(((1.0 - _q * _q) ** 2).unsqueeze(-1))
                print(f"  [{_r}-{_j}] 입자 {_i} 소속 {_m.numel()}개 "
                      f"방향 {_af_dirs[_r, _j].round(2).tolist()}", flush=True)
            _fd = torch.tensor(_af_tdir[_r], dtype=torch.float32, device=_x.device)
            # 알짜힘 0: 각 손잡이가 같은 크기의 힘을 내게 유효질량으로 나눈다.
            # (가속도를 같게 주면 공 크기가 달라 힘이 안 맞고 알짜힘이 남는다)
            _ms = wp.to_torch(mpm_solver.mpm_state.particle_mass)
            _M = torch.stack([(_w[_j].squeeze(-1) * _ms[_mem[_j]]).sum()
                              for _j in range(_af_N)])
            if _af_os.environ.get("AF_H_BALDIR") and _af_N == 4:
                # 힘 **크기는 넷 다 같게** 두고, 방향만 틀어 알짜힘을 0 으로 만든다.
                # sum_k M_k d_k = 0 은 변 길이가 M_k 인 닫힌 사각형과 같다.
                # 대각선 D 를 잡아 삼각형 둘로 쪼개면 정확히 닫힌다.
                _m = _M.detach().cpu().numpy().astype(float)
                _uu = _af_dirs[_r, 0]
                _vv = _af_dirs[_r, 2]
                _lo = max(abs(_m[0] - _m[1]), abs(_m[2] - _m[3]))
                _hi = min(_m[0] + _m[1], _m[2] + _m[3])
                _D = 0.5 * (_lo + _hi)

                def _tri(mA, mB, sgn):
                    ca = _af_np.clip((_D**2 + mA**2 - mB**2) / (2 * _D * mA), -1, 1)
                    cb = _af_np.clip((_D**2 + mB**2 - mA**2) / (2 * _D * mB), -1, 1)
                    sa = _af_np.sqrt(max(1 - ca * ca, 0.0))
                    sb = _af_np.sqrt(max(1 - cb * cb, 0.0))
                    return (sgn * ca * _uu + sa * _vv, sgn * cb * _uu - sb * _vv)

                _d0, _d1 = _tri(_m[0], _m[1], 1.0)
                _d2, _d3 = _tri(_m[2], _m[3], -1.0)
                _nd = _af_np.stack([_d0, _d1, _d2, _d3])
                _res = (_m[:, None] * _nd).sum(0)
                print(f"  [방향조정] 대각선 D {_D:.3f}, |d| "
                      f"{[round(float(_af_np.linalg.norm(x)), 4) for x in _nd]}, "
                      f"잔여 알짜힘/a {_af_np.abs(_res).max():.3e}", flush=True)
                _fd = torch.tensor(_nd, dtype=torch.float32, device=_x.device)
                _sc = torch.ones(_af_N, 1, device=_x.device)
            elif _af_os.environ.get("AF_H_NONORM"):
                # 알짜힘 맞추기를 끄고 네 손잡이에 **같은 가속도**를 그대로 준다
                _sc = torch.ones(_af_N, 1, device=_x.device)
            elif _af_force > 0.0:
                _sc = (_af_force / (_af_amax * _M)).unsqueeze(-1)
            else:
                _sc = (_M.mean() / _M).unsqueeze(-1)
            _net = (_M.unsqueeze(-1) * _sc * _fd).sum(0)
            print(f"  [알짜힘/a] {_net.abs().max().item():.3e} "
                  f"(유효질량 {[round(float(x), 4) for x in _M]})", flush=True)
            return (torch.tensor(_ids, device=_x.device), _fd.clone(),
                    _mem, _w, _fd * _sc)

        print(f"[손잡이] {_af_rounds}회 x {_af_N}개, 회차당 {_af_rf}프레임 중 "
              f"{_af_on_f}프레임만 힘, R {_af_R} "
              + (f"손잡이당 힘 {_af_force:g}" if _af_force > 0 else f"amax {_af_amax:g}"),
              flush=True)

''' + A1
assert A1 in s
s = s.replace(A1, B1, 1)

# ---------------------------------------------------------------- 2) 프레임당 힘
A2a = "        for step in range(step_per_frame):"
B2a = '''        if _af_on:
            if frame % _af_rf == 0 and frame // _af_rf < _af_rounds:
                print(f"[손잡이] {frame // _af_rf + 1}회차 (frame {frame}) 새로 잡음",
                      flush=True)
                (_af_hid_t, _af_udir, _af_mem, _af_w,
                 _af_fdir) = _af_pick(frame // _af_rf)
                _af_fscale = (_af_fdir.norm(dim=1, keepdim=True)
                              / _af_udir.norm(dim=1, keepdim=True).clamp(min=1e-12))
                _af_goal = (mpm_solver.export_particle_x_to_torch()[_af_hid_t]
                            + _af_D * _af_udir)
                if _af_kin:
                    # 위치 지정: 시작 -> 목표 궤적을 가속도·속도 상한으로 미리 계획
                    _ta = _af_vmax / _af_amax
                    _da = 0.5 * _af_amax * _ta * _ta
                    if 2.0 * _da <= _af_D:
                        _af_pk = _af_vmax
                        _af_ta = _ta
                        _af_tc = (_af_D - 2.0 * _da) / _af_vmax
                    else:
                        _af_ta = (_af_D / _af_amax) ** 0.5
                        _af_pk = _af_amax * _af_ta
                        _af_tc = 0.0
                    print(f"  [궤적] 거리 {_af_D} = 가속 {_af_ta:.4f}s + 등속 "
                          f"{_af_tc:.4f}s + 감속 {_af_ta:.4f}s, 최고속도 {_af_pk:.4f}",
                          flush=True)
            # 힘은 회차 내내 **방향이 고정**이다 (목표 지점 추종 없음).
            if _af_kin:
                _tt = (frame % _af_rf) * frame_dt
                if _tt < _af_ta:
                    _sp = _af_amax * _tt
                elif _tt < _af_ta + _af_tc:
                    _sp = _af_pk
                elif _tt < 2 * _af_ta + _af_tc:
                    _sp = _af_pk - _af_amax * (_tt - _af_ta - _af_tc)
                else:
                    _sp = 0.0
                _af_vpl = max(_sp, 0.0) * _af_udir
                _af_hist["hid"].append(_af_hid_t.detach().cpu().numpy().copy())
                _af_hist["hpos"].append(
                    mpm_solver.export_particle_x_to_torch()[_af_hid_t]
                    .detach().cpu().numpy().copy())
                _af_hist["hvel"].append(_af_vpl.detach().cpu().numpy().copy())
                _af_hist["R"].append(float(_af_Rbox[0]))
            # AF_H_GOAL=1 이면 고정 방향 대신 **목표점 방향**을 매 프레임 다시 잡는다
            if _af_os.environ.get("AF_H_GOAL"):
                _xh = mpm_solver.export_particle_x_to_torch()[_af_hid_t]
                _e = _af_goal - _xh
                _dn = _e.norm(dim=1, keepdim=True).clamp(min=1e-12)
                if _af_kv > 0.0:
                    # 위치지정 -> 속도(상한) -> 가속도(상한) 3단 캐스케이드
                    _vd = _af_kv * _e
                    _m = _vd.norm(dim=1, keepdim=True).clamp(min=1e-12)
                    _dir = _vd * (_m.clamp(max=1.0) / _m)
                else:
                    _dir = _e / _dn * (_dn / max(_af_tol, 1e-9)).clamp(max=1.0)
            else:
                _dir = _af_udir
            if _af_vmax > 0.0:
                # 속도 지정 + 가속도 상한: 크기만 서보가 정한다
                _vh = mpm_solver.export_particle_v_to_torch()[_af_hid_t]
                _acc = _af_ka * (_af_vmax * _dir - _vh)
                _n = _acc.norm(dim=1, keepdim=True).clamp(min=1e-12)
                _af_acc = _acc * (_n.clamp(max=_af_amax) / _n)
            elif _af_os.environ.get("AF_H_GOAL"):
                _af_acc = _af_amax * _dir * _af_fscale
            else:
                _af_acc = _af_amax * _af_fdir
            # 회차 앞쪽 _af_on_f 프레임에만 힘을 주고 나머지는 놓아 둔다
            if (frame % _af_rf) >= _af_on_f:
                _af_acc = _af_acc * 0.0
''' + A2a
assert A2a in s
s = s.replace(A2a, B2a, 1)

A2 = "            mpm_solver.p2g2p(frame, substep_dt, device=device)"
B2 = '''            if _af_on:
                _v = mpm_solver.export_particle_v_to_torch()
                for _k in range(len(_af_mem)):
                    if _af_kin:
                        # 위치 지정: 계획 속도로 **덮어쓴다** (가장자리로 갈수록 약하게)
                        _v[_af_mem[_k]] = (_v[_af_mem[_k]] * (1.0 - _af_w[_k])
                                           + _af_w[_k] * _af_vpl[_k])
                    else:
                        _v[_af_mem[_k]] += _af_w[_k] * _af_acc[_k] * substep_dt
''' + A2
assert A2 in s
s = s.replace(A2, B2, 1)

# ---------------------------------------------------------------- 3) 표식
A4 = "    if args.render_img and args.compile_video:"
B4 = '''    if _af_on and _af_os.environ.get("AF_H_DUMP"):
        import numpy as _af_np2
        _af_np2.savez_compressed(
            _af_os.environ["AF_H_DUMP"],
            hid=_af_np2.asarray(_af_hist["hid"]),
            hpos=_af_np2.asarray(_af_hist["hpos"], dtype=_af_np2.float32),
            hvel=_af_np2.asarray(_af_hist["hvel"], dtype=_af_np2.float32),
            R=_af_np2.asarray(_af_hist["R"], dtype=_af_np2.float32))
        print(f"[손잡이] 상태 저장 {_af_os.environ['AF_H_DUMP']}", flush=True)
''' + A4
assert A4 in s
s = s.replace(A4, B4, 1)

A3 = "            colors_precomp = convert_SH(shs, current_camera, gaussians, pos, rot)"
B3 = A3 + '''
            if _af_on:
                # 손잡이마다 고유색. 같은 색의 작고 진한 점 = 손잡이,
                # 화살표 = 그 손잡이에 주는 힘의 방향(고정), 사이의 점선 = 짝 표시.
                _pal = torch.tensor([[0.90, 0.10, 0.10], [0.10, 0.65, 0.20],
                                     [0.15, 0.35, 0.95], [0.85, 0.55, 0.05],
                                     [0.70, 0.15, 0.80], [0.10, 0.70, 0.70]],
                                    device=pos.device)
                _hx = mpm_solver.export_particle_x_to_torch()[_af_hid_t]
                _nh = _hx.shape[0]
                _cc = _pal[torch.arange(_nh, device=pos.device) % _pal.shape[0]]
                _nl = int(_af_os.environ.get("AF_H_LINE", 24))
                _s = torch.linspace(0.08, 0.92, _nl, device=pos.device).view(1, _nl, 1)
                _tip = (_af_goal if _af_os.environ.get("AF_H_GOAL")
                        else _hx + _af_D * _af_udir)
                _ln = (_hx.unsqueeze(1) * (1 - _s) + _tip.unsqueeze(1) * _s)
                _r0 = float(_af_os.environ.get("AF_H_MARK", 0.03)) / scale_origin
                _c6 = torch.tensor([1.0, 0, 0, 1.0, 0, 1.0], device=pos.device)
                # 힘이 닿는 반경 R 을 껍질(피보나치 구면)로 옅게 그린다
                _ns = int(_af_os.environ.get("AF_H_SPH", 260))
                if _ns > 0:
                    _k = torch.arange(_ns, device=pos.device, dtype=torch.float32)
                    _z = 1.0 - 2.0 * (_k + 0.5) / _ns
                    _rr = (1.0 - _z * _z).clamp(min=0).sqrt()
                    _ph = _k * 2.399963229728653
                    _sp = torch.stack([_rr * torch.cos(_ph),
                                       _rr * torch.sin(_ph), _z], -1)
                    _sh = _hx.unsqueeze(1) + _af_Rbox[0] * _sp.unsqueeze(0)
                    _sh = _sh.reshape(-1, 3)
                else:
                    _sh = _hx[:0]
                _sim = torch.cat([_hx, _tip, _ln.reshape(-1, 3), _sh], 0)
                _rad = torch.cat([
                    _r0 * torch.ones(_nh, device=pos.device),
                    0.7 * _r0 * torch.ones(_nh, device=pos.device),
                    0.30 * _r0 * torch.ones(_nh * _nl, device=pos.device),
                    0.22 * _r0 * torch.ones(_sh.shape[0], device=pos.device)])
                _op = torch.cat([
                    torch.ones(_nh, device=pos.device),
                    0.85 * torch.ones(_nh, device=pos.device),
                    0.55 * torch.ones(_nh * _nl, device=pos.device),
                    0.16 * torch.ones(_sh.shape[0], device=pos.device)]).unsqueeze(-1)
                _col = torch.cat([
                    _cc, _cc,
                    _cc.unsqueeze(1).expand(_nh, _nl, 3).reshape(-1, 3),
                    _cc.unsqueeze(1).expand(_nh, _ns, 3).reshape(-1, 3)
                    if _ns > 0 else _cc[:0]], 0)
                _pw = apply_inverse_rotations(
                    undotransform2origin(undoshift2center111(_sim), scale_origin,
                                         original_mean_pos), rotation_matrices)
                pos = torch.cat([pos, _pw], 0)
                cov3D = torch.cat([cov3D, (_rad * _rad).unsqueeze(-1) * _c6], 0)
                opacity = torch.cat([opacity, _op], 0)
                colors_precomp = torch.cat([colors_precomp, _col], 0)'''
assert A3 in s
s = s.replace(A3, B3, 1)

open(p, "w").write(s)
print(f"패치 완료: {p}")

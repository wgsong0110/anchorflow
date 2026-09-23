"""PhysGaussian 에 **물리적인 평행 집게**를 넣는다.

이전 판(`patch_pg_gripseq.py`)은 잡은 입자의 위치를 강제로 지정했다. 그건 집게가
물체를 통과하면서 그 자리의 재료를 끌고 가는 것이라, 상자 경계에서 재료가 찢겼다
(이웃거리 6~10배). 여기서는 집게 판을 **격자에 작용하는 움직이는 강체 충돌체**로
만든다 -- 판이 재료를 눌러 붙잡고, 마찰로 끌고 가며, 절대 뚫지 않는다.

솔버의 `grid_postprocess` 자리에 판마다 커널을 하나씩 얹고, 매 서브스텝 판의
자세·속도를 갱신한다 (`Dirichlet_collider` 구조체를 그대로 쓴다).

    "grip_plate": {"n":3, "arms":2, "seed":0, "approach":3, "close":3,
                   "hold":15, "rest":6, "jaw":0.10, "pad_w":0.10, "pad_d":0.10,
                   "thick":0.02, "grip":0.55, "friction":1.5,
                   "speed":[0.6,1.2], "twist":0.8}

  jaw    집게가 벌어진 간격(물체 지름 대비), grip 은 닫았을 때 그 비율
  thick  판 두께, friction 은 쿨롱 마찰계수
"""
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("--pg", required=True)
a = ap.parse_args()

MARK = "# [anchorflow] 물리 집게(판 충돌체)"

INIT = '''
    ''' + MARK + '''
    import json as _json_p
    _GP = _json_p.load(open(args.config)).get("grip_plate", None)
    _gp = None
    if _GP is not None:
        import numpy as _npp
        import sys as _sysp
        import warp as _wpp
        _sysp.path.insert(0, os.environ.get("AF_EXE", "/workspace/anchorflow/exe"))
        from control_traj import estimate_normals as _enp, surface_mask as _smp
        from grip_traj import random_moves as _rmp, _frame_from as _ffp
        from mpm_solver_warp.warp_utils import Dirichlet_collider as _DC

        @_wpp.kernel
        def _plate_collide(time: float, dt: float, state: MPMStateStruct,
                           model: MPMModelStruct, param: _DC):
            """강체 판 접촉: 파고든 깊이에 비례하는 **반발**과 쿨롱 마찰.

            입자를 손으로 옮기지 않는다. 판 안으로 들어오려는 격자 속도에
            깊이 비례 반발 속도를 더해, 재료가 스스로 밀려나게 한다.
            param.friction   쿨롱 마찰계수
            param.threshold  반발 강성 (깊이 1 칸당 몇 배속으로 밀어낼지)
            """
            gx, gy, gz = _wpp.tid()
            if time >= param.start_time and time < param.end_time:
                p = _wpp.vec3(float(gx) * model.dx, float(gy) * model.dx,
                              float(gz) * model.dx)
                rel = p - param.point
                a0 = _wpp.dot(rel, param.normal)
                b0 = _wpp.dot(rel, param.x_unit)
                c0 = _wpp.dot(rel, param.y_unit)
                h0 = param.direction[0]
                if (_wpp.abs(a0) < h0 + model.dx
                        and _wpp.abs(b0) < param.direction[1]
                        and _wpp.abs(c0) < param.direction[2]):
                    n = param.normal
                    if a0 < 0.0:
                        n = -param.normal                 # 가까운 면의 바깥 방향
                    depth = h0 - _wpp.abs(a0)             # >0 이면 판 안으로 들어옴
                    v = state.grid_v_out[gx, gy, gz] - param.velocity
                    vn = _wpp.dot(v, n)
                    if depth > 0.0:
                        # 침투를 한 번에 되돌리면 재료가 튕겨 폭발한다 (겪었다).
                        # 서브스텝마다 5% 만 회복하고, 속도 상한(param.threshold)을 둔다.
                        push = _wpp.min(0.05 * depth / dt, param.threshold)
                        if vn < push:
                            v = v + (push - vn) * n
                            vn = push
                    elif vn < 0.0:                        # 판을 향해 다가오는 중
                        vt = v - vn * n
                        lt = _wpp.length(vt)
                        if lt > 1e-12:
                            sc = _wpp.max(0.0, lt + vn * param.friction) / lt
                            vt = vt * sc
                        v = vt
                    state.grid_v_out[gx, gy, gz] = param.velocity + v

        _npl = 2 * int(_GP.get("arms", 2))
        _params = []
        for _i in range(_npl):
            _pp = _DC()
            _pp.start_time = 0.0
            _pp.end_time = -1.0                       # 기본은 꺼둔다
            _pp.friction = float(_GP.get("friction", 1.5))   # 쿨롱 마찰계수
            _pp.surface_type = 2
            _params.append(_pp)
            mpm_solver.collider_params.append(_pp)
            mpm_solver.grid_postprocess.append(_plate_collide)
            mpm_solver.modify_bc.append(None)
        _X0p = mpm_solver.mpm_state.particle_x.numpy().astype(_npp.float64)
        _EXT = float(_npp.linalg.norm(_X0p.max(0) - _X0p.min(0)))
        _GL = float(material_params.get("grid_lim", 2.0))
        _PAD = 5.0 * _GL / int(material_params["n_grid"])
        _rng = _npp.random.default_rng(int(_GP.get("seed", 0)))
        _sp = _GP.get("speed", [0.6, 1.2])
        _gp = dict(P=_params, ext=_EXT, arms=int(_GP.get("arms", 2)),
                   n=int(_GP.get("n", 3)), ap=int(_GP.get("approach", 3)),
                   close=int(_GP.get("close", 3)), hold=int(_GP.get("hold", 15)),
                   rest=int(_GP.get("rest", 6)),
                   jaw=float(_GP.get("jaw", 0.10)) * _EXT,
                   stiff=float(_GP.get("stiff", 2.0)),
                   lo=_PAD, hi=_GL - _PAD,
                   f_close=float(_GP.get("f_close", 3.0)),
                   v_cap=float(_GP.get("v_cap", 1.5)),
                   f_move=float(_GP.get("f_move", 6.0)),
                   damp=float(_GP.get("tool_damp", 0.999)),
                   dx=float(material_params.get("grid_lim", 2.0))
                   / int(material_params["n_grid"]),
                   grip=float(_GP.get("grip", 0.55)),
                   half=_npp.array([float(_GP.get("thick", 0.02)),
                                    float(_GP.get("pad_w", 0.10)),
                                    float(_GP.get("pad_d", 0.10))]) * _EXT,
                   mv=_rmp(_rng, int(_GP.get("n", 3)) * int(_GP.get("arms", 2)),
                           float(_sp[0]), float(_sp[1]),
                           float(_GP.get("twist", 0.8))),
                   cur=None, nxt=None, pose=[], log=[])
        _gp["mass"] = float(mpm_solver.mpm_state.particle_mass.numpy().sum()) \
            * float(_GP.get("mass_ratio", 0.5))
        _gp["floor"] = max([b.get("point", [0, 0, -1e9])[2] for b in bc_params
                            if b.get("type") == "surface_collider"
                            and abs(b.get("normal", [0, 0, 1])[2]) > 0.5] or [-1e9])
        os.makedirs(directory_to_save, exist_ok=True)

        def _pick(ei):
            """지금 모양의 표면에서 팔 수만큼 자리를 고른다 (집게를 놓을 자리)."""
            X = mpm_solver.mpm_state.particle_x.numpy().astype(_npp.float64)
            ok = _npp.isfinite(X).all(1)
            if not ok.all():
                X[~ok] = X[ok].mean(0) if ok.any() else 1.0
            nrm = _enp(X, k=16)
            srf = _npp.flatnonzero(_smp(X) & ok)
            r2 = _npp.random.default_rng(1000 + ei)
            pick = [int(srf[r2.integers(len(srf))])]
            while len(pick) < _gp["arms"]:
                d = _npp.min(_npp.linalg.norm(
                    X[srf][:, None, :] - X[pick][None, :, :], axis=-1), 1)
                pick.append(int(srf[int(_npp.argmax(d))]))
            out = []
            for ai, ci in enumerate(pick):
                # 집게는 **얇은 부위**를 물어야 한다. 두꺼운 데를 물면 안쪽 판이
                # 재료에 박힌 채 끌려가 물체를 갈라 버린다 (겪었다).
                # 후보 주변을 보고 가장 얇은 방향을 턱 축으로 삼고, 그 구간의
                # 가운데를 집게 중심으로 둔다. 너무 두꺼우면 다른 후보로 옮긴다.
                # 잡을 수 있는 자리를 고른다: 물체 **가장자리**에서 국소 두께가
                # 턱 최대 개방폭보다 얇은 곳. 두꺼운 한가운데를 누르면 판이
                # 몸통을 파고들어 재료를 밀어낸다 (겪었다).
                ctr = X.mean(0)
                rr = _npp.linalg.norm(X[srf] - ctr, axis=1)
                order = _npp.argsort(-rr)               # 바깥쪽부터
                best = None
                jmax = float(_GP.get("jaw_max", 0.22)) * _EXT
                for cj in [int(srf[o]) for o in order[::max(1, len(order)//60)]]:
                    sel = _npp.linalg.norm(X - X[cj], axis=1) < 0.6 * _gp["half"][1]
                    loc = X[sel]
                    if len(loc) < 30:
                        continue
                    c = loc.mean(0)
                    w, V = _npp.linalg.eigh(_npp.cov((loc - c).T)
                                            + 1e-12 * _npp.eye(3))
                    axis = V[:, 0]                      # 가장 얇은 방향 = 턱 축
                    t = (loc - c) @ axis
                    thick = float(t.max() - t.min())
                    if best is None or thick < best[0]:
                        best = (thick, c, axis)
                    if thick < 0.75 * jmax:             # 충분히 얇으면 채택
                        break
                thick, c, axis = best
                R = _ffp(axis, _npp.eye(3)[int(_npp.argmin(_npp.abs(axis)))])
                mv = _gp["mv"][min(ei * _gp["arms"] + ai, len(_gp["mv"]) - 1)]
                # 진짜 그리퍼처럼 **그 자리 두께에 맞춰 턱을 벌린다**. 턱보다
                # 두꺼운 데를 물면 판이 재료에 박혀 물체를 갈라 버린다 (겪었다).
                jw = min(0.5 * thick + 2.0 * _gp["dx"], 0.5 * jmax)
                ap_dir = (c - ctr)
                nrm_ap = _npp.linalg.norm(ap_dir)
                ap_dir = (ap_dir / nrm_ap) if nrm_ap > 1e-9 else R[0]
                ap_dir = ap_dir - R[0] * float(ap_dir @ R[0])   # 턱 축과 수직으로
                nrm_ap = _npp.linalg.norm(ap_dir)
                ap_dir = (ap_dir / nrm_ap) if nrm_ap > 1e-9 else R[1]
                out.append(dict(c0=c, R0=R, jaw=float(jw), ap_dir=ap_dir, **mv))
                print(f"    [팔 {ai+1}] 문 자리 두께 {thick:.3f} -> 턱 반간격 "
                      f"{jw:.3f}", flush=True)
            return out

        _gp["pick"] = _pick
        print(f"[물리집게] {_gp['n']} 번 x 팔 {_gp['arms']} 개, 판 반크기 "
              f"{_npp.round(_gp['half'], 3)}, 벌린 간격 {_gp['jaw']:.3f} -> 닫으면 "
              f"{_gp['jaw'] * _gp['grip']:.3f}, 마찰 {_GP.get('friction', 1.5)}",
              flush=True)
'''

POSE = '''
def _af_plate_step(gp, frame, dt_sub, x_np, v_np, m_np):
    """집게를 **강체로 적분**한다. 닫기도 이동도 힘으로 준다.

    각 팔은 질량 m 을 가진 강체이고 자유도는 (중심 3, 턱 간격 1) 이다.
    - 닫기: 일정한 파지력 F_close 로 턱을 좁힌다.
    - 이동: 목표 방향으로 F_move 를 준다.
    - 반력: 판 표면 안쪽으로 들어온 재료가 판을 되민다.
            F = sum_p m_p (v_p - v_plate)·n / dt  (그 순간 제거되는 운동량)
    그래서 두꺼운 데를 물면 스스로 멈추고, 무거운 걸 끌면 느려진다.
    """
    import numpy as np
    per = gp["ap"] + gp["close"] + gp["hold"] + gp["rest"]
    pi = frame % per
    out = []
    if gp["cur"] is None:
        return out
    for k, g in enumerate(gp["cur"]):
        st = g.setdefault("st", dict(c=g["c0"].copy(), v=np.zeros(3),
                                     gap=float(g["jaw"]), gv=0.0))
        R0, u = g["R0"], np.asarray(g["dir"])
        n0 = R0[0]                                   # 턱 축 (판 법선)
        half = gp["half"]
        # --- 재료가 판을 되미는 힘. 입자가 수십만이라 **GPU 에서** 센다
        # (매 서브스텝 CPU 로 복사하면 30 배 느려진다 -- 겪었다)
        import torch as _T
        dev = x_np.device
        Rt = _T.as_tensor(R0, dtype=_T.float32, device=dev)
        ct = _T.as_tensor(st["c"], dtype=_T.float32, device=dev)
        d = (x_np - ct) @ Rt.T
        F_c = np.zeros(3); F_g = 0.0
        for sgn in (1.0, -1.0):
            a = d[:, 0] - sgn * (st["gap"] + half[0])
            ins = ((a.abs() < half[0]) & (d[:, 1].abs() < half[1])
                   & (d[:, 2].abs() < half[2]))
            if not bool(ins.any()):
                continue
            vpl = _T.as_tensor(st["v"] + sgn * st["gv"] * n0,
                               dtype=_T.float32, device=dev)
            nn = _T.as_tensor(sgn * n0, dtype=_T.float32, device=dev)
            vn = (v_np[ins] - vpl) @ nn
            neg = vn < 0.0
            if bool(neg.any()):
                imp = float((m_np[ins][neg] * (-vn[neg])).sum()) / dt_sub
                F_c += imp * (sgn * n0)              # 판을 밀어내는 반력
                F_g += imp                           # 턱을 벌리려는 반력
        # --- 주는 힘
        F_a = np.zeros(3); Fg_a = 0.0
        if pi < gp["ap"]:                            # 접근: 표면 쪽으로
            F_a = np.asarray(g.get("ap_dir", n0)) * gp["f_move"]
        elif pi < gp["ap"] + gp["close"]:            # 닫기: 파지력
            Fg_a = -gp["f_close"]
        elif pi < gp["ap"] + gp["close"] + gp["hold"]:
            F_a = u * gp["f_move"]                   # 이동: 목표 방향
            Fg_a = -gp["f_close"]                    # 문 상태 유지
        else:
            Fg_a = +gp["f_close"]                    # 놓기: 벌린다
            F_a = -np.asarray(g.get("ap_dir", n0)) * 0.3 * gp["f_move"]
        # f_move / f_close 는 **가속도**로 해석한다 (질량에 비례해 힘을 준다는 뜻).
        # 힘을 절대값으로 주면 가벼운 물체에서 공구가 수십 배속으로 튀어 재료를
        # 뚫고 폭발한다 (겪었다). 접촉 반력은 질량으로 나눠 더한다.
        m = gp["mass"]
        vmax = gp["v_cap"]
        st["v"] = (st["v"] + (F_a + F_c / m) * dt_sub) * gp["damp"]
        sp_now = float(np.linalg.norm(st["v"]))
        if sp_now > vmax:
            st["v"] *= vmax / sp_now
        st["c"] = st["c"] + st["v"] * dt_sub
        st["gv"] = (st["gv"] + (Fg_a + F_g / (0.25 * m)) * dt_sub) * gp["damp"]
        st["gv"] = float(np.clip(st["gv"], -vmax, vmax))
        st["gap"] = float(np.clip(st["gap"] + st["gv"] * dt_sub,
                                  0.35 * float(g["jaw"]), 1.6 * float(g["jaw"])))
        if gp["floor"] > -1e8:                       # 지표면 아래로는 못 간다
            lo = gp["floor"] + float(half.max()) + st["gap"]
            if st["c"][2] < lo:
                st["c"][2] = lo
                st["v"][2] = max(0.0, st["v"][2])
        # --- 단측 구속: 판이 이미 재료가 있는 쪽으로는 **더 못 들어간다**.
        # 반력만으로는 늦어서(이미 박힌 뒤에 생긴다) 공구 속도를 직접 자른다.
        d2 = (x_np - _T.as_tensor(st["c"], dtype=_T.float32, device=dev)) @ Rt.T
        for sgn in (1.0, -1.0):
            a2 = d2[:, 0] - sgn * (st["gap"] + half[0])
            ins2 = ((a2.abs() < half[0]) & (d2[:, 1].abs() < half[1])
                    & (d2[:, 2].abs() < half[2]))
            cnt2 = int(ins2.sum())
            if cnt2 == 0:
                continue
            nn2 = sgn * n0                       # 판 바깥 방향
            vin = float(st["v"] @ (-nn2))        # 재료 쪽으로 들어가는 성분
            if vin > 0.0:
                st["v"] = st["v"] + vin * (-nn2) * -1.0 * 0.0 + nn2 * vin * 0.0
                st["v"] = st["v"] - (st["v"] @ (-nn2)) * (-nn2)   # 성분 제거
            # 턱도 더 닫히지 못하게 (재료를 물고 있으면 그 간격에서 멈춘다)
            if st["gv"] < 0.0:
                st["gv"] = 0.0
            # 이미 박힌 만큼은 천천히 빠져나온다
            depth2 = float((half[0] - a2[ins2].abs()).max())
            st["c"] = st["c"] + nn2 * min(depth2, 0.2 * gp["dx"])
        # --- 관통 해소: 판 안에 남은 입자를 표면으로 되돌린다.
        # 격자 속도 투영만으로는 재료가 흘러들어오는 것을 못 막는다 (겪었다).
        # 위치를 되돌린 만큼의 운동량은 공구에 반력으로 돌려준다 (양방향 결합).
        d3 = (x_np - _T.as_tensor(st["c"], dtype=_T.float32, device=dev)) @ Rt.T
        for sgn in (1.0, -1.0):
            face = sgn * (st["gap"] + half[0])
            a3 = d3[:, 0] - face
            ins3 = ((a3.abs() < half[0]) & (d3[:, 1].abs() < half[1])
                    & (d3[:, 2].abs() < half[2]))
            if not bool(ins3.any()):
                continue
            nn3 = _T.as_tensor(sgn * n0, dtype=_T.float32, device=dev)
            out_face = face + sgn * half[0]          # 바깥쪽 면
            push = (out_face - d3[ins3, 0]) * sgn    # 그 면까지 밀어낼 거리 (>0)
            x_np[ins3] = x_np[ins3] + push.unsqueeze(-1) * nn3
            vin3 = (v_np[ins3] @ nn3)
            neg3 = vin3 < 0.0
            if bool(neg3.any()):
                v_np[ins3] = _T.where(neg3.unsqueeze(-1),
                                      v_np[ins3] - vin3.unsqueeze(-1) * nn3,
                                      v_np[ins3])
                imp3 = float((m_np[ins3][neg3] * (-vin3[neg3])).sum()) / dt_sub
                st["v"] = st["v"] + (sgn * n0) * (imp3 / m) * dt_sub
        st["c"] = np.clip(st["c"], gp["lo"], gp["hi"])
        out.append((st["c"].copy(), R0, float(st["gap"]),
                    st["v"].copy(), np.zeros(3)))
    return out


def _af_plate_set(gp, poses, t_now, dt):
    """판 충돌체 파라미터를 지금 자세로 갱신한다 (판 2장 x 팔 개수)."""
    import warp as wp
    import numpy as np
    for ai, (c, R, gap, vel, om) in enumerate(poses):
        for si, sgn in enumerate((1.0, -1.0)):
            p = gp["P"][2 * ai + si]
            off = R[0] * (gap + gp["half"][0])       # gap = 절대 반간격
            cc = c + sgn * off
            p.point = wp.vec3(*[float(v) for v in cc])
            p.normal = wp.vec3(*[float(v) for v in (R[0] * sgn)])
            p.x_unit = wp.vec3(*[float(v) for v in R[1]])
            p.y_unit = wp.vec3(*[float(v) for v in R[2]])
            p.direction = wp.vec3(*[float(v) for v in gp["half"]])
            p.threshold = float(gp["stiff"])         # 반발 강성
            vv = vel
            p.velocity = wp.vec3(*[float(v) for v in vv])
            p.start_time = -1.0
            p.end_time = 1e9
'''

FRAME_HEAD = '''        if _gp is not None:
            _perp = _gp["ap"] + _gp["close"] + _gp["hold"] + _gp["rest"]
            _eip, _pip = frame // _perp, frame % _perp
            if _pip == 0 and _eip < _gp["n"]:
                _gp["cur"] = _gp["nxt"] if _gp["nxt"] is not None else _gp["pick"](_eip)
                _gp["nxt"] = None
                for _ai, _g in enumerate(_gp["cur"]):
                    _gp["log"].append(dict(ep=_eip, arm=_ai, frame=frame,
                                           center=_g["c0"].tolist(),
                                           R0=_g["R0"].tolist(),
                                           dir=_np_p.asarray(_g["dir"]).tolist(),
                                           speed=float(_g["speed"]),
                                           twist=float(_g["twist"]),
                                           ap=_gp["ap"], close=_gp["close"],
                                           hold=_gp["hold"], rest=_gp["rest"],
                                           jaw=_gp["jaw"], grip=_gp["grip"],
                                           half=_gp["half"].tolist(), ext=_gp["ext"]))
                    print(f"  [{_eip+1}번째 / 팔 {_ai+1}] 프레임 {frame}, 방향 "
                          f"{_np_p.round(_np_p.asarray(_g['dir']),2)}, 속도 "
                          f"{_g['speed']:.2f} 지름/s", flush=True)
            if _pip == _gp["ap"] + _gp["close"] + _gp["hold"] and _eip + 1 < _gp["n"]:
                _gp["nxt"] = _gp["pick"](_eip + 1)
'''

STEP_OLD = """        for step in range(step_per_frame):
            mpm_solver.p2g2p(frame, substep_dt, device=device)
"""
STEP_NEW = FRAME_HEAD + """        for step in range(step_per_frame):
            if _gp is not None:
                import warp as _wpq
                _xq = _wpq.to_torch(mpm_solver.mpm_state.particle_x)
                _vq = _wpq.to_torch(mpm_solver.mpm_state.particle_v)
                _mq = _wpq.to_torch(mpm_solver.mpm_state.particle_mass)
                _ps = _af_plate_step(_gp, frame, substep_dt, _xq, _vq, _mq)
                if _ps:
                    _af_plate_set(_gp, _ps, mpm_solver.time, substep_dt)
                if step == step_per_frame - 1:
                        _gp["pose"].append([(c.tolist(), R.tolist(), float(gap))
                                            for (c, R, gap, _v, _o) in _ps])
                        _np_p.save(os.path.join(directory_to_save,
                                                "gripseq_pose.npy"),
                                   _np_p.array(_gp["pose"], dtype=object),
                                   allow_pickle=True)
                        _np_p.save(os.path.join(directory_to_save, "gripseq.npy"),
                                   _np_p.array(_gp["log"], dtype=object),
                                   allow_pickle=True)
                # 집게가 재료를 격자 밖으로 밀어내면 p2g 가 남의 메모리를 건드려
                # CUDA 700 으로 죽는다 (겪었다). 매 서브스텝 도메인 안으로 자른다.
                import warp as _wpc
                _xx = _wpc.to_torch(mpm_solver.mpm_state.particle_x)
                _xx.nan_to_num_(nan=1.0, posinf=1.0, neginf=1.0)
                _gl2 = float(material_params.get("grid_lim", 2.0))
                _pd2 = 5.0 * _gl2 / int(material_params["n_grid"])
                _xx.clamp_(_pd2, _gl2 - _pd2)
                _vv3 = _wpc.to_torch(mpm_solver.mpm_state.particle_v)
                _vv3.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
            mpm_solver.p2g2p(frame, substep_dt, device=device)
"""

p = os.path.join(a.pg, "gs_simulation.py")
s = open(p).read()
if MARK in s:
    print("이미 되어 있다"); raise SystemExit
anc = "    for frame in tqdm(range(frame_num)):"
assert anc in s and s.count(STEP_OLD) == 1
if "import numpy as _np_p" not in s:
    s = s.replace("import numpy as np\n", "import numpy as np\nimport numpy as _np_p\n", 1)
if "from mpm_solver_warp.mpm_solver_warp import MPM_Simulator_WARP" in s:
    s = s.replace("from mpm_solver_warp.mpm_solver_warp import MPM_Simulator_WARP",
                  "from mpm_solver_warp.mpm_solver_warp import MPM_Simulator_WARP\n"
                  "from mpm_solver_warp.warp_utils import MPMStateStruct, MPMModelStruct", 1)
s = s.replace('if __name__ == "__main__":', POSE + '\n\nif __name__ == "__main__":', 1)
s = s.replace(anc, INIT + anc, 1).replace(STEP_OLD, STEP_NEW, 1)
open(p, "w").write(s)
import ast
ast.parse(s)
print(f"고쳤다: {p}")
print("PGPLATE_OK")

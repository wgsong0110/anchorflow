"""PhysGaussian 에 **SDF 기반 강체 집게**를 넣는다.

앞선 판(`patch_pg_plate.py`)은 "턱 축 방향 슬래브" 판정으로 접촉을 처리해서
깊이·방향이 근사였고, 두 판 사이 입자를 서로 반대쪽으로 밀어내는 모순이 생겨
판이 재료를 파고들었다 (판 안 입자 935~7,773 개). 여기서는 네 가지를 바꾼다.

1. 공구를 **해석적 상자 SDF** 로 둔다 -- 임의 점에서 부호 있는 거리 phi 와
   바깥 법선 grad phi 를 정확히 얻는다.
2. **격자 + 입자 이중 강제** -- 격자는 phi < dx 에서 상대속도의 안쪽 법선 성분을
   없애고 쿨롱 마찰을 걸며, 입자는 phi < 0 에서 `x <- x - phi * n` 으로 표면까지만
   밀어낸다 (최소 변위라 진동이 없다).
3. **양방향 결합** -- 없앤 운동량 총합을 공구에 반작용으로 돌려준다. 반력이
   가력을 넘으면 공구가 스스로 멈춘다 ("밀고 들어감" 대신 "닿으면 멈춤").
4. **충돌 없는 파지 자세 샘플링** -- 벌린 턱 부피 안에 재료가 없는 자세만 고른다.
   그래서 판이 처음부터 재료 밖에 있다.

    "grip_sdf": {"n":3, "arms":2, "seed":0, "approach":3, "close":4, "hold":15,
                 "rest":6, "pad_w":0.16, "pad_d":0.16, "thick":0.03,
                 "jaw_max":0.45, "friction":1.2, "f_close":1.5, "f_move":2.5,
                 "v_cap":1.0, "mass_ratio":0.5, "grip_force":0.6}

  python exe/patch_pg_sdf.py --pg /workspace/PhysGaussian_sdf
"""
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("--pg", required=True)
a = ap.parse_args()

MARK = "# [anchorflow] SDF 강체 집게"

INIT = '''
    ''' + MARK + '''
    import json as _json_s
    _GS = _json_s.load(open(args.config)).get("grip_sdf", None)
    _sg = None
    if _GS is not None:
        import numpy as _nps
        import sys as _syss
        import torch as _ts
        import warp as _wps
        _syss.path.insert(0, os.environ.get("AF_EXE", "/workspace/anchorflow/exe"))
        from grip_traj import random_moves as _rms, _frame_from as _ffs
        from mpm_solver_warp.warp_utils import Dirichlet_collider as _DCS

        @_wps.kernel
        def _sdf_collide(time: float, dt: float, state: MPMStateStruct,
                         model: MPMModelStruct, param: _DCS):
            """상자 SDF 접촉: 격자 속도에서 판 안쪽으로 파고드는 성분을 없앤다."""
            gx, gy, gz = _wps.tid()
            if time >= param.start_time and time < param.end_time:
                p = _wps.vec3(float(gx) * model.dx, float(gy) * model.dx,
                              float(gz) * model.dx)
                rel = p - param.point
                q0 = _wps.dot(rel, param.normal)
                q1 = _wps.dot(rel, param.x_unit)
                q2 = _wps.dot(rel, param.y_unit)
                h0 = param.direction[0]
                h1 = param.direction[1]
                h2 = param.direction[2]
                # 상자 SDF: 바깥은 남는 거리, 안쪽은 가장 가까운 면까지 (음수)
                e0 = _wps.abs(q0) - h0
                e1 = _wps.abs(q1) - h1
                e2 = _wps.abs(q2) - h2
                emax = _wps.max(e0, _wps.max(e1, e2))
                d0 = _wps.max(e0, 0.0)
                d1 = _wps.max(e1, 0.0)
                d2 = _wps.max(e2, 0.0)
                phi = _wps.sqrt(d0 * d0 + d1 * d1 + d2 * d2) + _wps.min(emax, 0.0)
                if phi < model.dx:
                    # 법선: 가장 가까운 면의 바깥 방향
                    n = param.normal * _wps.sign(q0)
                    if e1 > e0 and e1 > e2:
                        n = param.x_unit * _wps.sign(q1)
                    elif e2 > e0 and e2 > e1:
                        n = param.y_unit * _wps.sign(q2)
                    v = state.grid_v_out[gx, gy, gz] - param.velocity
                    vn = _wps.dot(v, n)
                    if vn < 0.0:
                        vt = v - vn * n
                        lt = _wps.length(vt)
                        if lt > 1e-12:
                            sc = _wps.max(0.0, lt + vn * param.friction) / lt
                            vt = vt * sc
                        v = vt
                    state.grid_v_out[gx, gy, gz] = param.velocity + v

        _nar = int(_GS.get("arms", 2))
        _params = []
        for _i in range(2 * _nar):
            _pp = _DCS()
            _pp.start_time = 0.0
            _pp.end_time = 1e9
            _pp.friction = float(_GS.get("friction", 1.2))
            _pp.surface_type = 2
            _params.append(_pp)
            mpm_solver.collider_params.append(_pp)
            mpm_solver.grid_postprocess.append(_sdf_collide)
            mpm_solver.modify_bc.append(None)
        _X0s = mpm_solver.mpm_state.particle_x.numpy().astype(_nps.float64)
        _EXT = float(_nps.linalg.norm(_X0s.max(0) - _X0s.min(0)))
        _GL = float(material_params.get("grid_lim", 2.0))
        _DX = _GL / int(material_params["n_grid"])
        _PAD = 5.0 * _DX
        _rng = _nps.random.default_rng(int(_GS.get("seed", 0)))
        _sg = dict(P=_params, arms=_nar, n=int(_GS.get("n", 3)), ext=_EXT, dx=_DX,
                   ap=int(_GS.get("approach", 3)), close=int(_GS.get("close", 4)),
                   hold=int(_GS.get("hold", 15)), rest=int(_GS.get("rest", 6)),
                   half=_nps.array([float(_GS.get("thick", 0.03)),
                                    float(_GS.get("pad_w", 0.16)),
                                    float(_GS.get("pad_d", 0.16))]) * _EXT,
                   jmax=float(_GS.get("jaw_max", 0.45)) * _EXT,
                   f_close=float(_GS.get("f_close", 1.5)),
                   f_move=float(_GS.get("f_move", 2.5)),
                   v_cap=float(_GS.get("v_cap", 1.0)),
                   grip_force=float(_GS.get("grip_force", 0.6)),
                   damp=float(_GS.get("tool_damp", 0.999)),
                   lo=_PAD, hi=_GL - _PAD, cur=None, nxt=None, pose=[], log=[],
                   mass=float(mpm_solver.mpm_state.particle_mass.numpy().sum())
                   * float(_GS.get("mass_ratio", 0.5)),
                   mv=_rms(_rng, int(_GS.get("n", 3)) * _nar, 0.4, 1.0, 0.5))
        _sg["floor"] = max([b.get("point", [0, 0, -1e9])[2] for b in bc_params
                            if b.get("type") == "surface_collider"
                            and abs(b.get("normal", [0, 0, 1])[2]) > 0.5] or [-1e9])

        def _pick_sdf(ei):
            """벌린 턱 부피 안에 재료가 **없는** 자세만 고른다 (충돌 없는 파지)."""
            X = mpm_solver.mpm_state.particle_x.numpy().astype(_nps.float64)
            ok = _nps.isfinite(X).all(1)
            if not ok.all():
                X[~ok] = X[ok].mean(0) if ok.any() else 1.0
            ctr = X.mean(0)
            half = _sg["half"]
            r2 = _nps.random.default_rng(2000 + ei)
            # 떨어져 나온 알갱이를 잡지 않도록 **가장 큰 덩어리**에서만 고른다.
            # 격자 점유로 연결 성분을 세고 제일 큰 것만 후보로 쓴다.
            cell = 3.0 * _sg["dx"]
            key = _nps.floor(X / cell).astype(_nps.int64)
            key -= key.min(0)
            shp = key.max(0) + 1
            occ = _nps.zeros(shp, bool)
            occ[key[:, 0], key[:, 1], key[:, 2]] = True
            try:
                from scipy import ndimage as _ndi
                lab, nlab = _ndi.label(occ)
                if nlab > 1:
                    big = 1 + int(_nps.argmax(_nps.bincount(
                        lab[occ].ravel())[1:]))
                    keep = lab[key[:, 0], key[:, 1], key[:, 2]] == big
                else:
                    keep = _nps.ones(len(X), bool)
            except Exception:
                keep = _nps.ones(len(X), bool)
            pool = _nps.flatnonzero(keep)
            print(f"    [본체] 가장 큰 덩어리 {len(pool)}/{len(X)} 입자", flush=True)
            out = []
            tries = 0
            while len(out) < _sg["arms"] and tries < 1200:
                tries += 1
                ci = int(pool[r2.integers(len(pool))])
                loc = X[_nps.linalg.norm(X - X[ci], axis=1) < 0.8 * half[1]]
                if len(loc) < max(300, int(0.002 * len(X))):   # 알갱이 배제
                    continue
                c = loc.mean(0)
                w, V = _nps.linalg.eigh(_nps.cov((loc - c).T) + 1e-12 * _nps.eye(3))
                axis = V[:, 0]                          # 가장 얇은 방향 = 턱 축
                t = (loc - c) @ axis
                thick = float(t.max() - t.min())
                if thick > 0.9 * _sg["jmax"]:           # 턱으로 감쌀 수 없다
                    continue
                R = _ffs(axis, _nps.eye(3)[int(_nps.argmin(_nps.abs(axis)))])
                gap = 0.5 * thick + 2.0 * _sg["dx"]     # 벌린 반간격
                # 벌린 두 판 부피 안에 재료가 있으면 그 자세는 버린다
                d = (X - c) @ R.T
                # 판 부피 안에 재료가 "거의" 없으면 채택한다. 0 을 요구하면 두툼한
                # 물체에서는 어떤 자세도 통과하지 못해 집게가 아예 안 움직인다 (겪었다).
                tol = max(50, int(3e-4 * len(X)))
                bad = False
                for sgn in (1.0, -1.0):
                    a0 = _nps.abs(d[:, 0] - sgn * (gap + half[0])) < half[0]
                    if (a0 & (_nps.abs(d[:, 1]) < half[1])
                            & (_nps.abs(d[:, 2]) < half[2])).sum() > tol:
                        bad = True
                        break
                if bad:
                    continue
                # 두 판 **사이**에 재료가 충분히 들어와야 실제로 집어진다.
                # 이 조건이 없으면 충돌 없는 자세만 고르다 물체 옆을 스친다 (겪었다).
                between = ((_nps.abs(d[:, 0]) < gap)
                           & (_nps.abs(d[:, 1]) < 0.8 * half[1])
                           & (_nps.abs(d[:, 2]) < 0.8 * half[2])).sum()
                if between < max(200, int(1e-3 * len(X))):
                    continue
                if any(float(_nps.linalg.norm(c - o["c0"])) < 1.5 * half[1]
                       for o in out):
                    continue
                ap_dir = c - ctr
                ap_dir = ap_dir - R[0] * float(ap_dir @ R[0])
                nn = _nps.linalg.norm(ap_dir)
                ap_dir = ap_dir / nn if nn > 1e-9 else R[1]
                mv = _sg["mv"][min(ei * _sg["arms"] + len(out), len(_sg["mv"]) - 1)]
                out.append(dict(c0=c, R0=R, jaw=float(gap), ap_dir=ap_dir, **mv))
                print(f"    [팔 {len(out)}] 두께 {thick:.3f} 반간격 {gap:.3f} "
                      f"사이 재료 {int(between)} 개", flush=True)
            return out

        _sg["pick"] = _pick_sdf
        print(f"[SDF집게] {_sg['n']} 번 x 팔 {_nar} 개, 판 반크기 "
              f"{_nps.round(_sg['half'], 3)}, 최대 개방 {_sg['jmax']:.3f}, "
              f"마찰 {_GS.get('friction', 1.2)}", flush=True)
'''

STEP_CODE = '''
def _af_sdf_phi(x, c, R, half):
    """상자 SDF 와 바깥 법선. x [N,3] (torch), c/R/half numpy."""
    import torch as T
    dev = x.device
    Rt = T.as_tensor(R, dtype=T.float32, device=dev)
    ct = T.as_tensor(c, dtype=T.float32, device=dev)
    ht = T.as_tensor(half, dtype=T.float32, device=dev)
    q = (x - ct) @ Rt.T
    e = q.abs() - ht
    outside = e.clamp(min=0.0)
    phi = outside.norm(dim=-1) + e.max(dim=-1).values.clamp(max=0.0)
    ax = e.argmax(dim=-1)                       # 가장 가까운 면
    sign = T.gather(q.sign(), 1, ax.unsqueeze(-1)).squeeze(-1)
    n = Rt[ax] * sign.unsqueeze(-1)
    return phi, n


def _af_sdf_step(sg, frame, dt_sub, x_t, v_t, m_t):
    """공구를 힘으로 적분하고, SDF 로 격자·입자를 이중 강제한다."""
    import numpy as np
    import torch as T
    per = sg["ap"] + sg["close"] + sg["hold"] + sg["rest"]
    pi = frame % per
    out = []
    if sg["cur"] is None:
        return out
    for k, g in enumerate(sg["cur"]):
        # 시작 위치는 목표 자세에서 **바깥쪽으로** 물체 하나 거리만큼 띄운다
        st = g.setdefault("st", dict(
            c=g["c0"] + np.asarray(g.get("ap_dir", g["R0"][1])) * (1.1 * sg["ext"]),
            v=np.zeros(3), gap=float(g["jaw"]), gv=0.0, locked=False))
        R0, u = g["R0"], np.asarray(g["dir"])
        n0, ap = R0[0], np.asarray(g.get("ap_dir", R0[1]))
        half = sg["half"]
        F_c = np.zeros(3); F_g = 0.0
        for sgn in (1.0, -1.0):
            pc = st["c"] + n0 * sgn * (st["gap"] + half[0])
            phi, nrm = _af_sdf_phi(x_t, pc, R0, half)
            ins = phi < 0.0
            if not bool(ins.any()):
                continue
            # (a) 입자를 표면까지만 밀어낸다 (최소 변위)
            x_t[ins] = x_t[ins] - phi[ins].unsqueeze(-1) * nrm[ins]
            # (b) 판 안으로 들어오는 법선 속도를 없애고 반력을 모은다
            vpl = T.as_tensor(st["v"] + n0 * sgn * st["gv"], dtype=T.float32,
                              device=x_t.device)
            vn = ((v_t[ins] - vpl) * nrm[ins]).sum(-1)
            neg = vn < 0.0
            if bool(neg.any()):
                idx = T.nonzero(ins).squeeze(-1)[neg]
                v_t[idx] = v_t[idx] - vn[neg].unsqueeze(-1) * nrm[ins][neg]
                imp = float((m_t[idx] * (-vn[neg])).sum()) / dt_sub
                F_c += imp * (n0 * sgn)
                F_g += imp
        # --- 주는 힘 (가속도로 해석한다)
        F_a = np.zeros(3); Fg_a = 0.0
        if pi < sg["ap"]:
            F_a = -ap * sg["f_move"]          # 바깥에서 물체 쪽으로 다가간다
        elif pi < sg["ap"] + sg["close"]:
            Fg_a = -sg["f_close"]
        elif pi < sg["ap"] + sg["close"] + sg["hold"]:
            F_a = u * sg["f_move"]
            Fg_a = -sg["f_close"]
        else:
            Fg_a = +sg["f_close"]
            F_a = ap * 0.3 * sg["f_move"]     # 놓고 바깥으로 물러난다
        m = sg["mass"]
        # 반력이 가력을 넘으면 공구가 멈춘다 -- 밀고 들어가지 않는다
        st["v"] = (st["v"] + (F_a + F_c / m) * dt_sub) * sg["damp"]
        sp = float(np.linalg.norm(st["v"]))
        if sp > sg["v_cap"]:
            st["v"] *= sg["v_cap"] / sp
        st["c"] = st["c"] + st["v"] * dt_sub
        # 목표 파지력에 닿으면 그 간격에서 유지 (힘 제어)
        if F_g / m > sg["grip_force"]:
            st["locked"] = True
        if not st["locked"]:
            st["gv"] = (st["gv"] + (Fg_a + F_g / (0.25 * m)) * dt_sub) * sg["damp"]
            st["gv"] = float(np.clip(st["gv"], -sg["v_cap"], sg["v_cap"]))
            st["gap"] = float(np.clip(st["gap"] + st["gv"] * dt_sub,
                                      0.3 * float(g["jaw"]), 1.8 * float(g["jaw"])))
        if pi >= sg["ap"] + sg["close"] + sg["hold"]:
            st["locked"] = False
        if sg["floor"] > -1e8:
            lo = sg["floor"] + float(half.max()) + st["gap"]
            if st["c"][2] < lo:
                st["c"][2] = lo
                st["v"][2] = max(0.0, st["v"][2])
        st["c"] = np.clip(st["c"], sg["lo"], sg["hi"])
        out.append((st["c"].copy(), R0, float(st["gap"]), st["v"].copy(),
                    np.zeros(3)))
    return out


def _af_sdf_set(sg, poses):
    """SDF 충돌체 파라미터를 지금 자세로 갱신한다 (판 2 장 x 팔 개수)."""
    import warp as wp
    for ai, (c, R, gap, vel, om) in enumerate(poses):
        for si, sgn in enumerate((1.0, -1.0)):
            p = sg["P"][2 * ai + si]
            cc = c + R[0] * sgn * (gap + sg["half"][0])
            p.point = wp.vec3(*[float(v) for v in cc])
            p.normal = wp.vec3(*[float(v) for v in R[0]])
            p.x_unit = wp.vec3(*[float(v) for v in R[1]])
            p.y_unit = wp.vec3(*[float(v) for v in R[2]])
            p.direction = wp.vec3(*[float(v) for v in sg["half"]])
            p.velocity = wp.vec3(*[float(v) for v in vel])
            p.start_time = -1.0
            p.end_time = 1e9
'''

FRAME_HEAD = '''        if _sg is not None:
            _pers = _sg["ap"] + _sg["close"] + _sg["hold"] + _sg["rest"]
            _eis, _pis = frame // _pers, frame % _pers
            if _pis == 0 and _eis < _sg["n"]:
                _sg["cur"] = _sg["nxt"] if _sg["nxt"] is not None else _sg["pick"](_eis)
                _sg["nxt"] = None
                for _ai, _g in enumerate(_sg["cur"]):
                    _sg["log"].append(dict(ep=_eis, arm=_ai, frame=frame,
                                           center=_g["c0"].tolist(),
                                           R0=_g["R0"].tolist(),
                                           jaw=float(_g["jaw"]),
                                           half=_sg["half"].tolist(),
                                           ext=_sg["ext"], ap=_sg["ap"],
                                           close=_sg["close"], hold=_sg["hold"],
                                           rest=_sg["rest"]))
            if _pis == _sg["ap"] + _sg["close"] + _sg["hold"] and _eis + 1 < _sg["n"]:
                _sg["nxt"] = _sg["pick"](_eis + 1)
'''

STEP_OLD = """        for step in range(step_per_frame):
            mpm_solver.p2g2p(frame, substep_dt, device=device)
"""
STEP_NEW = FRAME_HEAD + """        for step in range(step_per_frame):
            if _sg is not None and _sg["cur"]:
                import warp as _wpz
                _xz = _wpz.to_torch(mpm_solver.mpm_state.particle_x)
                _vz = _wpz.to_torch(mpm_solver.mpm_state.particle_v)
                _mz = _wpz.to_torch(mpm_solver.mpm_state.particle_mass)
                _psz = _af_sdf_step(_sg, frame, substep_dt, _xz, _vz, _mz)
                if _psz:
                    _af_sdf_set(_sg, _psz)
                    if step == step_per_frame - 1:
                        _sg["pose"].append([(c.tolist(), R.tolist(), float(gp))
                                            for (c, R, gp, _v, _o) in _psz])
                        _np_z.save(os.path.join(directory_to_save,
                                                "gripseq_pose.npy"),
                                   _np_z.array(_sg["pose"], dtype=object),
                                   allow_pickle=True)
                        _np_z.save(os.path.join(directory_to_save, "gripseq.npy"),
                                   _np_z.array(_sg["log"], dtype=object),
                                   allow_pickle=True)
                _xz.nan_to_num_(nan=1.0, posinf=1.0, neginf=1.0)
                _xz.clamp_(_sg["lo"], _sg["hi"])
                _vz.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
            mpm_solver.p2g2p(frame, substep_dt, device=device)
"""

p = os.path.join(a.pg, "gs_simulation.py")
s = open(p).read()
if MARK in s:
    print("이미 되어 있다"); raise SystemExit
anc = "    for frame in tqdm(range(frame_num)):"
assert anc in s, "프레임 루프를 못 찾았다"
assert s.count(STEP_OLD) == 1, "서브스텝 호출을 특정 못 했다"
if "import numpy as _np_z" not in s:
    s = s.replace("import numpy as np\n", "import numpy as np\nimport numpy as _np_z\n", 1)
if "from mpm_solver_warp.warp_utils import MPMStateStruct" not in s:
    s = s.replace("from mpm_solver_warp.mpm_solver_warp import MPM_Simulator_WARP",
                  "from mpm_solver_warp.mpm_solver_warp import MPM_Simulator_WARP\n"
                  "from mpm_solver_warp.warp_utils import MPMStateStruct, MPMModelStruct", 1)
s = s.replace('if __name__ == "__main__":', STEP_CODE + '\n\nif __name__ == "__main__":', 1)
s = s.replace(anc, INIT + anc, 1).replace(STEP_OLD, STEP_NEW, 1)
open(p, "w").write(s)
import ast
ast.parse(s)
print(f"고쳤다: {p}")
print("PGSDF_OK")

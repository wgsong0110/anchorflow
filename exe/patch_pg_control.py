"""PhysGaussian 의 `gs_simulation.py` 에 제어점 강제를 넣는다.

GF 쪽(`patch_gf_control.py`)과 같은 규칙이다 -- 표면에서 제어점을 고르고 반경 안
무리를 통째로, **매 서브스텝** 위치·속도를 궤적으로 박고, 한 프레임이 끝난 뒤에도
다시 박는다. `control.npz` 는 h5 옆에 저장한다.

PhysGaussian 은 입자를 재배열하지 않으므로 색인 되돌리기는 필요 없다.

  python exe/patch_pg_control.py --pg /workspace/PhysGaussian
"""
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("--pg", required=True)
a = ap.parse_args()

MARK = "# [anchorflow] 제어점 강제"

INIT = '''
    ''' + MARK + '''
    import json as _json_ctl
    _CC = _json_ctl.load(open(args.config)).get("control", None)
    _ctl = None
    if _CC is not None:
        import numpy as _np
        import sys as _sys
        import torch as _t
        import warp as _wp
        _sys.path.insert(0, os.environ.get("AF_EXE", "/workspace/anchorflow/exe"))
        from control_traj import ControlTraj as _CT, pick_control_points as _pcp
        _X0 = mpm_solver.mpm_state.particle_x.numpy().astype(_np.float64)
        _seed = int(_CC.get("seed", 0))
        _ci, _surf = _pcp(_X0, None, int(_CC.get("n_points", 4)), seed=_seed)
        _ext = float(_np.linalg.norm(_X0.max(0) - _X0.min(0)))
        _rad = float(_CC.get("radius", 0.0)) * _ext
        _mem, _off = [], []
        for _c in _ci:
            _m = (_np.flatnonzero(_np.linalg.norm(_X0 - _X0[_c], axis=1) <= _rad)
                  if _rad > 0 else _np.array([_c]))
            _mem.append(_m); _off.append((_X0[_m] - _X0[_c]).astype(_np.float32))
        _gl = float(material_params.get("grid_lim", 2.0))
        _dxg = _gl / int(material_params["n_grid"])
        _pad = 4.0 * _dxg + _rad
        _tr = _CT(_X0, _ci, dt=float(time_params["frame_dt"]),
                  steps=int(time_params["frame_num"]) + 1,
                  every_n=int(_CC.get("every_n", 8)),
                  p_touch=float(_CC.get("p_touch", 0.7)),
                  depth=float(_CC.get("depth", 0.1)),
                  v_max=float(_CC.get("v_max", 0.5)), seed=_seed,
                  bounds=(_np.full(3, _pad), _np.full(3, _gl - _pad)))
        os.makedirs(directory_to_save, exist_ok=True)
        _np.savez(os.path.join(directory_to_save, "control.npz"),
                  idx=_ci, pos=_tr.P.astype(_np.float32),
                  vel=_tr.V.astype(_np.float32), x0=_X0[_ci].astype(_np.float32),
                  members=_np.concatenate(_mem).astype(_np.int64),
                  member_ptr=_np.cumsum([0] + [len(m) for m in _mem]).astype(_np.int64),
                  cfg=_json_ctl.dumps(_CC))
        _ctl = dict(tr=_tr, mem=[_t.from_numpy(_np.ascontiguousarray(m)).cuda()
                                 for m in _mem],
                    off=[_t.from_numpy(o).cuda() for o in _off])
        print(f"[제어점] {len(_ci)} 개, 반경 {_rad:.4f}, 잡은 입자 "
              f"{[len(m) for m in _mem]}, 최대속도 {_tr.max_speed():.4f}", flush=True)
'''

STEP_OLD = """        for step in range(step_per_frame):
            mpm_solver.p2g2p(frame, substep_dt, device=device)
"""
STEP_NEW = """        _cp0 = _ctl["tr"].pos(frame) if _ctl else None
        _cp1 = _ctl["tr"].pos(frame + 1) if _ctl else None
        _cv = _ctl["tr"].vel(frame + 1) if _ctl else None
        for step in range(step_per_frame):
            if _ctl:
                import torch as _t2
                import warp as _wp2
                _w = (step + 1.0) / step_per_frame
                _c = _t2.from_numpy(
                    (_cp0 * (1.0 - _w) + _cp1 * _w).astype("float32")).cuda()
                _vv = _t2.from_numpy(_cv.astype("float32")).cuda()
                _tx = _wp2.to_torch(mpm_solver.mpm_state.particle_x)
                _tv = _wp2.to_torch(mpm_solver.mpm_state.particle_v)
                for _k in range(len(_ctl["mem"])):
                    _tx[_ctl["mem"][_k]] = _c[_k] + _ctl["off"][_k]
                    _tv[_ctl["mem"][_k]] = _vv[_k]
            mpm_solver.p2g2p(frame, substep_dt, device=device)

        if _ctl:
            import torch as _t3
            import warp as _wp3
            _c = _t3.from_numpy(_cp1.astype("float32")).cuda()
            _vv = _t3.from_numpy(_cv.astype("float32")).cuda()
            for _k in range(len(_ctl["mem"])):
                _wp3.to_torch(mpm_solver.mpm_state.particle_x)[
                    _ctl["mem"][_k]] = _c[_k] + _ctl["off"][_k]
                _wp3.to_torch(mpm_solver.mpm_state.particle_v)[
                    _ctl["mem"][_k]] = _vv[_k]
"""

p = os.path.join(a.pg, "gs_simulation.py")
s = open(p).read()
if MARK in s:
    print("이미 되어 있다"); raise SystemExit
anc = "    for frame in tqdm(range(frame_num)):"
assert anc in s, "프레임 루프를 못 찾았다"
assert STEP_OLD in s, "서브스텝 루프를 못 찾았다"
s = s.replace(anc, INIT + anc, 1).replace(STEP_OLD, STEP_NEW, 1)
open(p, "w").write(s)
import ast
ast.parse(s)
print(f"고쳤다: {p}")
print("PGCONTROL_OK")

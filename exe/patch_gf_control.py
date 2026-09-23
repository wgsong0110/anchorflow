"""3DGS 경로(`gs_simulation.py`)에도 **제어점 강제**를 넣는다.

입자 경로(`run_warp_mpm.py`)에는 제어점을 달았는데 3DGS 경로에는 없었다. 두
경로가 같은 솔버를 쓰므로 강제 방식도 같아야 한다 -- 표면 입자 몇 곳을 골라
반경 안 무리를 통째로, **매 서브스텝** 위치·속도를 궤적으로 박는다.

config 에 `control` 블록이 있을 때만 돈다 (없으면 원래 동작 그대로).

  python exe/patch_gf_control.py --gf <GaussianFluent> [--file <러너>]
"""
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("--gf", required=True)
ap.add_argument("--file", default=None)
a = ap.parse_args()

MARK = "# [anchorflow] 제어점 강제"

INIT = '''
    ''' + MARK + ''': 표면 입자 몇 곳을 골라 궤적으로 박는다
    _CC = _json_ctl.load(open(args.config)).get("control", None)
    _ctl = None
    if _CC is not None:
        import numpy as _np
        import sys as _sys
        _sys.path.insert(0, os.environ.get("AF_EXE", "/workspace/anchorflow/exe"))
        from control_traj import ControlTraj as _CT, pick_control_points as _pcp
        # 솔버는 이미 칸 순서로 재배열되어 있다. h5 와 control.npz 는 **처음
        # 번호**로 적으므로 여기서도 처음 순서로 되돌려 놓고 고른다
        _X0 = mpm_solver.mpm_state.particle_x.numpy().astype(_np.float64)
        _ao0 = getattr(mpm_solver, "af_orig", None)
        if _ao0 is not None:
            _X0 = _X0[_ao0.argsort().cpu().numpy()]
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
        import torch as _t
        _dev = "cuda:0"
        _ctl = dict(tr=_tr, mem=[_t.from_numpy(_np.ascontiguousarray(m)).to(_dev)
                                 for m in _mem],
                    off=[_t.from_numpy(o).to(_dev) for o in _off])
        print(f"[제어점] {len(_ci)} 개, 반경 {_rad:.4f}, 잡은 입자 "
              f"{[len(m) for m in _mem]}, 최대속도 {_tr.max_speed():.4f}", flush=True)
'''

STEP_OLD = """            for step in range(step_per_frame):
                mpm_solver.p2g2p(step, substep_dt, device=device"""
STEP_NEW = """            _mcur = None
            if _ctl:
                # af_sort_by_cell 이 매 프레임 입자를 재배열한다 -- 원래 번호로
                # 잡아둔 무리를 지금 자리로 옮겨야 한다
                import torch as _t1
                _ao = getattr(mpm_solver, "af_orig", None)
                if _ao is None:
                    _mcur = _ctl["mem"]
                else:
                    _inv = _t1.empty_like(_ao)
                    _inv[_ao] = _t1.arange(len(_ao), device=_ao.device)
                    _mcur = [_inv[m] for m in _ctl["mem"]]
            _cp0 = _ctl["tr"].pos(frame) if _ctl else None
            _cp1 = _ctl["tr"].pos(frame + 1) if _ctl else None
            _cv = _ctl["tr"].vel(frame + 1) if _ctl else None
            for step in range(step_per_frame):
                if _ctl:
                    import torch as _t2
                    _w = (step + 1.0) / step_per_frame
                    _c = _t2.from_numpy(
                        (_cp0 * (1.0 - _w) + _cp1 * _w).astype("float32")).to("cuda:0")
                    _vv = _t2.from_numpy(_cv.astype("float32")).to("cuda:0")
                    _tx = wp.to_torch(mpm_solver.mpm_state.particle_x)
                    _tv = wp.to_torch(mpm_solver.mpm_state.particle_v)
                    for _k in range(len(_ctl["mem"])):
                        _tx[_mcur[_k]] = _c[_k] + _ctl["off"][_k]
                        _tv[_mcur[_k]] = _vv[_k]
                mpm_solver.p2g2p(step, substep_dt, device=device"""

TAIL_OLD = """flip_pic=_flip_pic)
"""
files = [a.file] if a.file else [os.path.join(a.gf, "gs_simulation.py")]
for p in files:
    if not os.path.exists(p):
        print(f"[없음] {p}"); continue
    s = open(p).read()
    if MARK in s:
        print(f"이미 되어 있다: {p}"); continue
    if "import json as _json_ctl" not in s:
        s = s.replace("import json\n", "import json\nimport json as _json_ctl\n", 1)
    anc = "    for frame in tqdm(range(frame_num)):"
    if anc not in s:
        print(f"[건너뜀] {p}: 프레임 루프를 못 찾았다"); continue
    s = s.replace(anc, INIT + anc, 1)
    if STEP_OLD not in s:
        print(f"[건너뜀] {p}: 서브스텝 루프를 못 찾았다"); continue
    s = s.replace(STEP_OLD, STEP_NEW, 1)
    # 한 프레임이 끝난 뒤에도 다시 박는다 (g2p 가 옮겨 놓는다)
    s = s.replace("""        if args.output_ply or args.output_h5:""",
                  """            if _ctl:
                import torch as _t3
                _c = _t3.from_numpy(_cp1.astype("float32")).to("cuda:0")
                _vv = _t3.from_numpy(_cv.astype("float32")).to("cuda:0")
                for _k in range(len(_ctl["mem"])):
                    wp.to_torch(mpm_solver.mpm_state.particle_x)[
                        _mcur[_k]] = _c[_k] + _ctl["off"][_k]
                    wp.to_torch(mpm_solver.mpm_state.particle_v)[
                        _mcur[_k]] = _vv[_k]

        if args.output_ply or args.output_h5:""", 1)
    if "import warp as wp" not in s:
        s = s.replace("import json\n", "import json\nimport warp as wp\n", 1)
    open(p, "w").write(s)
    print(f"고쳤다: {p}")
print("GFCONTROL_OK")

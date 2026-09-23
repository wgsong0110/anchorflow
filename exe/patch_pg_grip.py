"""PhysGaussian 에 **평행 집게(그리퍼) 강제**를 넣는다.

config 에 `grips` 블록이 있으면 양 끝을 집게로 물고 SE(3) 로 움직인다 (제어점
블록과 따로 돌고, 둘 다 없으면 원래 동작 그대로다).

    "grips": {"jaw":0.12, "pad_w":0.35, "pad_d":0.18, "grab":0.12,
              "speed":0.15, "twist":0.0, "n_grips":2}

  python exe/patch_pg_grip.py --pg /workspace/PhysGaussian
"""
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("--pg", required=True)
a = ap.parse_args()

MARK = "# [anchorflow] 평행 집게 강제"

INIT = '''
    ''' + MARK + '''
    import json as _json_g
    _GG = _json_g.load(open(args.config)).get("grips", None)
    _grp = None
    if _GG is not None:
        import numpy as _npg
        import sys as _sysg
        import torch as _tg
        _sysg.path.insert(0, os.environ.get("AF_EXE", "/workspace/anchorflow/exe"))
        from grip_traj import pick_pinch_grips as _ppg, tear_traj as _tt
        _Xg = mpm_solver.mpm_state.particle_x.numpy().astype(_npg.float64)
        _gs, _ax, _ext = _ppg(_Xg, jaw=float(_GG.get("jaw", 0.12)),
                              pad_w=float(_GG.get("pad_w", 0.35)),
                              pad_d=float(_GG.get("pad_d", 0.18)),
                              grab=float(_GG.get("grab", 0.12)),
                              n_grips=int(_GG.get("n_grips", 2)))
        _P, _R, _V, _W = _tt(_gs, _ax, _ext, int(time_params["frame_num"]) + 1,
                             float(time_params["frame_dt"]),
                             speed=float(_GG.get("speed", 0.15)),
                             twist=float(_GG.get("twist", 0.0)))
        os.makedirs(directory_to_save, exist_ok=True)
        _mem = [g["members"] for g in _gs]
        _npg.savez(os.path.join(directory_to_save, "grips.npz"),
                   pos=_P.astype(_npg.float32), rot=_R.astype(_npg.float32),
                   vel=_V.astype(_npg.float32), omega=_W.astype(_npg.float32),
                   c0=_npg.stack([g["c"] for g in _gs]).astype(_npg.float32),
                   R0=_npg.stack([g["R"] for g in _gs]).astype(_npg.float32),
                   members=_npg.concatenate(_mem).astype(_npg.int64),
                   member_ptr=_npg.cumsum([0] + [len(m) for m in _mem]).astype(_npg.int64),
                   cfg=_json_g.dumps(_GG))
        # 렌더러가 잡힌 가우시안을 빨갛게 칠할 수 있게 같은 이름으로도 남긴다
        _npg.savez(os.path.join(directory_to_save, "control.npz"),
                   idx=_npg.array([int(m[0]) for m in _mem]),
                   pos=_P.astype(_npg.float32), vel=_V.astype(_npg.float32),
                   members=_npg.concatenate(_mem).astype(_npg.int64),
                   member_ptr=_npg.cumsum([0] + [len(m) for m in _mem]).astype(_npg.int64),
                   cfg=_json_g.dumps(_GG))
        _grp = dict(P=_P, R=_R, V=_V, W=_W,
                    mem=[_tg.from_numpy(_npg.ascontiguousarray(m)).cuda() for m in _mem],
                    off=[_tg.from_numpy(g["off"].astype("float32")).cuda() for g in _gs])
        print(f"[집게] {len(_gs)} 개, 잡은 입자 {[len(m) for m in _mem]}, "
              f"당김 {_GG.get('speed',0.15)} 지름/s, 비틀기 "
              f"{_GG.get('twist',0.0)} rad/s", flush=True)
'''

STEP_OLD = """            mpm_solver.p2g2p(frame, substep_dt, device=device)
"""
STEP_NEW = """            if _grp:
                import torch as _tg2
                import warp as _wpg
                _wf = (step + 1.0) / step_per_frame
                _f0, _f1 = frame, min(frame + 1, _grp["P"].shape[0] - 1)
                _tx = _wpg.to_torch(mpm_solver.mpm_state.particle_x)
                _tv = _wpg.to_torch(mpm_solver.mpm_state.particle_v)
                for _k in range(len(_grp["mem"])):
                    # 위치는 프레임 사이를 선형으로, 자세는 두 회전 사이를 그대로
                    # 섞지 않고 가까운 쪽(작은 각)이라 선형 보간 후 정규직교화한다
                    _Pm = (_grp["P"][_f0, _k] * (1 - _wf) + _grp["P"][_f1, _k] * _wf)
                    _Rm = (_grp["R"][_f0, _k] * (1 - _wf) + _grp["R"][_f1, _k] * _wf)
                    _u, _s, _vh = _np_g.linalg.svd(_Rm)
                    _Rm = _u @ _vh
                    _Rt = _tg2.from_numpy(_Rm.astype("float32")).cuda()
                    _Pt = _tg2.from_numpy(_Pm.astype("float32")).cuda()
                    _loc = _grp["off"][_k] @ _Rt.T
                    _tx[_grp["mem"][_k]] = _Pt + _loc
                    _w = _tg2.from_numpy(_grp["W"][_f1, _k].astype("float32")).cuda()
                    _tv[_grp["mem"][_k]] = (
                        _tg2.from_numpy(_grp["V"][_f1, _k].astype("float32")).cuda()
                        + _tg2.cross(_w.expand_as(_loc), _loc, dim=-1))
            mpm_solver.p2g2p(frame, substep_dt, device=device)
"""

p = os.path.join(a.pg, "gs_simulation.py")
s = open(p).read()
if MARK in s:
    print("이미 되어 있다"); raise SystemExit
anc = "    for frame in tqdm(range(frame_num)):"
assert anc in s, "프레임 루프를 못 찾았다"
assert s.count(STEP_OLD) == 1, "서브스텝 호출을 하나로 특정 못 했다"
if "import numpy as _np_g" not in s:
    s = s.replace("import numpy as np\n", "import numpy as np\nimport numpy as _np_g\n", 1)
s = s.replace(anc, INIT + anc, 1).replace(STEP_OLD, STEP_NEW, 1)
open(p, "w").write(s)
import ast
ast.parse(s)
print(f"고쳤다: {p}")
print("PGGRIP_OK")

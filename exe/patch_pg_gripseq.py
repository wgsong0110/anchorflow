"""PhysGaussian 에 **집게로 집었다 놓기를 반복하는 조작**을 넣는다 (팔 여러 개).

한 회차는 **접근 -> 집기 -> 물고 이동 -> 놓기 -> 다음 자리로 이동** 이다. 놓는
순간 다음에 잡을 자리를 그 시점 모양에서 골라 두고 쉬는 동안 그 자리로 **이어서**
움직이므로 집게가 순간이동하지 않는다.

집게는 잡은 덩어리를 강제로 끌고 갈 뿐 아니라 **판 두 장이 충돌체**로도 동작해서,
잡지 않은 입자를 뚫고 지나가지 않는다.

집게 크기는 **처음 물체 크기**로 고정한다 (도구는 물체가 늘어난다고 커지지 않는다).

    "grip_seq": {"n":3, "arms":2, "seed":0, "approach":10, "close":8,
                 "hold":60, "rest":24, "squeeze":0.25, "jaw":0.10,
                 "pad_w":0.22, "pad_d":0.22, "speed":[1.0,1.8], "twist":1.5,
                 "collide": true}

프레임마다 집게 자세를 `gripseq_pose.npy` 로 남기므로, 렌더는 그 값을 그대로
읽어 그리면 된다 (시뮬과 렌더가 어긋날 일이 없다).

  python exe/patch_pg_gripseq.py --pg /workspace/PhysGaussian_seq
"""
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("--pg", required=True)
a = ap.parse_args()

MARK = "# [anchorflow] 집었다 놓기 반복"

INIT = '''
    ''' + MARK + '''
    import json as _json_s
    _SQ = _json_s.load(open(args.config)).get("grip_seq", None)
    _sq = None
    if _SQ is not None:
        import numpy as _nps
        import sys as _syss
        import torch as _ts
        _syss.path.insert(0, os.environ.get("AF_EXE", "/workspace/anchorflow/exe"))
        from control_traj import estimate_normals as _en, surface_mask as _sm
        from grip_traj import pick_pad_at as _ppa, random_moves as _rm
        _nar = int(_SQ.get("arms", 2))
        _nep = int(_SQ.get("n", 3))
        _rng = _nps.random.default_rng(int(_SQ.get("seed", 0)))
        _spd = _SQ.get("speed", [1.0, 1.8])
        _mv = _rm(_rng, _nep * _nar, float(_spd[0]), float(_spd[1]),
                  float(_SQ.get("twist", 1.5)))
        _X00 = mpm_solver.mpm_state.particle_x.numpy().astype(_nps.float64)
        _EXT0 = float(_nps.linalg.norm(_X00.max(0) - _X00.min(0)))
        _GL = float(material_params.get("grid_lim", 2.0))
        _PAD = 5.0 * _GL / int(material_params["n_grid"])
        _sq = dict(mv=_mv, arms=_nar, n=_nep, ext=_EXT0,
                   ap=int(_SQ.get("approach", 10)), close=int(_SQ.get("close", 8)),
                   hold=int(_SQ.get("hold", 60)), rest=int(_SQ.get("rest", 24)),
                   squeeze=float(_SQ.get("squeeze", 0.25)),
                   jaw=float(_SQ.get("jaw", 0.10)),
                   pad_w=float(_SQ.get("pad_w", 0.22)),
                   pad_d=float(_SQ.get("pad_d", 0.22)),
                   collide=bool(_SQ.get("collide", True)),
                   lo=_PAD, hi=_GL - _PAD, cur=None, nxt=None, pose=[], log=[],
                   floor=max([b.get("point", [0, 0, -1e9])[2]
                              for b in bc_params
                              if b.get("type") == "surface_collider"
                              and abs(b.get("normal", [0, 0, 1])[2]) > 0.5]
                             or [-1e9]))
        os.makedirs(directory_to_save, exist_ok=True)

        def _pick_targets(_ei):
            """지금 모양의 표면에서 팔 수만큼, 서로 멀리 떨어진 자리를 고른다."""
            _Xn = mpm_solver.mpm_state.particle_x.numpy().astype(_nps.float64)
            _ok = _nps.isfinite(_Xn).all(1)
            if not _ok.all():          # 터진 입자는 무게중심으로 모아 둔다
                _Xn[~_ok] = _Xn[_ok].mean(0) if _ok.any() else 1.0
            _nrm = _en(_Xn, k=16)
            _srf = _nps.flatnonzero(_sm(_Xn) & _ok)
            _r2 = _nps.random.default_rng(1000 + _ei)
            _pick = [int(_srf[_r2.integers(len(_srf))])]
            while len(_pick) < _sq["arms"]:
                _d = _nps.min(_nps.linalg.norm(
                    _Xn[_srf][:, None, :] - _Xn[_pick][None, :, :], axis=-1), 1)
                _pick.append(int(_srf[int(_nps.argmax(_d))]))
            _out = []
            for _ai, _c in enumerate(_pick):
                _m, _R0 = _ppa(_Xn, _Xn[_c], _nrm[_c], jaw=_sq["jaw"],
                               pad_w=_sq["pad_w"], pad_d=_sq["pad_d"],
                               ext=_sq["ext"])
                _mvk = _sq["mv"][min(_ei * _sq["arms"] + _ai, len(_sq["mv"]) - 1)]
                _out.append(dict(m=_ts.from_numpy(_nps.ascontiguousarray(_m)).cuda(),
                                 offL=(_Xn[_m] - _Xn[_c]) @ _R0.T, R0=_R0.copy(),
                                 c0=_Xn[_c].copy(), n=len(_m), **_mvk))
            return _out

        _sq["pick"] = _pick_targets
        print(f"[반복 집기] {_nep} 번 x 팔 {_nar} 개 -- 접근 {_sq['ap']} / 집기 "
              f"{_sq['close']} / 이동 {_sq['hold']} / 놓고 이동 {_sq['rest']} "
              f"프레임, 무는 깊이 {_sq['squeeze']:.2f}, 충돌체 {_sq['collide']}",
              flush=True)
'''

FRAME_HEAD = '''        if _sq is not None:
            _per = _sq["ap"] + _sq["close"] + _sq["hold"] + _sq["rest"]
            _ei, _pi = frame // _per, frame % _per
            if _pi == 0 and _ei < _sq["n"]:
                # 다음 자리는 앞 회차를 놓을 때 이미 골라 뒀다 (이어서 가려고)
                _sq["cur"] = _sq["nxt"] if _sq["nxt"] is not None else _sq["pick"](_ei)
                _sq["nxt"] = None
                for _ai, _g in enumerate(_sq["cur"]):
                    _sq["log"].append(dict(ep=_ei, arm=_ai, frame=frame,
                                           center=_g["c0"].tolist(), n=int(_g["n"]),
                                           dir=_np_s.asarray(_g["dir"]).tolist(),
                                           speed=float(_g["speed"]),
                                           twist=float(_g["twist"]),
                                           R0=_g["R0"].tolist()))
                    print(f"  [{_ei+1}번째 / 팔 {_ai+1}] 프레임 {frame}, 입자 "
                          f"{_g['n']} 개, 방향 "
                          f"{_np_s.round(_np_s.asarray(_g['dir']),2)}, 속도 "
                          f"{_g['speed']:.2f} 지름/s, 비틀기 {_g['twist']:+.2f}",
                          flush=True)
            if _pi == _sq["ap"] + _sq["close"] + _sq["hold"] and _ei + 1 < _sq["n"]:
                _sq["nxt"] = _sq["pick"](_ei + 1)     # 놓는 순간 다음 자리 고르기
'''


def _pose_code():
    return '''
def _af_zclamp(sq, c, ext):
    """집게가 지표면 아래로 못 가게. 판 반경까지 고려해 중심을 들어 올린다."""
    if sq["floor"] < -1e8:
        return c
    r = ext * (max(sq["pad_w"], sq["pad_d"]) + sq["jaw"] + 0.035)
    c = c.copy()
    c[2] = max(float(c[2]), sq["floor"] + r)
    return c


def _af_pose(sq, frame, sub, dt, ext):
    """프레임(+서브스텝 진행도)에서 팔마다 (중심, 회전, 벌어짐, 속도, 각속도)."""
    import numpy as np
    per = sq["ap"] + sq["close"] + sq["hold"] + sq["rest"]
    ei, pi = frame // per, (frame % per) + sub
    out = []
    cur = sq["cur"]
    if cur is None:
        return out
    for k, g in enumerate(cur):
        u = np.asarray(g["dir"]); R0 = g["R0"]; c0 = g["c0"]
        if pi < sq["ap"]:                              # 접근 (벌린 채)
            w = pi / max(sq["ap"], 1)
            c = c0 - R0[0] * (0.55 * ext) * (1 - w)
            out.append((_af_zclamp(sq, c, ext), R0, 1.0, u * 0.0, u * 0.0, 0.0))
            continue
        if pi < sq["ap"] + sq["close"]:                # 집기
            w = (pi - sq["ap"]) / max(sq["close"], 1)
            out.append((c0, R0, 1.0 - w, u * 0.0, u * 0.0, w))
            continue
        tm = max(pi - sq["ap"] - sq["close"], 0.0) * dt
        tm_max = sq["hold"] * dt
        th = float(g["twist"]) * min(tm, tm_max)
        K = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
        R = (np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)) @ R0
        c = c0 + u * (float(g["speed"]) * ext * min(tm, tm_max))
        c = _af_zclamp(sq, np.clip(c, sq["lo"], sq["hi"]), ext)
        if pi < sq["ap"] + sq["close"] + sq["hold"]:   # 물고 이동
            out.append((c, R, 0.0, u * (float(g["speed"]) * ext),
                        u * float(g["twist"]), 1.0))
            continue
        # 놓고 다음 자리로 이어서 이동 (벌린 채)
        w = (pi - sq["ap"] - sq["close"] - sq["hold"]) / max(sq["rest"], 1)
        nx = sq["nxt"][k] if sq["nxt"] is not None else None
        if nx is None:
            out.append((_af_zclamp(sq, c - R[0] * (0.6 * ext) * min(w * 2, 1.0),
                                   ext), R, min(1.0, w * 3), u * 0.0, u * 0.0, 0.0))
            continue
        c1, R1 = nx["c0"] - nx["R0"][0] * (0.55 * ext), nx["R0"]
        lift = np.sin(np.pi * w) * 0.30 * ext
        cc = (1 - w) * c + w * c1 - ((1 - w) * R[0] + w * R1[0]) * lift
        M = (1 - w) * R + w * R1
        try:
            uu, _, vh = np.linalg.svd(M)
            Rn = uu @ vh
        except np.linalg.LinAlgError:
            Rn = R
        out.append((_af_zclamp(sq, cc, ext), Rn, 1.0, u * 0.0, u * 0.0, 0.0))
    return out


def _af_collide(state, sq, poses, ext, dt, held):
    """집게 판 두 장을 충돌체로. 판 안에 들어온 입자를 밖으로 밀어낸다."""
    import torch as T
    import warp as wp
    x = wp.to_torch(state.particle_x)
    v = wp.to_torch(state.particle_v)
    th = 0.035 * ext
    for (c, R, op, vel, om, _cl), g in zip(poses, sq["cur"]):
        gap = sq["jaw"] * ext * (1.0 + 1.8 * op)
        Rt = T.from_numpy(R.astype("float32")).cuda()
        ct = T.from_numpy(c.astype("float32")).cuda()
        vt = T.from_numpy(vel.astype("float32")).cuda()
        loc = (x - ct) @ Rt.T                       # 집게 좌표계
        side = T.sign(loc[:, 0])
        face = side * (gap + th)                    # 가까운 판의 중심면
        d = loc[:, 0] - face
        inside = ((d.abs() < th)
                  & (loc[:, 1].abs() < sq["pad_w"] * ext)
                  & (loc[:, 2].abs() < sq["pad_d"] * ext))
        if held is not None:
            inside = inside & held
        if not bool(inside.any()):
            continue
        push = side[inside] * (th - d[inside].abs() * T.sign(d[inside]) * 0)
        loc_new = loc[inside].clone()
        loc_new[:, 0] = face[inside] + T.sign(d[inside]) * th
        x[inside] = ct + loc_new @ Rt
        vn = (v[inside] - vt) @ Rt.T
        vn[:, 0] = 0.0                              # 판을 파고드는 성분 제거
        v[inside] = vt + vn @ Rt
'''


STEP_OLD = """        for step in range(step_per_frame):
            mpm_solver.p2g2p(frame, substep_dt, device=device)
"""
STEP_NEW = FRAME_HEAD + """        for step in range(step_per_frame):
            if _sq is not None and _sq["cur"]:
                import torch as _ts2
                import warp as _wps2
                _sub = (step + 1.0) / step_per_frame
                _ps = _af_pose(_sq, frame, _sub, frame_dt, _sq["ext"])
                _hold_mask = None
                for _gi, (_c, _R, _op, _vel, _om, _cl) in enumerate(_ps):
                    _g = _sq["cur"][_gi]
                    if _cl <= 0.0:
                        continue                      # 아직 안 물었거나 이미 놓았다
                    _oL = _g["offL"].copy()
                    _oL[:, 0] *= 1.0 - _sq["squeeze"] * min(_cl, 1.0)
                    _world = _oL @ _R
                    _xt = _ts2.from_numpy((_c[None, :] + _world).astype("float32")).cuda()
                    _wv = _ts2.from_numpy(_world.astype("float32")).cuda()
                    _omt = _ts2.from_numpy(_om.astype("float32")).cuda()
                    _vvt = _ts2.from_numpy(_vel.astype("float32")).cuda()
                    _wps2.to_torch(mpm_solver.mpm_state.particle_x)[_g["m"]] = _xt
                    _wps2.to_torch(mpm_solver.mpm_state.particle_v)[_g["m"]] = (
                        _vvt + _ts2.cross(_omt.expand_as(_wv), _wv, dim=-1))
                if _sq["collide"] and _ps:
                    _free = _ts2.ones(mpm_solver.n_particles, dtype=_ts2.bool,
                                      device="cuda")
                    for _gi, (_c, _R, _op, _vel, _om, _cl) in enumerate(_ps):
                        if _cl > 0.0:
                            _free[_sq["cur"][_gi]["m"]] = False
                    _af_collide(mpm_solver.mpm_state, _sq, _ps, _sq["ext"],
                                substep_dt, _free)
                # 안전망: 어떤 이유로든 입자가 격자를 벗어나면 p2g 가 남의 메모리를
                # 건드려 CUDA 700 으로 죽는다 (두 번 겪었다). 매 서브스텝 잘라 둔다.
                _xx = _wps2.to_torch(mpm_solver.mpm_state.particle_x)
                _xx.nan_to_num_(nan=1.0, posinf=1.0, neginf=1.0)
                _xx.clamp_(_sq["lo"], _sq["hi"])
                _vv2 = _wps2.to_torch(mpm_solver.mpm_state.particle_v)
                _vv2.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
                if step == step_per_frame - 1:
                    _sq["pose"].append([(c.tolist(), R.tolist(), float(op))
                                        for (c, R, op, _v, _o, _cl) in _ps])
                    _np_s.save(os.path.join(directory_to_save, "gripseq_pose.npy"),
                               _np_s.array(_sq["pose"], dtype=object),
                               allow_pickle=True)
                    _np_s.save(os.path.join(directory_to_save, "gripseq.npy"),
                               _np_s.array(_sq["log"], dtype=object),
                               allow_pickle=True)
            mpm_solver.p2g2p(frame, substep_dt, device=device)
"""

p = os.path.join(a.pg, "gs_simulation.py")
s = open(p).read()
if MARK in s:
    print("이미 되어 있다"); raise SystemExit
anc = "    for frame in tqdm(range(frame_num)):"
assert anc in s and s.count(STEP_OLD) == 1
if "import numpy as _np_s" not in s:
    s = s.replace("import numpy as np\n", "import numpy as np\nimport numpy as _np_s\n", 1)
s = s.replace('if __name__ == "__main__":', _pose_code() + '\n\nif __name__ == "__main__":', 1)
s = s.replace(anc, INIT + anc, 1).replace(STEP_OLD, STEP_NEW, 1)
open(p, "w").write(s)
import ast
ast.parse(s)
print(f"고쳤다: {p}")
print("PGGRIPSEQ_OK")

"""PG 클론에 `AF_H_SCEN` (시나리오 파일 구동) 을 넣는다. 여러 번 돌려도 안전하다.

시나리오 파일이 제어 입자 색인과 프레임별 지령 속도를 들고 있으므로, PG 는
컨트롤러를 계산하지 않고 그대로 읽어 쓴다 -- 다른 솔버와 구동이 글자 그대로 같다.

  python exe/patch_pg_scen.py --pg /home/dkta/work/PG_pgtraj
"""
import argparse
import os
import re

ap = argparse.ArgumentParser()
ap.add_argument("--pg", required=True)
a = ap.parse_args()
p = os.path.join(a.pg, "gs_simulation.py")
s = open(p).read()
if "AF_H_SCEN" in s:
    print("[패치] 이미 들어 있다")
    raise SystemExit(0)

# 1) 시나리오 적재
old = '        _af_hid_t = None\n'
new = ('        # [anchorflow] 시나리오 파일 구동. 제어 입자와 프레임별 지령 속도를\n'
       '        # 파일에서 읽어 그대로 쓴다 (모든 솔버가 같은 구동을 받게 한다).\n'
       '        _af_scen = None\n'
       '        if _af_os.environ.get("AF_H_SCEN"):\n'
       '            _af_scen = _af_np.load(_af_os.environ["AF_H_SCEN"])\n'
       '            _af_N = int(_af_scen["hid"].shape[0])\n'
       '            print(f"[손잡이] 시나리오 {_af_os.environ[\'AF_H_SCEN\']} "\n'
       '                  f"손잡이 {_af_N} 개, {_af_scen[\'vel\'].shape[0]} 프레임",\n'
       '                  flush=True)\n'
       '        _af_hid_t = None\n')
assert old in s
s = s.replace(old, new, 1)

# 2) 제어 입자를 파일에서 받는다
old = """            _Rtry = _af_R
            while True:"""
new = """            _Rtry = _af_R
            if _af_scen is not None:
                _pick = [int(q) for q in _af_scen["hid"]]
                _af_Rbox[0] = _Rtry
                _tries = 0
            while _af_scen is None:"""
assert old in s
s = s.replace(old, new, 1)

# 3) 지령 속도를 파일에서 받는다
old = """            if _af_kin and _af_mode == "randpt":
                _tt = (frame % _af_rf) * frame_dt"""
new = """            if _af_scen is not None:
                _vs = _af_scen["vel"]
                _af_vpl = torch.tensor(
                    _vs[min(frame, _vs.shape[0] - 1)], dtype=torch.float32,
                    device=mpm_solver.export_particle_x_to_torch().device)
                _af_hist["hid"].append(_af_hid_t.detach().cpu().numpy().copy())
                _af_hist["hpos"].append(
                    mpm_solver.export_particle_x_to_torch()[_af_hid_t]
                    .detach().cpu().numpy().copy())
                _af_hist["hvel"].append(_af_vpl.detach().cpu().numpy().copy())
                _af_hist["R"].append(float(_af_Rbox[0]))
            elif _af_kin and _af_mode == "randpt":
                _tt = (frame % _af_rf) * frame_dt"""
assert old in s
s = s.replace(old, new, 1)
open(p, "w").write(s)
print("[패치] AF_H_SCEN 추가 완료")

"""i-PhysGaussian 클론에 **같은 채우기**와 **시나리오 구동 손잡이**를 넣는다.

벤치가 성립하려면 입자 집합이 PG 와 글자 그대로 같아야 한다 (시나리오의 제어
입자 색인이 그 집합을 가리킨다). 손잡이는 PG 와 같은 규약으로 적용한다:

    v_p <- (1 - w_p) v_p + w_p v_cmd,   w = (1 - q^2)^2,  q = |x_p - c| / R

여러 번 돌려도 안전하다.

  python exe/patch_ipg_scen.py --ipg /root/work/i-physgaussian
"""
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("--ipg", required=True)
a = ap.parse_args()
p = os.path.join(a.ipg, "gs_simulation.py")
s = open(p).read()
if "AF_H_SCEN" in s:
    print("[패치] 이미 들어 있다")
    raise SystemExit(0)

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

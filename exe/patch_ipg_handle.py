"""i-PG 의 손잡이를 **하드 Dirichlet 격자 BC** 로 바꾼다.

전에는 스텝 전에 입자 속도를 감쇠 가중으로 섞었다(`patch_ipg_scen.py`). implicit
솔버에서는 그게 지켜지지 않는다 -- `bc_mask` 는 **`grid_postprocess` 가 바꾼
노드**에서만 잡히므로(implicit_mpm_solver.py 의 "Identify Dirichlet (BC) nodes"),
입자 쪽에서만 건드린 값은 Newton 반복이 그대로 풀어 버린다.

그래서 손잡이를 구면 Dirichlet 콜라이더로 등록한다:

    |x_node - c_j| < R  ->  grid_v_out[node] = v_cmd_j        (하드)

c_j 는 그 손잡이 입자의 **현재 위치**(PG 규약과 같다), v_cmd_j 는 시나리오 명령
속도다. 둘 다 프레임마다 호스트에서 콜라이더 파라미터에 써 넣는다.

여러 번 돌려도 안전하다.

  python exe/patch_ipg_handle.py --ipg /home/dkta/work/i-physgaussian
"""
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("--ipg", required=True)
a = ap.parse_args()

# ---------------------------------------------------------------- 1) 솔버 API
sp = os.path.join(a.ipg, "mpm_solver_warp", "mpm_solver_warp.py")
s = open(sp).read()
if "add_handle_sphere" in s:
    print("[패치] 솔버에 이미 들어 있다")
else:
    anchor = "    def add_bounding_box(self"
    assert anchor in s, "add_bounding_box 를 못 찾았다"
    new = '''    def add_handle_sphere(self, point, velocity, radius,
                          start_time=0.0, end_time=999.0):
        """움직이는 구면 **하드 Dirichlet**. 반경 안 격자 속도를 명령값으로 박는다.

        grid_postprocess 에 들어가므로 implicit 솔버의 bc_mask 가 이 노드를 잡고,
        Newton 잔차·탐색방향에서 0 이 되어 실제로 구속된다. 프레임마다 호스트에서
        `collider_params[k].point / .velocity` 를 갱신해 손잡이를 움직인다.
        """
        collider_param = Dirichlet_collider()
        collider_param.start_time = start_time
        collider_param.end_time = end_time
        collider_param.point = wp.vec3(float(point[0]), float(point[1]),
                                       float(point[2]))
        collider_param.velocity = wp.vec3(float(velocity[0]), float(velocity[1]),
                                          float(velocity[2]))
        collider_param.radius = float(radius)
        collider_param.surface_type = 0
        collider_param.friction = 0.0
        self.collider_params.append(collider_param)

        @wp.kernel
        def collide(
            time: float,
            dt: float,
            state: MPMStateStruct,
            model: MPMModelStruct,
            param: Dirichlet_collider,
        ):
            grid_x, grid_y, grid_z = wp.tid()
            if time >= param.start_time and time < param.end_time:
                offset = wp.vec3(
                    float(grid_x) * model.dx - param.point[0],
                    float(grid_y) * model.dx - param.point[1],
                    float(grid_z) * model.dx - param.point[2],
                )
                if wp.length(offset) < param.radius:
                    state.grid_v_out[grid_x, grid_y, grid_z] = wp.vec3(
                        param.velocity[0], param.velocity[1], param.velocity[2]
                    )

        self.grid_postprocess.append(collide)
        self.modify_bc.append(None)

'''
    s = s.replace(anchor, new + anchor, 1)
    open(sp, "w").write(s)
    print(f"[패치] add_handle_sphere 추가 -> {sp}")

# ------------------------------------------------------- 2) gs_simulation 연결
gp = os.path.join(a.ipg, "gs_simulation.py")
g = open(gp).read()
if "AF_H_HARD" in g:
    print("[패치] gs_simulation 에 이미 들어 있다")
    raise SystemExit(0)

# 예전 소프트 혼합 블록을 찾아 하드 BC 등록·갱신으로 교체한다.
old_key = "    _af_scen = None\n"
assert old_key in g, "patch_ipg_scen 의 블록을 못 찾았다 (먼저 그것을 적용할 것)"
# 소프트 혼합(프레임 루프 안의 _vr 조작)을 제거한다
import re
g2 = re.sub(
    r"\n        if _af_scen is not None:\n"
    r"(?:            .*\n)+?(?=        [^ ])",
    "\n", g, count=1)
assert g2 != g, "프레임 루프 안 소프트 혼합 블록을 못 찾았다"
g = g2
# 손잡이 소속·가중 계산도 필요 없다. 등록 + 프레임 갱신으로 바꾼다.
old_reg = g[g.index(old_key):g.index("    for frame in tqdm(range(frame_num)):")]
new_reg = '''    _af_scen = None
    if _af_os.environ.get('AF_H_SCEN'):
        # 하드 Dirichlet: 손잡이 입자 위치를 중심으로 반경 R 격자 속도를 박는다.
        _af_scen = _af_np.load(_af_os.environ['AF_H_SCEN'])
        _af_R = float(_af_os.environ.get('AF_H_R', 0.15))
        _af_hid = _af_t.as_tensor(_af_scen['hid'], dtype=_af_t.long)
        _af_vel = _af_scen['vel']
        _x0 = mpm_solver.export_particle_x_to_torch()
        _af_k0 = len(mpm_solver.collider_params)
        for _j in range(_af_hid.numel()):
            _c = _x0[_af_hid[_j]].tolist()
            mpm_solver.add_handle_sphere(_c, [0.0, 0.0, 0.0], _af_R)
        print(f'[손잡이-하드] {_af_hid.numel()} 개, R={_af_R}, '
              f'콜라이더 {_af_k0}..{_af_k0 + _af_hid.numel() - 1}, '
              f'시나리오 {_af_vel.shape[0]} 프레임  AF_H_HARD', flush=True)
'''
g = g.replace(old_reg, new_reg, 1)
# 프레임 루프 머리에 파라미터 갱신을 넣는다
old_loop = "    for frame in tqdm(range(frame_num)):\n"
new_loop = old_loop + '''        if _af_scen is not None:
            # 손잡이 중심은 **그 입자의 현재 위치**(PG 규약), 속도는 명령값
            _xc = mpm_solver.export_particle_x_to_torch()
            _vc = _af_vel[min(frame, _af_vel.shape[0] - 1)]
            for _j in range(_af_hid.numel()):
                _p = _xc[_af_hid[_j]].tolist()
                _cp = mpm_solver.collider_params[_af_k0 + _j]
                _cp.point = wp.vec3(float(_p[0]), float(_p[1]), float(_p[2]))
                _cp.velocity = wp.vec3(float(_vc[_j][0]), float(_vc[_j][1]),
                                       float(_vc[_j][2]))
'''
assert old_loop in g
g = g.replace(old_loop, new_loop, 1)
if "import warp as wp" not in g:
    g = g.replace("import torch", "import torch\nimport warp as wp", 1)
open(gp, "w").write(g)
print(f"[패치] gs_simulation 에 하드 손잡이 등록·갱신 추가 -> {gp}")

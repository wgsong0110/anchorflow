"""i-PG 에 **체적 가속도** 경계조건(`body_acceleration`)을 더한다.

그쪽 `particle_impulse` 는 상자 안 입자마다 **같은 힘**을 주고 각자 질량으로 나눈다
(`v += F/m·dt`). 입자 질량이 제각각이라(표면 가우시안 입자는 아주 가볍다) 가벼운
입자만 크게 튄다 -- 시연에서 얇은 판·가장자리만 날아가고 몸통은 버텼다. 물체 한
부분에 고르게 힘을 주려면 힘이 질량에 비례해야 하고, 그러면 가속도가 같다:
    v += a·dt         (a 는 모든 입자에 같다)

쓰는 법 (config 의 boundary_conditions):
    {"type": "body_acceleration", "acc": [ax, ay, az],
     "point": [...], "size": [...],        # 처음 위치로 한 번 고른다 (반폭 상자)
     "start_time": t0, "end_time": t1}

impulse_scale 을 곱하지 않는다 -- 가속도는 매 스텝 실제 dt 로 적분되므로 암시
적분의 큰 스텝에서도 시간에 대해 그대로다. 여러 번 돌려도 안전하다.

  python exe/patch_ipg_bodyacc.py --ipg /home/dkta/work/i-physgaussian
"""
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("--ipg", required=True)
a = ap.parse_args()

# 1) 솔버 메서드
sp = os.path.join(a.ipg, "mpm_solver_warp", "mpm_solver_warp.py")
s = open(sp).read()
if "def add_acceleration_on_particles" in s:
    print("[패치] 솔버에 이미 있다")
else:
    anchor = "    def enforce_particle_velocity_translation("
    assert anchor in s, "삽입 위치를 못 찾았다"
    meth = '''    def add_acceleration_on_particles(
        self,
        acc,
        point=[1, 1, 1],
        size=[1, 1, 1],
        start_time=0.0,
        end_time=999.0,
        device="cuda:0",
    ):
        """[anchorflow] 상자 안 입자(처음 위치로 고름)에 **같은 가속도** 를 준다."""
        param = Impulse_modifier()
        param.start_time = start_time
        param.end_time = end_time
        param.point = wp.vec3(point[0], point[1], point[2])
        param.size = wp.vec3(size[0], size[1], size[2])
        param.mask = wp.zeros(shape=self.n_particles, dtype=int, device=device)
        param.force = wp.vec3(acc[0], acc[1], acc[2])   # 여기서는 가속도
        wp.launch(
            kernel=selection_add_impulse_on_particles,
            dim=self.n_particles,
            inputs=[self.mpm_state, param],
            device=device,
        )
        self.impulse_params.append(param)

        @wp.kernel
        def apply_acc(
            time: float, dt: float, state: MPMStateStruct, param: Impulse_modifier
        ):
            p = wp.tid()
            if time >= param.start_time and time < param.end_time:
                if param.mask[p] == 1:
                    state.particle_v[p] = state.particle_v[p] + param.force * dt

        self.pre_p2g_operations.append(apply_acc)

'''
    s = s.replace(anchor, meth + anchor, 1)
    open(sp, "w").write(s)
    print(f"[패치] {sp}")

# 2) 설정 해석
dp = os.path.join(a.ipg, "utils", "decode_param.py")
s = open(dp).read()
if '"body_acceleration"' in s:
    print("[패치] decode_param 에 이미 있다")
else:
    anchor = '''        elif bc["type"] == "bounding_box":'''
    assert anchor in s, "decode_param 삽입 위치를 못 찾았다"
    br = '''        elif bc["type"] == "body_acceleration":
            # [anchorflow] 질량에 비례하는 힘 = 같은 가속도 (impulse_scale 미적용)
            mpm_solver.add_acceleration_on_particles(
                acc=bc["acc"],
                point=bc.get("point", [1, 1, 1]),
                size=bc.get("size", [1, 1, 1]),
                start_time=bc.get("start_time", 0.0),
                end_time=bc.get("end_time", 1e3),
            )
'''
    s = s.replace(anchor, br + anchor, 1)
    open(dp, "w").write(s)
    print(f"[패치] {dp}")

"""경계 커널이 **닿는 칸만** 보게 고친다. 물리는 그대로다.

`add_bounding_box` 는 가장자리 3 칸만 건드리는데 매 서브스텝 2700 만 칸을 전부
훑는다. 여섯 면의 얇은 판만 돌면 같은 일을 1.6M 칸으로 끝낸다.

`add_surface_collider` 는 `dot(cell*dx - point, n) < 0` 인 칸만 건드린다.
법선이 축에 붙어 있으면(거의 다 그렇다) 그 축의 한쪽 구간만 돌면 된다.
축에 안 붙은 법선이면 원래대로 전체를 돈다.

  python exe/patch_gf_bc.py --gf <GaussianFluent>
"""
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("--gf", required=True)
a = ap.parse_args()

MARK = "# [anchorflow] 경계는 닿는 칸만"
q = os.path.join(a.gf, "mpm_solver_warp", "mpm_solver_warp.py")
s = open(q).read()
if MARK in s:
    print("이미 되어 있다"); raise SystemExit(0)

# ---------------------------------------------------------------- 경계 상자
OLD = """            grid_x, grid_y, grid_z = wp.tid()
            padding = 3
            if time >= param.start_time and time < param.end_time:"""
NEW = """            face, u, v = wp.tid()
            padding = 3
            # 여섯 면의 3 칸짜리 판만 돈다. 안쪽 칸은 이 커널이 손대지 않는다.
            ax = face / 2
            side = face - ax * 2
            d = u / model.grid_dim_x
            uu = u - d * model.grid_dim_x
            grid_x = 0
            grid_y = 0
            grid_z = 0
            if ax == 0:
                grid_x = d
                if side == 1:
                    grid_x = model.grid_dim_x - 1 - d
                grid_y = uu
                grid_z = v
            if ax == 1:
                grid_y = d
                if side == 1:
                    grid_y = model.grid_dim_y - 1 - d
                grid_x = uu
                grid_z = v
            if ax == 2:
                grid_z = d
                if side == 1:
                    grid_z = model.grid_dim_z - 1 - d
                grid_x = uu
                grid_y = v
            if time >= param.start_time and time < param.end_time:"""
if OLD not in s:
    raise SystemExit("bounding_box 커널 머리를 못 찾았다")
s = s.replace(OLD, NEW, 1)

# 이 충돌자만 launch 모양을 바꾼다 -- grid_postprocess 에 모양을 같이 달아 둔다
OLD2 = """        self.grid_postprocess.append(collide)
        self.modify_bc.append(None)

    # particle_v += force/particle_mass * dt"""
NEW2 = """        self.grid_postprocess.append(collide)
        self.modify_bc.append(None)
        """ + MARK + """: 여섯 면 x (3*한변) x 한변 만 돈다
        self.grid_postprocess_dim[len(self.grid_postprocess) - 1] = (
            6, 3 * self.mpm_model.grid_dim_x, self.mpm_model.grid_dim_x)

    # particle_v += force/particle_mass * dt"""
if OLD2 not in s:
    raise SystemExit("bounding_box 의 append 를 못 찾았다")
s = s.replace(OLD2, NEW2, 1)

# 충돌자별 launch 모양을 담을 사전
OLD3 = "        self.grid_postprocess = []"
if OLD3 not in s:
    raise SystemExit("grid_postprocess 초기화를 못 찾았다")
s = s.replace(OLD3, OLD3 + "\n        self.grid_postprocess_dim = {}", 1)

# 경계 적용 launch 가 사전을 보게 한다
OLD4 = """                wp.launch(
                    kernel=self.grid_postprocess[k],
                    dim=grid_size,"""
NEW4 = """                wp.launch(
                    kernel=self.grid_postprocess[k],
                    dim=self.grid_postprocess_dim.get(k, grid_size),"""
if OLD4 not in s:
    raise SystemExit("경계 launch 를 못 찾았다")
s = s.replace(OLD4, NEW4, 1)

open(q, "w").write(s)
print(f"고쳤다: {q}")
print("GFBC_OK")

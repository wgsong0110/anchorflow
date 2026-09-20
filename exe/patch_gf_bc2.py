"""평면 충돌자도 **자기가 닿는 판만** 돌게 한다.

`add_surface_collider` 는 `dot(cell*dx - point, n) < 0` 인 칸만 건드린다.
법선이 축에 붙어 있으면 그 축의 한쪽 구간이 전부다. 수박 씬은 평면이 z=0 이라
격자 안에 조건을 만족하는 칸이 **하나도 없는데** 매 서브스텝 2700 만 칸을 훑는다.

구간은 `collider_param.size` 에 (축, 시작, 끝) 으로 실어 보낸다 -- 평면 충돌자는
size 를 쓰지 않으므로 남는 자리다. 축에 안 붙은 법선이면 축을 -1 로 두고 예전처럼
전체를 돈다.

  python exe/patch_gf_bc2.py --gf <GaussianFluent>
"""
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("--gf", required=True)
a = ap.parse_args()

MARK = "# [anchorflow] 평면은 자기 판만"
q = os.path.join(a.gf, "mpm_solver_warp", "mpm_solver_warp.py")
s = open(q).read()
if MARK in s:
    print("이미 되어 있다"); raise SystemExit(0)

OLD = """        self.collider_params.append(collider_param)

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
                    float(grid_x) * model.dx - param.point[0],"""
NEW = """        """ + MARK + """: 축에 붙은 법선이면 닿는 구간만 돈다
        _dx = self.mpm_model.grid_lim / self.mpm_model.n_grid
        _n = self.mpm_model.n_grid
        _ax = -1
        for _i in range(3):
            if abs(abs(normal[_i]) - 1.0) < 1e-9:
                _ax = _i
        _lo, _hi = 0, _n
        if _ax >= 0:
            _t = point[_ax] / _dx
            if normal[_ax] > 0:          # cell*dx < point  인 쪽
                _lo, _hi = 0, max(0, min(_n, int(_t) + 1))
            else:                         # cell*dx > point  인 쪽
                _lo, _hi = max(0, min(_n, int(_t))), _n
            if _hi <= _lo:
                _ax, _lo, _hi = 0, 0, 0
        collider_param.size = wp.vec3(float(_ax), float(_lo), float(_hi))
        self.collider_params.append(collider_param)

        @wp.kernel
        def collide(
            time: float,
            dt: float,
            state: MPMStateStruct,
            model: MPMModelStruct,
            param: Dirichlet_collider,
        ):
            gx, gy, gz = wp.tid()
            sax = wp.int(param.size[0])
            slo = wp.int(param.size[1])
            grid_x = gx
            grid_y = gy
            grid_z = gz
            if sax == 0:
                grid_x = gx + slo
            if sax == 1:
                grid_y = gy + slo
            if sax == 2:
                grid_z = gz + slo
            if time >= param.start_time and time < param.end_time:
                offset = wp.vec3(
                    float(grid_x) * model.dx - param.point[0],"""
if OLD not in s:
    raise SystemExit("평면 충돌자 커널 머리를 못 찾았다")
s = s.replace(OLD, NEW, 1)

OLD2 = """        self.grid_postprocess.append(collide)
        self.modify_bc.append(None)

    # a cubiod is a rectangular cube'"""
NEW2 = """        self.grid_postprocess.append(collide)
        self.modify_bc.append(None)
        _w = _hi - _lo
        if _ax == 0:
            self.grid_postprocess_dim[len(self.grid_postprocess) - 1] = (_w, _n, _n)
        elif _ax == 1:
            self.grid_postprocess_dim[len(self.grid_postprocess) - 1] = (_n, _w, _n)
        elif _ax == 2:
            self.grid_postprocess_dim[len(self.grid_postprocess) - 1] = (_n, _n, _w)

    # a cubiod is a rectangular cube'"""
if OLD2 not in s:
    raise SystemExit("평면 충돌자의 append 를 못 찾았다")
s = s.replace(OLD2, NEW2, 1)

open(q, "w").write(s)
print(f"고쳤다: {q}")
print("GFBC2_OK")

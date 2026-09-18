"""발산한 입자를 격리해 GaussianFluent 시뮬이 통째로 죽는 것을 막는다.

증상: 몇십 프레임 돌다가 `CUDA error 700 (illegal memory access)` 로 끝난다.
h5 를 열어보면 이미 프레임 8 부터 비유한 입자가 729 개 있고 프레임 30 에 1349 개로
늘다가, 32 에서 |v| 가 4.3 에서 366 으로 튀며 위치가 격자 밖(-0.116)으로 나간다.
즉 NaN 위치로 격자 색인을 만들다 메모리를 벗어나는 것이다.

해결: 매 서브스텝 앞에서 위치·속도가 유한하고 영역 안인지 보고, 아니면 그 입자의
`particle_selection` 을 1 로 올려 **시뮬에서 빼 버린다**. 격자를 건드리는 커널
(p2g, g2p, 응력 계산)이 전부 `particle_selection[p] == 0` 으로 막혀 있으므로
빠진 입자의 NaN 위치는 두 번 다시 읽히지 않는다. 위치를 손대지 않으므로 뒤에서
`isfinite` 로 거르는 우리 도구들과도 그대로 맞는다.

NaN 은 모든 비교가 거짓이라 `lo < x < hi` 한 번으로 NaN·inf·영역이탈이 함께 걸린다.
"""
from __future__ import annotations

import argparse
import os

MARK = "# [anchorflow] 발산한 입자를 시뮬에서 빼는 격리 커널"
KERNEL = '''

''' + MARK + '''
@wp.kernel
def quarantine_nonfinite(state: MPMStateStruct, lo: float, hi: float,
                         vmax: float):
    p = wp.tid()
    if state.particle_selection[p] == 0:
        x = state.particle_x[p]
        v = state.particle_v[p]
        ok = (x[0] > lo) and (x[0] < hi) and (x[1] > lo) and (x[1] < hi) \\
            and (x[2] > lo) and (x[2] < hi) \\
            and (v[0] > -vmax) and (v[0] < vmax) \\
            and (v[1] > -vmax) and (v[1] < vmax) \\
            and (v[2] > -vmax) and (v[2] < vmax)
        if not ok:
            state.particle_selection[p] = 1
            state.particle_v[p] = wp.vec3(0.0, 0.0, 0.0)
'''

CALL_OLD = """        wp.launch(
            kernel=zero_grid,
            dim=(grid_size),
            inputs=[self.mpm_state, self.mpm_model],
            device=device,
        )
"""
CALL_NEW = """        # [anchorflow] 발산한 입자를 먼저 빼낸다. 놔두면 NaN 위치로 격자 색인을
        # 만들다 프로세스가 통째로 죽는다.
        wp.launch(
            kernel=quarantine_nonfinite,
            dim=self.n_particles,
            inputs=[self.mpm_state, -0.5 * self.mpm_model.grid_lim,
                    1.5 * self.mpm_model.grid_lim, 1.0e5],
            device=device,
        )
""" + CALL_OLD

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True,
                help="GaussianFluent 또는 PhysGaussian 체크아웃 (구조가 같다)")
a = ap.parse_args()

pu = os.path.join(a.root, "mpm_solver_warp", "mpm_utils.py")
ps = os.path.join(a.root, "mpm_solver_warp", "mpm_solver_warp.py")
s = open(pu).read()
if MARK in s:
    raise SystemExit("이미 들어가 있다: " + pu)
open(pu, "w").write(s + KERNEL)

t = open(ps).read()
if CALL_OLD not in t:
    raise SystemExit("zero_grid 실행 지점을 못 찾았다")
open(ps, "w").write(t.replace(CALL_OLD, CALL_NEW, 1))
print("고쳤다:", pu, "와", ps, flush=True)
print("NANGUARD_OK", flush=True)

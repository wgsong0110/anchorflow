"""절 CM13.4: GF 공식 수박 러너를 그대로 돌리면서 프레임마다 입자 logJp (mpm_state.particle_Jp) 만 더 덤프한다.

GF 러너·솔버 소스는 건드리지 않는다. 러너가 `from mpm_solver_warp.engine_utils import *` 로 가져가는
save_data_at_frame 을 실행 전에 감싸, 원래 h5 저장 뒤 같은 프레임 번호로 jp/sc_<프레임:04d>.npy 를 쓴다
(입자 순서 = h5 의 x 순서 = mpm_state 순서). 결과 궤적은 원래 러너와 같은 코드 경로 (비결정 원자 합은 그대로).

  cd GaussianFluent && python <anchorflow>/exe/gf_wm_jp.py --jp_out DIR -- gs_simulation/watermelon/gs_simulation_watermelon.py --model_path .. --config .. --output_path .. --output_h5
"""
import os
import runpy
import sys

i = sys.argv.index("--")
JP = sys.argv[sys.argv.index("--jp_out") + 1]
runner, rest = sys.argv[i + 1], sys.argv[i + 2:]
os.makedirs(JP, exist_ok=True)
sys.path[:0] = [os.getcwd(), os.path.join(os.getcwd(), "gaussian-splatting")]

import numpy as np                                   # noqa: E402
import mpm_solver_warp.engine_utils as EU            # noqa: E402

_orig = EU.save_data_at_frame


def save_data_at_frame(mpm_solver, dir_name, frame, save_to_ply=True, save_to_h5=False):
    _orig(mpm_solver, dir_name, frame, save_to_ply=save_to_ply, save_to_h5=save_to_h5)
    jp = mpm_solver.mpm_state.particle_Jp.numpy().astype(np.float32)
    np.save(os.path.join(JP, f"sc_{frame:04d}.npy"), jp)
    fin = np.isfinite(jp)
    print(f"[jp] 프레임 {frame} 입자 {jp.size} 범위 {jp[fin].min():.4g}~{jp[fin].max():.4g} > 0 몫 {(jp > 0).mean():.4f} > 0.4 몫 {(jp > 0.4).mean():.4f}", flush=True)


EU.save_data_at_frame = save_data_at_frame
sys.argv = [runner] + rest
runpy.run_path(runner, run_name="__main__")

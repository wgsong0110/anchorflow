"""두 레포의 `gs_simulation.py` 에 **시뮬 구간 전용 타이머**를 박는다.

왜 필요한가: 벽시계 전체를 재면 PG 가 불리하다. PG 는 `--render_img` 가 없어도
프레임마다 카메라와 래스터라이저를 다시 만든다 (gs_simulation.py 의 for 루프
맨 앞, 조건 없음). i-PG 는 같은 블록을 `if args.render_img` 로 감싸 둔다.
거기에 h5 쓰기까지 섞이면 "프레임당 몇 ms" 가 솔버 비용이 아니게 된다.

그래서 **p2g2p 루프만** 재서 한 줄로 찍는다:
    [AF시간] 시뮬 X.XXXX 초, 프레임 N, 프레임당 Y.YYY ms

  python exe/patch_bench_timing.py --repo /home/dkta/work/PhysGaussian
"""
from __future__ import annotations

import argparse
import os
import shutil

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True)
a = ap.parse_args()

p = os.path.join(a.repo, "gs_simulation.py")
s = open(p).read()
if "[AF시간]" in s:
    print(f"이미 패치됨: {p}")
    raise SystemExit(0)
bak = p + ".pretime"
if not os.path.exists(bak):
    shutil.copy(p, bak)

# 1) 타이머 변수
s = s.replace("    for frame in tqdm(range(frame_num)):",
              "    _af_sim_t = 0.0\n    import time as _af_time\n"
              "    for frame in tqdm(range(frame_num)):", 1)

# 2) 서브스텝 루프를 감싼다 (PG: 단일 루프 / i-PG: implicit 분기 포함)
old_pg = """        for step in range(step_per_frame):
            mpm_solver.p2g2p(frame, substep_dt, device=device)
"""
new_pg = """        torch.cuda.synchronize()
        _af_t0 = _af_time.time()
        for step in range(step_per_frame):
            mpm_solver.p2g2p(frame, substep_dt, device=device)
        torch.cuda.synchronize()
        _af_sim_t += _af_time.time() - _af_t0
"""
old_ipg = """        if args.implicit:
            actual_dt = substep_dt * args.dt_multiplier"""
new_ipg = """        torch.cuda.synchronize()
        _af_t0 = _af_time.time()
        if args.implicit:
            actual_dt = substep_dt * args.dt_multiplier"""
old_ipg2 = """            for step in range(actual_steps_ex):
                mpm_solver.p2g2p(frame, actual_dt_ex, device=device)
"""
new_ipg2 = """            for step in range(actual_steps_ex):
                mpm_solver.p2g2p(frame, actual_dt_ex, device=device)
        torch.cuda.synchronize()
        _af_sim_t += _af_time.time() - _af_t0
"""
if old_pg in s:
    s = s.replace(old_pg, new_pg, 1)
elif old_ipg in s and old_ipg2 in s:
    s = s.replace(old_ipg, new_ipg, 1).replace(old_ipg2, new_ipg2, 1)
else:
    raise SystemExit("서브스텝 루프를 못 찾았다 -- 레포가 바뀌었는지 확인할 것")

# 3) 끝에서 한 줄 찍는다 (마지막 들여쓰기 없는 구문 앞)
tail = """
    print(f"[AF시간] 시뮬 {_af_sim_t:.4f} 초, 프레임 {frame_num}, "
          f"프레임당 {1000.0 * _af_sim_t / max(frame_num, 1):.3f} ms",
          flush=True)
"""
mark = "    if args.render_img and args.compile_video:"
if mark not in s:
    raise SystemExit("마무리 지점을 못 찾았다")
s = s.replace(mark, tail.lstrip("\n") + mark, 1)
open(p, "w").write(s)
print(f"패치 완료: {p} (원본 {bak})")

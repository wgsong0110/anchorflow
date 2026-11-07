"""절 CM34 진단 (결과 불변): 커널별 시간을 동기 벽시계로 재는 프로파일 감싸개.

옛 실행기 (fem_ao_rt_sim.py, 수박) 의 --prof 는 CUDA 이벤트 시간이 실제 재생 시간의 1/15 로 나와 (커널을 이벤트와 다른 스트림에 띄우는 것으로 보임) 쓸 수 없다.
여기서는 프로파일 프레임 (--prof, 그래프 밖 한 번, 상태 되돌림) 동안만 커널 하나마다 앞뒤 torch.cuda.synchronize + perf_counter 로 잰다 (실행 지연 몇 µs 포함).
그래프 재생 (실제 시간 측정) 은 건드리지 않는다.

사용: python exe/speed1_prof.py <실행기.py> <실행기 인자...>   (예: exe/fem_ao_rt_sim.py ... --prof 1)
"""
import os
import runpy
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import torch  # noqa: E402
from anchorflow import fem_an_rt_v0 as V0  # noqa: E402
from anchorflow import fem_an_rt as AN  # noqa: E402


class _T:
    def __init__(s, ms):
        s.ms = ms

    def elapsed_time(s, _other):
        return s.ms


def _patch(cls):
    ok = cls.k

    def k(s, name, n, args, b=128):
        pr = getattr(s, "prof", None)
        if pr is None:
            return ok(s, name, n, args, b)
        s.prof = None
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            return ok(s, name, n, args, b)
        finally:
            torch.cuda.synchronize()
            s.prof = pr
            pr.append((name, _T(1e3 * (time.perf_counter() - t0)), None))
    cls.k = k


_patch(V0.RTSolver)
_patch(AN.RTSolver)
sim = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(sim, run_name="__main__")

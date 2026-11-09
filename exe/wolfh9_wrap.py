"""절 CM87 묶음 wolfh9 감싸개: κ 0 씬 P1X 강성 무게 (lib/anchorflow/fem_wolfh9.py) 를 단 뒤 exe/speed12_wrap.py 로 넘김 (그쪽 인자 그대로; H8 사슬).

사용: python exe/wolfh9_wrap.py [--w9 1|2] <speed12_wrap 인자...> <실행기.py> <실행기 인자...>
설치: 실행기를 runpy 로 띄우는 순간 (안쪽 감싸개 = fem_wolfh6 (K) 설치 뒤, 가장 바깥). --w9 이 없거나 0 이면 모듈을 안 부름.
"""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lib"))
sys.setrecursionlimit(max(sys.getrecursionlimit(), 6000))                                 # 감싸개 한 겹 더 (첫 연기 f2eacb4: 기본 1000 에서 RecursionError)

argv = sys.argv[1:]
out = argv[argv.index("--out") + 1] if "--out" in argv else "."
os.makedirs(out, exist_ok=True)
O = dict(w9="0")
while argv and argv[0].startswith("--w9"):
    O[argv[0][2:]] = argv[1]
    del argv[:2]
MODE = int(float(O["w9"]))

_orig = runpy.run_path
_done = [False]


def run_path(path, *a, **kw):
    if MODE and not _done[0] and os.path.basename(path) in ("fem_an_rt_sim.py", "fem_ao_rt_sim.py"):
        _done[0] = True
        from anchorflow import fem_wolfh9 as W9
        W9.install(out, MODE)
    return _orig(path, *a, **kw)


runpy.run_path = run_path
sys.argv = [os.path.join(HERE, "speed12_wrap.py")] + argv
_orig(sys.argv[0], run_name="__main__")

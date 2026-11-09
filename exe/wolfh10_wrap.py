"""절 CM89 묶음 wolfh10 감싸개: 새 실행기에 뒤집힘 없는 보폭 필터 (옛 실행기 --ls_inv --ls_end --y0u --y0l 정의, lib/anchorflow/fem_wolfh10.py) 를 단 뒤 exe/wolfh9_wrap.py 로 넘김 (그쪽 인자 그대로; H8 사슬).

사용: python exe/wolfh10_wrap.py [--w10 1] [--w10dg 1] <wolfh9_wrap 인자...> <실행기.py> <실행기 인자...>
설치: 새 실행기 (fem_an_rt_sim.py) 를 runpy 로 띄우는 순간 (안쪽 감싸개들 설치 뒤 = 가장 바깥). --w10 이 없거나 0 이면 모듈을 안 부름. 옛 실행기 (수박) 는 안 건드림.
"""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lib"))
sys.setrecursionlimit(max(sys.getrecursionlimit(), 8000))                                 # 감싸개 한 겹 더 (절 CM87 첫 연기 RecursionError)

argv = sys.argv[1:]
out = argv[argv.index("--out") + 1] if "--out" in argv else "."
os.makedirs(out, exist_ok=True)
O = dict(w10="0", w10dg="0")
while argv and argv[0].startswith("--w10"):
    O[argv[0][2:]] = argv[1]
    del argv[:2]
MODE = int(float(O["w10"]))

_orig = runpy.run_path
_done = [False]


def run_path(path, *a, **kw):
    if MODE and not _done[0] and os.path.basename(path) == "fem_an_rt_sim.py":
        _done[0] = True
        from anchorflow import fem_wolfh10 as W10
        W10.install(out, MODE, int(float(O["w10dg"])))
    return _orig(path, *a, **kw)


runpy.run_path = run_path
sys.argv = [os.path.join(HERE, "wolfh9_wrap.py")] + argv
_orig(sys.argv[0], run_name="__main__")

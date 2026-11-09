"""절 CM94 묶음 wolfh13 감싸개: 확장 반영 MG 전처리기 (lib/anchorflow/fem_wolfh13.py) 를 단 뒤 exe/wolfh12_wrap.py 로 넘김 (그쪽 인자 그대로; H8 사슬).

사용: python exe/wolfh13_wrap.py [--w13 모드] <wolfh12_wrap 인자...> <실행기.py> <실행기 인자...>
설치: 새 실행기 fem_an_rt_sim.py (exe/s6/ 사본 포함) 를 runpy 로 띄우는 순간 (안쪽 감싸개들 설치 뒤 = 가장 바깥).
--w13 이 없거나 0 이면 모듈을 안 부름 (예전 그대로). 확장이 꺼진 판 (κ 1 씬) 은 모듈이 할당·커널을 하나도 안 띄움.
"""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lib"))
sys.setrecursionlimit(max(sys.getrecursionlimit(), 14000))                                # 감싸개 한 겹 더

argv = sys.argv[1:]
out = argv[argv.index("--out") + 1] if "--out" in argv else "."
os.makedirs(out, exist_ok=True)
O = dict(w13="0")
while argv and argv[0] in ("--w13",):
    O[argv[0][2:]] = argv[1]
    del argv[:2]
MODE = int(float(O["w13"]))

_orig = runpy.run_path
_done = [False]


def run_path(path, *a, **kw):
    if MODE and not _done[0] and os.path.basename(path) == "fem_an_rt_sim.py":
        _done[0] = True
        from anchorflow import fem_wolfh13 as W13
        W13.install(out, MODE)
    return _orig(path, *a, **kw)


runpy.run_path = run_path
sys.argv = [os.path.join(HERE, "wolfh12_wrap.py")] + argv
_orig(sys.argv[0], run_name="__main__")

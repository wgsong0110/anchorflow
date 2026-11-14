"""절 CM140 감싸개: --anx N (0 끔 기본, 1 Anderson 깊이 1, 2 질량 가중) 로 뉴턴 방향 가속 (lib/anchorflow/fem_anderson.py) 을 단 뒤 다음 감싸개로 넘김.
사용: python exe/anderson_wrap.py --anx N <다음 감싸개.py> <그 인자...> <실행기.py> <실행기 인자...>   (nlwhy_wrap 와 같은 설치 방식: 실행기 fem_an_rt_sim.py 를 띄우는 순간)"""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lib"))
argv = sys.argv[1:]
assert argv and argv[0] == "--anx", "anderson_wrap: 첫 인자는 --anx N"
mode = int(argv[1]); argv = argv[2:]
_orig = runpy.run_path
_done = [False]


def run_path(path, *a, **kw):
    if not _done[0] and os.path.basename(path) == "fem_an_rt_sim.py":
        _done[0] = True
        from anchorflow import fem_anderson as AN
        cl = AN.install(mode)
        print(f"[절 CM140] anderson 켬 -- 모드 {mode}, 감싼 클래스 {[c.__name__ for c in cl]}", flush=True)
    return _orig(path, *a, **kw)


runpy.run_path = run_path
assert argv and argv[0].endswith(".py"), "anderson_wrap: 다음 감싸개 .py"
sys.argv = argv
_orig(sys.argv[0], run_name="__main__")

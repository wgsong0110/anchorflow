"""절 CM146 감싸개: --rls N (0 끔 기본, 1 잔차 노름 Armijo 선탐색; 에너지 선탐색 끈 명령 위에서만) 로 잔차 노름 선탐색 (lib/anchorflow/fem_rls.py) 을 단 뒤 다음 감싸개로 넘김.
사용: python exe/rls_wrap.py --rls N <다음 감싸개.py> <그 인자...> <실행기.py> <실행기 인자...>   (nlwhy_wrap 와 같은 설치 방식: 실행기 fem_an_rt_sim.py 를 띄우는 순간)"""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lib"))
argv = sys.argv[1:]
assert argv and argv[0] == "--rls", "rls_wrap: 첫 인자는 --rls N"
mode = int(argv[1]); argv = argv[2:]
_orig = runpy.run_path
_done = [False]


def run_path(path, *a, **kw):
    if not _done[0] and os.path.basename(path) == "fem_an_rt_sim.py":
        _done[0] = True
        from anchorflow import fem_rls as RL
        cl = RL.install(mode)
        print(f"[절 CM146] rls 켬 -- 모드 {mode}, 감싼 클래스 {[c.__name__ for c in cl]}", flush=True)
    return _orig(path, *a, **kw)


runpy.run_path = run_path
assert argv and argv[0].endswith(".py"), "rls_wrap: 다음 감싸개 .py"
sys.argv = argv
_orig(sys.argv[0], run_name="__main__")

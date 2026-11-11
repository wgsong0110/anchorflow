"""절 CM111 묶음 nodepen 감싸개: 노드 덤프 진단 (lib/anchorflow/fem_nodepen.py, 읽기만, 결과 불변) 을 단 뒤 exe/wolfh12_wrap.py 로 넘김 (그쪽 인자 그대로).

사용: python exe/nodepen_wrap.py --npd 1 <wolfh12_wrap 인자...> <실행기.py> <실행기 인자...>
--npd 가 없거나 0 이면 모듈을 안 부름 (예전 그대로). 설치: 새 실행기 fem_an_rt_sim.py 를 runpy 로 띄우는 순간 (가장 바깥).
"""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lib"))
sys.setrecursionlimit(max(sys.getrecursionlimit(), 14000))

argv = sys.argv[1:]
out = argv[argv.index("--out") + 1] if "--out" in argv else "."
os.makedirs(out, exist_ok=True)
ON = 0
while argv and argv[0] == "--npd":
    ON = int(float(argv[1]))
    del argv[:2]

_orig = runpy.run_path
_done = [False]


def run_path(path, *a, **kw):
    if ON and not _done[0] and os.path.basename(path) == "fem_an_rt_sim.py":
        _done[0] = True
        from anchorflow import fem_nodepen as NP
        NP.install(out)
    return _orig(path, *a, **kw)


runpy.run_path = run_path
sys.argv = [os.path.join(HERE, "wolfh12_wrap.py")] + argv
_orig(sys.argv[0], run_name="__main__")

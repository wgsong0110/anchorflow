"""절 CM111 묶음 cellforce 감싸개: 칸 세로 압축 모드 항별 일반화 힘 기록 (lib/anchorflow/fem_cellforce.py, 읽기만, 결과 불변) 을 단 뒤 인자 첫 감싸개로 넘김.

사용: python exe/cellforce_wrap.py --cf 1 <다음 감싸개.py 인자...> <실행기.py> <실행기 인자...>
--cf 가 없거나 0 이면 모듈을 안 부름 (예전 그대로). 설치: 새 실행기 fem_an_rt_sim.py 를 runpy 로 띄우는 순간 (가장 바깥).
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
while argv and argv[0] == "--cf":
    ON = int(float(argv[1]))
    del argv[:2]

_orig = runpy.run_path
_done = [False]


def run_path(path, *a, **kw):
    if ON and not _done[0] and os.path.basename(path) == "fem_an_rt_sim.py":
        _done[0] = True
        from anchorflow import fem_cellforce as CF
        CF.install(out)
    return _orig(path, *a, **kw)


runpy.run_path = run_path
sys.argv = [os.path.abspath(argv[0])] + argv[1:]
_orig(sys.argv[0], run_name="__main__")

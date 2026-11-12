"""절 CM126 묶음 escwhy 감싸개: 입자 이탈 원인 걸음별 기록 (lib/anchorflow/fem_escwhy.py, 결과 불변) 을 단 뒤 다음 감싸개 (첫 인자) 로 넘김.
사용: python exe/escwhy_wrap.py <다음 감싸개.py> <그 인자...> <실행기.py> <실행기 인자...>
설치: 새 실행기 fem_an_rt_sim.py 를 runpy 로 띄우는 순간 (안쪽 감싸개들 설치 뒤 = 가장 바깥)."""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lib"))
sys.setrecursionlimit(max(sys.getrecursionlimit(), 16000))
argv = sys.argv[1:]
out = argv[argv.index("--out") + 1] if "--out" in argv else "."
os.makedirs(out, exist_ok=True)
_orig = runpy.run_path
_done = [False]


def run_path(path, *a, **kw):
    if not _done[0] and os.path.basename(path) == "fem_an_rt_sim.py":
        _done[0] = True
        from anchorflow import fem_escwhy as NL
        NL.install(out)
    return _orig(path, *a, **kw)


runpy.run_path = run_path
assert argv and argv[0].endswith(".py"), "escwhy_wrap: 다음 감싸개 .py 가 첫 인자"
sys.argv = argv
_orig(sys.argv[0], run_name="__main__")

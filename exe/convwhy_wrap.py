"""절 CM96 묶음 convwhy 감싸개: 접촉 걸음 수렴 진단 (lib/anchorflow/fem_convwhy.py) 을 단 뒤 exe/wolfh13_wrap.py 로 넘김 (그쪽 인자 그대로; H8 사슬).

사용: python exe/convwhy_wrap.py [--cwsave 'fi,..'] [--cwload 디렉토리|0|8|5] [--cwks c] [--cwnopl 0|1] <wolfh13_wrap 인자...> <실행기.py> <실행기 인자...>
설치: 새 실행기 fem_an_rt_sim.py 를 runpy 로 띄우는 순간 (안쪽 감싸개들 설치 뒤 = 가장 바깥).
--cwsave·--cwload 가 둘 다 없으면 모듈을 안 부름 (예전 그대로).
"""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lib"))
sys.setrecursionlimit(max(sys.getrecursionlimit(), 15000))                                # 감싸개 한 겹 더

argv = sys.argv[1:]
out = argv[argv.index("--out") + 1] if "--out" in argv else "."
os.makedirs(out, exist_ok=True)
O = dict(cwsave="", cwload="", cwks="1", cwnopl="0")
while argv and argv[0] in ("--cwsave", "--cwload", "--cwks", "--cwnopl"):
    O[argv[0][2:]] = argv[1]
    del argv[:2]
ON = bool(O["cwsave"] or O["cwload"])

_orig = runpy.run_path
_done = [False]


def run_path(path, *a, **kw):
    if ON and not _done[0] and os.path.basename(path) == "fem_an_rt_sim.py":
        _done[0] = True
        from anchorflow import fem_convwhy as CW
        CW.install(out, O["cwsave"], O["cwload"], float(O["cwks"]), int(float(O["cwnopl"])))
    return _orig(path, *a, **kw)


runpy.run_path = run_path
sys.argv = [os.path.join(HERE, "wolfh13_wrap.py")] + argv
_orig(sys.argv[0], run_name="__main__")

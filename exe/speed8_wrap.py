"""절 CM74 묶음 speed8 감싸개: 워프 모으기 W · X3g 후보 (lib/anchorflow/fem_speed8.py) 를 단 뒤 exe/speed7_wrap.py 로 넘김 (그쪽 인자 그대로).

사용: python exe/speed8_wrap.py [--s8w 1|2] [--s8g T] <speed7_wrap 인자...> <실행기.py> <실행기 인자...>
설치: 실행기를 runpy 로 띄우는 순간 (speed6·speed7 설치 뒤 = 가장 바깥). 인자가 없거나 0 이면 모듈을 안 부름.
"""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lib"))

argv = sys.argv[1:]
out = argv[argv.index("--out") + 1] if "--out" in argv else "."
os.makedirs(out, exist_ok=True)
O8 = dict(s8w="0", s8g="0")
while argv and argv[0].startswith("--s8"):
    O8[argv[0][2:]] = argv[1]
    del argv[:2]
ON = any(float(v) != 0 for v in O8.values())

_orig = runpy.run_path
_done = [False]


def run_path(path, *a, **kw):
    if ON and not _done[0] and os.path.basename(path) in ("fem_an_rt_sim.py", "fem_ao_rt_sim.py"):
        _done[0] = True
        from anchorflow import fem_speed8 as S8
        S8.install(out, w=int(float(O8["s8w"])), g=float(O8["s8g"]))
    return _orig(path, *a, **kw)


runpy.run_path = run_path
sys.argv = [os.path.join(HERE, "speed7_wrap.py")] + argv
_orig(sys.argv[0], run_name="__main__")

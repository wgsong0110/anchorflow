"""절 CM85 묶음 speed12 감싸개: 시작점 기준 고침·X2 신호 따로 (lib/anchorflow/fem_speed12.py) 를 단 뒤 exe/ext2_wrap.py 로 넘김 (그쪽 인자 그대로).

사용: python exe/speed12_wrap.py [--s12z 1] [--s12dg 1] <ext2_wrap 인자...> <실행기.py> <실행기 인자...>
설치: 옛 실행기 (fem_ao_rt_sim.py, 수박) 를 runpy 로 띄우는 순간 (안쪽 감싸개들 설치 뒤 = 가장 바깥). 인자가 없거나 0 이면 모듈을 안 부름. 새 실행기 씬은 안 건드림.
"""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lib"))

argv = sys.argv[1:]
out = argv[argv.index("--out") + 1] if "--out" in argv else "."
os.makedirs(out, exist_ok=True)
O = dict(s12z="0", s12dg="0")
while argv and argv[0].startswith("--s12"):
    O[argv[0][2:]] = argv[1]
    del argv[:2]
ON = any(float(v) != 0 for v in O.values())

_orig = runpy.run_path
_done = [False]


def run_path(path, *a, **kw):
    if ON and not _done[0] and os.path.basename(path) == "fem_ao_rt_sim.py":
        _done[0] = True
        from anchorflow import fem_speed12 as S12
        S12.install(out, z=int(float(O["s12z"])), dg=int(float(O["s12dg"])))
    return _orig(path, *a, **kw)


runpy.run_path = run_path
sys.argv = [os.path.join(HERE, "ext2_wrap.py")] + argv
_orig(sys.argv[0], run_name="__main__")

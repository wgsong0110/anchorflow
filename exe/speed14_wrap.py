"""절 CM88 묶음 speed14 감싸개: 진단·고침 (lib/anchorflow/fem_speed14.py) 을 단 뒤 exe/speed13_wrap.py 로 넘김 (그쪽 인자 그대로).

사용: python exe/speed14_wrap.py [--s14dg 1] [--s14a 1|2] [--s14ct 1] [--s14m 1] <speed13_wrap 인자...> <실행기.py> <실행기 인자...>
설치: 옛 실행기 (fem_ao_rt_sim.py, 수박) 를 runpy 로 띄우는 순간 (speed13 설치 뒤 = 가장 바깥). 인자가 없거나 0 이면 모듈을 안 부름.
"""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.setrecursionlimit(20000)                                                          # 감싸개 사슬이 깊어 (판마다 바깥 훅을 모두 거침) 한 겹 더하면 기본 1000 을 넘음
sys.path.insert(0, os.path.join(HERE, "..", "lib"))

argv = sys.argv[1:]
out = argv[argv.index("--out") + 1] if "--out" in argv else "."
os.makedirs(out, exist_ok=True)
O = dict(s14dg="0", s14a="0", s14ct="0", s14m="0")
while argv and argv[0].startswith("--s14"):
    O[argv[0][2:]] = argv[1]
    del argv[:2]
ON = any(float(v) != 0 for v in O.values())

_orig = runpy.run_path
_done = [False]


def run_path(path, *a, **kw):
    if ON and not _done[0] and os.path.basename(path) == "fem_ao_rt_sim.py":
        _done[0] = True
        from anchorflow import fem_speed14 as S14
        S14.install(out, dg=int(float(O["s14dg"])), a=int(float(O["s14a"])), ct=int(float(O["s14ct"])), m=int(float(O["s14m"])))
    return _orig(path, *a, **kw)


runpy.run_path = run_path
sys.argv = [os.path.join(HERE, "speed13_wrap.py")] + argv
_orig(sys.argv[0], run_name="__main__")

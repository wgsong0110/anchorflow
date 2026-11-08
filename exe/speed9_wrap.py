"""절 CM76 묶음 speed9 감싸개: 진단 D · 압축 C · 조기 끝 Y · W̄ 워프 모으기 E · 이어 하기 R · EW P (lib/anchorflow/fem_speed9.py) 를 단 뒤 exe/speed8_wrap.py 로 넘김 (그쪽 인자 그대로).

사용: python exe/speed9_wrap.py [--s9dg 1] [--s9c 1|2] [--s9y 1|2] [--s9e 1] [--s9r 1] [--s9p γ] <speed8_wrap 인자...> <실행기.py> <실행기 인자...>
설치: 실행기를 runpy 로 띄우는 순간 (speed6·7·8 설치 뒤 = 가장 바깥). 인자가 없거나 0 이면 모듈을 안 부름.
"""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lib"))

argv = sys.argv[1:]
out = argv[argv.index("--out") + 1] if "--out" in argv else "."
os.makedirs(out, exist_ok=True)
O9 = dict(s9dg="0", s9c="0", s9y="0", s9e="0", s9r="0", s9p="0")
while argv and argv[0].startswith("--s9"):
    O9[argv[0][2:]] = argv[1]
    del argv[:2]
ON = any(float(v) != 0 for v in O9.values())

_orig = runpy.run_path
_done = [False]


def run_path(path, *a, **kw):
    if ON and not _done[0] and os.path.basename(path) in ("fem_an_rt_sim.py", "fem_ao_rt_sim.py"):
        _done[0] = True
        from anchorflow import fem_speed9 as S9
        S9.install(out, dg=int(float(O9["s9dg"])), cm=int(float(O9["s9c"])), yl=int(float(O9["s9y"])), e3=int(float(O9["s9e"])), r=int(float(O9["s9r"])), ew=float(O9["s9p"]))
    return _orig(path, *a, **kw)


runpy.run_path = run_path
sys.argv = [os.path.join(HERE, "speed8_wrap.py")] + argv
_orig(sys.argv[0], run_name="__main__")

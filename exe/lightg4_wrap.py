"""절 CM75 묶음 lightg4 감싸개: 절 CM73 (A) 가벼움 벌점 몫에 재질 응집 몫 κ 를 곱함 (lib/anchorflow/fem_lightg4.py) -- 나머지는 exe/lightg3_wrap.py 로 넘김 (그쪽 인자 그대로).

사용: python exe/lightg4_wrap.py [--l4 1] <lightg3_wrap 인자 (--l3 ... 포함)> <실행기.py> <실행기 인자...>
  --l4 가 없거나 0 이면 lightg3_wrap 그대로. 설치는 실행기 runpy 순간, lightg3 설치 바로 뒤.
"""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lib"))

argv = sys.argv[1:]
out = argv[argv.index("--out") + 1] if "--out" in argv else "."
os.makedirs(out, exist_ok=True)
C = {}
while argv and argv[0].startswith("--l4"):
    C[argv[0][4:]] = argv[1]
    del argv[:2]
ON = int(float(C.get("", 0))) == 1

_orig = runpy.run_path
_st = dict(on=False)


def run_path(path, *a, **kw):
    b = os.path.basename(path)
    if ON and not _st["on"] and b in ("fem_an_rt_sim.py", "fem_ao_rt_sim.py"):                  # lightg3_wrap 의 run_path 가 먼저 L3 를 설치한 뒤 이리로 옴
        _st["on"] = True
        from anchorflow import fem_lightg4 as L4
        L4.install(out)
    return _orig(path, *a, **kw)


runpy.run_path = run_path
sys.argv = [os.path.join(HERE, "lightg3_wrap.py")] + argv
_orig(sys.argv[0], run_name="__main__")

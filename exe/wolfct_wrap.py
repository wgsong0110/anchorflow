"""절 CM108 묶음 wolfct 감싸개 (가장 바깥): 모래 반환 사상 일관 접선 (--ct 1) · 유한 차분 대조 (--ctfd 1, 다시 풀기 판 진단) = lib/anchorflow/fem_wolfct.py 를 단 뒤 exe/condwhy_wrap.py 로 넘김.

사용: python exe/wolfct_wrap.py [--ct 0|1] [--ctfd 0|1] [--cth h] <condwhy_wrap 인자...> <실행기.py> <실행기 인자...>
설치: 실행기 (fem_an_rt_sim.py · fem_ao_rt_sim.py) 를 runpy 로 띄우는 순간 -- condwhy 설치 (CW._read) 뒤라 FD 기록이 그 위에 얹힘.
--ct 0 --ctfd 0 이면 모듈을 안 부름 (예전 그대로).
"""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lib"))
sys.setrecursionlimit(max(sys.getrecursionlimit(), 16000))

argv = sys.argv[1:]
out = argv[argv.index("--out") + 1] if "--out" in argv else "."
os.makedirs(out, exist_ok=True)
O = dict(ct="0", ctfd="0", cth="1e-3")
while argv and argv[0] in ("--ct", "--ctfd", "--cth"):
    O[argv[0][2:]] = argv[1]
    del argv[:2]
ON = int(float(O["ct"])) != 0 or int(float(O["ctfd"])) != 0

_orig = runpy.run_path
_done = [False]


def run_path(path, *a, **kw):
    if ON and not _done[0] and os.path.basename(path) in ("fem_an_rt_sim.py", "fem_ao_rt_sim.py"):
        _done[0] = True
        from anchorflow import fem_wolfct as CT
        CT.install(out, int(float(O["ct"])), int(float(O["ctfd"])), float(O["cth"]))
    return _orig(path, *a, **kw)


runpy.run_path = run_path
sys.argv = [os.path.join(HERE, "condwhy_wrap.py")] + argv
_orig(sys.argv[0], run_name="__main__")

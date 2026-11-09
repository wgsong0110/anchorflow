"""절 CM92 묶음 finterp 감싸개: 입자 F 보간 교체 (--fp_interp) 의 부피 진단·부피 보존 보간 (lib/anchorflow/fem_finterp.py) 를 단 뒤 exe/wolfh11_wrap.py 로 넘김 (그쪽 인자 그대로; H8 사슬).

사용: python exe/finterp_wrap.py [--fi 0|1|2] [--fidg 1] <wolfh11_wrap 인자...> <실행기.py> <실행기 인자...>
설치: 실행기 (새 fem_an_rt_sim.py · 옛 fem_ao_rt_sim.py, exe/s6/ 사본 포함) 를 runpy 로 띄우는 순간 (안쪽 감싸개들 설치 뒤 = 가장 바깥).
--fi 와 --fidg 가 모두 없거나 0 이면 모듈을 안 부름 (예전 그대로).
"""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lib"))
sys.setrecursionlimit(max(sys.getrecursionlimit(), 10000))                                # 감싸개 한 겹 더

argv = sys.argv[1:]
out = argv[argv.index("--out") + 1] if "--out" in argv else "."
os.makedirs(out, exist_ok=True)
O = dict(fi="0", fidg="0")
while argv and argv[0] in ("--fi", "--fidg"):
    O[argv[0][2:]] = argv[1]
    del argv[:2]
MODE, DG = int(float(O["fi"])), int(float(O["fidg"]))

_orig = runpy.run_path
_done = [False]


def run_path(path, *a, **kw):
    if (MODE or DG) and not _done[0] and os.path.basename(path) in ("fem_an_rt_sim.py", "fem_ao_rt_sim.py"):
        _done[0] = True
        from anchorflow import fem_finterp as FI
        FI.install(out, MODE, DG)
    return _orig(path, *a, **kw)


runpy.run_path = run_path
sys.argv = [os.path.join(HERE, "wolfh11_wrap.py")] + argv
_orig(sys.argv[0], run_name="__main__")

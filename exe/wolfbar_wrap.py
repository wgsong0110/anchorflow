"""절 CM107 묶음 wolfbar 감싸개 (가장 바깥): wolf 바닥 장벽 칸별 부피 손실 기록 + 진단 판 장벽 강성 배수 (lib/anchorflow/fem_wolfbar.py).

사용: python exe/wolfbar_wrap.py [--wbdg 0|1] [--wbks 배수] <다음 감싸개.py> <그 인자...> <실행기.py> <실행기 인자...>
설치: 새 실행기 fem_an_rt_sim.py (exe/s6/ 사본 포함) 를 runpy 로 띄우는 순간 (안쪽 감싸개들 설치 뒤 = 가장 바깥, exe/wolfh12_wrap.py 와 같은 자리).
--wbdg 0 이고 --wbks 1 이면 모듈을 안 부름 (예전 그대로).
"""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lib"))
sys.setrecursionlimit(max(sys.getrecursionlimit(), 20000))

argv = sys.argv[1:]
out = argv[argv.index("--out") + 1] if "--out" in argv else "."
os.makedirs(out, exist_ok=True)
O = dict(wbdg="1", wbks="1")
while argv and argv[0] in ("--wbdg", "--wbks"):
    O[argv[0][2:]] = argv[1]
    del argv[:2]
DG, KSM = int(float(O["wbdg"])), float(O["wbks"])

_orig = runpy.run_path
_done = [False]


def run_path(path, *a, **kw):
    if (DG or KSM != 1.0) and not _done[0] and os.path.basename(path) == "fem_an_rt_sim.py":
        _done[0] = True
        from anchorflow import fem_wolfbar as WB
        WB.install(out, DG, KSM)
    return _orig(path, *a, **kw)


runpy.run_path = run_path
sys.argv = list(argv)
_orig(sys.argv[0], run_name="__main__")

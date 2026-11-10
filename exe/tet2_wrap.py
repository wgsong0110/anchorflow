"""절 CM103 묶음 tet2 감싸개 (가장 바깥): 사면체 판 장벽 (구조적 뒤집힘 방지 + 유계 왜곡) -- lib/anchorflow/fem_tet2.py.

사용: python exe/tet2_wrap.py --t2 0|1 --t2b σ_lo,σ_hi,d̂_lo,d̂_hi,d̂_J [--t2c c] [--t2dg 0|1] <다음 감싸개.py> <인자...>
  1 = 장벽 (요소 커널 + 닫힌 꼴 보폭 + 시작점 섞기), 0 = 기록만 (결과 불변 대조 판).
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
MODE, BAND, C, DG = None, None, 1.0, 1
while argv and argv[0] in ("--t2", "--t2b", "--t2c", "--t2dg"):
    k, v = argv[0], argv[1]
    if k == "--t2":
        MODE = int(float(v))
    elif k == "--t2b":
        BAND = [float(q) for q in v.split(",")]
    elif k == "--t2c":
        C = float(v)
    else:
        DG = int(float(v))
    del argv[:2]
assert MODE is not None and BAND is not None and len(BAND) == 5, "--t2 와 --t2b 다섯 값이 필요"
from anchorflow import fem_tet2 as T2
T2.install(out, MODE, BAND, C, DG)
sys.argv = list(argv)
runpy.run_path(sys.argv[0], run_name="__main__")

"""절 CM106 묶음 tetc 감싸개 (가장 바깥): 사면체 판 TC -- lib/anchorflow/fem_tetc.py.

사용: python exe/tetc_wrap.py --tcm 0|1 [--tcdg 0|1] exe/tet1_wrap.py --tet 1 ... <실행기.py> <실행기 인자...>
  1 = TC (사면체 단위 질량·운동량, 노드 ¼): 실행기 인자에 --gq_mass 와 마지막 --gq_oct 0 (κ 0), 사슬에 --tet 1 이 있어야 함 (없으면 멈춤).
  0 = 기록만 (T9S 등 대조 판, 결과 불변).
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
MODE, DG = None, 1
while argv and argv[0] in ("--tcm", "--tcdg"):
    if argv[0] == "--tcm":
        MODE = int(float(argv[1]))
    else:
        DG = int(float(argv[1]))
    del argv[:2]
assert MODE in (0, 1), "--tcm 0|1 이 필요"
if MODE == 1:
    ri = [i for i, a in enumerate(argv) if a.endswith("_rt_sim.py")]
    assert ri, "실행기 (..._rt_sim.py) 를 못 찾음"
    ra = argv[ri[-1] + 1:]
    oc = [ra[i + 1] for i, a in enumerate(ra) if a == "--gq_oct"]
    assert "--gq_mass" in ra and oc and float(oc[-1]) == 0.0, f"TC 는 실행기 --gq_mass 와 --gq_oct 0 (κ 0) 이 필요 (지금 gq_mass {'--gq_mass' in ra}, gq_oct {oc})"
    ti = [argv[i + 1] for i, a in enumerate(argv[:ri[-1]]) if a == "--tet"]
    assert ti and int(float(ti[-1])) == 1, "TC 는 사면체 기저 (tet1_wrap --tet 1) 위에서만"
from anchorflow import fem_tetc as TC
TC.install(out, MODE, DG)
sys.argv = list(argv)
runpy.run_path(sys.argv[0], run_name="__main__")

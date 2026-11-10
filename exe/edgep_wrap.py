"""절 CM105 묶음 edgep 감싸개 (가장 바깥): 노드 변 길이 상한 벌점 -- lib/anchorflow/fem_edgep.py.

사용: python exe/edgep_wrap.py --ep 0|1 --epc c [--epk 배율] [--epdg 0|1] [--epck 0|1] <다음 감싸개.py> <인자...>
  1 = 벌점 켬 (E = k Σ max(0, |r| − c L0)², k = ½ (2μ+λ) L0 x 배율), 0 = 기록만 (정규화 끈 대조 판, 결과 불변).
  --epck 1 = 세 프레임 뒤 유한차분·참값 검사 한 번 (epck.json, 해 상태 안 바꿈).
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
MODE, C, KS, DG, CK = None, None, 1.0, 1, 0
while argv and argv[0] in ("--ep", "--epc", "--epk", "--epdg", "--epck"):
    k, v = argv[0], argv[1]
    if k == "--ep":
        MODE = int(float(v))
    elif k == "--epc":
        C = float(v)
    elif k == "--epk":
        KS = float(v)
    elif k == "--epdg":
        DG = int(float(v))
    else:
        CK = int(float(v))
    del argv[:2]
assert MODE is not None and C is not None and C > 0, "--ep 와 --epc (> 0) 가 필요"
from anchorflow import fem_edgep as EP
EP.install(out, MODE, C, KS, DG, CK)
sys.argv = list(argv)
runpy.run_path(sys.argv[0], run_name="__main__")

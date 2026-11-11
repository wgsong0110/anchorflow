"""절 CM106.8 감싸개 (tetc_wrap.py 바깥): 뒤집힌 사면체 전부 기록 (lib/anchorflow/fem_tetc_inv.py) 을 켜고 exe/tetc_wrap.py 를 그대로 돌린다.
사용: python exe/tetc_inv_wrap.py --tcm 1 --tcdg 2 exe/tet1_wrap.py ... (tetc_wrap.py 와 같은 인자)"""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lib"))
from anchorflow import fem_tetc_inv                                                          # noqa: E402
fem_tetc_inv.install()
sys.argv = [os.path.join(HERE, "tetc_wrap.py")] + sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")

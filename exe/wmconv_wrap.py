"""절 CM90 묶음 wmconv 감싸개: 가벼운 판 덤프 (lib/anchorflow/fem_wmconv.py, 결과 불변) 를 단 뒤 exe/wmjitter_wrap.py 로 넘김 (그쪽 인자 그대로; jC·jA 명령 그대로).

사용: python exe/wmconv_wrap.py --wc 1 <wmjitter_wrap 인자...> <실행기.py> <실행기 인자...>
"""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.setrecursionlimit(max(sys.getrecursionlimit(), 20000))
sys.path.insert(0, os.path.join(HERE, "..", "lib"))

argv = sys.argv[1:]
out = argv[argv.index("--out") + 1] if "--out" in argv else "."
os.makedirs(out, exist_ok=True)
O = dict(wc="0")
while argv and argv[0].startswith("--wc"):
    O[argv[0][2:]] = argv[1]
    del argv[:2]
if int(float(O["wc"])):
    from anchorflow import fem_wmconv as WC
    WC.install(out)
sys.argv = [os.path.join(HERE, "wmjitter_wrap.py")] + argv
runpy.run_path(sys.argv[0], run_name="__main__")

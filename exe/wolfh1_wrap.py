"""절 CM52 묶음 wolfh1 감싸개: 고침 후보 (lib/anchorflow/fem_wolfh1.py) 를 붙인 뒤 exe/ghost3_wrap.py (--g3 없으면 손대지 않고) -> exe/escape4_wrap.py 로 넘김
(그쪽 --e5r·--e5a, escape3 의 --e3w, escape2·escape1·speed3 인자는 그대로 넘어감).

사용: python exe/wolfh1_wrap.py [--h1n 1] <escape4_wrap 인자...> <실행기.py> <실행기 인자...>
  --h1n 1   N: 국소 시작점 섞기 β 를 바닥 법선 성분에만 (접선 시작 = 관성 예측; --y0loc 위). 없으면 이 모듈을 안 부름.
"""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lib"))

argv = sys.argv[1:]
CAND = {}
while argv and argv[0].startswith("--h1"):
    CAND[argv[0][2:]] = argv[1]
    del argv[:2]
if int(float(CAND.get("h1n", 0))):
    assert "--y0loc" in argv, "--h1n 은 --y0loc 위"
    from anchorflow import fem_wolfh1 as H1
    H1.install(n=int(float(CAND["h1n"])))
    print(f"[절 CM52] 시작점 섞기 β 를 법선 성분에만 (N {H1.ST['n']})", flush=True)
sys.argv = [os.path.join(HERE, "ghost3_wrap.py")] + argv                                 # ghost3_wrap 은 --g3 없으면 그대로 escape4_wrap 으로 넘김
runpy.run_path(sys.argv[0], run_name="__main__")

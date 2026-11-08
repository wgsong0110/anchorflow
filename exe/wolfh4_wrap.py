"""절 CM68 묶음 wolfh4 감싸개: 인자 --w4* 를 떼고 exe/wolfh3_wrap.py (절 CM66, 그쪽 인자 --w3* 그대로) 를 띄움. 실행기 (fem_an_rt_sim.py) 를 runpy 로 띄우는 순간
(wolfh3 를 포함한 다른 감싸개가 모두 설치된 뒤) lib/anchorflow/fem_wolfh4.py 를 설치.

사용: python exe/wolfh4_wrap.py [--w4fb 1] [--w4dg 1] <wolfh3_wrap 인자...> <실행기.py> <실행기 인자...>
  --w4fb 1  F-bar: 칸 (팔분 8 가우스 점) 평균 증분 체적비로 X̄ = (j̄/j)^{1/3} X (응력·RM·커밋), 힘은 변분 꼴 (압력 칸 평균), 접선은 편차 점별 + 칸 순위 1 체적
  --w4dg 1  진단: 걸음 끝 가우스 점 RM 상태·소성 증분·압력·편차 응력·J 칸 안 분산을 바닥 높이 칸별로 (w4dg.json)
인자가 하나도 없으면 모듈을 안 부름 (wolfh3_wrap 그대로).
"""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lib"))

argv = sys.argv[1:]
out = argv[argv.index("--out") + 1] if "--out" in argv else "."
os.makedirs(out, exist_ok=True)
CAND = {}
while argv and argv[0].startswith("--w4"):
    CAND[argv[0][4:]] = int(float(argv[1]))
    del argv[:2]
ON = any(CAND.values())

_orig = runpy.run_path
_st = dict(done=False)


def run_path(path, *a, **kw):
    if ON and not _st["done"] and os.path.basename(path) in ("fem_an_rt_sim.py",):
        _st["done"] = True
        from anchorflow import fem_wolfh4 as W4
        W4.install(fb=CAND.get("fb", 0), dg=CAND.get("dg", 0), out=out)
    return _orig(path, *a, **kw)


runpy.run_path = run_path
sys.argv = [os.path.join(HERE, "wolfh3_wrap.py")] + argv
_orig(sys.argv[0], run_name="__main__")

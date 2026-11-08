"""절 CM66 묶음 wolfh3 감싸개: 인자 --w3* 를 떼고 exe/wolfh2_wrap.py (절 CM63, 그쪽 인자 --w2* 그대로) 를 띄움. 실행기 (fem_an_rt_sim.py) 를 runpy 로 띄우는 순간
(wolfh2 를 포함한 다른 감싸개가 모두 설치된 뒤) lib/anchorflow/fem_wolfh3.py 를 설치.

사용: python exe/wolfh3_wrap.py [--w3y 1] [--w3nt 1] [--w3dg 1] <wolfh2_wrap 인자...> <실행기.py> <실행기 인자...>
  --w3y 1   Y5: 국소 시작점 섞기의 장벽 에너지 판정을 바닥 법선 몫에만 (법선 1 차원 관성 + 장벽 최소), 접선은 실현 가능 규칙까지만
  --w3nt 1  Y5 위 N 꼴: 접선 = 관성 예측 그대로
  --w3dg 1  진단: 걸음별 높이 칸별 시작점·걸음 끝 실현 몫, --y0eb 가 올린 점, 묶인 점 (w3dg.json)
인자가 하나도 없으면 모듈을 안 부름 (wolfh2_wrap 그대로).
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
while argv and argv[0].startswith("--w3"):
    CAND[argv[0][4:]] = int(float(argv[1]))
    del argv[:2]
ON = any(CAND.values())
if CAND.get("y") or CAND.get("dg"):
    assert "--y0loc" in argv and "--fl_bar" in argv, "--w3y·--w3dg 는 --y0loc 장벽 판 위"
if CAND.get("y"):
    assert "--w2n" not in argv, "--w3y 와 --w2n 은 같이 안 씀 (Y5 위 N 은 --w3nt)"

_orig = runpy.run_path
_st = dict(done=False)


def run_path(path, *a, **kw):
    if ON and not _st["done"] and os.path.basename(path) in ("fem_an_rt_sim.py",):
        _st["done"] = True
        from anchorflow import fem_wolfh3 as W3
        W3.install(y=CAND.get("y", 0), nt=CAND.get("nt", 0), dg=CAND.get("dg", 0), out=out)
    return _orig(path, *a, **kw)


runpy.run_path = run_path
sys.argv = [os.path.join(HERE, "wolfh2_wrap.py")] + argv
_orig(sys.argv[0], run_name="__main__")

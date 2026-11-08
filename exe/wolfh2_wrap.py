"""절 CM63 묶음 wolfh2 감싸개: 인자 --w2* 를 떼고 exe/hhtc2_wrap.py (절 CM61, 그쪽 인자 그대로) 를 띄움. 실행기 (fem_an_rt_sim.py) 를 runpy 로 띄우는 순간
(다른 감싸개가 모두 설치된 뒤) lib/anchorflow/fem_wolfh2.py 를 설치.

사용: python exe/wolfh2_wrap.py [--w2n 1] [--w2s 1] [--w2gk0 1] [--w2gs 1] <hhtc2_wrap 인자...> <실행기.py> <실행기 인자...>
  --w2n 1    N: 시작점 섞기 β 를 바닥 법선 성분에만 (절 CM52)
  --w2s 1    S: 팔분 칸 높이 분포 위 정답 sticky 고착 몫 (--fl_pt 위)
  --w2gk0 1  진단: ghost penalty 강성 0 (고스트 질량만)
  --w2gs 1   바닥 고착 노드 (an > 0) 에 닿는 ghost penalty 면 빼기
인자가 하나도 없으면 모듈을 안 부름 (hhtc2_wrap 그대로).
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
while argv and argv[0].startswith("--w2"):
    CAND[argv[0][4:]] = int(float(argv[1]))
    del argv[:2]
ON = any(CAND.values())
cell = None
for i_, a_ in enumerate(argv):
    if a_ == "--cell":
        cell = float(argv[i_ + 1])                                                    # 마지막 --cell (실행기 argparse 와 같음)
if CAND.get("s"):
    assert "--fl_pt" in argv, "--w2s 는 --fl_pt 위"
    assert cell is not None, "--w2s: --cell 없음"
if CAND.get("n"):
    assert "--y0loc" in argv, "--w2n 은 --y0loc 위"
if CAND.get("gk0") or CAND.get("gs"):
    assert "--g4" in argv, "--w2gk0·--w2gs 는 ghost penalty (--g4) 위"

_orig = runpy.run_path
_st = dict(done=False)


def run_path(path, *a, **kw):
    if ON and not _st["done"] and os.path.basename(path) in ("fem_an_rt_sim.py",):
        _st["done"] = True
        from anchorflow import fem_wolfh2 as W2
        W2.install(n=CAND.get("n", 0), s=CAND.get("s", 0), gk0=CAND.get("gk0", 0), gs=CAND.get("gs", 0), cell=cell, out=out)
    return _orig(path, *a, **kw)


runpy.run_path = run_path
sys.argv = [os.path.join(HERE, "hhtc2_wrap.py")] + argv
_orig(sys.argv[0], run_name="__main__")

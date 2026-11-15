"""절 CM178 묶음 tetfix 감싸개 (가장 바깥): 사면체 TC 레고 발산 추적 (--tfx 1, 결과 불변) + 수정 플래그 (기본 0) -- lib/anchorflow/fem_tetfix.py.

사용: python exe/tetfix_wrap.py --tfx 0|1 [--tfw f0:f1] exe/tetc_wrap.py --tcm 1 ... <실행기 인자>
  --tff 1 [--tfth thinv]: 수정 (사면체 뒤집힘 불가능 선탐색, 기본 0). --tfx 1: 프레임 창 f0:f1 (0 부터, 기본 6:12) 의 걸음 추적 기록 -> <out>/tetfix.jsonl. 실행기 인자에 --nrec 이 있어야 함 (뉴턴 반복별 사본 버퍼).
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
MODE, WIN, FF, TH = None, (6, 12), 0, 1.125
while argv and argv[0] in ("--tfx", "--tfw", "--tff", "--tfth"):
    if argv[0] == "--tfx":
        MODE = int(float(argv[1]))
    elif argv[0] == "--tfw":
        WIN = tuple(int(q) for q in argv[1].split(":"))
    elif argv[0] == "--tff":
        FF = int(float(argv[1]))
    else:
        TH = float(argv[1])
    del argv[:2]
assert MODE in (0, 1), "--tfx 0|1 이 필요"
if MODE >= 1:
    assert "--nrec" in argv, "--tfx 1 은 실행기 --nrec 이 필요"
if MODE >= 1 or FF:
    from anchorflow import fem_tetfix as TF
    TF.install(out, MODE, WIN, FF, TH)
sys.argv = list(argv)
runpy.run_path(sys.argv[0], run_name="__main__")

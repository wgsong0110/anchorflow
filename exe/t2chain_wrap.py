"""절 CM56 묶음 t2chain 감싸개: 시간 예산 T2 덜 풀린 걸음 연쇄 끊기 (lib/anchorflow/fem_t2chain.py) 를 붙인 뒤 exe/speed4_wrap.py 로 넘김 (그쪽 인자는 그대로).

사용: python exe/t2chain_wrap.py --tc M <speed4_wrap 인자...> <실행기.py> <실행기 인자...>   (--tb 시간 예산 인자가 뒤 감싸개에 있어야 함)
  --tc M   비트 합: 0 진단만 (결과 불변) · 1 누적 이월 (마감 시각) · 4 한 프레임 이월 · 2 자른 걸음 몸 전체 후퇴 오일러 -> <out>/t2c.jsonl
--tc 가 없으면 이 모듈을 안 부름 (소스·그래프 그대로).
"""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lib"))

argv = sys.argv[1:]
out = argv[argv.index("--out") + 1] if "--out" in argv else "."
os.makedirs(out, exist_ok=True)
OPT = {}
while argv and argv[0].startswith("--tc"):
    OPT[argv[0][2:]] = int(float(argv[1]))
    del argv[:2]
if "tc" in OPT:
    assert "--tb" in argv, "--tc 는 시간 예산 (--tb) 판 위에서만"
    from anchorflow import fem_t2chain as TC
    TC.install(OPT["tc"], out)

sys.argv = [os.path.join(HERE, "speed4_wrap.py")] + argv
runpy.run_path(sys.argv[0], run_name="__main__")

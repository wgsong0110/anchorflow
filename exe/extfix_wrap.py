"""절 CM132 묶음 extfix 감싸개: 확장 B-스플라인 T1E2 외부 노드 정의 고침 (lib/anchorflow/fem_extfix.py) -- 새 플래그, 기본 0 (둘 다 0 이면 아무것도 안 바꿈).
사용: python exe/extfix_wrap.py [--xfl 0|1] [--xls 0|1|2] <다음 감싸개.py (보통 exe/ext2_wrap.py)> <그 인자...> <실행기.py> <실행기 인자...>
  --xfl 1 (다): 바닥 평면 아래 통째인 칸은 내부 칸 판정 (T1 26 이웃 점유) 에서 빈 칸으로 세지 않음.
  --xls 1 (가): 외부 노드 확장 계수 = P2G 입자·질량 가중 최소제곱 사영 (순수), 2 = 같은 사영을 아핀 재현 제약 아래.
설치: import 때 fem_wolfh8.install 을 감쌈 (ext2 -> ext1 -> wolfh8 설치가 그 자리에서 이 고침을 같이 닮). 안쪽 감싸개 순서·인자는 그대로."""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lib"))
argv = sys.argv[1:]
out = argv[argv.index("--out") + 1] if "--out" in argv else "."
os.makedirs(out, exist_ok=True)
C = dict(xfl="0", xls="0")
while argv and argv[0] in ("--xfl", "--xls"):
    C[argv[0][2:]] = argv[1]
    del argv[:2]
from anchorflow import fem_extfix as XF  # noqa: E402

XF.install(fl=int(float(C["xfl"])), ls=int(float(C["xls"])), out=out)
assert argv and argv[0].endswith(".py"), "extfix_wrap: 다음 감싸개 .py 가 첫 인자"
sys.argv = argv
runpy.run_path(sys.argv[0], run_name="__main__")

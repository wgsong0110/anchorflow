"""절 CM133 묶음 extth 감싸개: 확장 B-스플라인 T1E2 외부 노드를 채움 비 θ 로 원래 자유도와 연속으로 섞음 (lib/anchorflow/fem_extth.py) -- 새 플래그, 기본 0 (0 이면 아무것도 안 바꿈).
사용: python exe/extth_wrap.py [--xth 0|1|2] <다음 감싸개.py (보통 exe/ext2_wrap.py)> <그 인자...> <실행기.py> <실행기 인자...>
  --xth 1: θ = 점유 칸 안 기저 적분 몫 (0~1), 2: θ = m / (ρ̄ h³) (ρ̄ = 점유 영역 평균 밀도, 지시 식 그대로).
설치: import 때 fem_wolfh8.install 을 감쌈 (ext2 -> ext1 -> wolfh8 설치가 그 자리에서 같이 닮). 안쪽 감싸개 순서·인자는 그대로."""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lib"))
argv = sys.argv[1:]
out = argv[argv.index("--out") + 1] if "--out" in argv else "."
os.makedirs(out, exist_ok=True)
C = dict(xth="0")
while argv and argv[0] in ("--xth",):
    C[argv[0][2:]] = argv[1]
    del argv[:2]
from anchorflow import fem_extth as XT  # noqa: E402

XT.install(th=int(float(C["xth"])), out=out)
assert argv and argv[0].endswith(".py"), "extth_wrap: 다음 감싸개 .py 가 첫 인자"
sys.argv = argv
runpy.run_path(sys.argv[0], run_name="__main__")

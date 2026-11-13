"""절 CM138 감싸개: --ctan N (기본 0 = 옛 판) 을 환경 AF_CTAN 으로 두고 (fem_ao_rt 가 import 될 때 AO_SRC 에 매크로로 들어감) 다음 감싸개로 넘김.
N=1: 반환 사상 꼭짓점 (st 2, 응력 0) 인 가우스 점의 헤시안을 0 으로 (기울기와 일관).  N=2: 1 + 꼬리 (st 1, 마찰 원뿔 위) 점의 헤시안을 DP 반환 사상의 일관 접선 (대칭화) 으로.
사용: python exe/ctan_wrap.py --ctan N <다음 감싸개.py> <그 인자...> <실행기.py> <실행기 인자...>"""
import os
import runpy
import sys

argv = sys.argv[1:]
assert argv and argv[0] == "--ctan", "ctan_wrap: 첫 인자는 --ctan N"
os.environ["AF_CTAN"] = str(int(argv[1]))
argv = argv[2:]
assert argv and argv[0].endswith(".py"), "ctan_wrap: 다음 감싸개 .py"
sys.argv = argv
runpy.run_path(sys.argv[0], run_name="__main__")

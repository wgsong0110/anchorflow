"""절 CM189 감싸개: --ptan N (기본 0 = 옛 판) 을 환경 AF_PTAN 으로 두고 (fem_ao_rt 가 import 될 때 AO_SRC 에 매크로로 들어감) 다음 감싸개로 넘김.
N=1: 재질 커널 (aq_elem, 재질 1·3·5 = PG metal·foam·plasticine) 헤시안을 반환 사상과 일관된 접선 (합성 주값 응력 P(RM(σ)) 의 중심 차분) 으로. 모래 (ao_elem) 와 재질 0·7 은 그대로.
사용: python exe/ptan_wrap.py --ptan N <다음 감싸개.py> <그 인자...> <실행기.py> <실행기 인자...>"""
import os
import runpy
import sys

argv = sys.argv[1:]
assert argv and argv[0] == "--ptan", "ptan_wrap: 첫 인자는 --ptan N"
os.environ["AF_PTAN"] = str(int(argv[1]))
argv = argv[2:]
assert argv and argv[0].endswith(".py"), "ptan_wrap: 다음 감싸개 .py"
sys.argv = argv
runpy.run_path(sys.argv[0], run_name="__main__")

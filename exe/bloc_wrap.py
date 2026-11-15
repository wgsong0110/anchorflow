"""절 CM189 감싸개: --bloc N (기본 0 = 옛 판) 을 환경 AF_BLOC 으로 두고 (fem_ao_rt 가 import 될 때 AO_SRC 에 매크로로 들어감) 다음 감싸개로 넘김.
N=1: 재질 판 (장벽 바닥, 선탐색 없음) 뉴턴 보폭의 장벽 실현 가능 몫을 몸 전체 스칼라 대신 노드별로 (표본점 관통 쪽 몫 보수적 k_q -> 27 노드 최대; 가우스 점 뒤집힘 보폭은 전역 그대로).
N=2: N=1 + 뉴턴 시작점 국소 섞기 (--y0loc) 도 관통 쪽 몫 S⁻ 보수적 k_q (몸 전체 되돌림 없음, 고침 라운드 0).
사용: python exe/bloc_wrap.py --bloc N <다음 감싸개.py> <그 인자...> <실행기.py> <실행기 인자...>"""
import os
import runpy
import sys

argv = sys.argv[1:]
assert argv and argv[0] == "--bloc", "bloc_wrap: 첫 인자는 --bloc N"
os.environ["AF_BLOC"] = str(int(argv[1]))
argv = argv[2:]
assert argv and argv[0].endswith(".py"), "bloc_wrap: 다음 감싸개 .py"
sys.argv = argv
runpy.run_path(sys.argv[0], run_name="__main__")

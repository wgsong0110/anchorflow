"""절 CM128 감싸개: 뒤집힌 가우스 점 = 압축 (lib/anchorflow/fem_sinv.py) 을 단 뒤 다음 감싸개 (첫 인자) 로 넘김 (그쪽 인자 그대로).

사용: python exe/sinv_wrap.py [--xsinv 0|1] <다음 감싸개.py> <그 인자...> <실행기.py> <실행기 인자...>
  --xsinv 0 (기본) = 모듈을 안 부름 (예전 그대로) · 1 = 부호 있는 σ 변형률 (σ ≤ 0 은 σ − 1) 로 반환 사상·응력·헤시안·커밋/G2P F_E.
설치: 이 감싸개가 시작하자마자 (명령줄 맨 앞에 둠) cupy.RawModule 를 감쌈 -> 다른 감싸개 훅이 그 위에 쌓이므로 최종 CUDA 글자에 치환.
"""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lib"))
sys.setrecursionlimit(max(sys.getrecursionlimit(), 16000))

argv = sys.argv[1:]
out = argv[argv.index("--out") + 1] if "--out" in argv else "."
os.makedirs(out, exist_ok=True)
MODE = 0
while argv and argv[0] == "--xsinv":
    MODE = int(float(argv[1]))
    del argv[:2]
if MODE:
    from anchorflow import fem_sinv as SV
    SV.install(out, MODE)
assert argv and argv[0].endswith(".py"), "sinv_wrap: 다음 감싸개 .py 가 첫 인자"
sys.argv = argv
runpy.run_path(sys.argv[0], run_name="__main__")

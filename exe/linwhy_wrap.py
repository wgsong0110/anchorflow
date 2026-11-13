"""절 CM127 묶음 linwhy 감싸개: 뉴턴 선형계 측정 (lib/anchorflow/fem_linwhy.py; 고른 프레임만 그래프 대신 같은 커널을 바로 실행, 분석 뒤 버퍼 되돌림) 을 단 뒤 다음 감싸개 (첫 인자) 로 넘김.
사용: python exe/linwhy_wrap.py [--lw 0|1] <다음 감싸개.py> <그 인자...> <실행기.py> <실행기 인자...>   (env LW_F·LW_IT·LW_LZ·LW_NPCG·LW_ASM·LW_R)
--lw 가 없거나 0 이면 모듈을 안 부름. 설치: 새 실행기 fem_an_rt_sim.py 를 runpy 로 띄우는 순간 (안쪽 감싸개들 설치 뒤 = 가장 바깥)."""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lib"))
sys.setrecursionlimit(max(sys.getrecursionlimit(), 16000))
argv = sys.argv[1:]
out = argv[argv.index("--out") + 1] if "--out" in argv else "."
os.makedirs(out, exist_ok=True)
ON = 0
while argv and argv[0] == "--lw":
    ON = int(float(argv[1]))
    del argv[:2]
_orig = runpy.run_path
_done = [False]


def run_path(path, *a, **kw):
    if ON and not _done[0] and os.path.basename(path) == "fem_an_rt_sim.py":
        _done[0] = True
        from anchorflow import fem_linwhy as LW
        LW.install(out)
    return _orig(path, *a, **kw)


runpy.run_path = run_path
assert argv and argv[0].endswith(".py"), "linwhy_wrap: 다음 감싸개 .py 가 첫 인자"
sys.argv = [os.path.abspath(argv[0])] + argv[1:]
_orig(sys.argv[0], run_name="__main__")

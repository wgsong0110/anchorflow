"""절 CM129 감싸개: local-global PD 모래 판 (lib/anchorflow/fem_pd.py) 을 단 뒤 다음 감싸개 (첫 인자, 예 exe/wolfh12_wrap.py · exe/speed13_wrap.py) 로 넘김 (그쪽 인자 그대로).

사용: python exe/pd_wrap.py [--xpd 0|1] <다음 감싸개.py> <그 인자...> <실행기.py> <실행기 인자...>
  --xpd 0 (기본) = 모듈을 안 부름 (예전 그대로) · 1 = 모래 헤시안 자리에 상수 PD 행렬 (local = 반환 사상 힘).
설치: 새 실행기 fem_an_rt_sim.py 를 runpy 로 띄우는 순간 (가장 바깥 RawModule 훅 = 원래 글자를 먼저 봄).
"""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lib"))
sys.setrecursionlimit(max(sys.getrecursionlimit(), 14000))

argv = sys.argv[1:]
out = argv[argv.index("--out") + 1] if "--out" in argv else "."
os.makedirs(out, exist_ok=True)
MODE = 0
while argv and argv[0] == "--xpd":
    MODE = int(float(argv[1]))
    del argv[:2]

_orig = runpy.run_path
_done = [False]


def run_path(path, *a, **kw):
    if MODE and not _done[0] and os.path.basename(path) == "fem_an_rt_sim.py":
        _done[0] = True
        from anchorflow import fem_pd as LG
        LG.install(out, MODE)
    return _orig(path, *a, **kw)


runpy.run_path = run_path
assert argv and argv[0].endswith(".py"), "pd_wrap: 다음 감싸개 .py 가 첫 인자"
sys.argv = argv
_orig(sys.argv[0], run_name="__main__")

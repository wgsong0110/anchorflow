"""절 CM69 묶음 lightg 감싸개: 노드 가벼움 연속 가중 벌점 (lib/anchorflow/fem_lightg.py) 을 단 뒤 exe/mfix_wrap.py 로 넘김 (그쪽 인자 그대로).

사용: python exe/lightg_wrap.py [--lg γ̄_L] [--lgagg 0|1] [--lgdg 0|1] [--lgfd 프레임,...] <mfix_wrap 인자...> <실행기.py> <실행기 인자...>
  --lg 가 없으면 모듈을 안 부름 (mfix_wrap 그대로). m̄ = 0 인 씬 (레고·ficus) 은 모듈이 켜져도 아무 커널도 안 띄움 (P1X 그대로).
모듈 설치는 실행기 (fem_an_rt_sim.py / fem_ao_rt_sim.py) 를 runpy 로 띄우는 순간 (다른 훅 설치 뒤 = 가장 바깥).
"""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lib"))

argv = sys.argv[1:]
out = argv[argv.index("--out") + 1] if "--out" in argv else "."
os.makedirs(out, exist_ok=True)
C = {}
while argv and argv[0].startswith("--lg"):
    C[argv[0][4:]] = argv[1]
    del argv[:2]
if "" in C:
    os.environ["AF_LG_G"] = str(float(C[""]))
    os.environ["AF_LG_AGG"] = str(int(float(C.get("agg", 0))))
    os.environ["AF_LG_DG"] = str(int(float(C.get("dg", 0))))
    os.environ["AF_LG_FD"] = C.get("fd", "")

_orig = runpy.run_path
_st = dict(on=False)


def run_path(path, *a, **kw):
    b = os.path.basename(path)
    if "" in C and not _st["on"] and b in ("fem_an_rt_sim.py", "fem_ao_rt_sim.py"):
        _st["on"] = True
        from anchorflow import fem_lightg as LG
        LG.install(out)
    return _orig(path, *a, **kw)


runpy.run_path = run_path
sys.argv = [os.path.join(HERE, "mfix_wrap.py")] + argv
_orig(sys.argv[0], run_name="__main__")

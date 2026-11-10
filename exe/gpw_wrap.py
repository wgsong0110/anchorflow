"""절 CM95 묶음 gpw 감싸개: P1X 면 가중 w_F = 1 − φ_min (lib/anchorflow/fem_gpw.py) 를 단 뒤 다음 감싸개로 넘김 (그쪽 인자 그대로; H5 사슬).

사용: python exe/gpw_wrap.py [--gpw 모드] [--gpwdg 1] <다음 감싸개.py> <그 인자...> <실행기.py> <실행기 인자...>
  모드 1 = (A) 모든 씬, 2 = (B) κ 0 씬 (wolf) 만. --gpw 가 없거나 0 이면 모듈을 안 부름 (예전 그대로).
설치: 실행기 fem_an_rt_sim.py · fem_ao_rt_sim.py (exe/s6/ 사본 포함) 를 runpy 로 띄우는 순간 (안쪽 감싸개들 = lightg3·lightg4 설치 뒤, 가장 바깥).
"""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lib"))
sys.setrecursionlimit(max(sys.getrecursionlimit(), 16000))                                # 감싸개 한 겹 더

argv = sys.argv[1:]
out = argv[argv.index("--out") + 1] if "--out" in argv else "."
os.makedirs(out, exist_ok=True)
O = dict(gpw="0", gpwdg="0")
while argv and argv[0] in ("--gpw", "--gpwdg"):
    O[argv[0][2:]] = argv[1]
    del argv[:2]
MODE = int(float(O["gpw"]))
if int(float(O["gpwdg"])):
    os.environ["AF_GPW_DG"] = "1"

_orig = runpy.run_path
_done = [False]


def run_path(path, *a, **kw):
    if MODE and not _done[0] and os.path.basename(path) in ("fem_an_rt_sim.py", "fem_ao_rt_sim.py"):
        _done[0] = True
        from anchorflow import fem_gpw as GPW
        GPW.DG = int(os.environ.get("AF_GPW_DG", "0"))
        GPW.install(out, MODE)
    return _orig(path, *a, **kw)


runpy.run_path = run_path
sys.argv = list(argv)
_orig(sys.argv[0], run_name="__main__")

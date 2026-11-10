"""절 CM99 묶음 gpw2 감싸개: 절 CM95 면 가중 w_F = 1 − φ_min (lib/anchorflow/fem_gpw.py) + 칸 점유 φ 새 정의 (lib/anchorflow/fem_gpw2.py) 를 단 뒤 다음 감싸개로 넘김 (H5 사슬).

사용: python exe/gpw2_wrap.py [--gpw 모드] [--gpwdg 1] [--gphi 정의] <다음 감싸개.py> <그 인자...> <실행기.py> <실행기 인자...>
  --gpw: 1 = (A) 모든 씬, 2 = (B) κ 0 씬 (wolf) 만 (절 CM95 그대로). --gphi: 1 = (G) 팔분 기하 채움, 2 = (N) 첫 프레임 정규화 min(1, φ/φ_0).
  --gpw 가 없거나 0 이면 아무 모듈도 안 부름. --gphi 0 이면 절 CM95 그대로.
설치: 실행기 fem_an_rt_sim.py · fem_ao_rt_sim.py (exe/s6/ 사본 포함) 를 runpy 로 띄우는 순간 (안쪽 감싸개들 뒤, 가장 바깥): fem_gpw.install 다음 fem_gpw2.install.
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
O = dict(gpw="0", gpwdg="0", gphi="0")
while argv and argv[0] in ("--gpw", "--gpwdg", "--gphi"):
    O[argv[0][2:]] = argv[1]
    del argv[:2]
MODE = int(float(O["gpw"]))
PHI = int(float(O["gphi"]))
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
        if PHI:
            from anchorflow import fem_gpw2 as GP2
            GP2.install(out, PHI)
    return _orig(path, *a, **kw)


runpy.run_path = run_path
sys.argv = list(argv)
_orig(sys.argv[0], run_name="__main__")

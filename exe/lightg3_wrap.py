"""절 CM73 묶음 lightg3 감싸개: 가벼움 연속 가중 벌점의 위상 기준 정의 (A)·(B) (lib/anchorflow/fem_lightg3.py) 를 단 뒤 exe/mfix_wrap.py 로 넘김 (그쪽 인자 그대로).

사용: python exe/lightg3_wrap.py [--l3 γ̄_L] [--l3sc 0|1] [--l3fr 0|1|2] [--l3r0 0|1] [--l3dg 0|1] [--l3ck 0|1] [--l3fd 프레임,...] <mfix_wrap 인자...> <실행기.py> <실행기 인자...>
  --l3sc 0 = 가벼움 몫을 후보 면 전부에, 1 = P1X 밖 면에만. --l3fr 0 = 절 CM71 λ_free, 1 = (A) 원래 고정 판정에서 자유인 노드만, 2 = (B) (A) + 장벽 받침 표시 노드 낀 면 뺌.
  --l3r0 1 = 진단 전용 (가중 λ 몫 0 = G'', λ 만 기록). --l3ck 1 = 금 면 진단.
  --l3 가 없으면 모듈을 안 부름 (mfix_wrap 그대로). m̄ = 0 인 씬 (레고·ficus) 은 모듈이 켜져도 아무 커널도 안 띄움 (P1X 그대로).
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
while argv and argv[0].startswith("--l3"):
    C[argv[0][4:]] = argv[1]
    del argv[:2]
if "" in C:
    os.environ["AF_L3_G"] = str(float(C[""]))
    os.environ["AF_L3_SC"] = str(int(float(C.get("sc", 0))))
    os.environ["AF_L3_FR"] = str(int(float(C.get("fr", 1))))
    os.environ["AF_L3_R0"] = str(int(float(C.get("r0", 0))))
    os.environ["AF_L3_DG"] = str(int(float(C.get("dg", 0))))
    os.environ["AF_L3_CK"] = str(int(float(C.get("ck", 0))))
    os.environ["AF_L3_FD"] = C.get("fd", "")

_orig = runpy.run_path
_st = dict(on=False)


def run_path(path, *a, **kw):
    b = os.path.basename(path)
    if "" in C and not _st["on"] and b in ("fem_an_rt_sim.py", "fem_ao_rt_sim.py"):
        _st["on"] = True
        from anchorflow import fem_lightg3 as L3
        L3.install(out)
    return _orig(path, *a, **kw)


runpy.run_path = run_path
sys.argv = [os.path.join(HERE, "mfix_wrap.py")] + argv
_orig(sys.argv[0], run_name="__main__")

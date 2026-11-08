"""절 CM67 묶음 mfix 감싸개: 저질량 고정 노드 정의 (lib/anchorflow/fem_mfix.py, --mf1 1 연속 고정 (관성 덧질량) · --mf2 1 장벽 표본점 27 노드 고정 빼기) 를 단 뒤 exe/wmfloor_wrap.py 로 넘김 (그쪽 인자 그대로).

사용: python exe/mfix_wrap.py [--mf1 0|1] [--mf2 0|1] <wmfloor_wrap 인자...> <실행기.py> <실행기 인자...>
  인자가 하나도 없으면 모듈을 안 부름 (wmfloor_wrap 그대로). 둘 다 0 이면 진단 mf.jsonl 만 (결과 불변 -- 커널 안 띄움).
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
while argv and argv[0].startswith("--mf"):
    C[argv[0][4:]] = argv[1]
    del argv[:2]
if C:
    os.environ["AF_MF1"] = str(int(float(C.get("1", 0))))
    os.environ["AF_MF2"] = str(int(float(C.get("2", 0))))

_orig = runpy.run_path
_st = dict(on=False)


def run_path(path, *a, **kw):
    b = os.path.basename(path)
    if C and not _st["on"] and b in ("fem_an_rt_sim.py", "fem_ao_rt_sim.py"):
        _st["on"] = True
        from anchorflow import fem_mfix as MF
        MF.install(out)
    return _orig(path, *a, **kw)


runpy.run_path = run_path
sys.argv = [os.path.join(HERE, "wmfloor_wrap.py")] + argv
_orig(sys.argv[0], run_name="__main__")

"""절 CM64 묶음 wmfloor 감싸개: 고스트 질량 면 넓히기 (lib/anchorflow/fem_wmfloor.py, --wfm 0|5) + 진단 D6·검사를 단 뒤 exe/hhtc2_wrap.py 로 넘김 (그쪽 인자 그대로).

사용: python exe/wmfloor_wrap.py [--wfm M] [--wfd6 1] [--wfchk 프레임,...] <hhtc2_wrap 인자...> <실행기.py> <실행기 인자...>
  --wfm M    0 = M0 (고스트 질량 면 = 후보 면 전부) · 5 = M5 (한쪽 이상 두꺼운 칸인 후보 면) · -1 = 더한 면 없음 (진단만, 결과 불변). 강성 벌점 면은 P1X 그대로.
  --wfd6 1   (결과 불변) 프레임 끝 바닥 밑 입자 기록 -> <out>/wf.jsonl 'd6'
  --wfchk L  (결과 불변) 프레임 목록 L 끝에 면 수·곱 검사 -> wf.jsonl 'chk'
  인자가 하나도 없으면 모듈을 안 부름 (hhtc2_wrap 그대로).
모듈 설치는 실행기 (fem_an_rt_sim.py / fem_ao_rt_sim.py) 를 runpy 로 띄우는 순간 (ghost6·hhtc2 설치 뒤 = 가장 바깥).
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
while argv and argv[0].startswith("--wf"):
    C[argv[0][4:]] = argv[1]
    del argv[:2]
if C:
    os.environ["AF_WF_MM"] = str(int(float(C.get("m", -1))))
    os.environ["AF_WF_D6"] = str(int(float(C.get("d6", 0))))
    os.environ["AF_WF_CHK"] = C.get("chk", "")

_orig = runpy.run_path
_st = dict(on=False)


def run_path(path, *a, **kw):
    b = os.path.basename(path)
    if C and not _st["on"] and b in ("fem_an_rt_sim.py", "fem_ao_rt_sim.py"):
        _st["on"] = True
        from anchorflow import fem_wmfloor as WF
        WF.install(out)
    return _orig(path, *a, **kw)


runpy.run_path = run_path
sys.argv = [os.path.join(HERE, "hhtc2_wrap.py")] + argv
_orig(sys.argv[0], run_name="__main__")

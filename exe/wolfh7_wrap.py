"""절 CM79 묶음 wolfh7 감싸개 (exe/wolfh6_wrap.py 복사): (Y7) P1X 강성 × 면 항복 가중 (lib/anchorflow/fem_wolfh7.py) 을 단 뒤
공통 후보 사슬로 넘김: exe/lightg4_wrap.py ('--l4') -> lightg3_wrap ('--l3') -> wolfh5_wrap ('--w5', S_구속·C) -> [첫 mfix_wrap 자리를 한 번 exe/speed9_wrap.py (--s9 있을 때) 또는 speed8_wrap.py 로]
-> (speed9 ->) speed8 -> speed7_wrap ('--s7'·'--s6') -> mfix_wrap ('--mf') -> wmfloor_wrap -> hhtc2_wrap -> ...

사용: python exe/wolfh7_wrap.py [--w7y 1] <lightg4_wrap 인자 ... (--l4 · --l3 · --w5 · --s9 · --s8 · --s7 · --s6 · --mf ... 이 차례로)> <실행기.py> <실행기 인자...>
  '--w7y' 가 없거나 0 이면 모듈을 안 부름 (사슬만). '--s6|--s7|--s8|--s9' 가 하나도 없으면 mfix_wrap 을 안 바꿈 (속도 판 없는 사슬 = 절 CM75 판).
설치: 실행기를 runpy 로 띄우는 순간 (다른 감싸개가 다 설치된 뒤 = 가장 바깥).
"""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lib"))

argv = sys.argv[1:]
out = argv[argv.index("--out") + 1] if "--out" in argv else "."
os.makedirs(out, exist_ok=True)
C = dict(y="0")
while argv and argv[0].startswith("--w7"):
    C[argv[0][4:]] = argv[1]
    del argv[:2]
ON = any(float(v) != 0 for v in C.values())
HEAD = argv[:argv.index("--out")]
SP = any(a.startswith(("--s6", "--s7", "--s8", "--s9")) for a in HEAD if a.startswith("--"))
S9 = any(a.startswith("--s9") for a in HEAD if a.startswith("--"))

_orig = runpy.run_path
_st = dict(done=False, nmf=0)


def run_path(path, *a, **kw):
    b = os.path.basename(path)
    if b == "mfix_wrap.py" and SP:
        _st["nmf"] += 1
        if _st["nmf"] == 1:                                                                     # wolfh5_wrap 의 다음 = 속도 사슬 (그 끝이 진짜 mfix_wrap)
            path = os.path.join(HERE, "speed9_wrap.py" if S9 else "speed8_wrap.py")
            sys.argv[0] = path
    if ON and not _st["done"] and b in ("fem_an_rt_sim.py", "fem_ao_rt_sim.py"):
        _st["done"] = True
        from anchorflow import fem_wolfh7 as W7
        W7.install(out, Y=int(float(C["y"])))
    return _orig(path, *a, **kw)


runpy.run_path = run_path
sys.argv = [os.path.join(HERE, "lightg4_wrap.py")] + argv
_orig(sys.argv[0], run_name="__main__")

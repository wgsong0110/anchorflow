"""절 CM77 묶음 wolfh6 감싸개: (K) P1X 강성 × 재질 응집 몫 κ · (L) 가벼운 자유 노드 HHT·후퇴 오일러 섞기 (lib/anchorflow/fem_wolfh6.py) 를 단 뒤
공통 후보 H4 사슬로 넘김: exe/lightg4_wrap.py ('--l4') -> lightg3_wrap ('--l3') -> wolfh5_wrap ('--w5', S_구속·C) -> [mfix_wrap 자리를 한 번 exe/speed8_wrap.py 로]
-> speed7_wrap ('--s8'·'--s7'·'--s6') -> mfix_wrap ('--mf') -> wmfloor_wrap -> hhtc2_wrap -> ...

사용: python exe/wolfh6_wrap.py [--w6k 1] [--w6l 1] [--w6x 1] <lightg4_wrap 인자 ... (--l4 · --l3 · --w5 · --s8 · --s7 · --s6 · --mf ... 이 차례로)> <실행기.py> <실행기 인자...>
  '--s6|--s7|--s8' 인자가 있으면 wolfh5_wrap 이 부르는 첫 mfix_wrap.py 를 speed8_wrap.py 로 바꿔 띄움 (speed7_wrap 이 끝에 mfix_wrap 을 부름; 인자는 그대로 이어짐).
  '--w6*' 가 없거나 0 이면 모듈을 안 부름 (사슬만).
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
C = dict(k="0", l="0", x="0")
while argv and argv[0].startswith("--w6"):
    C[argv[0][4:]] = argv[1]
    del argv[:2]
ON = any(float(v) != 0 for v in C.values())
SP = any(a.startswith(("--s6", "--s7", "--s8")) for a in argv[:argv.index("--out")] if a.startswith("--"))

_orig = runpy.run_path
_st = dict(done=False, nmf=0)


def run_path(path, *a, **kw):
    b = os.path.basename(path)
    if b == "mfix_wrap.py" and SP:
        _st["nmf"] += 1
        if _st["nmf"] == 1:                                                                     # wolfh5_wrap 의 다음 = speed8 사슬 (그 끝이 진짜 mfix_wrap)
            path = os.path.join(HERE, "speed8_wrap.py")
            sys.argv[0] = path
    if ON and not _st["done"] and b in ("fem_an_rt_sim.py", "fem_ao_rt_sim.py"):
        _st["done"] = True
        from anchorflow import fem_wolfh6 as W6
        W6.install(out, K=int(float(C["k"])), L=int(float(C["l"])), X=int(float(C["x"])))
    return _orig(path, *a, **kw)


runpy.run_path = run_path
sys.argv = [os.path.join(HERE, "lightg4_wrap.py")] + argv
_orig(sys.argv[0], run_name="__main__")

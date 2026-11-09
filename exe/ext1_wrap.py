"""절 CM83 묶음 ext1 감싸개 (exe/wolfh8_wrap.py 복사): 확장 B-스플라인 T1E2 를 두 실행기에 (lib/anchorflow/fem_ext1.py -- 새 실행기 = fem_wolfh8 훅, 옛 실행기 = fem_ext1 훅)
+ (K) P1X 강성 × κ (lib/anchorflow/fem_wolfh6.py, 절 CM77) 을 단 뒤
공통 후보 사슬로 넘김: exe/lightg4_wrap.py ('--l4') -> lightg3_wrap ('--l3') -> wolfh5_wrap ('--w5', S_구속·C) -> [첫 mfix_wrap 자리를 한 번 exe/speed9_wrap.py (--s9 있을 때) 또는 speed8_wrap.py 로]
-> (speed9 ->) speed8 -> speed7_wrap ('--s7'·'--s6') -> mfix_wrap ('--mf') -> wmfloor_wrap -> hhtc2_wrap -> ...

사용: python exe/ext1_wrap.py [--w8e 1] [--w8d 차수 (기본 2)] [--w8t 내부 칸 정의 0|1|2] [--w8x 1 고립 노드 0 차 확장] [--w8k 1] [--x1ck 1 검사] <lightg4_wrap 인자 ... (--l4 · --l3 · --w5 · --s9 · --s8 · --s7 · --s6 · --mf ... 이 차례로)> <실행기.py> <실행기 인자...>
  '--w8e'·'--w8k' 가 없거나 0 이면 모듈을 안 부름 (사슬만). '--s6|--s7|--s8|--s9' 가 하나도 없으면 mfix_wrap 을 안 바꿈 (속도 판 없는 사슬 = 절 CM75 판).
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
C = dict(e="0", d="2", k="0", t="0", x="0", ck="0")
while argv and argv[0].startswith(("--w8", "--x1")):
    C[argv[0][4:]] = argv[1]
    del argv[:2]
ON = float(C["e"]) != 0 or float(C["k"]) != 0
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
        if float(C["k"]) != 0:
            from anchorflow import fem_wolfh6 as W6
            W6.install(out, K=1)
        if float(C["e"]) != 0:
            from anchorflow import fem_ext1 as X1
            X1.install(out, D=int(float(C["d"])), T=int(float(C["t"])), X=int(float(C["x"])), ck=int(float(C["ck"])))
    return _orig(path, *a, **kw)


runpy.run_path = run_path
sys.argv = [os.path.join(HERE, "lightg4_wrap.py")] + argv
_orig(sys.argv[0], run_name="__main__")

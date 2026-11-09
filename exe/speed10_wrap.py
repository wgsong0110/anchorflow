"""절 CM78 묶음 speed10 감싸개: 노드 상자 압축 N · 첫 할당 상자 배수 K (lib/anchorflow/fem_speed10.py) 를 단 뒤 공통 후보 H5 사슬로 넘김:
exe/lightg4_wrap.py ('--l4') -> lightg3_wrap ('--l3') -> wolfh5_wrap -> [첫 mfix_wrap 자리를 exe/speed9_wrap.py 로] speed9_wrap ('--s9') -> speed8_wrap ('--s8')
-> speed7_wrap ('--s7'·'--s6') -> mfix_wrap ('--mf') -> wmfloor_wrap -> hhtc2_wrap -> ... (exe/wolfh6_wrap.py 와 같은 꼴, 다음이 speed9 사슬).

사용: python exe/speed10_wrap.py [--s10n 1|2] [--s10k all|이름,..] [--s10g 블록] [--s10cap M] <lightg4_wrap 인자 ...> <실행기.py> <실행기 인자...>
  '--s10*' 가 없거나 0 이면 모듈을 안 부름 (사슬만). 소스 고침 (N) 은 감싸개가 뜨는 순간 (가장 안쪽), 나머지 설치는 실행기를 runpy 로 띄우는 순간 (가장 바깥).
"""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lib"))

argv = sys.argv[1:]
out = argv[argv.index("--out") + 1] if "--out" in argv else "."
os.makedirs(out, exist_ok=True)
O = dict(s10n="0", s10k="all", s10g="1024", s10cap="0")
while argv and argv[0].startswith("--s10"):
    O[argv[0][2:]] = argv[1]
    del argv[:2]
ON = int(float(O["s10n"])) != 0 or float(O["s10cap"]) > 0
SP = any(a.startswith(("--s6", "--s7", "--s8", "--s9")) for a in argv[:argv.index("--out")] if a.startswith("--"))
if ON:
    from anchorflow import fem_speed10 as S10
    S10.pre_install(n=int(float(O["s10n"])), k=O["s10k"], g=int(float(O["s10g"])))

_orig = runpy.run_path
_st = dict(done=False, nmf=0)


def run_path(path, *a, **kw):
    b = os.path.basename(path)
    if b == "mfix_wrap.py" and SP:
        _st["nmf"] += 1
        if _st["nmf"] == 1:                                                                     # wolfh5_wrap 의 다음 = speed9 사슬 (그 끝이 진짜 mfix_wrap)
            path = os.path.join(HERE, "speed9_wrap.py")
            sys.argv[0] = path
    if ON and not _st["done"] and b in ("fem_an_rt_sim.py", "fem_ao_rt_sim.py"):
        _st["done"] = True
        S10.install(out, cap=float(O["s10cap"]))
    return _orig(path, *a, **kw)


runpy.run_path = run_path
sys.argv = [os.path.join(HERE, "lightg4_wrap.py")] + argv
_orig(sys.argv[0], run_name="__main__")

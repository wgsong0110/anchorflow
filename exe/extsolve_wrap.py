"""절 CM117 묶음 extsolve 감싸개: 줄어든 문제 Ã = Eᵀ A E 의 층 0 체비셰프 평활 (lib/anchorflow/fem_extsolve.py, wolfh13 모드 2 위) 을 단 뒤
다음 감싸개 (첫 인자, 예 exe/wolfh12_wrap.py (wolf H9 사슬) · exe/speed13_wrap.py (레고·ficus·bread H9 사슬)) 로 넘김 (그쪽 인자 그대로).

사용: python exe/extsolve_wrap.py [--xs 차수] [--xslo 0.1] [--xshi 1.1] <다음 감싸개.py> <그 인자...> <실행기.py> <실행기 인자...>
설치: 새 실행기 fem_an_rt_sim.py 를 runpy 로 띄우는 순간 (안쪽 감싸개들 설치 뒤 = 가장 바깥). --xs 가 없거나 0 이면 모듈을 안 부름 (예전 그대로).
확장이 꺼진 판 (κ 1 씬) 은 wolfh13·extsolve 가 할당·커널을 하나도 안 띄움 (확장 판 표시가 없으면 원래 V 순환으로 넘김).
"""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lib"))
sys.setrecursionlimit(max(sys.getrecursionlimit(), 14000))

argv = sys.argv[1:]
out = argv[argv.index("--out") + 1] if "--out" in argv else "."
os.makedirs(out, exist_ok=True)
O = dict(xs="0", xslo="0.1", xshi="1.1", xabs="0", xapex="0", xct="0", xfd="0", xtr="0")
while argv and argv[0] in ("--xs", "--xslo", "--xshi", "--xabs", "--xapex", "--xct", "--xfd", "--xtr"):
    O[argv[0][2:]] = argv[1]
    del argv[:2]
K = int(float(O["xs"])); ABS = int(float(O["xabs"])); APX = int(float(O["xapex"])); CT = int(float(O["xct"])); FD = int(float(O["xfd"])); TRL = int(float(O["xtr"]))

_orig = runpy.run_path
_done = [False]


def run_path(path, *a, **kw):
    if (K or ABS or APX or CT or FD or TRL) and not _done[0] and os.path.basename(path) == "fem_an_rt_sim.py":
        _done[0] = True
        from anchorflow import fem_extsolve as XS
        if K:
            XS.install(out, k=K, lo=float(O["xslo"]), hi=float(O["xshi"]))
        if ABS:
            XS.install_abs(out)
        if APX:
            XS.install_apex(out)
        if CT == 3:                                                                   # 절 CM120: Levenberg 이동 (wolfct ct 1 글자 위에; 먼저 설치 = 안쪽)
            from anchorflow import fem_ctlm as LM
            LM.install()
        if TRL:                                                                       # 절 CM121: 비 판정 + 전역 Levenberg
            from anchorflow import fem_trlm as TRM
            TRM.install(out)
        if CT or FD:
            from anchorflow import fem_wolfct as WCT
            WCT.install(out, ct=(1 if CT in (1, 3) else 0), fd=FD)
    return _orig(path, *a, **kw)


runpy.run_path = run_path
assert argv and argv[0].endswith(".py"), "extsolve_wrap: 다음 감싸개 .py 가 첫 인자"
sys.argv = argv
_orig(sys.argv[0], run_name="__main__")

"""절 CM109 묶음 xfer 감싸개 (가장 바깥): 입자 <-> 격자 전달 정의 -- lib/anchorflow/fem_xfer.py.

사용: python exe/xfer_wrap.py --xa 0|1 --xb 0|1 [--xdg 0|1] [--xck 0|1] <다음 감싸개.py> <인자...>
  --xa 1 = (A) P2G 받침 판정 (φ < ½ 노드를 가중 이웃에 합치고 묶음), --xb 1 = (B) G2P APIC (Jiang 2015), 둘 다 0 이면 기록만 (결과 불변).
설치: 실행기 (fem_an_rt_sim.py) 를 runpy 로 띄우는 순간 (다른 감싸개가 다 설치된 뒤 = 훅이 가장 바깥, exe/ext2_wrap.py 와 같은 방식).
"""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lib"))
sys.setrecursionlimit(max(sys.getrecursionlimit(), 20000))

argv = sys.argv[1:]
out = argv[argv.index("--out") + 1] if "--out" in argv else "."
os.makedirs(out, exist_ok=True)
C = dict(xa="0", xb="0", xdg="1", xck="0")
while argv and argv[0] in ("--xa", "--xb", "--xdg", "--xck"):
    C[argv[0][2:]] = argv[1]
    del argv[:2]
_orig = runpy.run_path
_st = dict(done=False)


def run_path(path, *a, **kw):
    if not _st["done"] and os.path.basename(path) == "fem_an_rt_sim.py":
        _st["done"] = True
        from anchorflow import fem_xfer as XF
        XF.install(out, a=int(float(C["xa"])), b=int(float(C["xb"])), dg=int(float(C["xdg"])), ck=int(float(C["xck"])))
    assert os.path.basename(path) != "fem_ao_rt_sim.py", "절 CM109 xfer: 옛 실행기 (수박) 는 안 고침"
    return _orig(path, *a, **kw)


runpy.run_path = run_path
sys.argv = list(argv)
_orig(sys.argv[0], run_name="__main__")

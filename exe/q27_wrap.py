"""절 CM101 묶음 q27 감싸개: 요소 적분을 칸마다 가우스 3x3x3 (27 점) 으로 바꾸는 모듈 (lib/anchorflow/fem_q27.py) 을 단 뒤 다음 감싸개로 넘김 (그쪽 인자 그대로; H5 사슬).

사용: python exe/q27_wrap.py [--q27 0|1] [--q27cap 1] [--q27dg 1] <다음 감싸개.py> <그 인자...> <실행기.py> <실행기 인자...>
  --q27 1 = 27 점 적분 (정의 변경), 0 이나 없음 = 2x2x2 그대로. --q27cap 1 = 컴파일되는 CUDA 소스를 모두 <out>/q27src 에 갈무리 (결과 불변).
  --q1pt 1 = 절 CM179 칸마다 가우스 1x1x1 (칸 가운데 1 점, 무게 = 칸 부피; 모래시계 대조군; 기본 0, 정의 변경).
  --q27ph all|src|cls = 절 CM179: src = CUDA 글자 고침만 (가장 안쪽), cls = 풀이기 클래스 고침만 (가장 바깥); 기본 all = 예전 그대로.
  --q27jit 1 = 수박 꿈틀거림 (강체 뺀 프레임 사이 속도 RMS, exe/q27_jit.py 와 같은 정의) 을 실행 중에 jit.json 으로 (덤프 없이, 결과 불변).
  --q27dg 1 = 진단 기록 (q27.jsonl). 셋 다 없으면 모듈을 안 부름 (예전 그대로); '--q27 0' 은 모듈을 불러 풀이기 정보 (q27solver.pt) 만 저장 (결과 불변, 대조 판).
설치: 실행기 fem_an_rt_sim.py · fem_ao_rt_sim.py (exe/s6/ 사본 포함) 를 runpy 로 띄우는 순간 (안쪽 감싸개들 설치 뒤, 가장 바깥 = cp.RawModule 를 마지막에 감쌈).
"""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lib"))
sys.setrecursionlimit(max(sys.getrecursionlimit(), 18000))                                # 감싸개 한 겹 더

argv = sys.argv[1:]
out = argv[argv.index("--out") + 1] if "--out" in argv else "."
os.makedirs(out, exist_ok=True)
O = dict(q27="0", q27cap="0", q27dg="0", q27jit="0", q1pt="0", q27ph="all")
SEEN = False
while argv and argv[0] in ("--q27", "--q27cap", "--q27dg", "--q27jit", "--q1pt", "--q27ph"):
    O[argv[0][2:]] = argv[1]
    del argv[:2]
    SEEN = True
MODE, CAP, DG = int(float(O["q27"])), int(float(O["q27cap"])), int(float(O["q27dg"]))
if int(float(O["q1pt"])):                                                         # 절 CM179: 1x1x1 (칸 가운데 점 하나; 기본 0)
    MODE = 2

_orig = runpy.run_path
_done = [False]


print("[절 CM179 q27_wrap] 단계 %s 실행 (SEEN %s)" % (O["q27ph"], SEEN), flush=True)
if SEEN and O["q27ph"] == "src":                                                     # 글자 고침만: 실행기 순간을 기다리지 않고 지금 (다른 감싸개들의 훅보다 먼저 = 가장 안쪽 RawModule 감싸개)
    from anchorflow import fem_q27 as Q27S
    Q27S.install(out, MODE, CAP, DG, "fem_an_rt_sim.py", int(float(O["q27jit"])), "src")
    _done = [True]


def run_path(path, *a, **kw):
    if os.path.basename(path) in ("fem_an_rt_sim.py", "fem_ao_rt_sim.py"):
        print("[절 CM179 q27_wrap] 단계 %s 훅이 실행기 경로를 봄 (done %s)" % (O["q27ph"], _done[0]), flush=True)
    if SEEN and not _done[0] and os.path.basename(path) in ("fem_an_rt_sim.py", "fem_ao_rt_sim.py"):
        _done[0] = True
        from anchorflow import fem_q27 as Q27
        Q27.install(out, MODE, CAP, DG, os.path.basename(path), int(float(O["q27jit"])), O["q27ph"])
    return _orig(path, *a, **kw)


runpy.run_path = run_path
sys.argv = list(argv)
_orig(sys.argv[0], run_name="__main__")

"""절 CM62 묶음 speed5 감싸개: 프레임 시각·단계 시간 기록·시뮬/렌더 겹치기 (lib/anchorflow/fem_speed5.py) 를 단 뒤 exe/hhtc_wrap.py 로 넘김 (그쪽 인자 그대로).

사용: python exe/speed5_wrap.py [--s5m 1] [--s5p 1] [--s5o 1] <hhtc_wrap 인자...> <실행기.py> <실행기 인자...>
  --s5m 1  프레임 시각 기록만 (결과 불변) -> <out>/s5.jsonl (per·int·lat)
  --s5p 1  + 단계 시간 기록 (결과 불변, 그래프 안 globaltimer 표식) -> s5.jsonl 'sp'
  --s5o 1  + 시뮬 (비차단 스트림) · 래스터화 (기본 스트림) 겹치기 (풀이 커널·순서 같음; Rf 가 빼는 렌더 몫은 시뮬 스트림에 드러난 몫)
설치는 실행기 (fem_an_rt_sim.py / fem_ao_rt_sim.py) 를 runpy 로 띄우는 순간 -- 다른 감싸개가 모두 설치된 뒤 (가장 바깥).
"""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lib"))

argv = sys.argv[1:]
out = argv[argv.index("--out") + 1] if "--out" in argv else "."
os.makedirs(out, exist_ok=True)
OPT = dict(s5m=0, s5p=0, s5o=0)
while argv and argv[0].startswith("--s5"):
    OPT[argv[0][2:]] = int(float(argv[1]))
    del argv[:2]
ON = any(OPT.values())

_orig = runpy.run_path
_done = [False]


def run_path(path, *a, **kw):
    if ON and not _done[0] and os.path.basename(path) in ("fem_an_rt_sim.py", "fem_ao_rt_sim.py"):
        _done[0] = True
        from anchorflow import fem_speed5 as S5
        S5.install(out, p=OPT["s5p"], o=OPT["s5o"])
    return _orig(path, *a, **kw)


runpy.run_path = run_path
sys.argv = [os.path.join(HERE, "hhtc_wrap.py")] + argv
_orig(sys.argv[0], run_name="__main__")

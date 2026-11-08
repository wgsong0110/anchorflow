"""절 CM65 묶음 speed6 감싸개: 프레임 벽시계 기록·가벼운 판·프로파일 (lib/anchorflow/fem_speed6.py) 을 단 뒤 exe/hhtc2_wrap.py 로 넘김 (그쪽 인자 그대로).

사용: python exe/speed6_wrap.py [--s6w 1] [--s6o 1] [--s6l 1] [--s6prof F0,F1] <hhtc2_wrap 인자...> <실행기.py> <실행기 인자...>
  --s6w 1      프레임 벽시계 기록만 (결과 불변) -> <out>/s6.jsonl
  --s6o 1      + 시뮬 (비차단 스트림) · 래스터화 (기본 스트림) 겹치기 O (절 CM62 fem_speed5.ov_install 그대로)
  --s6l 1      + 가벼운 판: 기록 전용 프레임 훅을 장치 일만 하는 판으로 (다른 감싸개 설치 전에 바꿈; 실행기는 stab 이 exe/s6/ 사본으로)
  --s6prof a,b 프레임 a ~ b−1 을 torch.profiler 로 -> trace.json.gz + s6prof.json
설치: 가벼운 판 바꾸기는 지금 (다른 감싸개 앞), 벽시계 기록·O 는 실행기를 runpy 로 띄우는 순간 (다른 감싸개가 모두 설치된 뒤 = 가장 바깥).
"""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lib"))

argv = sys.argv[1:]
out = argv[argv.index("--out") + 1] if "--out" in argv else "."
os.makedirs(out, exist_ok=True)
OPT = dict(s6w="0", s6o="0", s6l="0", s6prof="")
while argv and argv[0].startswith("--s6"):
    OPT[argv[0][2:]] = argv[1]
    del argv[:2]
ON = any(v not in ("", "0") for v in OPT.values())
if ON and OPT["s6l"] not in ("", "0"):
    from anchorflow import fem_speed6 as S6
    S6.C.lean = 1
    S6.lean_preinstall()

_orig = runpy.run_path
_done = [False]


def run_path(path, *a, **kw):
    if ON and not _done[0] and os.path.basename(path) in ("fem_an_rt_sim.py", "fem_ao_rt_sim.py"):
        _done[0] = True
        from anchorflow import fem_speed6 as S6
        S6.install(out, o=int(float(OPT["s6o"])), lean=int(float(OPT["s6l"])), prof=OPT["s6prof"] or None)
    return _orig(path, *a, **kw)


runpy.run_path = run_path
sys.argv = [os.path.join(HERE, "hhtc2_wrap.py")] + argv
_orig(sys.argv[0], run_name="__main__")

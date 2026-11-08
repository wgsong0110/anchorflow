"""절 CM72 묶음 speed7 감싸개: 커널 프로파일·결과 불변 가속 (lib/anchorflow/fem_speed7.py) + 절 CM65 speed6 (벽시계 기록·O·L·B·Rw) 을 단 뒤
exe/mfix_wrap.py 로 넘김 (G'' = G' + 절 CM67 (2) '--mf2 1'; 그쪽 인자 그대로). speed6_wrap.py 와 같은 글자에 다음 감싸개만 mfix_wrap.

사용: python exe/speed7_wrap.py [--s7kp F0,F1] [--s7nm 1] [--s6o 1 --s6l 1 --s6b 1 --s6rw 1 ...] <mfix_wrap 인자...> <실행기.py> <실행기 인자...>
설치: 가벼운 판 바꾸기는 지금 (다른 감싸개 앞), speed6 (벽시계 기록·O) 와 speed7 은 실행기를 runpy 로 띄우는 순간 (가장 바깥; speed7 이 speed6 뒤).
"""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lib"))

argv = sys.argv[1:]
out = argv[argv.index("--out") + 1] if "--out" in argv else "."
os.makedirs(out, exist_ok=True)
O7 = dict(s7kp="", s7nm="0")
OPT = dict(s6w="0", s6o="0", s6l="0", s6prof="", s6b="0", s6r="0", s6rw="0")
while argv and (argv[0].startswith("--s7") or argv[0].startswith("--s6")):
    k_ = argv[0][2:]
    (O7 if k_.startswith("s7") else OPT)[k_] = argv[1]
    del argv[:2]
ON7 = any(v not in ("", "0") for v in O7.values())
ON = ON7 or any(v not in ("", "0") for v in OPT.values())
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
        S6.install(out, o=int(float(OPT["s6o"])), lean=int(float(OPT["s6l"])), prof=OPT["s6prof"] or None,
                   b=int(float(OPT["s6b"])), r=int(float(OPT["s6r"])), rw=int(float(OPT["s6rw"])))
        if ON7:
            from anchorflow import fem_speed7 as S7
            S7.install(out, kp=O7["s7kp"] or None, nm=int(float(O7["s7nm"])))
    return _orig(path, *a, **kw)


runpy.run_path = run_path
sys.argv = [os.path.join(HERE, "mfix_wrap.py")] + argv
_orig(sys.argv[0], run_name="__main__")

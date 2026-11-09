"""절 CM90 묶음 wmjitter 감싸개: 진단 기록 (lib/anchorflow/fem_wmjitter.py, 결과 불변) 을 단 뒤 exe/speed13_wrap.py 로 넘김 (그쪽 인자 그대로; H9 사슬).

사용: python exe/wmjitter_wrap.py [--wj 1] [--wj_sub 64] [--wj_gsub 8] [--wj_f0 29] [--wj_dfr 1] <speed13_wrap 인자...> <실행기.py> <실행기 인자...>
설치: 시작하자마자 fem_speed6.Defer (가벼운 판 '--s6l 1' 의 프레임 기록) 의 keep·finish 를 감쌈. --wj 이 없거나 0 이면 모듈을 안 부름.
"""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.setrecursionlimit(max(sys.getrecursionlimit(), 20000))                                # 감싸개 한 겹 더 (절 CM86·CM87 첫 연기: 재귀 한도)
sys.path.insert(0, os.path.join(HERE, "..", "lib"))

argv = sys.argv[1:]
out = argv[argv.index("--out") + 1] if "--out" in argv else "."
os.makedirs(out, exist_ok=True)
O = dict(wj="0", wj_sub="64", wj_gsub="8", wj_f0="29", wj_dfr="0")
while argv and argv[0].startswith("--wj"):
    O[argv[0][2:]] = argv[1]
    del argv[:2]
if int(float(O["wj"])):
    from anchorflow import fem_wmjitter as WJ
    WJ.install(out, sub=int(float(O["wj_sub"])), gsub=int(float(O["wj_gsub"])), f0=int(float(O["wj_f0"])), dfr=int(float(O["wj_dfr"])))
sys.argv = [os.path.join(HERE, "speed13_wrap.py")] + argv
runpy.run_path(sys.argv[0], run_name="__main__")

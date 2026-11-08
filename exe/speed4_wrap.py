"""절 CM50 묶음 speed4 감싸개: 단계 시간 기록·결과 불변 가속 (lib/anchorflow/fem_speed4.py) 을 붙인 뒤 exe/escape4_wrap.py 로 넘김 (그쪽 인자는 그대로).

사용: python exe/speed4_wrap.py [--s4p 1] [--s4w 1|2] [--s4x 1] <escape4_wrap 인자...> <실행기.py> <실행기 인자...>
  --s4p 1   단계 시간 기록 (결과 불변; 그래프 안 globaltimer 표식) -> <out>/s4.jsonl 'sp'
  --s4w 1   S1 국소 보폭 확인 라운드 조기 끝 (결과 불변), 2 = 확인 판 (건너뛰지 않고 어긋남 셈) -> s4.jsonl 'ws4'
  --s4x 1   X1 무너진 첫 시도 꼬리 건너뛰기 (결과 불변)
인자가 하나도 없으면 이 모듈을 안 부름 (소스·그래프 그대로).
"""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lib"))

argv = sys.argv[1:]
out = argv[argv.index("--out") + 1] if "--out" in argv else "."
os.makedirs(out, exist_ok=True)
OPT = {}
while argv and argv[0].startswith("--s4"):
    OPT[argv[0][2:]] = int(float(argv[1]))
    del argv[:2]
OPT = {k: v for k, v in OPT.items() if v}
if OPT:
    from anchorflow import fem_speed4 as S4
    if OPT.get("s4p"):
        S4.sp_install()
    if OPT.get("s4w"):
        S4.ws4_install(OPT["s4w"])
    if OPT.get("s4x"):
        S4.x1_install()
    S4.frame_install(out)
    print(f"[절 CM50] speed4 {OPT}", flush=True)

sys.argv = [os.path.join(HERE, "escape4_wrap.py")] + argv
runpy.run_path(sys.argv[0], run_name="__main__")

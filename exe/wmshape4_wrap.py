"""절 CM46 묶음 wmshape4 감싸개: 진단 D4·후보 T (lib/anchorflow/fem_wmshape4.py) 를 붙인 뒤 exe/escape3_wrap.py 로 넘김
(그쪽 --e3w W̄ · AF_E3_DG, escape2 의 --e2* · AF_E2_DG, escape1·speed3 인자는 그대로 넘어감).

사용: python exe/wmshape4_wrap.py [--w4t 1] [--w4dg 3,4,5] <escape3_wrap 인자...> <실행기.py> <실행기 인자...>
  --w4t 1        후보 T: X3c 속도 수준 접촉 (X2) 이 지우는 몫을 띠 몫 대신 '걸음 중 바닥 (d̂) 에 닿은 뒤 몫' 으로 (결과 바뀜, 무너진 프레임만)
  --w4dg 프레임  진단 D4 (결과 불변): 그 프레임 (1 부터) 의 X3c 시도별 층별 관성 예측·목표·시작점·반복별 위치·끝 -> <out>/w4.jsonl
"""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lib"))

argv = sys.argv[1:]
out = argv[argv.index("--out") + 1] if "--out" in argv else "."
os.makedirs(out, exist_ok=True)
CAND = {}
while argv and argv[0].startswith("--w4"):                                              # 감싸개 인자 (맨 앞에만)
    CAND[argv[0][2:]] = argv[1]
    del argv[:2]
T = int(float(CAND.get("w4t", 0))) == 1
DG = [int(q) for q in CAND.get("w4dg", "").split(",") if q.strip()]
if T or DG:
    from anchorflow import fem_wmshape4 as W4
    W4.install(t=T, dg=DG, out=out)
    print(f"[절 CM46] 후보 T {int(T)}, 진단 D4 프레임 {DG}", flush=True)

sys.argv = [os.path.join(HERE, "escape3_wrap.py")] + argv
runpy.run_path(sys.argv[0], run_name="__main__")

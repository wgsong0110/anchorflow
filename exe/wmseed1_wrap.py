"""절 CM60 묶음 wmseed1 감싸개 (수박 씨 영역 β 3e9 를 입자 수준으로): 실행기를 고치지 않고 모듈 함수를 감싼 뒤 exe/wmcrack1_wrap.py 로 넘김.

사용: python exe/wmseed1_wrap.py [--sd1 1] <wmcrack1_wrap 인자...> <실행기.py> <실행기 인자...>
  --sd1 1  씨 입자 몫 (정의 lib/anchorflow/fem_wmseed.py, doc 절 CM60.2): 씨 입자 F·logJp 는 입자 수준 (G2P 입자 RM, β 3e9, −inf 출구 그대로),
           가우스 점 응력·에너지 = (1 − φ_s) 소성 몫 (비씨 입자만) + φ_s 씨 몫 (씨 입자 F 평균, 얼은 몫 −inf 따로). --seedb·--fp_interp·--gq_oct 0 위에서만.
  인자 없으면 아무것도 안 바꿈.
"""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lib"))

argv = sys.argv[1:]
CAND = {}
while argv and argv[0].startswith("--sd1"):
    CAND[argv[0][2:]] = argv[1]
    del argv[:2]

if int(float(CAND.get("sd1", 0))):
    def _v(name):
        return argv[argv.index(name) + 1] if name in argv else None
    assert "--seedb" in argv, "절 CM60 --sd1: --seedb (씨 마스크) 위에서만"
    assert "--fp_interp" in argv, "절 CM60 --sd1: --fp_interp 위에서만"
    assert _v("--gq_oct") == "0", "절 CM60 --sd1: --gq_oct 0 (빈 팔분 점 V_q = 0) 위에서만"
    assert "--det" not in argv, "절 CM60 --sd1: 비결정 판만"
    from anchorflow import fem_wmseed as SDM
    SDM.install()
    print("[절 CM60] --sd1 1: 씨 입자 F·logJp 입자 수준 (G2P 입자 RM β 3e9), 가우스 점 = 소성 몫 (비씨) + 씨 몫 (φ_s, 얼은 몫 −inf), 보간 교체는 비씨 입자·비씨 점만", flush=True)

sys.argv = [os.path.join(HERE, "wmcrack1_wrap.py")] + argv
runpy.run_path(sys.argv[0], run_name="__main__")

"""절 CM39 묶음 speed3 감싸개: 실행기 (exe/fem_an_rt_sim.py · exe/fem_ao_rt_sim.py) 를 고치지 않고 모듈 함수를 감싸 진단·후보를 붙인다.

사용: python exe/speed3_wrap.py <실행기.py> <실행기 인자...>   -- 켜는 것은 환경 변수로:
  AF_S3_UL=1  wolf 입자 이탈 진단 (lib/anchorflow/fem_speed3.py ULDiag, 결과 불변; 새 실행기 --ksplit 판만 -- split_step 뒤 호스트 읽기)
"""
import os
import runpy
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from anchorflow import fem_an_rt as ART  # noqa: E402
from anchorflow import fem_speed3 as S3  # noqa: E402

sim = sys.argv[1]
argv = sys.argv[1:]
out = argv[argv.index("--out") + 1] if "--out" in argv else "."

if os.environ.get("AF_S3_UL", "0") == "1":
    _o_split = ART.split_step
    _st = dict(fi=0, d=None)

    def split_step(s, t, h, depth):
        r = _o_split(s, t, h, depth)
        if _st["d"] is None:
            _st["d"] = S3.ULDiag(out)
        try:
            _st["d"].frame(s, int(round(t / h)))
        except Exception:                                                              # 진단 실패는 풀이에 영향 없음 -- 기록만
            import traceback
            with open(os.path.join(out, "ul_err.txt"), "a") as f_:
                f_.write(traceback.format_exc()[-2000:] + "\n")
        return r
    ART.split_step = split_step

sys.argv = argv
runpy.run_path(sim, run_name="__main__")

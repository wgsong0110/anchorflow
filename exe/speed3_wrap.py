"""절 CM39 묶음 speed3 감싸개: 실행기 (exe/fem_an_rt_sim.py · exe/fem_ao_rt_sim.py) 를 고치지 않고 모듈 함수·풀이기 메서드를 감싸 진단·후보를 붙인다.

사용: python exe/speed3_wrap.py [감싸개 인자...] <실행기.py> <실행기 인자...>
  --tb B      후보 T 시간 예산 (ms; 프레임 GPU 시계로 반복 xf 이상을 B − m 안에서만) -- lib/anchorflow/fem_speed3.py tb_install (결과 바뀜)
  --tb_m m    예산 여유 (ms, 기본 0.5)
  --w1 1      후보 W1: 덜 풀린 걸음의 가속도 기억 0 을 몸 전체로 (--ue_loc 국소 갱신 커널; 결과 바뀜)
  --inv M     가속 I (수박 옛 실행기, 결과 불변): ws_inv_q 에 f32 증명 빠른 길 (1), 확인 판 (2: 빠른 길 없이 증명 참 & k > 0 인 점 수를 셈)
  --x3e T     후보 X3e (옛 실행기): X3c 판정을 반복 0 시작 잔차 ‖g‖²/F² > T 로 앞당기고 걸음 시작부터만 다시 (결과 바뀜)
  환경 AF_S3_UL=1  wolf 입자 이탈 진단 (fem_speed3.ULDiag, 결과 불변; 새 실행기 --ksplit 판만 -- split_step 뒤 호스트 읽기)
"""
import os
import runpy
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from anchorflow import fem_an_rt as ART  # noqa: E402
from anchorflow import fem_speed3 as S3  # noqa: E402

opt, i = {}, 1
while i < len(sys.argv) and not sys.argv[i].endswith(".py"):
    opt[sys.argv[i].lstrip("-")] = sys.argv[i + 1]
    i += 2
sim = sys.argv[i]
argv = sys.argv[i:]
out = argv[argv.index("--out") + 1] if "--out" in argv else "."
os.makedirs(out, exist_ok=True)

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

if "tb" in opt:
    S3.tb_install(float(opt["tb"]), float(opt.get("tb_m", 0.5)), out)
    print(f"[절 CM39 후보 T] 시간 예산 {opt['tb']} ms (여유 {opt.get('tb_m', 0.5)} ms)", flush=True)
if "x3e" in opt:
    S3.x3e_install(float(opt["x3e"]))
    print(f"[절 CM39 후보 X3e] 반복 0 잔차 문턱 {opt['x3e']}", flush=True)
if opt.get("inv", "0") != "0":
    S3.inv_install(int(opt["inv"]))
    print(f"[절 CM39 가속 I] ws_inv_q f32 증명 판 {opt['inv']}", flush=True)
if opt.get("w1", "0") != "0":
    S3.w1_install()
    print("[절 CM39 후보 W1] 덜 풀린 걸음 가속도 기억 0 몸 전체", flush=True)
if opt:
    S3.frame_hook_install(out)

sys.argv = argv
runpy.run_path(sim, run_name="__main__")

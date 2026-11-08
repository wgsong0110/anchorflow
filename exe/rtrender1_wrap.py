"""절 CM54 묶음 rtrender1 감싸개: 시뮬 루프 안 실시간 3DGS 렌더 시간 (lib/anchorflow/rtrender1.py) 을 붙인 뒤 exe/speed4_wrap.py 로 넘김 (그쪽 인자 그대로).

사용: python exe/rtrender1_wrap.py --rr pg|gf --rr_shape 이름 --rr_traj 궤적 --rr_cfg 설정 [--rr_model GF 모델] [--rr_gf GF 소스] [--rr_fast 0|1] [--rr_chk 1,10,..]
         <speed4_wrap 인자...> <실행기.py> <실행기 인자...>
다른 감싸개가 쪼개기 함수를 다 바꾼 뒤에 훅을 걸도록, 실행기 (fem_an_rt_sim.py / fem_ao_rt_sim.py) 를 runpy 로 띄우는 순간에 설치.
훅은 풀이 상태를 읽기만 함 (결과 불변). 결과 <out>/rr.jsonl (프레임마다 a 시뮬 · b 갱신 · c 래스터화 ms), rr_meta.json, (확인) rrimg/·rr_chk.jsonl.
"""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lib"))

argv = sys.argv[1:]
out = argv[argv.index("--out") + 1] if "--out" in argv else "."
os.makedirs(out, exist_ok=True)
O = dict(rr="pg", rr_shape="lego", rr_traj="", rr_cfg="", rr_model="", rr_gf="", rr_fast="0", rr_chk="")
while argv and argv[0].startswith("--rr"):
    O[argv[0][2:]] = argv[1]
    del argv[:2]

_orig = runpy.run_path
_done = [False]


def run_path(path, *a, **kw):
    if not _done[0] and os.path.basename(path) in ("fem_an_rt_sim.py", "fem_ao_rt_sim.py"):
        _done[0] = True
        from anchorflow import rtrender1 as RR
        RR.install(O["rr"], O["rr_shape"], O["rr_traj"], O["rr_cfg"], O["rr_model"], O["rr_gf"], int(O["rr_fast"]), O["rr_chk"], out)
    return _orig(path, *a, **kw)


runpy.run_path = run_path
sys.argv = [os.path.join(HERE, "speed4_wrap.py")] + argv
_orig(sys.argv[0], run_name="__main__")

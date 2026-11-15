"""절 CM102 묶음 tet1 감싸개: 공용 실행기 기저를 사면체 일차 기저로 (lib/anchorflow/fem_tet1.py) 바꾼 뒤 다음 감싸개로 넘김 (H9 사슬 그대로).

사용: python exe/tet1_wrap.py [--tet 0|1] [--tmx 0|1|2|3] <다음 감싸개.py> <그 인자...> <실행기.py> <실행기 인자...>
  1 = 사면체 기저 (소스 글자 바꿈 + 기저 표), 0 = 소스 그대로 (컴파일 소스 해시만 tet1info.json 에 -- 끈 판 확인용).
설치: 가장 바깥 (다른 감싸개가 소스를 다 바꾼 뒤 컴파일 순간에 바꾸도록 cupy.RawModule 을 감쌈).
"""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lib"))
sys.setrecursionlimit(max(sys.getrecursionlimit(), 20000))

argv = sys.argv[1:]
out = argv[argv.index("--out") + 1] if "--out" in argv else "."
os.makedirs(out, exist_ok=True)
MODE, MIX = None, 0
while argv and argv[0] in ("--tet", "--toct", "--tmx"):
    if argv[0] == "--tet":
        MODE = int(float(argv[1]))
    elif argv[0] == "--tmx":
        MIX = int(float(argv[1]))                                                  # 절 CM180 혼합 판 (기본 0): 탄성 사면체 + 입자<->노드 전달 이차 B-스플라인
    else:
        os.environ["AF_TET1_OCT"] = argv[1]                                        # 옛 실행기 사면체 점유 가중 κ (--gq_mass 없는 수박 판)
    del argv[:2]
if MODE == 1 or os.environ.get("AF_TET1_OCT"):
    os.environ["AF_TET1"] = "1"                                                     # 옛 실행기 CM30 assert: --gq_mass 없이 --ws_bar 허용
if MODE is not None:
    from anchorflow import fem_tet1 as TET
    assert not MIX or MODE == 1, "--tmx 1|2|3 은 --tet 1 위에서만"
    TET.install(out, MODE, MIX)
sys.argv = list(argv)
runpy.run_path(sys.argv[0], run_name="__main__")

"""절 CM184 묶음 hexff 감싸개 (가장 바깥): 가우스 점 뒤집힘 불가능 보폭·시작점 확인 (lib/anchorflow/fem_hexff.py) 을 설치한 뒤 다음 감싸개로.

사용: python exe/hexff_wrap.py --hff 1 [--hff_tau τ] <다음 감싸개 / 실행기 인자...>   (--hff 0 = 설치 안 함, 기존 판 그대로)
"""
import atexit
import json
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lib"))
sys.setrecursionlimit(max(sys.getrecursionlimit(), 20000))

argv = sys.argv[1:]
out = argv[argv.index("--out") + 1] if "--out" in argv else "."
ON, TAU, LM, KF, MT = None, 0.0, 0, 1, (2,)
while argv and argv[0] in ("--hff", "--hff_tau", "--hff_lmix", "--hff_kfe", "--hff_skip"):
    if argv[0] == "--hff":
        ON = int(float(argv[1]))
    elif argv[0] == "--hff_skip":            # 쉼표 재질 번호 (기본 7 = GF NACC), all = 모든 재질 (탄성 실행기 포함)
        MT = None if argv[1] == "all" else tuple(int(q) for q in argv[1].split(","))
    elif argv[0] == "--hff_kfe":
        KF = int(float(argv[1]))
    elif argv[0] == "--hff_lmix":
        LM = int(float(argv[1]))
    else:
        TAU = float(argv[1])
    del argv[:2]
assert ON in (0, 1), "--hff 0|1 이 필요"
if ON:
    from anchorflow import fem_hexff as HF
    HF.install(TAU, LM, KF, MT)

    def _fin():
        o = dict(tau=HF.INFO["tau"], mods=HF.INFO["mods"], n_kfe=HF.INFO["n_kfe"], n_chk=HF.INFO["n_chk"], n_lmix=HF.INFO["n_lmix"], rel_err=HF.INFO.get("rel_err"), bad_args=sorted(map(list, HF.INFO.get("bad_args", set()))))
        print("[절 CM184 hexff] " + json.dumps(o), flush=True)
        try:
            os.makedirs(out, exist_ok=True)
            json.dump(o, open(os.path.join(out, "hffinfo.json"), "w"))
        except Exception:
            pass
    atexit.register(_fin)
if os.environ.get("AF_NANPROBE", "0") == "1":      # 절 CM184 진단 (결과 불변)
    from anchorflow import fem_nanprobe as NPB
    os.makedirs(out, exist_ok=True)
    NPB.install(out)
sys.argv = list(argv)
runpy.run_path(sys.argv[0], run_name="__main__")

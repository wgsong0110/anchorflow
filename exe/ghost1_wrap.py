"""절 CM47 묶음 ghost1 감싸개: B-스플라인 작은 받침 안정화 ghost penalty (lib/anchorflow/fem_ghost1.py) 를 붙인 뒤 exe/escape3_wrap.py 로 넘김
(그쪽 --e3*·--e2*·escape1·speed3 인자는 그대로 넘어감).

사용: python exe/ghost1_wrap.py [--g1 1] <escape3_wrap 인자...> <실행기.py> <실행기 인자...>
  --g1 1  ghost penalty 켬 (γ_a = (2μ+λ)·1e-4, Hansbo·Larson·Larsson 2017; 경계 칸 내부 면의 2 차 법선 미분 점프). 없거나 0 이면 이 모듈을 안 부름.
  환경 AF_G1_DG=1 (결과 불변, 프레임마다 장치 계수 읽기만): <out>/g1.jsonl 에 프레임별 경계 칸 수·벌점 면 수·찬 칸 수.
"""
import json
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lib"))

argv = sys.argv[1:]
out = argv[argv.index("--out") + 1] if "--out" in argv else "."
os.makedirs(out, exist_ok=True)
CAND = {}
while argv and argv[0].startswith("--g1"):
    CAND[argv[0][2:]] = argv[1]
    del argv[:2]
ON = int(float(CAND.get("g1", 0)))
if ON:
    from anchorflow import fem_ghost1 as G1
    G1.install()
    print(f"[절 CM47] ghost penalty 켬 (γ_a = (2μ+λ)·{G1.GAMMA_A:g}, k = γ_a h)", flush=True)
    if os.environ.get("AF_G1_DG", "0") == "1":
        from anchorflow import fem_an_rt as ART
        from anchorflow import fem_an_rt_v0 as V0
        LOG = open(os.path.join(out, "g1.jsonl"), "w")
        CNT = [0]

        def wrap_step(cls):
            _os = cls.step

            def step(s, t, *a, **kw):
                d = getattr(s, "_g1dep", 0)
                s._g1dep = d + 1
                try:
                    r = _os(s, t, *a, **kw)
                finally:
                    s._g1dep = d
                if d == 0 and getattr(s, "gpcnt", None) is not None:
                    try:
                        c = s.gpcnt.cpu().tolist()
                        CNT[0] += 1
                        LOG.write(json.dumps(dict(f=CNT[0], ncut=int(c[0]), nface=int(c[1]), nc=int(s.IP[11].item()), k=float(s._gpk))) + "\n")
                        LOG.flush()
                    except Exception as e_:
                        with open(os.path.join(out, "g1_err.txt"), "a") as f_:
                            f_.write(repr(e_) + "\n")
                return r
            cls.step = step
        for cls_ in (ART.RTSolver, V0.RTSolver):
            wrap_step(cls_)

sys.argv = [os.path.join(HERE, "escape3_wrap.py")] + argv
runpy.run_path(sys.argv[0], run_name="__main__")

"""절 CM51 묶음 ghost3 감싸개: ghost penalty 3 (lib/anchorflow/fem_ghost3.py -- 고스트 질량 + 얇은 구조 면 빼기 + 칸별 커널) 을 붙인 뒤 exe/escape4_wrap.py 로 넘김
(그쪽 --e5r·--e5a, escape3 의 --e3w, escape2·escape1·speed3 인자는 그대로 넘어감).

사용: python exe/ghost3_wrap.py [--g3 g] [--g3m 1] [--g3t 1] [--g3fd 1,5,13] <escape4_wrap 인자...> <실행기.py> <실행기 인자...>
  --g3 g     ghost penalty 3 켬, γ̄ = g (강성 k = g (2μ+λ) h, 질량 k_M = g ρ̄ h³). g = 0 = 커널은 다 돌고 더하는 값 0 (결정적 판 허용, 해시 대조용). 없으면 이 모듈을 안 부름.
  --g3m 0|1  고스트 질량 (기본 1), --g3t 0|1 얇은 구조 면 빼기 (기본 1).
  --g3fd L   (결과 불변) 프레임 목록 L 의 걸음 뒤 그래프 밖 검사·비교·시간 (fem_ghost3.check) -> <out>/g3fd.jsonl
  환경 AF_G1_DG=1 (결과 불변): <out>/g1.jsonl 에 프레임별 경계 칸·면 (빼기 뒤·앞)·두꺼운 칸 수.
"""
import atexit
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
while argv and argv[0].startswith("--g3"):
    CAND[argv[0][2:]] = argv[1]
    del argv[:2]
if "g3" in CAND:
    os.environ["AF_G3_GAMMA"] = str(float(CAND["g3"]))
    os.environ["AF_G3_MASS"] = str(int(float(CAND.get("g3m", 1))))
    os.environ["AF_G3_THIN"] = str(int(float(CAND.get("g3t", 1))))
    from anchorflow import fem_ghost3 as G3
    from anchorflow import fem_an_rt as ART
    from anchorflow import fem_an_rt_v0 as V0
    G3.install()
    print(f"[절 CM51] ghost penalty 3 켬 (γ̄ {G3.GAMMA:g}, 질량 {G3.MASS}, 얇은 구조 빼기 {G3.THIN}, 칸별 커널)", flush=True)
    FD = {int(q) for q in CAND.get("g3fd", "").split(",") if q.strip()}
    DG1 = os.environ.get("AF_G1_DG", "0") == "1"
    L1 = open(os.path.join(out, "g1.jsonl"), "w") if DG1 else None
    ST = dict(n=0, done=set())

    def err(tag, e_):
        with open(os.path.join(out, "g3_err.txt"), "a") as f_:
            import traceback
            f_.write(f"{tag}: {e_!r}\n{traceback.format_exc()}\n")

    def torch_ok(s):
        import torch
        return bool(torch.isfinite(s.y).all())

    def wrap_step(cls):
        _os = cls.step

        def step(s, t, *a, **kw):
            d = getattr(s, "_g3dep", 0)
            s._g3dep = d + 1
            try:
                r = _os(s, t, *a, **kw)
            finally:
                s._g3dep = d
            if d != 0 or getattr(s, "gpcnt", None) is None:
                return r
            ST["n"] += 1
            h = float(getattr(s, "h", 0.0) or 0.0)
            fi = int(round(t / h)) + 1 if h > 0 else ST["n"]
            try:
                if L1 is not None:
                    c = s.gpcnt.cpu().tolist()
                    L1.write(json.dumps(dict(f=fi, ncut=int(c[0]), nface=int(c[1]), nface_all=int(c[2]), nthick=int(c[3]), nc=int(s.IP[11].item()),
                                             k=float(s._gpk), kM=float(s._g3kM), it2=float(getattr(s, "_g3it2", 0.0)))) + "\n"); L1.flush()
                if fi in FD and fi not in ST["done"] and bool(torch_ok(s)):
                    ST["done"].add(fi)
                    rs = G3.check(s, fi, out)
                    print(f"[절 CM51 검사] 프레임 {fi}: {json.dumps(rs, ensure_ascii=False)[:1500]}", flush=True)
            except Exception as e_:
                err(f"프레임 {fi}", e_)
            return r
        cls.step = step

    for cls_ in (ART.RTSolver, V0.RTSolver):
        wrap_step(cls_)

    @atexit.register
    def _info():
        try:
            json.dump(dict(gamma=G3.GAMMA, mass=G3.MASS, thin=G3.THIN, nk=G3.ST["nk"], said=G3.ST["said"], nmass=G3.ST["nmass"], nstep=G3.ST["nstep"]),
                      open(os.path.join(out, "g3info.json"), "w"), ensure_ascii=False)
        except Exception as e_:
            err("info", e_)

sys.argv = [os.path.join(HERE, "escape4_wrap.py")] + argv
runpy.run_path(sys.argv[0], run_name="__main__")

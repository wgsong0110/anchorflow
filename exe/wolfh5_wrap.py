"""절 CM70 묶음 wolfh5 감싸개: 인자 --w5* 를 떼고 exe/mfix_wrap.py (절 CM67, G'' 의 '--mf2 1' 등 그쪽 인자 그대로) 를 띄움.
실행기 (fem_an_rt_sim.py) 를 runpy 로 띄우는 순간 (다른 감싸개가 모두 설치된 뒤 = 가장 바깥) lib/anchorflow/fem_wolfh5.py 를 설치.

사용: python exe/wolfh5_wrap.py [--w5dg 1] [--w5s 1] <mfix_wrap 인자...> <실행기.py> <실행기 인자...>
  --w5dg 1  진단 (결과 불변): 걸음마다 방출 앞면 거리 칸 x 높이 칸 별 구속 몫·노드 속도·재질 힘·ghost 힘 (입자) + 가우스 점 응력 (w5dg.json)
  --w5c 1   C: 구속 받침 안 저질량 고정 노드를 구속 예측에 (Ũ 대신)
  --w5g 1   B_g: ghost penalty 면 중 순차 방출 구속 노드 (k_구속 < 1) 에 닿는 면 빼기
  --w5gk0 1 진단: ghost penalty 강성 0 (고스트 질량 그대로)
  --w5fb 1  절 CM68 F-bar 를 같이 설치 (lib/anchorflow/fem_wolfh4.py fb)
  --w5gw 1  B_gk: ghost penalty 면 가중 = 면 36 노드 k_구속 최솟값
  --w5s 1   S_구속: 순차 방출 구속 노드 몫을 정답 dx 격자 받침에서 (정답 노드 k_g -> 입자 κ_p -> 우리 노드 질량 가중)
인자가 하나도 없으면 모듈을 안 부름 (mfix_wrap 그대로). 방출 매개 (축·끝 위치·반폭·앞면 속도·정답 dx) 는 --data 의 cfg 에서.
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
while argv and argv[0].startswith("--w5"):
    CAND[argv[0][4:]] = int(float(argv[1]))
    del argv[:2]
ON = any(CAND.values())


def rel_of(path):
    """cfg 의 release_particles_sequentially (exe/fem_an_rt_sim.py 의 구속 상자와 같은 매개) -- 없으면 빈 dict."""
    import torch
    try:
        d = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    except Exception:                                                             # noqa: BLE001
        d = torch.load(path, map_location="cpu", weights_only=False)
    cfg = d["cfg"]
    o = dict(dxg=float(cfg.get("grid_lim", 2.0)) / int(cfg.get("n_grid", 100)))
    for bc in (cfg.get("boundary_conditions") or []):
        if bc.get("type") == "release_particles_sequentially":
            ax = [k for k in range(3) if bc["normal"][k] != 0][0]
            H0 = abs(float(bc["start_position"]) - float(bc["end_position"]))
            o.update(ax=ax, c0=float(bc["end_position"]), H0=H0, vf=H0 / float(bc["end_time"]))
    del d
    return o


_orig = runpy.run_path
_st = dict(done=False)


def run_path(path, *a, **kw):
    if ON and not _st["done"] and os.path.basename(path) in ("fem_an_rt_sim.py",):
        _st["done"] = True
        if CAND.get("fb", 0):                                                   # 절 CM68 F-bar (칸 평균 증분 체적) 를 같이 (모래 경로만)
            from anchorflow import fem_wolfh4 as W4
            W4.install(fb=1, dg=0, out=out)
        from anchorflow import fem_wolfh5 as W5
        W5.install(dg=CAND.get("dg", 0), s=CAND.get("s", 0), c=CAND.get("c", 0), g=CAND.get("g", 0), gk0=CAND.get("gk0", 0), gw=CAND.get("gw", 0), out=out, rel=rel_of(argv[argv.index("--data") + 1]))
    return _orig(path, *a, **kw)


runpy.run_path = run_path
sys.argv = [os.path.join(HERE, "mfix_wrap.py")] + argv
_orig(sys.argv[0], run_name="__main__")

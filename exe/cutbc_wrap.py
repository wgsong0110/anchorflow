"""GaussianFluent `cut` 경계 (surface_collider surface="cut", GF surface_type 11) 를 우리 판 노드에 거는 감싸개.

사용: python -u exe/cutbc_wrap.py --cut_bc 1 [--cut_slab fix|free] [<다음 감싸개.py> <그 인자...>] exe/fem_an_rt_sim.py <실행기 인자 ...>
      (예: python -u exe/cutbc_wrap.py --cut_bc 1 exe/impulse_wrap.py --impulse_bc 1 exe/fem_an_rt_sim.py --data ... --floor none --cfg_bc ...)
--cut_bc 가 없거나 0 이면 아무것도 안 함 (결과 불변). 실행기가 --floor vel 로 첫 surface_collider 를 바닥으로 읽으므로 cut 이 있는 씬은 --floor none 필수 (아니면 중단).

GF 정의 (mpm_solver_warp.py 의 collide 커널 surface_type 11): 시각 start <= t < end 에 격자 노드 (좌표 = 인덱스*dx = 시뮬 공간 좌표) 마다
  (x_n - point) . normal < 0 인 노드만 대상.  대상 중 z < 0.4 또는 z > 0.53 이면 v = 0, 아니면 v = (vx, 0, vz) * 0.3.
  이 후처리는 GF 에서 매 서브스텝 (substep_dt) 격자 속도에 걸린다.

우리 구현 (해석, 아래 근거):
  - v = 0 노드 = 기존 --cfg_bc 의 상자 구속 노드 (rt_bc_box -> fixd, vbc = 0; 걸음 변위 증분 0 = Dirichlet) 로. 새 CUDA 커널 없음.
    노드 위치는 격자 노드 기준 위치 A*y (rt_bc_box 가 쓰는 것과 같음). 상자는 법선이 축 정렬일 때만 가능 (아니면 중단).
  - 슬랩 (0.4 <= z <= 0.53) 의 "x0.3, y 0" 은 GF 에서 매 서브스텝 곱해진다. 우리 한 걸음 tau = frame_dt/K 는 GF 서브스텝 n = tau/substep_dt 개를 덮으므로
    걸음 전체 몫은 0.3^n (예: n = 300 이면 1e-157 = 0). 즉 걸음 단위에서 정확한 몫은 "변위 증분 0" (Dirichlet) 의 극한. 그래서 --cut_slab fix (기본) = 슬랩 노드도 구속 (0).
    --cut_slab free = 슬랩 노드를 자유로 둠 (비교용; GF 와 다름).
  - 못 옮기는 것: GF 의 0.3 배는 서브스텝마다 일어나므로 정상상태에 v = 0.43 dt a 의 누설이 있다 (노드 가속도 a). 이것은 걸음 한 번짜리 음해법에서 표현 못 함 (fix 는 누설 0).
    y 성분만 0 (x, z 는 자유) 인 노드별 성분 구속도 fixd 가 노드 단위라 표현 불가 -- 위 극한 때문에 fix 에서는 필요 없음.
"""
import math
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lib"))
sys.setrecursionlimit(max(sys.getrecursionlimit(), 15000))

ZLO, ZHI, SLAB_F = 0.4, 0.53, 0.3
BIG = 20.0


def cut_rule_factor(xn, point, normal, n_sub):
    """GF 규칙 그대로 노드별 속도 몫 (N,3, numpy): 대상 아님 1, z<0.4 또는 z>0.53 이면 0, 아니면 (0.3^n, 0, 0.3^n) (n = 걸음이 덮는 GF 서브스텝 수; n=1 이면 GF 한 서브스텝 그대로)."""
    import numpy as np
    nv = np.asarray(normal, dtype=np.float64); nv = nv / np.linalg.norm(nv)
    tgt = ((xn - np.asarray(point, dtype=np.float64)) @ nv) < 0
    out = np.ones_like(xn)
    z = xn[:, 2]
    zero = tgt & ((z < ZLO) | (z > ZHI))
    slab = tgt & ~zero
    out[zero] = 0.0
    f = SLAB_F ** n_sub
    out[slab] = np.array([f, 0.0, f])
    return out


def cut_boxes(point, normal, t0, t1, loff, slab):
    """cut 대상 영역을 rt_bc_box 상자 (중심, 반 크기, 속도, t0, t1) 목록으로 (시뮬 공간 + 띄운 거리 loff)."""
    nv = [float(q) for q in normal]
    nn = math.sqrt(sum(q * q for q in nv)); nv = [q / nn for q in nv]
    ax = [i for i in range(3) if abs(nv[i]) > 1e-9]
    if len(ax) != 1 or abs(abs(nv[ax[0]]) - 1.0) > 1e-6:
        raise SystemExit(f"[cutbc] 법선 {normal} 이 축 정렬이 아님 -- 상자 구속으로 못 옮김")
    k, sg = ax[0], (1.0 if nv[ax[0]] > 0 else -1.0)
    pk = float(point[k])
    lo, hi = (-BIG, pk) if sg > 0 else (pk, BIG)                                    # (x_k - p_k) * sg < 0 영역

    def box(lo_, hi_):
        c = [0.0, 0.0, 0.0]; s = [BIG, BIG, BIG]
        c[k] = 0.5 * (lo_ + hi_); s[k] = 0.5 * (hi_ - lo_)
        return ([c[a] + loff[a] for a in range(3)], s, [0.0, 0.0, 0.0], t0, t1)
    if slab == "fix":
        return [box(lo, hi)]
    assert k == 2, "--cut_slab free 는 법선이 z 축일 때만 (GF 의 z 문턱이 z 축)"
    out = []
    for a_, b_ in ((-BIG, ZLO), (ZHI, BIG)):                                         # z < 0.4, z > 0.53 중 대상 영역과 겹치는 부분
        l2, h2 = max(a_, lo), min(b_, hi)
        if l2 < h2:
            out.append(box(l2, h2))
    return out


def install(slab):
    from anchorflow import fem_an_rt as ART
    orig = ART.RTSolver.__init__
    done = [False]

    def init(s, *a, **kw):
        if not done[0]:
            done[0] = True
            g = None
            f = sys._getframe(1)
            while f is not None:
                if "cfg" in f.f_globals and "d" in f.f_globals and "h" in f.f_globals and f.f_globals.get("__name__") == "__main__":
                    g = f.f_globals; break
                f = f.f_back
            assert g is not None, "[cutbc] 실행기 전역 (cfg) 을 못 찾음"
            cfg = g["cfg"]
            cuts = [b for b in (cfg.get("boundary_conditions") or []) if b.get("type") == "surface_collider" and b.get("surface") == "cut"]
            if cuts:
                assert kw.get("floor") is None, "[cutbc] cut 씬은 --floor none 필수 (실행기가 첫 surface_collider 를 바닥으로 읽음)"
                lo = g.get("LOFF")
                loff = [float(q) for q in lo.flatten().tolist()] if lo is not None else [0.0, 0.0, 0.0]
                tau = float(g["h"]) / float(kw["K"] if "K" in kw else a[11])
                nsub = tau / float(cfg["substep_dt"])
                bc = list(kw.get("bc") or [])
                for c in cuts:
                    bx = cut_boxes(c["point"], c["normal"], float(c.get("start_time", 0.0)), float(c.get("end_time", 1e3)), loff, slab)
                    bc += bx
                    print(f"[cutbc] cut point {c['point']} normal {c['normal']} t [{bx[0][3]}, {bx[0][4]}) 상자 {len(bx)} 개, 걸음당 GF 서브스텝 n = {nsub:.1f}, 슬랩 몫 0.3^n = {SLAB_F ** nsub:.3e}, 슬랩 {slab}", flush=True)
                    if slab == "fix" and SLAB_F ** nsub > 1e-6:
                        print("[cutbc] 경고: 0.3^n 이 작지 않음 -- Dirichlet 0 극한이 GF 와 많이 다를 수 있음", flush=True)
                kw["bc"] = bc
        return orig(s, *a, **kw)

    ART.RTSolver.__init__ = init


if __name__ == "__main__":
    argv = sys.argv[1:]
    ON, SLAB = 0, "fix"
    while argv and argv[0] in ("--cut_bc", "--cut_slab"):
        if argv[0] == "--cut_bc":
            ON = int(float(argv[1]))
        else:
            SLAB = argv[1]; assert SLAB in ("fix", "free")
        del argv[:2]
    _orig = runpy.run_path
    _done = [False]

    def run_path(path, *a, **kw):
        if ON and not _done[0] and os.path.basename(path) == "fem_an_rt_sim.py":
            _done[0] = True
            install(SLAB)
        return _orig(path, *a, **kw)

    runpy.run_path = run_path
    sys.argv = [os.path.abspath(argv[0])] + argv[1:]
    _orig(sys.argv[0], run_name="__main__")

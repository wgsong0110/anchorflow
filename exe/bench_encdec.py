"""인코딩/디코딩 한 번에 걸리는 시간을, 두 가중치 방식에 대해 잰다.

표현 잔차는 잘린 가우시안이 낮았다. 남는 질문은 값을 치르고 얻은 것인지다 --
잘린 쪽은 가우시안마다 붙는 앵커 수가 다르고(학습 후 평균 17.6, 최대 39),
softmax 는 상한이 k 로 묶여 있다. 짝 수가 다르면 비용도 다르다.

네 조각을 따로 잰다. 롤아웃에서 호출되는 빈도가 서로 다르기 때문이다.

  refresh    KD-트리 질의로 후보 짝을 다시 만든다. 스케줄로 가끔만 돈다.
  prepare    가중치·B·B^-1 을 만든다. 앵커가 움직이면 매번 필요하다.
  ls_factor  인코더의 촐레스키 분해. prepare 당 한 번이면 되고 프레임마다
             재활용된다 -- 그래서 인코딩 자체와 나눠 잰다.
  project_ls / gaussian_pos   실제 인코딩과 디코딩.

CUDA 는 비동기라 synchronize 없이 재면 커널 발사 시간만 재게 된다. 워밍업 후
동기화해서 잰다.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

_lib = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib")
sys.path.insert(0, _lib)

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--h5_dir", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--geom", default=None, help="학습된 잘린-가우시안 기하 .pt")
ap.add_argument("--geom_softmax", default=None, help="학습된 softmax 기하 .pt")
ap.add_argument("--softmax_k", type=int, default=16)
ap.add_argument("--n_anchors", type=int, default=512)
ap.add_argument("--K", type=int, default=8)
ap.add_argument("--n_pts", type=int, default=40000)
ap.add_argument("--stride", type=int, default=2)
ap.add_argument("--c", type=float, default=0.25)
ap.add_argument("--eig_floor", type=float, default=0.02)
ap.add_argument("--warmup", type=int, default=10)
ap.add_argument("--reps", type=int, default=50)
ap.add_argument("--substeps", type=int, default=40,
                help="한 프레임의 서브스텝 수. 프레임 비용은 이것 곱하기 서브스텝 비용")
ap.add_argument("--mpm", action="store_true",
                help="같은 입자 집합 위에서 MPM 한 서브스텝도 잰다")
ap.add_argument("--student", action="store_true",
                help="같은 앵커 수에서 학생 스테퍼 한 스텝도 잰다")
ap.add_argument("--hidden", type=int, default=128)
ap.add_argument("--depth", type=int, default=4)
ap.add_argument("--heads", type=int, default=4)
ap.add_argument("--chunk", type=int, default=1)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

dev = "cuda"
torch.set_grad_enabled(False)

from anchorflow import gf_scene                   # noqa: E402
from anchorflow.anchor_sparse import AnchorSparse  # noqa: E402

cfg = gf_scene.read_cfg(a.config)
TR, EXT = gf_scene.load_traj(a.h5_dir, stride=a.stride, n_pts=a.n_pts,
                             seed=a.seed, dev=dev)
X0 = TR[0].contiguous()
# 변형된 상태를 인코딩하는 비용을 재야 한다. 정준 배치는 잔차가 0 에 가까워
# 촐레스키가 비정상적으로 편한 경우가 될 수 있다.
XT = TR[-1].contiguous()
sc = gf_scene.build_scene(X0, cfg, n_anchors=a.n_anchors, K=a.K,
                          eig_floor=a.eig_floor, dev=dev)
del TR


def timeit(fn, warmup, reps):
    """-> (평균 ms, 표준편차 ms)"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.mean(ts)), float(np.std(ts))


def bench(name, softmax_w, geom):
    fit = AnchorSparse(sc, c=a.c, eig_floor=a.eig_floor, softmax_w=softmax_w,
                       softmax_k=a.softmax_k).to(dev)
    if geom:
        st = torch.load(geom, map_location=dev, weights_only=False)
        for k in ("pos", "log_s", "quat", "log_amp"):
            getattr(fit, k).data.copy_(st[k].to(dev))
    fit.refresh()
    fit.set_B_ref()
    P = int(fit.pair_g.shape[0])
    cache = fit.prepare()
    fac = fit.ls_factor(cache)

    r = {"pairs": P, "pairs_per_gaussian": P / fit.N}
    r["refresh"], r["refresh_sd"] = timeit(fit.refresh, 3, max(5, a.reps // 5))
    fit.refresh()
    r["prepare"], r["prepare_sd"] = timeit(lambda: fit.prepare(), a.warmup, a.reps)
    cache = fit.prepare()
    r["ls_factor"], r["ls_factor_sd"] = timeit(lambda: fit.ls_factor(cache),
                                               a.warmup, a.reps)
    fac = fit.ls_factor(cache)
    r["encode"], r["encode_sd"] = timeit(lambda: fit.project_ls(XT, cache, fac),
                                         a.warmup, a.reps)
    p = fit.project_ls(XT, cache, fac)
    r["decode"], r["decode_sd"] = timeit(lambda: fit.gaussian_pos(p, cache),
                                         a.warmup, a.reps)
    # 한 프레임을 왕복하는 실제 비용: 앵커가 움직였으니 prepare 는 다시 해야 하고,
    # 인코더 분해는 그 prepare 당 한 번이다.
    # 물리 한 서브스텝. 인코딩/디코딩과 달리 매 서브스텝 돌아야 하는 비용이다.
    v0 = torch.zeros_like(fit.pos)
    w_, rc_, q_, Binv_, blocked_, mass_ = cache
    m_ = mass_.unsqueeze(-1)
    keep_ = (~fit.fixed).unsqueeze(-1).to(fit.pos.dtype)
    pp = fit.pos.detach().clone()
    r["substep"], r["substep_sd"] = timeit(
        lambda: fit.substep(pp, v0, w_, rc_, q_, Binv_, blocked_, m_, keep_),
        a.warmup, a.reps)
    r["frame_sim"] = r["substep"] * a.substeps
    r["roundtrip"] = r["prepare"] + r["ls_factor"] + r["encode"] + r["decode"]
    r["roundtrip_cached"] = r["encode"] + r["decode"]
    print(f"[{name}]  짝 {P} ({P/fit.N:.1f}/가우시안)\n"
          f"    refresh {r['refresh']:7.2f}  prepare {r['prepare']:6.2f}  "
          f"ls_factor {r['ls_factor']:6.2f}\n"
          f"    encode  {r['encode']:7.2f}  decode  {r['decode']:6.2f}  "
          f"-> 왕복 {r['roundtrip']:.2f} ms "
          f"(prepare 재활용 시 {r['roundtrip_cached']:.2f})\n"
          f"    substep {r['substep']:6.2f} x{a.substeps} = "
          f"프레임 물리 {r['frame_sim']:.1f} ms", flush=True)
    del fit
    torch.cuda.empty_cache()
    return r


CASES = [("잘린 가우시안 · 학습 전", False, None)]
if a.geom:
    CASES.append(("잘린 가우시안 · 학습 후", False, a.geom))
CASES.append((f"kNN softmax k={a.softmax_k} · 학습 전", True, None))
if a.geom_softmax:
    CASES.append((f"kNN softmax k={a.softmax_k} · 학습 후", True, a.geom_softmax))

res = {n: bench(n, sm, g) for n, sm, g in CASES}

# ---- 학생 스테퍼 --------------------------------------------------------
# 학생은 앵커 시뮬의 한 프레임(서브스텝 40 회)을 한 번의 순전파로 대신한다.
# 그래서 "서브스텝 대 스텝"이 아니라 **프레임 대 프레임**으로 견줘야 한다.
# 비용은 앵커 수와 구조가 정하고 가우시안 수와 무관하다 (스키닝만 예외).
if a.student:
    from anchorflow.nextstate import NextStep, apply_step   # noqa: E402
    fit = AnchorSparse(sc, c=a.c, eig_floor=a.eig_floor).to(dev)
    fit.refresh(); fit.set_B_ref()
    cache = fit.prepare()
    FRAME_DT = float(sc.sub_dt) * a.substeps
    net = NextStep(hidden=a.hidden, depth=a.depth, heads=a.heads,
                   use_accel=False, scale=EXT,
                   vel_scale=EXT / max(FRAME_DT, 1e-6),
                   chunk=a.chunk, zero_init=True).to(dev).eval()
    pS = fit.pos.detach().clone()
    vS = torch.zeros_like(pS)
    fwd, fwd_sd = timeit(
        lambda: apply_step(net, pS, vS, None, FRAME_DT, fit.fixed), a.warmup, a.reps)
    dec, _ = timeit(lambda: fit.gaussian_pos(pS, cache), a.warmup, a.reps)
    res["학생 스테퍼"] = {"forward": fwd, "forward_sd": fwd_sd, "decode": dec,
                      "frame": fwd + dec, "hidden": a.hidden, "depth": a.depth,
                      "heads": a.heads, "chunk": a.chunk}
    print(f"[학생 스테퍼]  순전파 {fwd:.2f} ms + 디코딩 {dec:.2f} ms = "
          f"프레임 {fwd+dec:.2f} ms  (앵커 {int(sc.M)}, hidden {a.hidden}, "
          f"depth {a.depth}, chunk {a.chunk})", flush=True)
os.makedirs(a.out, exist_ok=True)
json.dump(dict(n_pts=int(X0.shape[0]), n_anchors=int(sc.M), reps=a.reps,
               gpu=torch.cuda.get_device_name(0), cases=res),
          open(os.path.join(a.out, "encdec_speed.json"), "w"), indent=1,
          ensure_ascii=False)
# ---- MPM ---------------------------------------------------------------
# 같은 입자 집합·같은 격자 위에서 재야 비교가 된다. 앵커 시뮬은 앵커 512 개를
# 적분하고 가우시안을 통해 힘을 모으는 반면, MPM 은 입자 전부를 격자에 뿌렸다
# 되받는다 -- 그 차이가 그대로 비용 차이다.
if a.mpm:
    import warp as wp                                   # noqa: E402

    wp.init()
    from anchorflow.mpm_teacher import MPMTeacher       # noqa: E402

    T = MPMTeacher(sc, horizon=a.substeps * float(sc.sub_dt) * 4)
    dt_ = float(sc.sub_dt)
    ms, sd = timeit(lambda: T.solver.p2g2p(None, dt_, device=T.wp_dev),
                    a.warmup, a.reps)
    res["MPM"] = {"substep": ms, "substep_sd": sd,
                  "frame": ms * a.substeps, "n_grid": int(sc.n_grid),
                  "n_particles": int(sc.keep.sum())}
    print(f"[MPM]  substep {ms:.2f} ms x{a.substeps} = 프레임 {ms*a.substeps:.1f} ms "
          f"(입자 {int(sc.keep.sum())}, 격자 {int(sc.n_grid)}^3)", flush=True)

print(f"[저장] {os.path.join(a.out, 'encdec_speed.json')}", flush=True)
print("SPEED_OK")

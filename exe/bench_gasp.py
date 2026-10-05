"""GASP 를 **같은 품질기준 규약**으로 잰다.

GASP 의 시뮬은 GaMeS(`gs_flat`) 모델에서 만든 **가짜 메시(pseudomesh)의 꼭짓점**을
taichi_elements 의 MPM 솔버로 굴리는 것이다 (`taichi_examples/demo/3d.py` 가 하는 일).
여기서는 그 솔버를 그대로 쓰되 장면을 우리 규약(바닥 z=0.1 위 낙하, 프레임 1/60,
앞 24 프레임 버리고 충돌 10 프레임 비교)에 맞춘다.

서브스텝 손잡이는 그쪽 솔버의 `dt_scale` 이다:
    default_dt = 2e-2 * dx / size * dt_scale,  substeps = int(frame_dt/default_dt)+1
그쪽이 고정해 둔 값(nu=0.2, rho=1000)은 **그대로 둔다**.

  python exe/bench_gasp.py --shape mic --material elastic --run run5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

W = "/home/dkta/work"
sys.path.insert(0, f"{W}/taichi_elements")
sys.path.insert(0, os.path.dirname(__file__))
from bench_qfps import ladder, search, rms_rel        # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--shape", required=True)
ap.add_argument("--material", default="elastic",
                choices=["elastic", "elastoplastic", "viscoplastic",
                         "fracture"])
ap.add_argument("--run", default="run5")
ap.add_argument("--frames", type=int, default=10)
ap.add_argument("--skip", type=int, default=24)
ap.add_argument("--floor", type=float, default=0.1)
ap.add_argument("--n_grid", type=int, default=100)
ap.add_argument("--tol", type=float, default=5e-3)
ap.add_argument("--tau", type=float, default=1e-3)
ap.add_argument("--s0", type=int, default=400, help="사다리 시작 서브스텝")
ap.add_argument("--max_mul", type=int, default=8)
ap.add_argument("--phase", choices=["search", "time", "both"], default="both")
ap.add_argument("--out", default="")
a = ap.parse_args()

# taichi_elements 의 MPM 이 가진 재질: elastic / sand / snow / water.
# 우리 축(탄성·탄소성·점소성·파괴)과 맞는 것만 잰다 -- 나머지는 **미지원**이다
# (그쪽에 그 구성식이 없다. 임의로 다른 재질을 갖다 붙이면 비교가 아니다).
MAT = {"elastic": "elastic", "elastoplastic": "snow", "viscoplastic": "sand"}
if a.material not in MAT:
    print(f"[미지원] GASP(taichi_elements)에는 {a.material} 구성식이 없다",
          flush=True)
    if a.out:
        json.dump(dict(method="gasp", shape=a.shape, material=a.material,
                       unsupported=True,
                       reason="taichi_elements MPM 에 해당 구성식 없음"),
                  open(a.out, "w"), indent=1)
    raise SystemExit(0)

VP = f"{W}/gamesout/{a.shape}/pseudomesh_info/ours_30000/vertices.pt"
if not os.path.exists(VP):
    raise SystemExit(f"가짜 메시가 없다: {VP} (create_pseudomesh.py 를 먼저)")
V = torch.load(VP, map_location="cpu").cpu().numpy().reshape(-1, 3)

# 우리 장면 좌표로 옮긴다: 앵커 채우기 입자집합과 **같은 바운딩박스**에 맞춘다
X0 = np.load(f"{W}/anfill_{a.shape}.npy")
lo_t, hi_t = X0.min(0), X0.max(0)
lo_v, hi_v = V.min(0), V.max(0)
sc = float((hi_t - lo_t).max() / (hi_v - lo_v).max())
P0 = (V - (lo_v + hi_v) / 2.0) * sc + (lo_t + hi_t) / 2.0
L = float(np.linalg.norm(X0.max(0) - X0.min(0)))
print(f"[설정] gasp {a.shape} {a.material}  꼭짓점 {P0.shape[0]}  지름 L {L:.4f}",
      flush=True)

import taichi as ti                                   # noqa: E402
from engine.mpm_solver import MPMSolver               # noqa: E402

ti.init(arch=ti.gpu, device_memory_fraction=0.7, log_level=ti.ERROR)
MATID = {"elastic": MPMSolver.material_elastic,
         "snow": MPMSolver.material_snow,
         "sand": MPMSolver.material_sand}


def run(s, frames=None):
    """서브스텝 s 로 굴린다 -> (궤적 [T,N,3], 시뮬 시간)."""
    frames = a.frames if frames is None else frames
    nrun = frames + a.skip
    dx = 2.0 / a.n_grid
    # substeps = int(frame_dt / (2e-2*dx/size*dt_scale)) + 1 == s 가 되게
    dt_scale = (1.0 / 60.0) / (2e-2 * dx / 2.0) / float(s)
    mpm = MPMSolver(res=(a.n_grid,) * 3, size=2, dt_scale=dt_scale,
                    E_scale=1.0, unbounded=False, support_plasticity=True)
    mpm.set_gravity((0.0, 0.0, -9.8))
    mpm.add_surface_collider(point=(0.0, 0.0, a.floor), normal=(0.0, 0.0, 1.0),
                             surface=MPMSolver.surface_sticky)
    mpm.add_particles(particles=P0.astype(np.float32),
                      material=MATID[MAT[a.material]])
    X, t_sim = [], 0.0
    for f in range(nrun + 1):
        if f >= a.skip:
            X.append(mpm.particle_info()["position"].copy())
        if f == nrun:
            break
        ti.sync()
        t0 = time.time()
        mpm.step(1.0 / 60.0)
        ti.sync()
        t_sim += time.time() - t0
    return np.stack(X), t_sim


SJ = a.out or f"{W}/bench/{a.run}/gasp_{a.shape}_{a.material}.json"
d = json.load(open(SJ)) if os.path.exists(SJ) else {}

if a.phase in ("search", "both"):
    Xc, sc_, hist = ladder(run, a.s0, a.tau, L, max_mul=a.max_mul)
    print(f"[기준] s={sc_} 최고정밀 궤적 확보", flush=True)
    best = search(run, Xc, L, a.tol, a.s0 // 4, sc_)
    d.update(method="gasp", shape=a.shape, material=a.material,
             n_particles=int(P0.shape[0]), L=L, s_conv=int(sc_),
             n_grid=a.n_grid, frames=a.frames, tol=a.tol, tau=a.tau,
             ladder=[(int(q), float(w)) for q, w in hist])
    if best is None:
        print("[결과] 합격 설정 없음", flush=True)
        d["s"] = None
    else:
        s, _t, e1, eT = best
        d.update(s=int(s), e1=float(e1), eT=float(eT))
        print(f"[탐색] s={s}, 한프레임 {100 * e1:.4f}% 누적 {100 * eT:.4f}%",
              flush=True)
    json.dump(d, open(SJ, "w"), indent=1)

if a.phase in ("time", "both") and d.get("s"):
    s = int(d["s"])
    _, tL = run(s, 60)
    t_per = tL / float(60 + a.skip)
    d.update(ms_per_frame=1000 * t_per, ms_per_substep=1000 * t_per / s,
             fps=1.0 / t_per)
    json.dump(d, open(SJ, "w"), indent=1)
    print(f"[결과] gasp {a.shape} {a.material}: s={s}, {1.0 / t_per:.2f} FPS "
          f"({1000 * t_per:.1f} ms/프레임, 서브스텝당 "
          f"{1000 * t_per / s:.3f} ms)", flush=True)

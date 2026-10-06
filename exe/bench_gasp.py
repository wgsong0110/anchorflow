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
ap.add_argument("--v0", type=float, default=0.0, help="초기 z 속도")
ap.add_argument("--tol", type=float, default=5e-3)
ap.add_argument("--tau", type=float, default=1e-3)
ap.add_argument("--s0", type=int, default=400, help="사다리 시작 서브스텝")
ap.add_argument("--max_mul", type=int, default=8)
ap.add_argument("--phase", choices=["search", "time", "both", "smoke"],
                default="both")
ap.add_argument("--out", default="")
a = ap.parse_args()

# taichi_elements 의 MPM 이 가진 재질: elastic / sand / snow / water.
# 우리 축(탄성·탄소성·점소성·파괴)과 맞는 것만 잰다 -- 나머지는 **미지원**이다
# (그쪽에 그 구성식이 없다. 임의로 다른 재질을 갖다 붙이면 비교가 아니다).
# 파괴는 그쪽에 없어서 **우리가 더한 재질**이다 (exe/patch_ti_fracture.py 가
# CD-MPM(Cam-Clay+Borden)을 taichi 솔버에 한 갈래 붙인다).
MAT = {"elastic": "elastic", "elastoplastic": "snow",
       "viscoplastic": "sand", "fracture": "fracture"}
if a.material == "fracture":                 # PG 파괴 칸과 같은 충돌 세기
    if a.v0 == 0.0:
        a.v0 = -6.0
# ⚠ 파괴 칸의 **격자는 올리지 않는다** (PG 는 200 으로 올렸다). GASP 의 입자는
# 가짜 메시 꼭짓점으로 **개수가 고정**이라, 격자를 2 배로 올리면
#   (1) p_vol = dx^3 이라 입자 질량이 8 배 작아져 수치가 터지고, 터진 F 를
#       taichi 의 svd 가 받으면 반복이 끝나지 않는다 (서브스텝 하나에서 영구 정지)
#   (2) 서브스텝당 2.0 초 (격자 100 의 0.1 초) 라 궤적 하나에 15 시간이다
# 둘 다 실측했다 (2026-10-06, 네 형상 전부 50 분 돌려 한 서브스텝도 못 나갔다).
# 그래서 GASP 파괴는 **그쪽 격자(100)** 에 초기속도 -6 만 넣어 잰다.

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

# 카드를 통째로 요구하면 같은 GPU 의 다른 잡과 부딪혀 초기화가 죽는다.
# (실측: 0.7 로 두니 "materialize_runtime" 에서 CUDA 오류)
ti.init(arch=ti.gpu, log_level=ti.ERROR,
        device_memory_fraction=float(os.environ.get("AF_TI_FRAC", 0.35)))
MATID = {"elastic": MPMSolver.material_elastic,
         "snow": MPMSolver.material_snow,
         "sand": MPMSolver.material_sand,
         "fracture": getattr(MPMSolver, "material_fracture", -1)}
if MATID[MAT[a.material]] < 0:
    raise SystemExit("[실패] taichi 솔버에 파괴 재질이 없다. "
                     "exe/patch_ti_fracture.py 를 먼저 돌릴 것")


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
                      material=MATID[MAT[a.material]],
                      velocity=([0.0, 0.0, a.v0] if a.v0 else None))
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

import joblock                                           # noqa: E402
joblock.take(f"gasp_{a.shape}_{a.material}",
             out_exists=("s" in d and a.phase == "search"))

if a.phase == "smoke":
    # 한 궤적만 굴려 **터지지 않는지**와 **실제로 깨지는지**를 본다.
    # 격자를 200 으로 올렸을 때 (p_vol = dx^3 이라 질량이 8 배 작아진다)
    # 수치가 터지고 taichi 의 svd 가 끝나지 않는 일이 있었다 -- 대량 실행
    # 전에 한 칸을 이렇게 먼저 본다.
    X, tl = run(a.s0, a.frames)
    c0 = X[0].mean(0)
    d0 = float(np.linalg.norm(X[0].max(0) - X[0].min(0)))
    for f in range(X.shape[0]):
        bb = float(np.linalg.norm(X[f].max(0) - X[f].min(0)))
        far = float((np.linalg.norm(X[f] - c0, axis=1) > 0.75 * d0).mean())
        print(f"  [연기] 프레임 {f:2d} 지름 {bb / d0:.3f}배  멀어진 입자 "
              f"{100 * far:.2f}%  유한 {bool(np.isfinite(X[f]).all())}",
              flush=True)
    print(f"[연기] s={a.s0} {X.shape[0]} 프레임 {tl:.1f}초 "
          f"(프레임당 {tl / X.shape[0]:.2f}초)", flush=True)
    raise SystemExit(0)

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

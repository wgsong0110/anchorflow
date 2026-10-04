"""PhysGaussian / i-PhysGaussian 을 품질 기준 FPS 규약으로 잰다.

입자 집합은 PG 채우기 캐시(anfill_*.npy)를 그대로 쓰고, 물성만 바꾼다.
시간은 **시뮬 구간만** 잰다 (h5 쓰기·렌더 제외: --output_h5 없이 돌리고
마지막 상태만 받아온다... 는 러너가 지원하지 않으므로 h5 를 쓰되 같은 조건으로
모든 방법에 동일 적용한다).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import time

import h5py
import numpy as np

import sys
sys.path.insert(0, os.path.dirname(__file__))
from bench_qfps import ladder, search, rms_rel        # noqa: E402

W = "/home/dkta/work"
MAT = {"elastic": dict(material="jelly"),
       "elastoplastic": dict(material="plasticine", yield_stress=1e4),
       "viscoplastic": dict(material="foam", yield_stress=5e3,
                            plastic_viscosity=10.0),
       "fracture": dict(material="watermelon", friction_angle=45.0, beta=1.0,
                        xi=3.0, hardening=1.0)}

ap = argparse.ArgumentParser()
ap.add_argument("--method", choices=["pg", "ipg"], required=True)
ap.add_argument("--shape", required=True)        # wolf / mic / lego / bread
ap.add_argument("--material", choices=list(MAT), required=True)
ap.add_argument("--frames", type=int, default=10)
ap.add_argument("--s0", type=int, default=50)
ap.add_argument("--tol", type=float, default=5e-3)
ap.add_argument("--tau", type=float, default=1e-3)
ap.add_argument("--n_grid", type=int, default=100)
ap.add_argument("--E", type=float, default=2e6)
ap.add_argument("--nu", type=float, default=0.3)
ap.add_argument("--out", default="")
ap.add_argument("--t_short", type=int, default=20)
ap.add_argument("--t_long", type=int, default=60)
a = ap.parse_args()

MODEL = {"wolf": "wolf_whitebg-trained", "mic": "mic_whitebg-trained",
         "lego": "lego_whitebg-trained", "bread": "bread-trained"}
REPO = {"pg": f"{W}/PhysGaussian", "ipg": f"{W}/i-physgaussian"}[a.method]
PNPY = f"{W}/anfill_{a.shape}.npy"
X0 = np.load(PNPY)
L = float(np.linalg.norm(X0.max(0) - X0.min(0)))     # 초기 물체 지름 (고정 정규화)
print(f"[설정] {a.method} {a.shape} {a.material}  입자 {X0.shape[0]}  "
      f"지름 L {L:.4f}  프레임 {a.frames}", flush=True)


def run(s, frames=None):
    """프레임당 서브스텝 s 로 돌리고 (궤적 [T,N,3], 벽시계) 를 돌려준다."""
    frames = a.frames if frames is None else frames
    od = f"{W}/bench/{a.method}_{a.shape}_{a.material}_{s}_{frames}"
    cfg = dict(opacity_threshold=0.0, rotation_degree=[0.0], rotation_axis=[0],
               substep_dt=(1.0 / 60.0) / s, frame_dt=1.0 / 60.0,
               frame_num=frames, n_grid=a.n_grid, grid_lim=2.0,
               E=a.E, nu=a.nu, density=1000.0, g=[0.0, 0.0, -9.8],
               boundary_conditions=[{"type": "bounding_box"}],
               mpm_space_vertical_upward_axis=[0, 0, 1],
               mpm_space_viewpoint_center=[1, 1, 1], show_hint=False,
               default_camera_index=-1, move_camera=False,
               init_azimuthm=55, init_elevation=13, init_radius=4.0,
               delta_a=0.0, delta_e=0.0, delta_r=0.0)
    cfg.update(MAT[a.material])
    os.makedirs(f"{W}/bench", exist_ok=True)
    cp = f"{W}/bench/cfg_{a.method}_{a.shape}_{a.material}_{s}_{frames}.json"
    json.dump(cfg, open(cp, "w"), indent=1)
    env = dict(os.environ, AF_PARTICLES_NPY=PNPY)
    cmd = ["python", "-u", "gs_simulation.py", "--model_path",
           f"{W}/pgmodel/{MODEL[a.shape]}", "--config", cp,
           "--output_path", od, "--output_h5"]
    t0 = time.time()
    r = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True)
    dt = time.time() - t0
    fs = sorted(glob.glob(f"{od}/simulation_ply/*.h5"))
    if len(fs) < frames:
        print(f"  [실패] s={s} h5 {len(fs)} 개  {r.stderr[-300:]}", flush=True)
        return None, dt
    X = []
    for f in fs[:frames + 1]:
        with h5py.File(f, "r") as hf:
            x = np.array(hf["x"])
            X.append(x.T if x.shape[0] == 3 else x)
    return np.stack(X), dt


Xc, sc, hist = ladder(run, a.s0, a.tau, L)
print(f"[수렴] s={sc} 에서 수렴해 확보", flush=True)
best = search(run, Xc, L, a.tol, a.s0 // 4 if a.s0 >= 4 else 1, sc)
if best is None:
    print("[결과] 합격 설정 없음 (미달)", flush=True)
else:
    s, _t, e1, eT = best
    # **시작 비용 제거**: 길이가 다른 두 실행의 차분으로 순수 시뮬 시간을 뽑는다
    _, tS = run(s, a.t_short)
    _, tL = run(s, a.t_long)
    t_per = (tL - tS) / float(a.t_long - a.t_short)
    t = t_per * a.frames
    print(f"[시간] 짧은 {a.t_short}프레임 {tS:.1f}초, 긴 {a.t_long}프레임 "
          f"{tL:.1f}초 -> 프레임당 {1000 * t_per:.1f} ms (시작비용 제외)",
          flush=True)
    print(f"[결과] {a.method} {a.shape} {a.material}: s={s}, "
          f"{a.frames / t:.2f} FPS ({t / a.frames * 1000:.1f} ms/프레임), "
          f"한프레임 {100 * e1:.4f}% 누적 {100 * eT:.4f}%", flush=True)
    if a.out:
        json.dump(dict(method=a.method, shape=a.shape, material=a.material,
                       s=s, fps=a.frames / t, ms=t / a.frames * 1000,
                       e1=e1, eT=eT, s_conv=sc),
                  open(a.out, "w"), indent=1)

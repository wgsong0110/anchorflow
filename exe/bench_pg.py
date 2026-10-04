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
# 물성별 설정. 파괴(CD-MPM)는 GaussianFluent 의 watermelon 설정을 **그쪽 값
# 그대로** 쓴다 (E 2e3 / nu 0.38 / 밀도 1 / g -15). E=2e6 로는 항복면에 닿지
# 않아 아예 깨지지 않는다.
MAT = {"elastic": dict(material="jelly"),
       "elastoplastic": dict(material="plasticine", yield_stress=1e4),
       "viscoplastic": dict(material="foam", yield_stress=5e3,
                            plastic_viscosity=10.0),
       "fracture": dict(material="watermelon", friction_angle=45.0, beta=1.0,
                        xi=3.0, hardening=1.0, alpha_0=-0.04,
                        E=2e3, nu=0.38, density=1.0, g=[0.0, 0.0, -15.0])}

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
ap.add_argument("--vid", default="", help="합격 설정의 궤적을 영상으로 남긴다")
# 탐색(서브스텝 찾기)은 병렬로 돌려도 되지만 **시간 측정은 단독 실행**이어야 한다.
ap.add_argument("--phase", choices=["search", "time", "both"], default="both")
# **떨어뜨려서 부딪히는 창**에서 기준을 잰다. 바닥이 z=--floor 에 있고 물체는
# (1,1,1) 중심이라 20~26 프레임쯤에 닿는다 -- 처음 10 프레임만 보면 자유낙하라
# 서브스텝을 아무리 줄여도 수렴해 보여 기준이 무의미해진다.
ap.add_argument("--skip", type=int, default=24, help="앞 몇 프레임을 버리는가")
ap.add_argument("--floor", type=float, default=0.1, help="바닥 z (0=바닥 없음)")
# i-PG 의 본체는 **암시적 적분**이다. 설명서의 실행 예가
#   --implicit --solver newton_gmres --dt_multiplier k --impulse_scale 1/k
# 라서, 명시적으로 돌리면 PG 와 수치가 똑같이 나온다 (실제로 s 가 같았다).
ap.add_argument("--solver", default="newton_gmres",
                choices=["picard", "picard_vanilla", "newton_gmres"])
ap.add_argument("--explicit", action="store_true", help="i-PG 를 명시적으로")
ap.add_argument("--max_mul", type=int, default=8, help="사다리 상한 배수")
a = ap.parse_args()

MODEL = {"wolf": "wolf_whitebg-trained", "mic": "mic_whitebg-trained",
         "lego": "lego_whitebg-trained", "bread": "bread-trained"}
REPO = {"pg": f"{W}/PhysGaussian", "ipg": f"{W}/i-physgaussian"}[a.method]
PNPY = f"{W}/anfill_{a.shape}.npy"
X0 = np.load(PNPY)
L = float(np.linalg.norm(X0.max(0) - X0.min(0)))     # 초기 물체 지름 (고정 정규화)
print(f"[설정] {a.method} {a.shape} {a.material}  입자 {X0.shape[0]}  "
      f"지름 L {L:.4f}  프레임 {a.frames}", flush=True)


RUN = os.environ.get("AF_BENCH_RUN", "run2")


def run(s, frames=None):
    """프레임당 서브스텝 s 로 돌리고 (궤적 [T,N,3], 벽시계) 를 돌려준다."""
    frames = a.frames if frames is None else frames
    nrun = frames + a.skip                 # 실제로 돌리는 프레임 수
    od = (f"{W}/bench/{RUN}/sim/{a.method}_{a.shape}_{a.material}"
          f"_{s}_{frames}")
    # **직전 회차의 h5 를 절대 재사용하지 않는다.** 실행이 터져도 옛 h5 가 남아
    # 있으면 개수 검사를 통과해 다른 입자 집합의 궤적을 읽는다 (입자 수가
    # 37855 와 311361 로 엇갈려 터진 원인이다).
    import shutil
    shutil.rmtree(od, ignore_errors=True)
    cfg = dict(opacity_threshold=0.0, rotation_degree=[0.0], rotation_axis=[0],
               substep_dt=(1.0 / 60.0) / s, frame_dt=1.0 / 60.0,
               frame_num=nrun, n_grid=a.n_grid, grid_lim=2.0,
               E=a.E, nu=a.nu, density=1000.0, g=[0.0, 0.0, -9.8],
               boundary_conditions=(
                   [{"type": "bounding_box"}]
                   + ([] if a.floor <= 0 else
                      [{"type": "surface_collider",
                        "point": [1.0, 1.0, a.floor],
                        "normal": [0.0, 0.0, 1.0], "surface": "sticky",
                        "friction": 0.0, "start_time": 0,
                        "end_time": 1000.0}])),
               mpm_space_vertical_upward_axis=[0, 0, 1],
               mpm_space_viewpoint_center=[1, 1, 1], show_hint=False,
               default_camera_index=-1, move_camera=False,
               init_azimuthm=55, init_elevation=13, init_radius=4.0,
               delta_a=0.0, delta_e=0.0, delta_r=0.0)
    cfg.update(MAT[a.material])
    os.makedirs(f"{W}/bench", exist_ok=True)
    os.makedirs(f"{W}/bench/{RUN}", exist_ok=True)
    cp = (f"{W}/bench/{RUN}/cfg_{a.method}_{a.shape}_{a.material}"
          f"_{s}_{frames}.json")
    json.dump(cfg, open(cp, "w"), indent=1)
    # warp 커널 캐시를 셀마다 분리한다 (같이 쓰면 동시 컴파일이 캐시를 깨뜨려
    # 모듈 적재 실패/불법 주소 접근으로 터진다)
    # 캐시는 **프로세스마다** 따로 쓴다. 같은 디렉토리를 둘이 쓰면 한쪽이
    # 쓰는 중에 다른 쪽이 읽어 "Failed to lookup kernel function" 으로 죽는다
    # (closure 로 만들어지는 collide 커널에서 실제로 터졌다).
    wc = f"{W}/wpcache/{a.method}_{a.shape}_{a.material}_{os.getpid()}"
    os.makedirs(wc, exist_ok=True)
    # 로케일을 UTF-8 로 못 박는다. 떼어낸 잡은 LANG 이 없어 ascii 가 되고,
    # warp 가 생성한 .cu 를 쓸 때 UnicodeEncodeError 로 죽는다 (PG 16 셀 전멸 원인)
    env = dict(os.environ, AF_PARTICLES_NPY=PNPY, WARP_CACHE_PATH=wc,
               PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    cmd = ["python", "-u", "gs_simulation.py", "--model_path",
           f"{W}/pgmodel/{MODEL[a.shape]}", "--config", cp,
           "--output_path", od, "--output_h5"]
    if a.method == "ipg" and not a.explicit:
        cmd += ["--implicit", "--solver", a.solver]
    t0 = time.time()
    r = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True)
    dt = time.time() - t0
    # **시뮬 구간만** 재서 쓴다 (PG 는 --render_img 없이도 프레임마다 카메라와
    # 래스터라이저를 다시 만든다 -- 벽시계로 재면 그게 섞여 FPS 가 뒤집힌다)
    for ln in r.stdout.splitlines():
        if ln.startswith("[AF시간]"):
            try:
                dt = float(ln.split("시뮬")[1].split("초")[0])
            except Exception:
                pass
    fs = sorted(glob.glob(f"{od}/simulation_ply/*.h5"))
    if len(fs) < nrun:
        print(f"  [실패] s={s} h5 {len(fs)} 개 (필요 {nrun})\n"
              f"{r.stderr[-1500:]}", flush=True)
        return None, dt
    X = []
    for f in fs[a.skip:a.skip + frames + 1]:
        with h5py.File(f, "r") as hf:
            x = np.array(hf["x"])
            X.append(x.T if x.shape[0] == 3 else x)
    return np.stack(X), dt


SJ = a.out or f"{W}/bench/{a.method}_{a.shape}_{a.material}.json"


def load():
    return json.load(open(SJ)) if os.path.exists(SJ) else {}


if a.phase in ("search", "both"):
    Xc, sc, hist = ladder(run, a.s0, a.tau, L, max_mul=a.max_mul)
    print(f"[수렴] s={sc} 에서 수렴해 확보", flush=True)
    best = search(run, Xc, L, a.tol, a.s0 // 4 if a.s0 >= 4 else 1, sc)
    d = load()
    d.update(method=a.method, shape=a.shape, material=a.material,
             n_particles=int(X0.shape[0]), L=L, s_conv=int(sc),
             n_grid=a.n_grid, E=a.E, nu=a.nu, frames=a.frames,
             tol=a.tol, tau=a.tau,
             ladder=[(int(q), float(w)) for q, w in hist])
    if best is None:
        print("[결과] 합격 설정 없음 (미달)", flush=True)
        d["s"] = None
    else:
        s, _t, e1, eT = best
        d.update(s=int(s), e1=float(e1), eT=float(eT))
        print(f"[탐색] s={s} (수렴 s={sc}), 한프레임 {100 * e1:.4f}% "
              f"누적 {100 * eT:.4f}%", flush=True)
    json.dump(d, open(SJ, "w"), indent=1)

if a.phase in ("time", "both"):
    d = load()
    if not d.get("s"):
        print("[시간] 합격 설정이 없어 건너뜀", flush=True)
        raise SystemExit(0)
    s = int(d["s"])
    # 계측된 시뮬 시간을 그대로 쓴다 (레포 안에서 p2g2p 루프만 잰다).
    # 길이차분은 시작비용은 지웠지만 **프레임마다 들어가는 렌더 준비**는 못
    # 지워서 i-PG wolf 가 61 FPS, mic 이 2.3 FPS 로 나오는 식이었다.
    nf = a.t_long
    _, tL = run(s, nf)
    t_per = tL / float(nf + a.skip)
    print(f"[시간] {nf + a.skip}프레임 시뮬 {tL:.2f}초 -> 프레임당 "
          f"{1000 * t_per:.2f} ms (p2g2p 루프만)", flush=True)
    d.update(ms_per_frame=1000 * t_per, ms_per_substep=1000 * t_per / float(s),
             fps=1.0 / t_per, t_short=tS, t_long=tL)
    json.dump(d, open(SJ, "w"), indent=1)
    print(f"[결과] {a.method} {a.shape} {a.material}: s={s} "
          f"(수렴 s={d['s_conv']}), {1.0 / t_per:.2f} FPS "
          f"({1000 * t_per:.1f} ms/프레임, 서브스텝당 "
          f"{1000 * t_per / s:.3f} ms), 한프레임 {100 * d['e1']:.4f}% "
          f"누적 {100 * d['eT']:.4f}%", flush=True)
    if a.vid:
        Xb, _ = run(s, max(a.frames, 40))
        import torch
        tp = a.vid.replace(".mp4", ".pt")
        torch.save({"x": torch.as_tensor(Xb)}, tp)
        import subprocess as sp
        sp.run(["python", "-u", f"{W}/anchorflow/exe/vid_traj.py",
                "--traj", tp, "--out", a.vid, "--sub", "120000", "--s", "0.6",
                "--label", f"{a.method} {a.shape} {a.material} (s={s}, "
                           f"{1000 * t_per:.0f} ms/프레임)"],
               cwd=f"{W}/anchorflow",
               env=dict(os.environ, PYTHONPATH=f"{W}/anchorflow/lib"))
        print(f"[영상] {a.vid}", flush=True)

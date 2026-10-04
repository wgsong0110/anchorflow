"""품질기준 FPS 벤치의 **시각 품질** 팔: 렌더해서 FID/FVD/KVD 를 잰다.

규약
  * 한 칸(방법×형상×물성)마다 두 번 렌더한다 -- 그 방법의 **수렴 서브스텝**
    s_conv(참조)과 **기준 통과 서브스텝** s(대상). 두 벌의 프레임으로
    FID/FVD/KVD 를 잰다. 참조는 언제나 **자기 수렴해**라서 MPM 쪽에 유리하게
    기울지 않는다.
  * 시뮬 입자는 그 레포의 **공식 채우기**를 쓴다 (`particle_filling`,
    `wmats/<형상>_fillonly.json` 의 블록 그대로). 시간 측정 팔이 쓰는
    앵커 채우기 npy 는 가우시안을 갈아끼워 렌더가 불가능하기 때문이다.
    따라서 s 는 위치 기준으로 **앵커 집합에서** 찾은 값을 그대로 가져다 쓴다.
  * 물성·E·nu·density·경계는 시간 측정 팔과 같다.

  python exe/bench_vq.py --method pg --shape mic --material elastic \
      --run run3 --frames 40
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess

W = "/home/dkta/work"

ap = argparse.ArgumentParser()
ap.add_argument("--method", choices=["pg", "ipg"], required=True)
ap.add_argument("--shape", required=True)
ap.add_argument("--material", required=True)
ap.add_argument("--run", default="run3")
ap.add_argument("--frames", type=int, default=60)
ap.add_argument("--floor", type=float, default=0.1)
ap.add_argument("--s", type=int, default=0, help="탐색 결과 대신 쓸 서브스텝")
ap.add_argument("--s_ref", type=int, default=0, help="참조(수렴) 서브스텝")
ap.add_argument("--only_test", action="store_true",
                help="대상만 렌더하고 지표는 건너뛴다 (영상만 볼 때)")
ap.add_argument("--win", type=int, default=16)
ap.add_argument("--E", type=float, default=2e6)
ap.add_argument("--nu", type=float, default=0.3)
ap.add_argument("--n_grid", type=int, default=100)
a = ap.parse_args()

MAT = {"elastic": dict(material="jelly"),
       "elastoplastic": dict(material="plasticine", yield_stress=1e4),
       "viscoplastic": dict(material="foam", yield_stress=5e3,
                            plastic_viscosity=10.0),
       # 파괴는 GF watermelon 설정 그대로 (E 2e6 로는 항복면에 닿지 않는다)
       "fracture": dict(material="watermelon", friction_angle=45.0, beta=1.0,
                        xi=3.0, hardening=1.0, alpha_0=-0.04,
                        E=2e3, nu=0.38, density=1.0, g=[0.0, 0.0, -15.0])}
MODEL = {"wolf": "wolf_whitebg-trained", "mic": "mic_whitebg-trained",
         "lego": "lego_whitebg-trained", "bread": "bread-trained"}
REPO = {"pg": f"{W}/PhysGaussian", "ipg": f"{W}/i-physgaussian"}[a.method]

O = f"{W}/bench/{a.run}"
sj = f"{O}/{a.method}_{a.shape}_{a.material}.json"
d = json.load(open(sj)) if os.path.exists(sj) else {}
s_test = a.s or d.get("s")
s_ref = a.s_ref or d.get("s_conv")
if not s_test or not s_ref:
    raise SystemExit(f"[건너뜀] 서브스텝을 모른다 ({sj}). --s/--s_ref 로 주거나 "
                     f"탐색을 먼저 돌릴 것")
s_test, s_ref = int(s_test), int(s_ref)

fill = json.load(open(f"{W}/wmats/{a.shape}_fillonly.json"))
cam = {k: fill[k] for k in ("default_camera_index", "init_azimuthm",
                            "init_elevation", "init_radius", "move_camera",
                            "delta_a", "delta_e", "delta_r") if k in fill}
cam["move_camera"] = False


def build(s):
    cfg = dict(opacity_threshold=0.0, rotation_degree=[0.0], rotation_axis=[0],
               substep_dt=(1.0 / 60.0) / s, frame_dt=1.0 / 60.0,
               frame_num=a.frames, n_grid=a.n_grid, grid_lim=2.0,
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
               mpm_space_viewpoint_center=fill.get(
                   "mpm_space_viewpoint_center", [1, 1, 1]),
               show_hint=False, scale=fill.get("scale", 1.0),
               particle_filling=fill["particle_filling"])
    cfg.update(MAT[a.material]); cfg.update(cam)
    return cfg


def render(s, tag):
    od = f"{O}/vq/{a.method}_{a.shape}_{a.material}_{tag}_{s}"
    shutil.rmtree(od, ignore_errors=True)
    os.makedirs(od, exist_ok=True)
    cp = f"{O}/vq/cfg_{a.method}_{a.shape}_{a.material}_{tag}_{s}.json"
    json.dump(build(s), open(cp, "w"), indent=1)
    wc = f"{W}/wpcache/{a.method}_{a.shape}_{a.material}"
    os.makedirs(wc, exist_ok=True)
    env = dict(os.environ, WARP_CACHE_PATH=wc, PYTHONUTF8="1",
               PYTHONIOENCODING="utf-8")
    env.pop("AF_PARTICLES_NPY", None)          # 공식 채우기를 쓴다
    cmd = ["python", "-u", "gs_simulation.py", "--model_path",
           f"{W}/pgmodel/{MODEL[a.shape]}", "--config", cp,
           "--output_path", od, "--render_img", "--white_bg", "--output_h5"]
    r = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True)
    n = len(glob.glob(f"{od}/*.png"))
    print(f"[렌더] {tag} s={s} -> png {n} 장  {od}", flush=True)
    if n < a.frames:
        print(r.stdout[-800:], flush=True)
        print(r.stderr[-800:], flush=True)
    return od, n


print(f"[칸] {a.method} {a.shape} {a.material}  대상 s={s_test} / "
      f"참조 s_conv={s_ref}  프레임 {a.frames}", flush=True)
os.makedirs(f"{O}/vq", exist_ok=True)
if a.only_test:
    dt, nt = render(s_test, "test")
    mp4 = f"{O}/vq/{a.method}_{a.shape}_{a.material}_test.mp4"
    subprocess.run(["ffmpeg", "-y", "-framerate", "30", "-pattern_type",
                    "glob", "-i", f"{dt}/*.png", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", mp4], capture_output=True, text=True)
    print(f"[영상] {mp4}\nVQ_CELL_DONE", flush=True)
    raise SystemExit(0)
dr, nr = render(s_ref, "ref")
dt, nt = render(s_test, "test")
if min(nr, nt) < a.win:
    raise SystemExit(f"[실패] 프레임이 부족하다 (ref {nr}, test {nt})")

mj = f"{O}/vq/{a.method}_{a.shape}_{a.material}_vq.json"
subprocess.run(["python", "-u", f"{W}/anchorflow/exe/vq_metrics.py",
                "--ref", dr, "--test", dt, "--out", mj, "--win", str(a.win),
                "--label", f"{a.method} {a.shape} {a.material} "
                           f"s={s_test} vs s_conv={s_ref}"],
               cwd=f"{W}/anchorflow",
               env=dict(os.environ, PYTHONPATH=f"{W}/anchorflow/lib",
                        PYTHONUTF8="1"))
for src, tag in ((dt, "test"), (dr, "ref")):
    mp4 = f"{O}/vq/{a.method}_{a.shape}_{a.material}_{tag}.mp4"
    subprocess.run(["ffmpeg", "-y", "-framerate", "30", "-pattern_type",
                    "glob", "-i", f"{src}/*.png", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", mp4],
                   capture_output=True, text=True)
    print(f"[영상] {mp4}", flush=True)
print("VQ_CELL_DONE", flush=True)

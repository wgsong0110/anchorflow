"""i-PG 로 소성·점성·파괴 **시연 영상**을 뽑는다 (3DGS 공식 래스터라이저).

세 씬은 사용자가 정한 그대로다.

  소성  한 물체를 반으로 나눠 양쪽 절반에 **반대 방향 힘**을 준다 (인장 -> 목이
        가늘어지고 끊어진다). 중력·바닥 없이 순수 인장으로 본다.
  점성  한 물체 **위에 같은 물체를 떨어뜨려** 붙기를 기다린 뒤, 두 덩이에
        반대 방향 힘을 준다 (점성 실이 늘어난다).
  파괴  위에서 바닥으로 던진다 (초기 하강속도 -6, 격자 200).

구동은 i-PG 에 이미 들어 있는 **하드 Dirichlet 손잡이**(`AF_H_SCEN`)로 한다 --
반경 R 안의 격자 속도를 명령값으로 박으므로 암시적 솔버의 Newton 반복이
구속을 풀어 버리지 않는다. 손잡이 입자 번호는 **공식 채우기 캐시**에서 고르므로
(같은 캐시를 읽어 돌리니 번호가 일치한다) 렌더가 가능한 실행 그대로다.

점성 씬은 물체가 둘이라 가우시안도 둘이어야 한다. 그래서 3DGS ply 를 수직으로
옮겨 복제한 모델을 만들고(`--dup`), 거기에 공식 채우기를 다시 돌린다.

  python exe/demo_vid_ipg.py --scene plastic --shape lego
  python exe/demo_vid_ipg.py --scene viscous --shape lego
  python exe/demo_vid_ipg.py --scene fracture --shape lego
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess

import numpy as np

W = "/home/dkta/work"
MODEL = {"wolf": "wolf_whitebg-trained", "mic": "mic_whitebg-trained",
         "lego": "lego_whitebg-trained", "bread": "bread-trained"}
# 물성은 벤치와 **같은 값**을 쓴다 (exe/bench_vq.py 의 MAT)
MAT = {"plastic": dict(material="plasticine", yield_stress=1e4),
       "viscous": dict(material="foam", yield_stress=5e3,
                       plastic_viscosity=10.0),
       "fracture": dict(material="watermelon", friction_angle=45.0, beta=1.0,
                        xi=3.0, hardening=1.0, alpha_0=-0.04,
                        E=2e3, nu=0.38, density=1.0, g=[0.0, 0.0, -15.0])}

ap = argparse.ArgumentParser()
ap.add_argument("--scene", choices=list(MAT), required=True)
ap.add_argument("--shape", default="lego")
ap.add_argument("--method", default="ipg", choices=["ipg", "pg"])
ap.add_argument("--out", default=f"{W}/demo")
ap.add_argument("--frames", type=int, default=0, help="0 이면 씬 기본값")
ap.add_argument("--s", type=int, default=0, help="0 이면 씬 기본값")
ap.add_argument("--pull", type=float, default=0.25, help="손잡이 속도 (단위/초)")
ap.add_argument("--hold", type=int, default=30,
                help="점성: 붙기를 기다리는 프레임 수")
ap.add_argument("--radius", type=float, default=0.0,
                help="손잡이 반경 (0 이면 물체 길이의 0.3 배)")
ap.add_argument("--gap", type=float, default=0.15,
                help="점성: 두 물체 사이 간격 (물체 높이 대비)")
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--only_fill", action="store_true", help="채우기만 하고 끝")
ap.add_argument("--tag", default="", help="출력 이름에 붙일 꼬리말")
ap.add_argument("--azim", type=float, default=-999,
                help="카메라 방위각 (기본: 당기는 씬은 채우기 설정 +90 도 -- "
                     "그래야 당기는 축이 화면 가로로 보인다)")
# i-PG 의 본체는 **암시적 적분기**다. 명시로 돌리면 전진 오일러라 PG 와 같아져
# i-PG 라고 부를 수 없다. 그래서 기본이 암시이고, 그쪽 설명서의 레시피대로
# **큰 스텝**을 쓴다: dt_multiplier k 로 스텝을 k 배 키우고(프레임당 서브스텝은
# 1/k 로 줄고) impulse_scale 1/k 로 임펄스를 맞춘다. 논문은 20 배를 든다.
ap.add_argument("--explicit", action="store_true",
                help="전진 오일러로 (그러면 PG 와 같다 -- 비교용으로만)")
ap.add_argument("--dt_mult", type=float, default=20.0,
                help="암시 스텝을 명시 대비 몇 배로 키우는가")
ap.add_argument("--solver", default="newton_gmres",
                choices=["newton_gmres", "picard", "picard_vanilla"])
a = ap.parse_args()

REPO = {"pg": f"{W}/PhysGaussian", "ipg": f"{W}/i-physgaussian"}[a.method]
FRAMES = a.frames or {"plastic": 90, "viscous": 120, "fracture": 60}[a.scene]
# 벤치에서 그 물성이 통과한 서브스텝을 그대로 쓴다 (lego: 소성 1393, 점성 1854,
# 파괴는 기준 미달이라 수렴 실행값 6400)
SUB = a.s or {"plastic": 1393, "viscous": 1854, "fracture": 6400}[a.scene]
OD = (f"{a.out}/{a.method}_{a.shape}_{a.scene}"
      + (f"_{a.tag}" if a.tag else ""))
os.makedirs(a.out, exist_ok=True)


# --------------------------------------------------------- 1) 모델 (점성은 복제)
def dup_model(src, dst, gap):
    """3DGS ply 를 수직(+z)으로 옮겨 **두 벌**로 만든다.

    모델 좌표의 z 가 시뮬의 수직축이다 (config 의 rotation_degree 가 비어 있어
    회전이 항등이고 mpm_space_vertical_upward_axis 가 [0,0,1]).
    """
    from plyfile import PlyData, PlyElement
    if os.path.exists(f"{dst}/point_cloud/iteration_30000/point_cloud.ply"):
        print(f"[복제] 이미 있다: {dst}", flush=True)
        return dst
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("point_cloud"))
    sp = f"{src}/point_cloud/iteration_30000/point_cloud.ply"
    v = PlyData.read(sp)["vertex"].data
    dz = float(v["z"].max() - v["z"].min())
    off = dz * (1.0 + gap)
    up = v.copy()
    up["z"] = up["z"] + off
    both = np.concatenate([v, up])
    os.makedirs(f"{dst}/point_cloud/iteration_30000", exist_ok=True)
    PlyData([PlyElement.describe(both, "vertex")]).write(
        f"{dst}/point_cloud/iteration_30000/point_cloud.ply")
    print(f"[복제] 가우시안 {len(v)} -> {len(both)} 개, z 를 {off:.4f} 옮겨 "
          f"얹었다 -> {dst}", flush=True)
    return dst


MP = f"{W}/pgmodel/{MODEL[a.shape]}"
FILL = f"{W}/pgfill_{a.shape}.npy"
if a.scene == "viscous":
    MP = dup_model(MP, f"{W}/pgmodel/{MODEL[a.shape]}-x2", a.gap)
    FILL = f"{W}/pgfill_{a.shape}_x2.npy"


# ------------------------------------------------------------------ 2) 설정
fill = json.load(open(f"{W}/wmats/{a.shape}_fillonly.json"))
cam = {k: fill[k] for k in ("default_camera_index", "init_azimuthm",
                            "init_elevation", "init_radius", "move_camera",
                            "delta_a", "delta_e", "delta_r") if k in fill}
cam["move_camera"] = False
if a.azim != -999:
    cam["init_azimuthm"] = a.azim
elif a.scene in ("plastic", "viscous"):
    cam["init_azimuthm"] = float(cam.get("init_azimuthm", 0.0)) + 90.0
FLOOR = [q["point"][2] for q in fill["boundary_conditions"]
         if q["type"] == "surface_collider"]
FLOOR = FLOOR[0] if FLOOR else 0.48


def build(frames, sub, with_floor, gravity):
    cfg = dict(opacity_threshold=fill.get("opacity_threshold", 0.02),
               rotation_degree=[0.0], rotation_axis=[0],
               substep_dt=(1.0 / 60.0) / sub, frame_dt=1.0 / 60.0,
               frame_num=frames,
               n_grid=(200 if a.scene == "fracture" else 100), grid_lim=2.0,
               E=2e6, nu=0.3, density=1000.0,
               g=[0.0, 0.0, -9.8 if gravity else 0.0],
               boundary_conditions=(
                   [{"type": "bounding_box"}]
                   + ([{"type": "surface_collider",
                        "point": [1.0, 1.0, FLOOR],
                        "normal": [0.0, 0.0, 1.0], "surface": "sticky",
                        "friction": 0.0, "start_time": 0,
                        "end_time": 1000.0}] if with_floor else [])),
               mpm_space_vertical_upward_axis=[0, 0, 1],
               mpm_space_viewpoint_center=fill.get(
                   "mpm_space_viewpoint_center", [1, 1, 1]),
               show_hint=False, scale=fill.get("scale", 1.0),
               particle_filling=fill["particle_filling"])
    cfg.update(MAT[a.scene]); cfg.update(cam)
    if a.scene == "fracture":
        cfg["init_velocity"] = [0.0, 0.0, -6.0]
    return cfg


GRAV = a.scene != "plastic"          # 소성은 순수 인장 (중력·바닥 없음)
FLOOR_ON = a.scene != "plastic"


def run(cfg, od, scen=None, radius=0.0, render=True):
    shutil.rmtree(od, ignore_errors=True)
    os.makedirs(od, exist_ok=True)
    cp = f"{od}.json"
    json.dump(cfg, open(cp, "w"), indent=1)
    wc = f"{W}/wpcache/demo_{a.scene}_{os.getpid()}"
    os.makedirs(wc, exist_ok=True)
    env = dict(os.environ, WARP_CACHE_PATH=wc, PYTHONUTF8="1",
               PYTHONIOENCODING="utf-8", AF_PGFILL_NPY=FILL)
    env.pop("AF_PARTICLES_NPY", None)            # 공식 채우기를 쓴다
    if scen:
        env["AF_H_SCEN"] = scen
        env["AF_H_R"] = str(radius)
    else:
        env.pop("AF_H_SCEN", None)
    cmd = ["python", "-u", "gs_simulation.py", "--model_path", MP,
           "--config", cp, "--output_path", od, "--output_h5"]
    if render:
        cmd += ["--render_img", "--white_bg"]
    if a.method == "ipg" and not a.explicit:
        cmd += ["--implicit", "--solver", a.solver,
                "--dt_multiplier", str(a.dt_mult),
                "--impulse_scale", str(1.0 / a.dt_mult)]
    r = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True)
    n = len(glob.glob(f"{od}/*.png"))
    print(f"[실행] {os.path.basename(od)} png {n} 장", flush=True)
    if n < (cfg["frame_num"] if render else 0):
        print(r.stdout[-2500:], flush=True)
        print(r.stderr[-1500:], flush=True)
    return n


# ------------------------------------------- 3) 채우기 캐시 (손잡이 번호의 기준)
if not os.path.exists(FILL):
    print("[채우기] 캐시가 없다 -- 한 프레임만 돌려 공식 채우기를 떨군다",
          flush=True)
    run(build(1, 200, FLOOR_ON, GRAV), f"{OD}_fill", render=False)
    if not os.path.exists(FILL):
        raise SystemExit("[중단] 채우기 캐시가 안 생겼다")
X0 = np.load(FILL)
ext = X0.max(0) - X0.min(0)
print(f"[채우기] 입자 {X0.shape[0]} 개  범위 {ext.round(3)}  "
      f"중심 {X0.mean(0).round(3)}", flush=True)
if a.only_fill:
    raise SystemExit(0)


# -------------------------------------------------------------- 4) 손잡이 명령
def principal(X):
    """**수평** 주축 (수직 성분은 뺀다).

    그냥 주축을 쓰면 쌓아 둔 두 덩이(점성 씬)에서 세로축이 잡혀 손잡이가 둘을
    위아래로 떼어 놓는다 -- 붙기를 기다리는 씬이 성립하지 않는다.
    """
    C = X - X.mean(0)
    C[:, 2] = 0.0
    w, V = np.linalg.eigh(C.T @ C / len(X))
    ax = V[:, int(np.argmax(w))]
    ax[2] = 0.0
    return ax / (np.linalg.norm(ax) + 1e-12)


SCEN = None
R = a.radius
if a.scene in ("plastic", "viscous"):
    ax = principal(X0)
    ax = ax * np.sign(ax[int(np.argmax(np.abs(ax)))])      # 부호 고정
    t = X0 @ ax
    if a.scene == "plastic":
        # 한 물체를 반으로: 주축 양 끝 입자를 손잡이로 잡고 반대로 당긴다
        hid = np.array([int(np.argmin(t)), int(np.argmax(t))])
        v = np.stack([-ax, ax]) * a.pull                   # [2,3]
        vel = np.repeat(v[None], FRAMES, 0)                # [T,2,3]
        R = R or 0.3 * float(t.max() - t.min())
    else:
        # 두 덩이: 아래쪽에서 하나, 위쪽에서 하나 잡고 **붙은 뒤** 반대로 당긴다
        zmid = 0.5 * (X0[:, 2].min() + X0[:, 2].max())
        lo = np.where(X0[:, 2] < zmid)[0]
        hi = np.where(X0[:, 2] >= zmid)[0]
        hid = np.array([int(lo[np.argmin(t[lo])]), int(hi[np.argmax(t[hi])])])
        v = np.stack([-ax, ax]) * a.pull
        vel = np.zeros((FRAMES, 2, 3), np.float32)
        vel[a.hold:] = v                                   # 기다린 뒤 당긴다
        R = R or 0.22 * float(t.max() - t.min())
    SCEN = f"{OD}_scen.npz"
    np.savez(SCEN, hid=hid.astype(np.int64), vel=vel.astype(np.float32))
    print(f"[손잡이] 입자 {hid.tolist()}  위치 {X0[hid].round(3).tolist()}  "
          f"반경 {R:.3f}  주축 {ax.round(3)}  속도 {a.pull}"
          + (f"  (앞 {a.hold} 프레임은 대기)" if a.scene == "viscous" else ""),
          flush=True)


# ------------------------------------------------------------------- 5) 실행
n = run(build(FRAMES, SUB, FLOOR_ON, GRAV), OD, scen=SCEN, radius=R)
if n:
    mp4 = OD + ".mp4"
    subprocess.run(["python", "-u", f"{W}/anchorflow/exe/pngs2mp4.py",
                    "--dir", OD, "--out", mp4, "--fps", str(a.fps)],
                   cwd=f"{W}/anchorflow")
    print(f"[영상] {mp4}  ({n} 프레임, {a.fps} fps)", flush=True)
print("DEMO_DONE", flush=True)

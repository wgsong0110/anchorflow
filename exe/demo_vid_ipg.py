"""i-PG 로 소성·점성·파괴 **시연 영상**을 뽑는다 (3DGS 공식 래스터라이저).

세 씬은 사용자가 정한 그대로다.

  소성  한 물체를 반으로 나눠 양쪽 절반에 **반대 방향 힘**을 준다 (인장 -> 목이
        가늘어지고 끊어진다). 중력·바닥 없이 순수 인장으로 본다.
  점성  한 물체 **위에 같은 물체를 떨어뜨려** 붙기를 기다린 뒤, 두 덩이에
        반대 방향 힘을 준다 (점성 실이 늘어난다).
  파괴  위에서 바닥으로 던진다 (초기 하강속도 -6, 격자 200).

구동은 **힘**이다 (속도를 박지 않는다). 질량에 비례하는 힘, 즉 영역 전체에 같은
가속도를 준다 (`exe/patch_ipg_bodyacc.py` 가 붙이는 `body_acceleration` 경계조건 --
상자 안 입자를 **처음 위치로 한 번 골라** 정해진 시간 동안 `v += a·dt`). 소성은 주축 기준 왼쪽 절반 전체에 왼쪽
힘, 오른쪽 절반 전체에 오른쪽 힘을 준다. 점성은 아래 덩이 전체와 위 덩이 전체에
반대 방향 힘을 준다. 가속도는 매 스텝 실제 dt 로 적분되므로 암시 적분의 큰
스텝에서도 시간에 대해 그대로다. 좌우는 카메라를 돌리지 않고 **원래 카메라의 화면
가로축**(--force_cam)으로 정한다.

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
       # 점성은 **흘러내려야** 한다. 벤치 값(항복 5e3)은 제 무게(ρgh≈5e3)를
       # 겨우 버텨 강체처럼 보였다. 항복을 무게 응력의 1/3 로 낮추고 점성을
       # 키워 꿀처럼 천천히 처지게 한다.
       "viscous": dict(material="foam", E=2e5, yield_stress=1500.0,
                       plastic_viscosity=100.0),
       # 입상: PhysGaussian 공식 wolf 설정(config/wolf_config.json)의 모래 그대로
       "granular": dict(material="sand", E=5e7, nu=0.3, density=2000.0,
                        friction_angle=30.0),
       "fracture": dict(material="watermelon", friction_angle=45.0, beta=1.0,
                        xi=3.0, hardening=1.0, alpha_0=-0.04,
                        E=2e3, nu=0.38, density=1.0, g=[0.0, 0.0, -15.0])}

ap = argparse.ArgumentParser()
ap.add_argument("--scene", choices=list(MAT), required=True)
ap.add_argument("--shape", default="lego")
ap.add_argument("--method", default="ipg", choices=["ipg", "pg", "gf"])
ap.add_argument("--out", default=f"{W}/demo")
ap.add_argument("--frames", type=int, default=0, help="0 이면 씬 기본값")
ap.add_argument("--s", type=int, default=0, help="0 이면 씬 기본값")
ap.add_argument("--acc", type=float, default=0.0,
                help="양쪽에 주는 가속도 (0 이면 소성 10, 점성 5)")
ap.add_argument("--force_frames", type=int, default=0,
                help="힘을 주는 프레임 수 (0 이면 소성 20, 점성 40)")
ap.add_argument("--hold", type=int, default=30,
                help="점성: 붙기를 기다리는 프레임 수")
ap.add_argument("--gap", type=float, default=0.15,
                help="점성: 두 물체 사이 간격 (물체 높이 대비)")
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--only_fill", action="store_true", help="채우기만 하고 끝")
ap.add_argument("--tag", default="", help="출력 이름에 붙일 꼬리말")
ap.add_argument("--azim", type=float, default=-999,
                help="카메라 방위각 (기본 45 -- 옆모습, 당기는 축이 화면 가로)")
ap.add_argument("--elev", type=float, default=15.0)
ap.add_argument("--scale", type=float, default=0.0,
                help="물체 크기 (0 이면 1.0). 채우기 캐시는 크기마다 따로 둔다")
ap.add_argument("--n_grid", type=int, default=0, help="0 이면 파괴 200, 나머지 100")
ap.add_argument("--v0", type=float, default=-12.0,
                help="파괴: 초기 하강속도 (실제로 깨지게 벤치의 -6 보다 크게)")
ap.add_argument("--force_cam", default=f"{W}/demo/cam_gt.json",
                help="화면 가로축을 읽을 카메라 (exe/dump_demo_cam.py 출력)")
ap.add_argument("--cam_r", type=float, default=7.0)
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

REPO = {"pg": f"{W}/PhysGaussian", "ipg": f"{W}/i-physgaussian",
        "gf": f"{W}/GaussianFluent"}[a.method]
# GaussianFluent 로 돌릴 때 그쪽 공식 수박 설정(config/watermelon_config.json)의
# 솔버 값: FLIP/PIC 0.7, 격자 300. 재질 상수는 MAT 의 watermelon 과 같다.
GF_FRACTURE = dict(flip_pic_ratio=0.7)
FRAMES = a.frames or {"plastic": 90, "viscous": 120, "granular": 120,
                       "fracture": 60}[a.scene]
# 벤치에서 그 물성이 통과한 서브스텝을 그대로 쓴다 (lego: 소성 1393, 점성 1854,
# 파괴는 기준 미달이라 수렴 실행값 6400)
SUB = a.s or {"plastic": 1393, "viscous": 1854, "granular": 833,
              "fracture": 6400}[a.scene]   # 입상은 wolf 설정의 2e-5 초
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


SCALE = a.scale or 1.0
MP = f"{W}/pgmodel/{MODEL[a.shape]}"
FILL = (f"{W}/pgfill_{a.shape}.npy" if SCALE == 1.0
        else f"{W}/pgfill_{a.shape}_s{SCALE:g}.npy")
TWO = a.scene in ("viscous", "granular")      # 두 덩이 씬
if TWO:
    MP = dup_model(MP, f"{W}/pgmodel/{MODEL[a.shape]}-x2", a.gap)
    FILL = (f"{W}/pgfill_{a.shape}_x2.npy" if SCALE == 1.0
            else f"{W}/pgfill_{a.shape}_x2_s{SCALE:g}.npy")


# ------------------------------------------------------------------ 2) 설정
fill = json.load(open(f"{W}/wmats/{a.shape}_fillonly.json"))
cam = {k: fill[k] for k in ("default_camera_index", "init_azimuthm",
                            "init_elevation", "init_radius", "move_camera",
                            "delta_a", "delta_e", "delta_r") if k in fill}
cam["move_camera"] = False
# 궤도 카메라를 쓴다. 채우기 설정의 default_camera_index=0 이 남아 있으면 러너가
# 학습 카메라 0 번을 그대로 써서 방위각이 먹지 않는다 (실측: 260 을 줘도 정면).
# 방위 45 도는 +x 쪽에서 보는 옆모습이라 lego 의 긴 축(y)이 화면 가로가 된다.
# 카메라는 **원래 채우기 설정 그대로** 둔다 (학습 카메라 0 번 = 피팅 GT 와 같은 시점).
# 좌우는 카메라를 돌려 맞추지 않고, 힘을 주는 영역과 방향을 이 카메라의 화면
# 가로축에 맞춘다 (아래 --force_cam). --azim 을 줄 때만 궤도 카메라로 바꾼다.
if a.azim != -999:
    cam["default_camera_index"] = -1
    cam["init_azimuthm"] = a.azim
    cam["init_elevation"] = a.elev
    cam["init_radius"] = a.cam_r
    cam["delta_a"] = cam["delta_e"] = cam["delta_r"] = 0.0
FLOOR = [q["point"][2] for q in fill["boundary_conditions"]
         if q["type"] == "surface_collider"]
FLOOR = FLOOR[0] if FLOOR else 0.48


def build(frames, sub, with_floor, gravity, extra_bc=()):
    cfg = dict(opacity_threshold=fill.get("opacity_threshold", 0.02),
               rotation_degree=[0.0], rotation_axis=[0],
               substep_dt=(1.0 / 60.0) / sub, frame_dt=1.0 / 60.0,
               frame_num=frames,
               n_grid=(a.n_grid or (200 if a.scene in ("fracture", "granular")
                                    else 100)),
               grid_lim=2.0,
               E=2e6, nu=0.3, density=1000.0,
               g=[0.0, 0.0, -9.8 if gravity else 0.0],
               boundary_conditions=(
                   [{"type": "bounding_box"}]
                   + ([{"type": "surface_collider",
                        "point": [1.0, 1.0, FLOOR],
                        "normal": [0.0, 0.0, 1.0], "surface": "sticky",
                        "friction": 0.0, "start_time": 0,
                        "end_time": 1000.0}] if with_floor else [])
                   + list(extra_bc)),
               mpm_space_vertical_upward_axis=[0, 0, 1],
               mpm_space_viewpoint_center=fill.get(
                   "mpm_space_viewpoint_center", [1, 1, 1]),
               show_hint=False, scale=SCALE,
               particle_filling=fill["particle_filling"])
    cfg.update(MAT[a.scene]); cfg.update(cam)
    if a.method == "gf" and a.scene == "fracture":
        cfg.update(GF_FRACTURE)
        if not a.n_grid:
            cfg["n_grid"] = 300
    if a.scene == "fracture":
        # i-PG 러너는 exe/patch_ipg_initv.py 를 적용해야 이 값을 읽는다
        cfg["init_velocity"] = [0.0, 0.0, a.v0]
    return cfg


GRAV = a.scene != "plastic"          # 소성은 순수 인장 (중력·바닥 없음)
FLOOR_ON = a.scene != "plastic"


def run(cfg, od, render=True):
    shutil.rmtree(od, ignore_errors=True)
    os.makedirs(od, exist_ok=True)
    cp = f"{od}.json"
    json.dump(cfg, open(cp, "w"), indent=1)
    wc = f"{W}/wpcache/demo_{a.scene}_{os.getpid()}"
    os.makedirs(wc, exist_ok=True)
    env = dict(os.environ, WARP_CACHE_PATH=wc, PYTHONUTF8="1",
               PYTHONIOENCODING="utf-8", AF_PGFILL_NPY=FILL)
    env.pop("AF_PARTICLES_NPY", None)            # 공식 채우기를 쓴다
    env.pop("AF_H_SCEN", None)                   # 속도 손잡이는 쓰지 않는다
    cmd = ["python", "-u", "gs_simulation.py", "--model_path", MP,
           "--config", cp, "--output_path", od, "--output_h5"]
    if render:
        cmd += ["--render_img", "--white_bg"]
    if a.method == "ipg" and not a.explicit:
        cmd += ["--implicit", "--solver", a.solver,
                "--dt_multiplier", str(a.dt_mult),
                "--impulse_scale", "1.0"]
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


EXTRA = []
if a.scene in ("plastic", "viscous", "granular"):
    # 화면 오른쪽 방향(월드) = 카메라 회전의 첫 열. 시뮬 공간은 모델 공간을
    # 평행이동·양의 배율로만 옮기므로(회전 항등) 방향이 그대로다.
    _c = json.load(open(a.force_cam))
    r = np.array(_c["R"])[:, 0]
    r[2] = 0.0
    r = r / np.linalg.norm(r)
    k = int(np.argmax(np.abs(r)))               # 상자는 축 정렬 -> 지배 축으로 고른다
    e = r                                       # 오른쪽 절반이 받는 방향 (화면 오른쪽)
    ACC = a.acc or (100.0 if a.scene == "plastic" else 5.0)
    NF = a.force_frames or (20 if a.scene == "plastic" else 40)
    c = X0.mean(0)
    BIG = 4.0                                   # 다른 축은 전부 덮는다
    if a.scene == "plastic":
        cut = float(np.median(X0[:, k]))        # 정확히 반으로
        t0 = 0.0
        sr = float(np.sign(r[k]))                # 화면 오른쪽이 축 k 의 어느 쪽인가
        groups = [("화면 왼쪽 절반", (X0[:, k] - cut) * sr < 0, -1.0),
                  ("화면 오른쪽 절반", (X0[:, k] - cut) * sr >= 0, +1.0)]
        boxes = []
        for sgn in (-1.0, +1.0):                 # sgn: 화면 기준 왼(-)/오른(+)
            pt = c.copy(); pt[k] = cut + sgn * sr * BIG / 2
            sz = np.full(3, BIG); sz[k] = BIG / 2
            boxes.append((pt, sz, sgn))
    else:
        zmid = 0.5 * (X0[:, 2].min() + X0[:, 2].max())
        t0 = a.hold / 60.0
        groups = [("아래 덩이", X0[:, 2] < zmid, -1.0),
                  ("위 덩이", X0[:, 2] >= zmid, +1.0)]
        boxes = []
        for sgn in (-1.0, +1.0):
            pt = c.copy(); pt[2] = zmid + sgn * BIG / 2
            sz = np.full(3, BIG); sz[2] = BIG / 2
            boxes.append((pt, sz, sgn))
    for pt, sz, sgn in boxes:
        # 질량에 비례하는 힘 = 같은 가속도 (exe/patch_ipg_bodyacc.py). 입자마다
        # 같은 힘을 주는 particle_impulse 는 가벼운 표면 입자만 튀게 한다.
        EXTRA.append({"type": "body_acceleration",
                      "acc": (sgn * ACC * e).tolist(),
                      "point": pt.tolist(), "size": sz.tolist(),
                      "start_time": t0, "end_time": t0 + NF / 60.0})
    for nm, mk, sgn in groups:
        print(f"[힘] {nm}: 입자 {int(mk.sum())} 개, 방향 화면 "
              f"{'오른쪽' if sgn > 0 else '왼쪽'} {np.round(sgn * e, 3).tolist()}, "
              f"가속도 {ACC}, "
              f"{t0 * 60:.0f}~{t0 * 60 + NF:.0f} 프레임", flush=True)


# ------------------------------------------------------------------- 5) 실행
n = run(build(FRAMES, SUB, FLOOR_ON, GRAV, EXTRA), OD)
if n:
    mp4 = OD + ".mp4"
    subprocess.run(["python", "-u", f"{W}/anchorflow/exe/pngs2mp4.py",
                    "--dir", OD, "--out", mp4, "--fps", str(a.fps)],
                   cwd=f"{W}/anchorflow")
    print(f"[영상] {mp4}  ({n} 프레임, {a.fps} fps)", flush=True)
print("DEMO_DONE", flush=True)

"""RAF(Scene-Level Heterogeneous Physics, CVPR-F 2026)가 쓰는 **Genesis** 로
파괴·소성유동·접합·인열 네 씬을 돌린다.

먼저 분명히 해 둘 것. Genesis 의 MPM 재질은 Elastic / ElastoPlastic(von Mises) /
Sand(Drucker-Prager) / Snow / Liquid / Muscle 뿐이고 **손상·연화 파라미터가 없다**
(`materials/MPM/elasto_plastic.py` 의 필드: E, nu, rho, yield_lower, yield_higher,
use_von_mises, von_mises_yield_stress). 즉 PhysGaussian 과 같은 자리에 있고,
**파괴와 인열은 원리상 안 나온다** -- 항복하면 응력이 상한에 묶일 뿐 재료가 약해지지
않아, 끊어지는 대신 늘어나거나 퍼진다. 그래서 이 스크립트의 목적은 "되는 것을
보여주기" 가 아니라 **되는 것과 안 되는 것을 같은 조건에서 보여주기** 다.

  소성유동  Sand 기둥 붕괴            -- 된다 (Drucker-Prager)
  접합      두 덩이를 밀어붙이기      -- 된다 (MPM 은 격자를 공유해 저절로 붙는다)
  인열      붙은 덩이를 양쪽으로 당김 -- 기본 재질로는 목만 가늘어진다
  파괴      공을 바닥에 내리꽂기      -- 기본 재질로는 조각이 아니라 퍼진다

**막는 것은 RAF 의 얼개가 아니라 재질 목록이다.** RAF 는 3DGS 를 입자로 추상화하고
입자 시뮬을 돌린 뒤 되돌려 스키닝하므로, 파괴되는 입자 재질을 끼우면 파괴 씬도
나온다. 그래서 `exe/patch_genesis_cdmpm.py` 로 CD-MPM(비연관 Cam-Clay + 항복면
경화)을 Genesis 재질로 넣고, **같은 씬을 재질만 바꿔** 나란히 돌린다.

  python exe/run_raf_scenes.py --out DIR [--scene all]
"""
import argparse
import os
import time

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--scene", default="all",
                choices=("all", "flow", "bond", "tear", "break"))
ap.add_argument("--frames", type=int, default=150)
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--res", type=int, default=640)
ap.add_argument("--grid_density", type=int, default=64)
a = ap.parse_args()

import genesis as gs

gs.init(backend=gs.gpu, logging_level="warning")
os.makedirs(a.out, exist_ok=True)


def new_scene(dt=2e-4, substeps=10,
              # MPM 영역은 안쪽으로 세 칸 물러서므로, 아래 끝을 바닥(z=0) 보다
              # 그만큼 낮춰 둬야 물체가 바닥에 닿을 수 있다.
              bound=((-0.5, -0.5, -0.06), (0.5, 0.5, 0.94))):
    return gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt, substeps=substeps),
        mpm_options=gs.options.MPMOptions(
            lower_bound=bound[0], upper_bound=bound[1],
            grid_density=a.grid_density),
        vis_options=gs.options.VisOptions(show_world_frame=False),
        renderer=gs.renderers.Rasterizer(),
        show_viewer=False,
    )


def record(scene, cam, name, steps, on_step=None):
    t0 = time.time()
    cam.start_recording()
    for i in range(steps):
        if on_step is not None:
            on_step(i)
        scene.step()
        cam.render()
    p = os.path.join(a.out, f"{name}.mp4")
    cam.stop_recording(save_to_filename=p, fps=a.fps)
    print(f"[저장] {p}  {steps} 프레임  {time.time()-t0:.0f}s", flush=True)
    return p


def add_cam(scene, pos=(0.9, -0.9, 0.55), lookat=(0.0, 0.0, 0.18)):
    return scene.add_camera(res=(a.res, int(a.res * 0.78)), pos=pos,
                            lookat=lookat, fov=40, GUI=False)


def drive_vel(ent, v):
    """엔티티의 모든 입자 속도를 v 로 고정한다 (set_velocity 는 입자별 배열을 받는다)."""
    ent.set_velocity(np.tile(np.asarray(v, np.float32), (ent.n_particles, 1)))


# GF 수박과 같은 음속(61 m/s)을 갖도록 SI 단위로 옮긴 CD-MPM 물성
def cd_mat(E=2e6, nu=0.38, rho=1000.0, beta=1.0, xi=3.0):
    return gs.materials.MPM.CDMPM(E=E, nu=nu, rho=rho, friction_angle=45.0,
                                  beta=beta, xi=xi, hardening=1.0, alpha_0=-0.04)


def ep_mat(E=2e6, nu=0.38, rho=1000.0, ys=1e4):
    return gs.materials.MPM.ElastoPlastic(E=E, nu=nu, rho=rho,
                                          use_von_mises=True,
                                          von_mises_yield_stress=ys)


# --------------------------------------------------------------- 소성 유동
def scene_flow():
    sc = new_scene()
    sc.add_entity(gs.morphs.Plane())
    sc.add_entity(
        material=gs.materials.MPM.Sand(E=1e6, nu=0.2, rho=1000.0,
                                       friction_angle=30.0),
        morph=gs.morphs.Box(pos=(0.0, 0.0, 0.22), size=(0.12, 0.12, 0.42)),
        surface=gs.surfaces.Default(color=(0.85, 0.72, 0.42)),
    )
    cam = add_cam(sc)
    sc.build()
    return record(sc, cam, "raf_flow", a.frames)


# ------------------------------------------------------------------- 접합
def _bond(tag, mat):
    sc = new_scene()
    sc.add_entity(gs.morphs.Plane())
    left = sc.add_entity(material=mat,
                         morph=gs.morphs.Box(pos=(-0.11, 0.0, 0.12),
                                             size=(0.14, 0.14, 0.14)),
                         surface=gs.surfaces.Default(color=(0.9, 0.35, 0.3)))
    right = sc.add_entity(material=mat,
                          morph=gs.morphs.Box(pos=(0.11, 0.0, 0.12),
                                              size=(0.14, 0.14, 0.14)),
                          surface=gs.surfaces.Default(color=(0.3, 0.5, 0.9)))
    cam = add_cam(sc)
    sc.build()

    def drive(i):
        if i < 40:                       # 서로를 향해 밀어붙인다
            drive_vel(left, (0.35, 0.0, 0.0))
            drive_vel(right, (-0.35, 0.0, 0.0))
    return record(sc, cam, f"raf_bond_{tag}", a.frames, drive)


def scene_bond():
    _bond("cd", cd_mat())


# ------------------------------------------------------------------- 인열
def _tear(tag, mat):
    sc = new_scene()
    sc.add_entity(gs.morphs.Plane())
    left = sc.add_entity(material=mat,
                         morph=gs.morphs.Box(pos=(-0.08, 0.0, 0.30),
                                             size=(0.16, 0.12, 0.12)),
                         surface=gs.surfaces.Default(color=(0.9, 0.35, 0.3)))
    right = sc.add_entity(material=mat,
                          morph=gs.morphs.Box(pos=(0.08, 0.0, 0.30),
                                              size=(0.16, 0.12, 0.12)),
                          surface=gs.surfaces.Default(color=(0.3, 0.5, 0.9)))
    cam = add_cam(sc, pos=(0.0, -1.0, 0.45), lookat=(0.0, 0.0, 0.30))
    sc.build()

    def drive(i):
        if i < 30:                       # 먼저 붙인다
            drive_vel(left, (0.30, 0.0, 0.0))
            drive_vel(right, (-0.30, 0.0, 0.0))
        else:                            # 그리고 끊어질 때까지 당긴다
            drive_vel(left, (-0.25, 0.0, 0.0))
            drive_vel(right, (0.25, 0.0, 0.0))
    return record(sc, cam, f"raf_tear_{tag}", a.frames, drive)


def scene_tear():
    _tear("ep", ep_mat())      # Genesis 기본 재질
    _tear("cd", cd_mat())      # CD-MPM


# ------------------------------------------------------------------- 파괴
def _break(tag, mat):
    sc = new_scene(dt=1e-4, substeps=10)
    sc.add_entity(gs.morphs.Plane())
    ball = sc.add_entity(
        material=mat,
        morph=gs.morphs.Sphere(pos=(0.0, 0.0, 0.45), radius=0.10),
        surface=gs.surfaces.Default(color=(0.35, 0.75, 0.4)),
    )
    cam = add_cam(sc, pos=(0.8, -0.8, 0.40), lookat=(0.0, 0.0, 0.12))
    sc.build()
    drive_vel(ball, (0.0, 0.0, -6.0))    # GF 수박과 같은 초속
    return record(sc, cam, f"raf_break_{tag}", a.frames)


def scene_break():
    _break("ep", ep_mat())
    _break("cd", cd_mat())


JOBS = {"flow": scene_flow, "bond": scene_bond,
        "tear": scene_tear, "break": scene_break}
todo = list(JOBS) if a.scene == "all" else [a.scene]
for k in todo:
    print(f"=== {k} ===", flush=True)
    try:
        JOBS[k]()
    except Exception as e:
        import traceback
        print(f"[{k}] 실패: {e}", flush=True)
        traceback.print_exc()
print("RAF_SCENES_DONE", flush=True)

"""RAF 방식으로 **실제 3DGS 자산**을 입자로 바꿔 Genesis MPM 에 먹인다.

RAF 는 3DGS 를 입자셋으로 추상화하고(`gaussian.py: load_3dgs_ply`), 입자 시뮬을
돌린 뒤 결과로 3DGS 를 스키닝한다. 여기서는 그 첫 두 단계를 그대로 하고
(`gs.morphs.PointCloud` 로 입자를 직접 넣는다), 그리는 것은 입자다 -- 가우시안
렌더는 RAF 가 Omniverse 로 하는데 그건 따로 깔아야 한다.

입자 위치는 h5 로 떨군다. Genesis 의 래스터라이저는 포인트클라우드 엔티티에
`_vmesh` 가 없어 그리다 죽고, 무엇보다 **GF 비교에 쓰던 렌더러와 인열 잣대를
그대로 쓰려면** 같은 형식이어야 한다 (`exe/render_gf_traj.py`,
`exe/measure_gf_tearing.py`).

  파괴    수박(GaussianFluent 자산) + CD-MPM
  소성유동 wolf(PhysGaussian 자산) + Sand
  인열    빵(PhysGaussian 자산) 을 좌우로 갈라 양쪽에서 당긴다

**시간 설정 주의.** Genesis 는 `scene.step()` 한 번이 `SimOptions.dt` 만큼 간다
(서브스텝 크기가 아니다). 앞서 dt 2e-4 로 220 스텝을 돌려 0.044 초밖에 안 갔고
영상이 거의 안 움직였다. 그래서 여기서는 **초 단위로** 길이를 준다.

  python exe/run_raf_gs_scenes.py --scene break --ply <model.ply> --out DIR
"""
import argparse
import os
import time

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--scene", required=True, choices=("break", "flow", "tear"))
ap.add_argument("--ply", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--tag", default=None)
ap.add_argument("--material", default=None, choices=(None, "cdmpm", "sand", "ep"))
ap.add_argument("--n_pts", type=int, default=250000, help="입자 수 (솎는다)")
ap.add_argument("--size", type=float, default=0.30, help="물체 최대 변 (m)")
ap.add_argument("--sim_time", type=float, default=1.2, help="돌릴 시간 (초)")
ap.add_argument("--dt", type=float, default=5e-4)
ap.add_argument("--substeps", type=int, default=10)
ap.add_argument("--render_every", type=int, default=8)
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--res", type=int, default=640)
ap.add_argument("--grid_density", type=int, default=96)
ap.add_argument("--v0", type=float, default=-3.0, help="파괴 씬의 초기 낙하 속도")
ap.add_argument("--pull", type=float, default=0.25, help="인열 씬에서 당기는 속도")
ap.add_argument("--opacity_min", type=float, default=0.02)
a = ap.parse_args()

import h5py
import genesis as gs
from plyfile import PlyData


def load_gs_points(path, n_max, opacity_min):
    """RAF 의 load_3dgs_ply 와 같은 자리 -- 위치와 불투명도만 쓴다."""
    p = PlyData.read(path)["vertex"]
    xyz = np.stack([np.asarray(p[k]) for k in ("x", "y", "z")], 1).astype(np.float64)
    if "opacity" in p.data.dtype.names:
        op = 1.0 / (1.0 + np.exp(-np.asarray(p["opacity"])))
        xyz = xyz[op > opacity_min]
    xyz = xyz[np.isfinite(xyz).all(1)]
    # 꼬리 1% 를 떼고 중심·크기를 잡는다 (RAF 의 robust_center 와 같은 뜻)
    lo, hi = np.percentile(xyz, 0.5, axis=0), np.percentile(xyz, 99.5, axis=0)
    keep = ((xyz > lo) & (xyz < hi)).all(1)
    xyz = xyz[keep]
    if n_max and len(xyz) > n_max:
        idx = np.random.default_rng(0).choice(len(xyz), n_max, replace=False)
        xyz = xyz[np.sort(idx)]
    return xyz


def place(xyz, size, z_bottom):
    c = 0.5 * (xyz.min(0) + xyz.max(0))
    s = size / max(xyz.max(0) - xyz.min(0))
    q = (xyz - c) * s
    q[:, 2] += z_bottom - q[:, 2].min()
    return q.astype(np.float32)


gs.init(backend=gs.gpu, logging_level="warning")
os.makedirs(a.out, exist_ok=True)
tag = a.tag or a.scene

MATS = {
    # GF 수박 물성을 SI 로 옮긴 것 (음속 61 m/s 가 같아지게 E 를 1000 배)
    "cdmpm": lambda: gs.materials.MPM.CDMPM(E=2e6, nu=0.38, rho=1000.0,
                                            friction_angle=45.0, beta=1.0,
                                            xi=3.0, hardening=1.0, alpha_0=-0.04),
    "sand": lambda: gs.materials.MPM.Sand(E=1e6, nu=0.2, rho=1000.0,
                                          friction_angle=30.0),
    "ep": lambda: gs.materials.MPM.ElastoPlastic(E=2e6, nu=0.38, rho=1000.0,
                                                 use_von_mises=True,
                                                 von_mises_yield_stress=1e4),
}
mat_name = a.material or {"break": "cdmpm", "flow": "sand", "tear": "cdmpm"}[a.scene]

pts = load_gs_points(a.ply, a.n_pts, a.opacity_min)
print(f"[자산] {os.path.basename(a.ply)} -> 입자 {len(pts)}", flush=True)

sc = gs.Scene(
    sim_options=gs.options.SimOptions(dt=a.dt, substeps=a.substeps),
    mpm_options=gs.options.MPMOptions(lower_bound=(-0.5, -0.5, -0.06),
                                      upper_bound=(0.5, 0.5, 0.94),
                                      grid_density=a.grid_density),
    show_viewer=False,
)
sc.add_entity(gs.morphs.Plane())

ents = []
if a.scene == "tear":
    q = place(pts, a.size, 0.30)
    mid = np.median(q[:, 0])
    for m, col in ((q[:, 0] < mid, (0.9, 0.35, 0.3)), (q[:, 0] >= mid, (0.3, 0.5, 0.9))):
        ents.append(sc.add_entity(material=MATS[mat_name](),
                                  morph=gs.morphs.PointCloud(points=q[m]),
                                  surface=gs.surfaces.Default(color=col)))

else:
    z0 = 0.45 if a.scene == "break" else 0.02
    q = place(pts, a.size, z0)
    ents.append(sc.add_entity(material=MATS[mat_name](),
                              morph=gs.morphs.PointCloud(points=q),
                              surface=gs.surfaces.Default(color=(0.4, 0.75, 0.45))))

sc.build()


def drive_vel(ent, v):
    ent.set_velocity(np.tile(np.asarray(v, np.float32), (ent.n_particles, 1)))


if a.scene == "break":
    drive_vel(ents[0], (0.0, 0.0, a.v0))

steps = int(a.sim_time / a.dt)
bond_steps = int(0.15 / a.dt)          # 인열: 먼저 0.15 초 붙인다
print(f"[설정] 재질 {mat_name}, {a.sim_time} s = {steps} 스텝, "
      f"{a.render_every} 스텝마다 렌더 -> {steps // a.render_every} 프레임", flush=True)

odir = os.path.join(a.out, f"gs_{tag}")
os.makedirs(odir, exist_ok=True)


def snap(k):
    q = np.concatenate([np.asarray(e.get_particles_pos().cpu()).reshape(-1, 3)
                        for e in ents])
    with h5py.File(os.path.join(odir, f"sim_{k:010d}.h5"), "w") as h:
        h.create_dataset("x", data=q.T.astype(np.float32))
    return q


t0 = time.time()
k = 0
snap(k); k += 1
for i in range(steps):
    if a.scene == "tear":
        if i < bond_steps:
            drive_vel(ents[0], (a.pull, 0.0, 0.0))
            drive_vel(ents[1], (-a.pull, 0.0, 0.0))
        else:
            drive_vel(ents[0], (-a.pull, 0.0, 0.0))
            drive_vel(ents[1], (a.pull, 0.0, 0.0))
    sc.step()
    if (i + 1) % a.render_every == 0:
        snap(k); k += 1
print(f"[저장] {odir}  {k} 프레임  {time.time()-t0:.0f}s", flush=True)

# 얼마나 움직였는지 숫자로도 남긴다 (눈으로만 보면 또 틀린다)
with h5py.File(os.path.join(odir, "sim_0000000000.h5"), "r") as h:
    q0 = np.array(h["x"]).T
q1 = np.concatenate([np.asarray(e.get_particles_pos().cpu()).reshape(-1, 3)
                     for e in ents])
d = np.linalg.norm(q1 - q0, axis=1)
ext = float(np.linalg.norm(q0.max(0) - q0.min(0)))
print(f"[움직임] 평균 {100*d.mean()/ext:.2f}% 최대 {100*d.max()/ext:.2f}% (물체 지름 대비)",
      flush=True)
print("GS_SCENE_DONE", flush=True)

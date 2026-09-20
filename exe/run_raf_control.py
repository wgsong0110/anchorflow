"""RAF 방식(3DGS -> 통일 입자셋 -> Genesis MPM)에 **제어점 조작**을 붙인다.

RAF 는 자산을 하나의 입자셋으로 추상화하고 그 위에서 물리를 푼다. 여기서는
물리속성이 지정된 3DGS 를 입자로 바꾸고, 표면 입자 몇 곳을 제어점으로 잡아
warp 교사와 **같은 궤적 규칙**(웨이포인트 + Catmull-Rom + 속도 상한)으로 강제한다.

Genesis 는 `set_particles_vel` 이 NotImplementedError 라 입자 부분집합을 직접
못 박는다. 그래서 **제어점 무리를 각각 별도 엔티티**로 만든다 -- MPM 은 격자를
공유하므로 물리는 하나로 풀리고, 엔티티 단위 `set_velocity` 는 동작한다.

저장 형식은 warp 교사와 같게 맞춘다 (sim_*.h5 + control.npz). 그래야 같은 학습기·
같은 렌더러·같은 잣대를 그대로 쓸 수 있다.

  python exe/run_raf_control.py --ply <3dgs.ply> --config <json> --out DIR
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import h5py
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--ply", required=True, help="물리속성을 줄 3DGS")
ap.add_argument("--config", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--n_pts", type=int, default=120000)
ap.add_argument("--frames", type=int, default=None)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import genesis as gs
from plyfile import PlyData
from control_traj import ControlTraj, pick_control_points

cfg = json.load(open(a.config))
gs.init(backend=gs.gpu, logging_level="warning")
os.makedirs(a.out, exist_ok=True)


def load_gs(path, n_max, op_min):
    """RAF 의 3DGS -> 입자 추상화와 같은 자리 (위치와 불투명도만 쓴다)."""
    p = PlyData.read(path)["vertex"]
    x = np.stack([np.asarray(p[k]) for k in ("x", "y", "z")], 1).astype(np.float64)
    if "opacity" in p.data.dtype.names:
        x = x[1.0 / (1.0 + np.exp(-np.asarray(p["opacity"]))) > op_min]
    x = x[np.isfinite(x).all(1)]
    lo, hi = np.percentile(x, 0.5, 0), np.percentile(x, 99.5, 0)
    x = x[((x > lo) & (x < hi)).all(1)]
    if n_max and len(x) > n_max:
        i = np.random.default_rng(0).choice(len(x), n_max, replace=False)
        x = x[np.sort(i)]
    return x


pts = load_gs(a.ply, a.n_pts, float(cfg.get("opacity_threshold", 0.02)))
size = float(cfg.get("size", 0.30))
c = 0.5 * (pts.min(0) + pts.max(0))
pts = (pts - c) * (size / max(pts.max(0) - pts.min(0)))
pts[:, 2] += float(cfg.get("z_bottom", 0.05)) - pts[:, 2].min()
pts = pts.astype(np.float32)
EXT = float(np.linalg.norm(pts.max(0) - pts.min(0)))
print(f"[자산] {os.path.basename(a.ply)} -> 입자 {len(pts)}, 지름 {EXT:.4f}", flush=True)

# --------------------------------------------------------------- 제어점
CC = cfg.get("control", {})
K = int(CC.get("n_points", 4))
seed = int(CC.get("seed", a.seed))
ctrl_idx, surf = pick_control_points(pts.astype(np.float64), None, K, seed=seed)
rad = float(CC.get("radius", 0.10)) * EXT
mem = []
for ci in ctrl_idx:
    mem.append(np.flatnonzero(np.linalg.norm(pts - pts[ci], axis=1) <= rad))
free = np.setdiff1d(np.arange(len(pts)), np.concatenate(mem))
print(f"[제어점] {K} 개, 반경 {rad:.4f} (지름의 {CC.get('radius',0.10)}), "
      f"잡은 입자 {[len(m) for m in mem]}, 자유 {len(free)}", flush=True)

dt = float(cfg.get("dt", 5e-4))
frames = a.frames if a.frames is not None else int(cfg.get("frame_num", 40))
sub = int(cfg.get("substeps", 10))
traj = ControlTraj(pts.astype(np.float64), ctrl_idx, dt=dt, steps=frames + 1,
                   every_n=int(CC.get("every_n", 8)),
                   p_touch=float(CC.get("p_touch", 0.7)),
                   depth=float(CC.get("depth", 0.10)),
                   v_max=float(CC.get("v_max", 0.5)), seed=seed,
                   bounds=(np.array([-0.45, -0.45, -0.04]) + rad,
                           np.array([0.45, 0.45, 0.9]) - rad))
print(f"[궤적] {frames+1} 스텝, 실제 최대속도 {traj.max_speed():.4f}", flush=True)

MATS = {
    "cdmpm": lambda m: gs.materials.MPM.CDMPM(
        E=m["E"], nu=m["nu"], rho=m["rho"], friction_angle=m.get("friction_angle", 45.0),
        beta=m.get("beta", 1.0), xi=m.get("xi", 3.0), hardening=m.get("hardening", 1.0),
        alpha_0=m.get("alpha_0", -0.04)),
    "plasticine": lambda m: gs.materials.MPM.ElastoPlastic(
        E=m["E"], nu=m["nu"], rho=m["rho"], use_von_mises=True,
        von_mises_yield_stress=m.get("yield_stress", 1e4)),
    "sand": lambda m: gs.materials.MPM.Sand(
        E=m["E"], nu=m["nu"], rho=m["rho"],
        friction_angle=m.get("friction_angle", 40.0)),
}
mk = MATS[cfg.get("material", "plasticine")]
mp = cfg.get("params", dict(E=2e6, nu=0.3, rho=1000.0))

sc = gs.Scene(
    sim_options=gs.options.SimOptions(dt=dt, substeps=sub),
    mpm_options=gs.options.MPMOptions(lower_bound=(-0.5, -0.5, -0.06),
                                      upper_bound=(0.5, 0.5, 0.94),
                                      grid_density=int(cfg.get("grid_density", 96))),
    show_viewer=False,
)
sc.add_entity(gs.morphs.Plane())
# 자유 입자 한 덩이 + 제어점 무리 K 덩이. MPM 은 격자를 공유하므로 물리는 하나다.
ent_free = sc.add_entity(material=mk(mp),
                         morph=gs.morphs.PointCloud(points=pts[free]),
                         surface=gs.surfaces.Default(color=(0.45, 0.72, 0.5)))
ent_grip = [sc.add_entity(material=mk(mp),
                          morph=gs.morphs.PointCloud(points=pts[m]),
                          surface=gs.surfaces.Default(color=(0.95, 0.2, 0.15)))
            for m in mem]
sc.build()

order = np.concatenate([free] + mem)          # h5 안에서의 자리
inv = np.argsort(order)
ctrl_local = np.array([int(np.flatnonzero(order == ci)[0]) for ci in ctrl_idx])
mptr = np.cumsum([0] + [len(m) for m in mem]) + len(free)
np.savez(os.path.join(a.out, "control.npz"),
         idx=ctrl_local, pos=traj.P.astype(np.float32),
         vel=traj.V.astype(np.float32), x0=pts[ctrl_idx],
         members=np.concatenate([np.arange(mptr[i], mptr[i + 1])
                                 for i in range(K)]).astype(np.int64),
         member_ptr=(mptr - mptr[0] + len(free)).astype(np.int64),
         cfg=json.dumps(CC))


# 제어점 무리는 제어점과 같은 offset 으로 움직인다 (warp 교사와 같은 규칙)
OFF = [pts[m] - pts[ci] for m, ci in zip(mem, ctrl_idx)]


def drive(k, e, cpos, v):
    """위치와 속도를 **둘 다** 박는다. 속도만 주면 격자를 거쳐 흘러 어긋난다."""
    e.set_velocity(np.tile(np.asarray(v, np.float32), (e.n_particles, 1)))
    e.set_particles_pos((cpos[None, :] + OFF[k]).astype(np.float32))


def snap(k):
    q = np.concatenate([np.asarray(e.get_particles_pos().cpu()).reshape(-1, 3)
                        for e in [ent_free] + ent_grip])
    with h5py.File(os.path.join(a.out, f"sim_{k:010d}.h5"), "w") as h:
        h.create_dataset("x", data=q.T.astype(np.float32))
    return q


t0 = time.time()
snap(0)
for f in range(frames):
    v = traj.vel(f + 1)
    cp = traj.pos(f + 1)
    for k, e in enumerate(ent_grip):
        drive(k, e, cp[k], v[k])
    sc.step()
    for k, e in enumerate(ent_grip):     # step 이 옮겼으니 다시 박는다
        drive(k, e, cp[k], v[k])
    snap(f + 1)
    if (f + 1) % 10 == 0:
        print(f"  f{f+1:4d}  {time.time()-t0:.0f}s", flush=True)
print(f"[저장] {a.out}  {frames+1} 프레임  {time.time()-t0:.0f}s", flush=True)
print("RAF_CTRL_DONE", flush=True)

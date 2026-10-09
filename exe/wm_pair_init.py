"""두 수박 정면 충돌 장면 초기 상태: GaussianFluent 공식 내부 채우기를 한 수박(0 프레임 입자) 두 벌.

시뮬 상자를 grid_lim 3 으로 넓히고 격자 450 (dx 0.00667, GF 수박과 같은 칸 크기). 두 수박을 x 로 --gap 만큼
떨어뜨려 놓고 서로를 향해 ±--v 로 보낸다. 물성·중력·FLIP 은 GF 수박 config 그대로 (g -15).
충돌 속도·간격·프레임 간격은 공식 장면이 없어 정한 값.

  python exe/wm_pair_init.py --h5 <GF sim_0000000000.h5> --config <GF config.json> --out DIR
"""
import argparse, json, os
import h5py
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--h5", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--gap", type=float, default=0.6)
ap.add_argument("--v", type=float, default=6.0)
ap.add_argument("--zc", type=float, default=1.2, help="두 수박 중심 높이")
ap.add_argument("--frame_dt", type=float, default=0.01)
ap.add_argument("--frames", type=int, default=100)
a = ap.parse_args()
with h5py.File(a.h5) as h:
    x = np.array(h["x"]); x = (x.T if x.shape[0] == 3 else x).astype(np.float64)
c = 0.5 * (x.min(0) + x.max(0)); w = x[:, 0].max() - x[:, 0].min()
d = w / 2 + a.gap / 2
SH = [np.array([1.5 - d, 1.5, a.zc]) - c, np.array([1.5 + d, 1.5, a.zc]) - c]
X = np.concatenate([x + SH[0], x + SH[1]])
V = np.concatenate([np.tile([a.v, 0, 0], (len(x), 1)), np.tile([-a.v, 0, 0], (len(x), 1))])
OBJ = np.repeat([0, 1], len(x)).astype(np.int32)
assert X.min() > 0.05 and X.max() < 2.95, (X.min(0), X.max(0))
os.makedirs(a.out, exist_ok=True)
with h5py.File(f"{a.out}/init.h5", "w") as h:
    h.create_dataset("x", data=X.T.astype(np.float32)); h.create_dataset("v", data=V.T.astype(np.float32))
    h.create_dataset("obj", data=OBJ)
cfg = json.load(open(a.config))
cfg.update(grid_lim=3.0, n_grid=450, frame_dt=a.frame_dt, frame_num=a.frames, init_velocity=[0.0, 0.0, 0.0])
cfg["boundary_conditions"] = [{"type": "bounding_box"}]
cfg.pop("particle_filling", None)
cfg["repflow_note"] = dict(scene="두 수박 정면 충돌", gap=a.gap, v=a.v, zc=a.zc)
json.dump(cfg, open(f"{a.out}/config.json", "w"), indent=1)
json.dump(dict(shifts=[s.tolist() for s in SH], n_each=len(x), width=float(w)), open(f"{a.out}/meta.json", "w"), indent=1)
print(f"[두 수박] 한 벌 {len(x)} 입자, 폭 {w:.3f}, 중심 x {1.5 - d:.3f} / {1.5 + d:.3f}, 간격 {a.gap}, ±{a.v} "
      f"-> {a.gap / (2 * a.v):.3f}s 뒤 충돌", flush=True)

"""쌓인 더미가 **산을 이루는지** 잰다 (안식각).

모래가 평평해져 버리면 Drucker-Prager 의 마찰이 일을 못 한 것이다. 그 항복
조건은 `|dev| + (3λ+2μ)/(2μ)·tr·α ≤ 0` 이라 **전단 한계가 압력에 비례**하는데,
E 가 너무 작으면 압력이 사실상 0 이라 마찰각을 아무리 올려도 흘러내린다.

재는 것은 마지막 프레임의 높이 대 반경이다.
  h = (z 의 95 분위) - 바닥,  R = 중심에서의 수평거리 95 분위
  안식각 = atan(h / R)
평평하면 0 도에 가깝고, 마찰각 30 도짜리 원뿔이면 30 도 근처여야 한다.

  python exe/measure_repose.py --h5_dir DIR
"""
import argparse
import glob
import json
import os

import h5py
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--h5_dir", required=True)
ap.add_argument("--tag", default="run")
ap.add_argument("--q", type=float, default=95.0)
ap.add_argument("--out", default=None)
a = ap.parse_args()

fs = sorted(glob.glob(os.path.join(a.h5_dir, "**", "sim_*.h5"), recursive=True))
if not fs:
    raise SystemExit(f"h5 가 없다: {a.h5_dir}")


def rd(p):
    with h5py.File(p, "r") as h:
        x = np.array(h["x"])
    x = (x.T if x.shape[0] == 3 else x).astype(np.float64)
    return x[np.isfinite(x).all(1)]


rows = []
z_floor = rd(fs[0])[:, 2].min()
for i in (0, len(fs) // 2, len(fs) - 1):
    x = rd(fs[i])
    c = np.median(x[:, :2], axis=0)
    r = np.linalg.norm(x[:, :2] - c, axis=1)
    h = np.percentile(x[:, 2], a.q) - z_floor
    R = np.percentile(r, a.q)
    ang = float(np.degrees(np.arctan2(max(h, 0.0), max(R, 1e-9))))
    rows.append(dict(frame=i, h=float(h), R=float(R), angle=ang))
    print(f"  f{i:4d}  높이 {h:.4f}  반경 {R:.4f}  안식각 {ang:5.1f}도", flush=True)

f0, fl = rows[0], rows[-1]
print(f"[판정] {a.tag}: 안식각 {f0['angle']:.1f}도 -> {fl['angle']:.1f}도, "
      f"반경 {f0['R']:.3f} -> {fl['R']:.3f} "
      f"({'평평해졌다' if fl['angle'] < 10 else '산을 이룬다'})", flush=True)
if a.out:
    json.dump(dict(tag=a.tag, rows=rows), open(a.out, "w"), indent=1)
print("REPOSE_OK", flush=True)

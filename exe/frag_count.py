"""충돌 궤적의 파괴 정도: 물체마다 입자를 dx 격자에 올려 26-연결 성분을 센다.

  python exe/frag_count.py --sim DIR/sim --dx 0.02
"""
import argparse, glob
import h5py
import numpy as np
from scipy import ndimage

ap = argparse.ArgumentParser()
ap.add_argument("--sim", required=True)
ap.add_argument("--dx", type=float, default=0.02)
ap.add_argument("--frames", default="0,15,20,30,45,60")
a = ap.parse_args()
fs = sorted(glob.glob(f"{a.sim}/sim_*.h5"))
for t in [int(q) for q in a.frames.split(",")]:
    if t >= len(fs):
        continue
    with h5py.File(fs[t]) as h:
        x = h["x"][()].T; ob = h["obj"][()]
    out = []
    for o in (0, 1):
        p = x[ob == o]; p = p[np.isfinite(p).all(1)]
        k = np.floor((p - p.min(0)) / a.dx).astype(int)
        occ = np.zeros(k.max(0) + 1, bool); occ[tuple(k.T)] = True
        lab, n = ndimage.label(occ, structure=np.ones((3, 3, 3)))
        sz = np.bincount(lab[tuple(k.T)])[1:]
        sz = np.sort(sz)[::-1] / len(p)
        out.append(f"물체{o}: 조각 {n} (1% 넘는 것 {(sz > 0.01).sum()}), 가장 큰 조각 {100 * sz[0]:.1f}%")
    print(f"  f{t:3d}  " + " | ".join(out), flush=True)

"""찰흙 덩어리 3DGS 자산을 직접 찍어낸다 (학습 없이).

가진 자산이 식빵·늑대·화분 같은 것뿐이라 "찰흙 씬" 이 없다. 그런데 MPM 이 필요로
하는 것은 위치·불투명도·공분산뿐이고 색은 보기 위한 것이라, 격자에 점을 깔고
필드를 채우면 그대로 쓸 수 있는 ply 가 된다.

기본은 가로로 긴 막대다 (양끝을 잡고 당겨 끊는 장면을 만들기 위해). 표면에만
점을 두지 않고 속까지 채우므로 particle_filling 없이도 부피가 있다.

SH 는 0 차만 쓴다 (f_dc 3 개 + f_rest 45 개를 0 으로). 3DGS 의 색 규약은
c = 0.5 + C0 * f_dc, C0 = 0.28209479 이므로 f_dc = (c - 0.5) / C0 이다.
"""
from __future__ import annotations

import argparse
import os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--size", type=float, nargs=3, default=[0.6, 0.18, 0.18],
                help="막대의 가로x세로x높이 (월드 단위)")
ap.add_argument("--spacing", type=float, default=0.006,
                help="점 간격. 작을수록 입자가 많아진다")
ap.add_argument("--color", type=float, nargs=3, default=[0.62, 0.44, 0.34],
                help="찰흙 색 (0~1 RGB)")
ap.add_argument("--color_jitter", type=float, default=0.04)
ap.add_argument("--round", type=float, default=0.35,
                help="모서리를 둥글리는 정도 (0 이면 직육면체, 1 이면 타원체)")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

rng = np.random.RandomState(a.seed)
sx, sy, sz = a.size
n = [max(2, int(round(s / a.spacing))) for s in (sx, sy, sz)]
g = np.stack(np.meshgrid(
    np.linspace(-sx / 2, sx / 2, n[0]),
    np.linspace(-sy / 2, sy / 2, n[1]),
    np.linspace(-sz / 2, sz / 2, n[2]), indexing="ij"), -1).reshape(-1, 3)
# 격자 그대로 두면 MPM 에서 줄무늬가 생긴다. 간격의 1/4 만큼 흔들어 둔다.
g = g + rng.uniform(-a.spacing / 4, a.spacing / 4, g.shape)

if a.round > 0:
    # 직육면체와 타원체를 섞는다 (superquadric 대신 단순 보간)
    q = (2 * g / np.array([sx, sy, sz]))
    r = np.linalg.norm(q, axis=1)
    keep = r <= (1.0 / a.round if a.round < 1 else 1.0)
    inf = np.abs(q).max(1) <= 1.0
    g = g[keep & inf]
else:
    g = g[np.abs(2 * g / np.array([sx, sy, sz])).max(1) <= 1.0]

N = g.shape[0]
C0 = 0.28209479177387814
col = np.clip(np.array(a.color) + rng.normal(0, a.color_jitter, (N, 3)), 0, 1)
f_dc = (col - 0.5) / C0
f_rest = np.zeros((N, 45), dtype=np.float32)
opacity = np.full((N, 1), 6.0, dtype=np.float32)          # sigmoid(6) ~ 0.9975
scale = np.full((N, 3), np.log(a.spacing * 0.6), dtype=np.float32)
rot = np.tile(np.array([1.0, 0, 0, 0], dtype=np.float32), (N, 1))
normals = np.zeros((N, 3), dtype=np.float32)

fields = (["x", "y", "z", "nx", "ny", "nz"]
          + [f"f_dc_{i}" for i in range(3)]
          + [f"f_rest_{i}" for i in range(45)]
          + ["opacity", "scale_0", "scale_1", "scale_2",
             "rot_0", "rot_1", "rot_2", "rot_3"])
data = np.concatenate([g, normals, f_dc, f_rest, opacity, scale, rot],
                      1).astype(np.float32)
assert data.shape[1] == len(fields), (data.shape, len(fields))

os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
with open(a.out, "wb") as f:
    f.write(b"ply\nformat binary_little_endian 1.0\n")
    f.write(f"element vertex {N}\n".encode())
    for k in fields:
        f.write(f"property float {k}\n".encode())
    f.write(b"end_header\n")
    f.write(data.tobytes())
print(f"[저장] {a.out}  점 {N}, 크기 {sx}x{sy}x{sz}, 간격 {a.spacing}", flush=True)
print(f"  범위 x[{g[:,0].min():.3f},{g[:,0].max():.3f}] "
      f"y[{g[:,1].min():.3f},{g[:,1].max():.3f}] "
      f"z[{g[:,2].min():.3f},{g[:,2].max():.3f}]", flush=True)
print("CLAYPLY_OK", flush=True)

"""메시를 PNG 로 미리 본다 (의존성 없이 numpy z-버퍼 + 평면 셰이딩).

trimesh/open3d 같은 것을 깔지 않고 확인만 하려는 용도다. 여러 시점을 한 장에
붙여 낸다.
"""
from __future__ import annotations

import argparse

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--mesh", required=True, help="gs_to_mesh 가 남긴 .npz")
ap.add_argument("--out", required=True)
ap.add_argument("--res", type=int, default=420)
ap.add_argument("--views", type=int, default=3)
a = ap.parse_args()

import imageio.v2 as imageio                                     # noqa: E402

d = np.load(a.mesh)
V, F = d["v"].astype(np.float64), d["f"].astype(np.int64)
c = 0.5 * (V.min(0) + V.max(0))
r = float(np.linalg.norm(V.max(0) - V.min(0))) * 0.75
tiles = []
for vi in range(a.views):
    az = np.deg2rad(35 + 120 * vi)
    el = np.deg2rad(20)
    eye = c + r * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az),
                            np.sin(el)])
    fwd = (c - eye); fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1.0]); right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    M = np.stack([right, up, fwd])                 # 세계 -> 시야
    P = (V - eye) @ M.T
    f = a.res * 0.9 / r
    sx = P[:, 0] / np.maximum(P[:, 2], 1e-6) * f + a.res / 2
    sy = -P[:, 1] / np.maximum(P[:, 2], 1e-6) * f + a.res / 2
    z = P[:, 2]
    img = np.ones((a.res, a.res), np.float32)
    zb = np.full((a.res, a.res), 1e9, np.float32)
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    nl = np.linalg.norm(n, axis=1, keepdims=True)
    n = n / np.maximum(nl, 1e-12)
    light = np.array([0.4, 0.5, 0.75]); light /= np.linalg.norm(light)
    shade = np.clip(0.25 + 0.75 * np.abs(n @ light), 0, 1)
    # 삼각형을 무게중심 한 점으로 찍는다 (미리보기라 충분하다)
    tz = z[F].mean(1)
    tx = sx[F].mean(1); ty = sy[F].mean(1)
    ok = (tz > 0) & (tx > 0) & (tx < a.res - 1) & (ty > 0) & (ty < a.res - 1)
    xi = tx[ok].astype(int); yi = ty[ok].astype(int)
    zi = tz[ok]; si = shade[ok]
    order = np.argsort(-zi)                        # 먼 것부터 덮어쓴다
    img[yi[order], xi[order]] = si[order]
    tiles.append((img * 255).astype(np.uint8))
out = np.concatenate(tiles, 1)
imageio.imwrite(a.out, np.stack([out] * 3, -1))
print(f"[저장] {a.out}  정점 {V.shape[0]} 삼각형 {F.shape[0]}", flush=True)
print("PREVIEW_OK", flush=True)

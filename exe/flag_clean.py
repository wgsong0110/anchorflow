"""깃발 3DGS 의 흰 군더더기 가우시안 제거 (흰 배경 학습이라 남은 것): 형상 밖 + 바늘 모양.

남기는 것: 천 직사각형(x 0~1.5, z 1.15~2.15) 근처 |y| < --dy 이거나 깃대 원기둥 근처인 가우시안 중
가장 긴 축이 --max_axis 이하인 것. 흰 체크무늬(천 위)는 위치로 남는다.
  python exe/flag_clean.py --src flag_gs --dst flag_gs_clean --meta flag_ns/meta.json
"""
import argparse, json, os, shutil
import numpy as np
from plyfile import PlyData, PlyElement

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True); ap.add_argument("--dst", required=True); ap.add_argument("--meta", required=True)
ap.add_argument("--dy", type=float, default=0.02); ap.add_argument("--margin", type=float, default=0.02)
ap.add_argument("--max_axis", type=float, default=0.03, help="m")
a = ap.parse_args()
meta = json.load(open(a.meta)); s, c = meta["scale"], np.array(meta["center_yup_to_zup"])
p = f"{a.src}/point_cloud/iteration_30000/point_cloud.ply"
pl = PlyData.read(p); v = pl.elements[0].data
X = np.stack([v["x"], v["y"], v["z"]], 1) / s + c
ax = np.exp(np.stack([v[f"scale_{i}"] for i in range(3)], 1)).max(1) / s
m = a.margin
cloth = (X[:, 0] > -m) & (X[:, 0] < 1.5 + m) & (X[:, 2] > 1.15 - m) & (X[:, 2] < 2.15 + m) & (np.abs(X[:, 1]) < a.dy)
pole = (np.hypot(X[:, 0] + 0.03, X[:, 1]) < 0.025 + m) & (X[:, 2] > -m) & (X[:, 2] < 2.2 + m)
keep = (cloth | pole) & (ax < a.max_axis)
op = 1 / (1 + np.exp(-v["opacity"]))
print(f"[정리] 가우시안 {len(X)} -> {int(keep.sum())} (형상 밖 {int((~(cloth | pole)).sum())}, 바늘 {int(((cloth | pole) & (ax >= a.max_axis)).sum())}; "
      f"지운 것 중 불투명 > 0.1: {int(((~keep) & (op > 0.1)).sum())})")
os.makedirs(f"{a.dst}/point_cloud/iteration_30000", exist_ok=True)
PlyData([PlyElement.describe(v[keep], "vertex")]).write(f"{a.dst}/point_cloud/iteration_30000/point_cloud.ply")
for f in ("cfg_args", "cameras.json"):
    shutil.copy(f"{a.src}/{f}", f"{a.dst}/{f}")

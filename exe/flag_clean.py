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
ap.add_argument("--max_axis", type=float, default=0.02, help="m: 이보다 긴 가우시안만 아래 두 조건으로 본다")
ap.add_argument("--needle", type=float, default=6.0, help="가장 긴 축 / 두 번째 축 이 이 이상이면 바늘")
a = ap.parse_args()
meta = json.load(open(a.meta)); s, c = meta["scale"], np.array(meta["center_yup_to_zup"])
p = f"{a.src}/point_cloud/iteration_30000/point_cloud.ply"
pl = PlyData.read(p); v = pl.elements[0].data
X = np.stack([v["x"], v["y"], v["z"]], 1) / s + c
sc3 = np.sort(np.exp(np.stack([v[f"scale_{i}"] for i in range(3)], 1)), 1) / s   # 작은 -> 큰
ax, mid = sc3[:, 2], sc3[:, 1]
m = a.margin
cloth = (X[:, 0] > -m) & (X[:, 0] < 1.5 + m) & (X[:, 2] > 1.15 - m) & (X[:, 2] < 2.15 + m) & (np.abs(X[:, 1]) < a.dy)
pole = (np.hypot(X[:, 0] + 0.03, X[:, 1]) < 0.025 + m) & (X[:, 2] > -m) & (X[:, 2] < 2.2 + m)
# 흰 군더더기만: 색(SH 0 차)이 흰색이고 긴 것. 천 가우시안 대부분이 가늘고 길어서(평면 학습) 모양만으로 고르면 천까지 지운다
rgb = np.stack([v[f"f_dc_{i}"] for i in range(3)], 1) * 0.28209479177387814 + 0.5
white = rgb.min(1) > 0.8
needle = (ax >= a.max_axis) & white
# 천 가장자리 밖으로 번지는 큰 가우시안 (위쪽 흰 안개): 중심 + 가장 긴 축이 직사각형을 3 cm 넘게 벗어난다
spill = cloth & (ax >= a.max_axis) & ((X[:, 2] + ax > 2.15 + 0.03) | (X[:, 2] - ax < 1.15 - 0.03) | (X[:, 0] + ax > 1.5 + 0.03))
keep = (cloth | pole) & ~needle & ~spill
op = 1 / (1 + np.exp(-v["opacity"]))
print(f"[정리] 가우시안 {len(X)} -> {int(keep.sum())} (형상 밖 {int((~(cloth | pole)).sum())}, 흰 긴 것 {int(((cloth | pole) & needle).sum())}, 가장자리 번짐 {int((spill & ~needle).sum())}; "
      f"지운 것 중 불투명 > 0.1: {int(((~keep) & (op > 0.1)).sum())})")
os.makedirs(f"{a.dst}/point_cloud/iteration_30000", exist_ok=True)
PlyData([PlyElement.describe(v[keep], "vertex")]).write(f"{a.dst}/point_cloud/iteration_30000/point_cloud.ply")
for f in ("cfg_args", "cameras.json"):
    shutil.copy(f"{a.src}/{f}", f"{a.dst}/{f}")

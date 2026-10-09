"""깃발 메쉬 (glb): 1.5 x 1.0 m 직사각형 천(세로로 선 xz 평면) + 왼쪽 깃대(회색 원기둥).

천 텍스처는 Poly Haven CC0 gingham_check 확산 맵을 3 x 2 로 이어 붙인 한 장 (UV 0~1).
천은 격자 --nx x --nz 로 나눈다 (시뮬 메쉬로도 쓴다). 깃대는 천 왼쪽 변 바로 옆.

  python exe/make_flag.py --tex gingham_check_diff_1k.jpg --out flag.glb
"""
import argparse
import numpy as np
import trimesh
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--tex", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--w", type=float, default=1.5)
ap.add_argument("--h", type=float, default=1.0)
ap.add_argument("--nx", type=int, default=90)
ap.add_argument("--nz", type=int, default=60)
ap.add_argument("--pole_r", type=float, default=0.025)
ap.add_argument("--pole_len", type=float, default=2.2)
a = ap.parse_args()
t = Image.open(a.tex).convert("RGB")
W, H = t.size
big = Image.new("RGB", (3 * W, 2 * H))
for i in range(3):
    for j in range(2):
        big.paste(t, (i * W, j * H))
xs, zs = np.linspace(0, a.w, a.nx + 1), np.linspace(0, a.h, a.nz + 1)
X, Z = np.meshgrid(xs, zs, indexing="xy")
V = np.stack([X.ravel(), np.zeros(X.size), Z.ravel() + (a.pole_len - a.h - 0.05)], 1)   # 깃대 위쪽에 단다
UV = np.stack([X.ravel() / a.w, Z.ravel() / a.h], 1)
idx = lambda i, j: j * (a.nx + 1) + i                                     # noqa: E731
F = []
for j in range(a.nz):
    for i in range(a.nx):
        F += [[idx(i, j), idx(i + 1, j), idx(i + 1, j + 1)], [idx(i, j), idx(i + 1, j + 1), idx(i, j + 1)]]
cloth = trimesh.Trimesh(V, np.array(F), process=False,
                        visual=trimesh.visual.TextureVisuals(uv=UV, material=trimesh.visual.material.PBRMaterial(baseColorTexture=big)))
pole = trimesh.creation.cylinder(radius=a.pole_r, height=a.pole_len, sections=32)
pole.apply_translation([-a.pole_r - 0.005, 0.0, a.pole_len / 2])
gray = Image.new("RGB", (8, 8), (140, 140, 145))
pole.visual = trimesh.visual.TextureVisuals(uv=np.full((len(pole.vertices), 2), 0.5),
                                            material=trimesh.visual.material.PBRMaterial(baseColorTexture=gray))
# glTF 는 y 위 규약이다 (mesh_to_nerfsynth 가 y 위 -> z 위로 돌린다). z 위로 만든 것을 y 위로 바꿔 내보낸다
_zy = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], float)   # (x, y, z) -> (x, z, -y)
cloth.apply_transform(_zy); pole.apply_transform(_zy)
sc = trimesh.Scene(); sc.add_geometry(cloth, node_name="flag_cloth"); sc.add_geometry(pole, node_name="flag_pole")
sc.export(a.out)
print(f"[깃발] 천 {len(V)} 꼭짓점 {len(F)} 삼각형, 깃대 높이 {a.pole_len} m -> {a.out}")

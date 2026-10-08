"""glTF 메쉬(Poly Haven CC0)를 NeRF-synthetic 형식 다시점 영상으로 렌더한다 -- 공식 3DGS 학습 입력.

Fracture-GS 의 Teapot & Table 장면 자산이 공개되지 않아, 공개 CC0 메쉬로 3DGS 를 직접 학습하기 위한 것.
  - glTF 는 y 위, 여기서는 z 위로 바꾼다 (x, y, z) -> (x, -z, y)
  - 중심을 원점에, 가장 긴 변을 --size 로 맞춘다 (원래 크기 배율은 meta.json 에 남긴다)
  - 흰 배경 800x800, camera_angle_x 0.6911 (NeRF-synthetic 과 같다), 위쪽 반구 + 약간 아래
  - 빛은 세계 고정 (보는 방향과 무관한 색이 되게): 환경광 0.55 + 위쪽 점광원

  python exe/mesh_to_nerfsynth.py --gltf A/a_1k.gltf --nodes teapot_01,teapot_01_lid --out DIR
"""
import argparse
import json
import math
import os

import imageio.v2 as imageio
import numpy as np
import torch
import trimesh
from pytorch3d.renderer import (FoVPerspectiveCameras, MeshRasterizer, MeshRenderer, PointLights,
                                RasterizationSettings, SoftPhongShader, TexturesUV, look_at_view_transform)
from pytorch3d.renderer.blending import BlendParams
from pytorch3d.structures import Meshes, join_meshes_as_scene
from tqdm import tqdm

ap = argparse.ArgumentParser()
ap.add_argument("--gltf", required=True)
ap.add_argument("--nodes", default="", help="쓸 노드 이름에 들어간 문자열 (쉼표). 비우면 전부")
ap.add_argument("--out", required=True)
ap.add_argument("--size", type=float, default=2.0)
ap.add_argument("--n_train", type=int, default=150)
ap.add_argument("--n_test", type=int, default=20)
ap.add_argument("--res", type=int, default=800)
a = ap.parse_args()
dev = torch.device("cuda")

sc = trimesh.load(a.gltf)
keys = [q for q in a.nodes.split(",") if q]
parts = []
for name in sc.graph.nodes_geometry:
    if keys and not any(name.endswith(k) for k in keys):
        continue
    T, gk = sc.graph[name]
    m = sc.geometry[gk].copy(); m.apply_transform(T)
    parts.append((name, m))
print("[노드]", [p[0] for p in parts], flush=True)
Ylup = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], np.float64)          # y 위 -> z 위
allv = np.concatenate([p[1].vertices @ Ylup.T for p in parts])
c = 0.5 * (allv.min(0) + allv.max(0)); s = a.size / float((allv.max(0) - allv.min(0)).max())
meshes = []
for name, m in parts:
    v = (m.vertices @ Ylup.T - c) * s
    uv = np.asarray(m.visual.uv, np.float32)
    img = np.asarray(m.visual.material.baseColorTexture.convert("RGB"), np.float32) / 255.0
    meshes.append(Meshes(verts=[torch.tensor(v, dtype=torch.float32, device=dev)],
                         faces=[torch.tensor(m.faces, dtype=torch.int64, device=dev)],
                         textures=TexturesUV(maps=torch.tensor(img, device=dev)[None],
                                             faces_uvs=[torch.tensor(m.faces, dtype=torch.int64, device=dev)],
                                             verts_uvs=[torch.tensor(uv, device=dev)])))
mesh = join_meshes_as_scene(meshes)
fov = 0.6911112070083618
R0 = 1.25 * a.size / math.tan(fov / 2) * 0.5 * 1.15                 # 물체가 화면의 약 80% 를 채우는 거리
lights = PointLights(device=dev, location=[[0.0, 0.0, 6.0]], ambient_color=[[0.55] * 3],
                     diffuse_color=[[0.45] * 3], specular_color=[[0.0] * 3])
rast = MeshRasterizer(raster_settings=RasterizationSettings(image_size=a.res, blur_radius=0.0, faces_per_pixel=1))
os.makedirs(a.out, exist_ok=True)
rng = np.random.default_rng(0)


def cams(n, seed):
    r = np.random.default_rng(seed)
    out = []
    for i in range(n):
        az = 2 * math.pi * (i + r.random()) / n
        el = math.radians(r.uniform(-15, 75))
        out.append(np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)]) * R0)
    return out


for split, n, seed in (("train", a.n_train, 0), ("test", a.n_test, 1)):
    os.makedirs(f"{a.out}/{split}", exist_ok=True)
    frames = []
    for i, eye in enumerate(tqdm(cams(n, seed), desc=split)):
        up = (0, 0, 1)
        R, T = look_at_view_transform(eye=[eye.tolist()], at=[[0, 0, 0]], up=[up], device=dev)
        cam = FoVPerspectiveCameras(device=dev, R=R, T=T, fov=fov, degrees=False)
        rnd = MeshRenderer(rast, SoftPhongShader(device=dev, cameras=cam, lights=lights,
                                                 blend_params=BlendParams(background_color=(1.0, 1.0, 1.0))))
        im = rnd(mesh, cameras=cam)[0]
        rgb = im[..., :3].clamp(0, 1); al = (im[..., 3] > 0).float()
        rgba = torch.cat([rgb, al[..., None]], -1)
        imageio.imwrite(f"{a.out}/{split}/r_{i}.png", (rgba.cpu().numpy() * 255).astype(np.uint8))
        f = eye / np.linalg.norm(eye) * -1                                  # 앞 = 원점 쪽
        rgt = np.cross(f, up); rgt /= np.linalg.norm(rgt); u = np.cross(rgt, f)
        c2w = np.eye(4); c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = rgt, u, -f, eye   # Blender/OpenGL
        frames.append(dict(file_path=f"./{split}/r_{i}", transform_matrix=c2w.tolist()))
    json.dump(dict(camera_angle_x=fov, frames=frames), open(f"{a.out}/transforms_{split}.json", "w"), indent=1)
json.dump(dict(gltf=a.gltf, nodes=[p[0] for p in parts], center_yup_to_zup=c.tolist(), scale=s,
               real_extent_m=((allv.max(0) - allv.min(0))).tolist()), open(f"{a.out}/meta.json", "w"), indent=1)
print(f"[완료] {a.out}  배율 {s:.4f} (실제 크기 {np.round(allv.max(0) - allv.min(0), 3)} m)", flush=True)

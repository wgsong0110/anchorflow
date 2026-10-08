"""glTF 메쉬(Poly Haven CC0)를 NeRF-synthetic 형식 다시점 영상으로 렌더한다 -- 공식 3DGS 학습 입력.

Fracture-GS 의 Teapot & Table 장면 자산이 공개되지 않아, 공개 CC0 메쉬로 3DGS 를 직접 학습하기 위한 것.
  - glTF 는 y 위, 여기서는 z 위로 바꾼다 (x, y, z) -> (x, -z, y)
  - 중심을 원점에, 가장 긴 변을 --size 로 맞춘다 (원래 크기 배율은 meta.json 에 남긴다)
  - 흰 배경 800x800, camera_angle_x 0.6911 (NeRF-synthetic 과 같다), 고도 -15~75 도
  - 빛은 세계 고정 (보는 방향과 무관한 색이 되게): 환경광 0.55 + 방향광 0.45 (양면)
  - 래스터화는 kaolin (simp env; af env 의 kaolin 은 warp 버전이 안 맞고 pytorch3d 는 렌더러가 없다)

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
from kaolin.render.mesh import rasterize, texture_mapping
from tqdm import tqdm

ap = argparse.ArgumentParser()
ap.add_argument("--gltf", required=True)
ap.add_argument("--nodes", default="", help="쓸 노드 이름의 끝 문자열 (쉼표). 비우면 전부")
ap.add_argument("--out", required=True)
ap.add_argument("--size", type=float, default=2.0)
ap.add_argument("--n_train", type=int, default=150)
ap.add_argument("--n_test", type=int, default=20)
ap.add_argument("--res", type=int, default=800)
ap.add_argument("--flip_y", action="store_true", help="kaolin 출력 행 순서가 뒤집혀 있으면")
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

V, Fc, UV, TEX, FT = [], [], [], [], []
off = 0
for ti, (name, m) in enumerate(parts):
    v = (m.vertices @ Ylup.T - c) * s
    V.append(v); Fc.append(m.faces + off); off += len(v)
    UV.append(np.asarray(m.visual.uv, np.float32)[m.faces])                  # [F,3,2]
    TEX.append(np.asarray(m.visual.material.baseColorTexture.convert("RGB"), np.float32) / 255.0)
    FT.append(np.full(len(m.faces), ti))
V = torch.tensor(np.concatenate(V), dtype=torch.float32, device=dev)
Fc = torch.tensor(np.concatenate(Fc), dtype=torch.int64, device=dev)
UV = torch.tensor(np.concatenate(UV), device=dev)
FT = torch.tensor(np.concatenate(FT), device=dev)
TEXT = [torch.tensor(t, device=dev).permute(2, 0, 1)[None].contiguous() for t in TEX]
fn = torch.cross(V[Fc[:, 1]] - V[Fc[:, 0]], V[Fc[:, 2]] - V[Fc[:, 0]], dim=-1)
fn = fn / fn.norm(dim=-1, keepdim=True).clamp_min(1e-12)
LDIR = torch.tensor([0.3, 0.2, 1.0], device=dev); LDIR = LDIR / LDIR.norm()
shade_f = 0.55 + 0.45 * (fn @ LDIR).abs()                                    # 양면 방향광
fov = 0.6911112070083618
R0 = 1.25 * a.size / math.tan(fov / 2) * 0.5 * 1.15                 # 물체가 화면의 약 80% 를 채우는 거리
FEAT = torch.cat([UV, FT[:, None, None].float().expand(-1, 3, 1), shade_f[:, None, None].expand(-1, 3, 1)], -1)


def render(c2w):
    w2c = torch.tensor(np.linalg.inv(c2w), dtype=torch.float32, device=dev)
    pc = V @ w2c[:3, :3].T + w2c[:3, 3]                                 # 카메라: x 오른, y 위, -z 앞
    f_ = 1.0 / math.tan(fov / 2)
    ndc = torch.stack([f_ * pc[:, 0] / (-pc[:, 2]), f_ * pc[:, 1] / (-pc[:, 2])], -1)
    fz = pc[Fc][None, ..., 2].contiguous(); fi = ndc[Fc][None].contiguous()
    out, fidx = rasterize(a.res, a.res, fz, fi, FEAT[None].contiguous())
    out = out[0]; mask = fidx[0] >= 0
    rgb = torch.ones(a.res, a.res, 3, device=dev)
    for ti, T_ in enumerate(TEXT):
        sel = mask & (out[..., 2].round() == ti)
        if sel.any():
            uv = out[..., :2][sel]
            col = texture_mapping(uv[None, :, None, :], T_, mode="bilinear")[0, :, 0]
            rgb[sel] = (col * out[..., 3][sel][:, None]).clamp(0, 1)
    if a.flip_y:
        rgb, mask = rgb.flip(0), mask.flip(0)
    return rgb, mask


def cams(n, seed):
    r = np.random.default_rng(seed)
    out = []
    for i in range(n):
        az = 2 * math.pi * (i + r.random()) / n
        el = math.radians(r.uniform(-15, 75))
        out.append(np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)]) * R0)
    return out


os.makedirs(a.out, exist_ok=True)
for split, n, seed in (("train", a.n_train, 0), ("test", a.n_test, 1)):
    os.makedirs(f"{a.out}/{split}", exist_ok=True)
    frames = []
    for i, eye in enumerate(tqdm(cams(n, seed), desc=split)):
        up = np.array([0.0, 0.0, 1.0])
        f = -eye / np.linalg.norm(eye)                                      # 앞 = 원점 쪽
        rgt = np.cross(f, up); rgt /= np.linalg.norm(rgt); u = np.cross(rgt, f)
        c2w = np.eye(4); c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = rgt, u, -f, eye   # Blender/OpenGL
        rgb, al = render(c2w)
        rgba = torch.cat([rgb, al.float()[..., None]], -1)
        imageio.imwrite(f"{a.out}/{split}/r_{i}.png", (rgba.cpu().numpy() * 255).astype(np.uint8))
        frames.append(dict(file_path=f"./{split}/r_{i}", transform_matrix=c2w.tolist()))
    json.dump(dict(camera_angle_x=fov, frames=frames), open(f"{a.out}/transforms_{split}.json", "w"), indent=1)
json.dump(dict(gltf=a.gltf, nodes=[p[0] for p in parts], center_yup_to_zup=c.tolist(), scale=s,
               real_extent_m=((allv.max(0) - allv.min(0))).tolist()), open(f"{a.out}/meta.json", "w"), indent=1)
print(f"[완료] {a.out}  배율 {s:.4f} (실제 크기 {np.round(allv.max(0) - allv.min(0), 3)} m)", flush=True)
